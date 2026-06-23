"""Geometry-conditioned pix2pixHD model for v5 (+ local piece-crop objective).

Generator input:
  concat(A_synthetic RGB, onehot(seg), geom)

Stage-A piece-legibility fix
----------------------------
The board/occupancy is solved but pieces render as smeared blobs. The global
losses (GAN + feature-matching + VGG + masked L1) reward average texture and
global realism, not per-type 3D piece shape. The frozen square-classifier loss is
weak and gameable (runs only on the generator output, no class balancing, rewards
a "classifiable blob").

This model adds a LOCAL, CLASS-CONDITIONAL objective operating on occupied-square
crops, sampled with class balancing so the rare/confused tall pieces (R,B,Q,K and
black rook) actually get gradient:

  * netDp  : class-conditional PatchGAN on 96px occupied-square crops. Real = crop
             from real_B, fake = crop from fake_B, conditioned (concat) on the
             piece class one-hot. Punishes blobs that pass global realism but are
             not recognizable as the correct type.
  * G_PGAN : generator fools netDp on its crops.
  * G_PFM  : feature matching between netDp(fake crop) and netDp(real crop) -- the
             alignment-tolerant structure term that fights blur.
  * G_PVGG : VGG perceptual on the crops (mid-frequency piece structure).
  * G_PCLS : the frozen classifier loss, now on the SAME class-balanced crop subset
             and down-weighted (a light direct identity anchor).

All of the above is TRAIN ONLY. Inference is unchanged: G consumes the single
synthetic render + seg/geom derived from the same FEN/synthetic state.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base_model import BaseModel
from . import networks
from .paired_hd_model import VGGPerceptualLoss, MultiscaleDiscriminator, NLayerDiscriminatorFeat


CLASS_FROM_SEG = {
    3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6,
    9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12,
}

# Piece class index 0..11 (seg id - 3) -> name, for the hard-class boost.
#   0:P 1:N 2:B 3:R 4:Q 5:K  6:p 7:n 8:b 9:r 10:q 11:k
# Boost the confused/low-recall + worst double-head classes (independent visual review
# of silABC): BLACK officers double-head worst -> black knight n=7, black queen q=10,
# black king k=11; plus prior black rook r=9, black bishop b=8, white B/R/Q.
HARD_CLASS_BOOST = torch.tensor(
    [1.0, 1.0, 1.6, 1.6, 1.6, 1.0, 1.0, 1.8, 1.4, 2.0, 1.8, 1.8], dtype=torch.float32)

# Per-class local-crop window (px) around the cell centre. Tall pieces need a larger
# window so their leaning crown/mitre is captured (the old fixed 112 clipped them).
#                P    N    B    R    Q    K    p    n    b    r    q    k
WIN_BY_CLASS = [112, 128, 160, 128, 160, 160, 112, 128, 160, 128, 160, 160]


def resnet18_square_classifier(num_classes=13):
    try:
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    except TypeError:
        return torchvision.models.resnet18(pretrained=False, num_classes=num_classes)


class FrozenPieceClassifierLoss(nn.Module):
    """Frozen real-square classifier. Now scores a PRE-CROPPED, class-balanced set
    of 64px crops (cropping/sampling is done once in the model)."""

    def __init__(self, ckpt, device):
        super().__init__()
        self.net = resnet18_square_classifier().to(device)
        self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, crops64, classes):
        # crops64: K,3,64,64 in [-1,1]; classes: K piece-class ids 0..11.
        if crops64.numel() == 0:
            return crops64.new_tensor(0.0)
        logits = self.net(crops64)
        targets = classes + 1  # classifier label space: 0=empty, 1..12=pieces
        return self.ce(logits, targets)


class PairedGeomHDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--lambda_GAN", type=float, default=1.0)
        parser.add_argument("--lambda_feat", type=float, default=10.0)
        parser.add_argument("--lambda_VGG", type=float, default=3.0)
        parser.add_argument("--lambda_L1", type=float, default=10.0)
        parser.add_argument("--lambda_piece_cls", type=float, default=0.5)
        parser.add_argument("--l1_piece_w", type=float, default=0.1)
        parser.add_argument("--lambda_edge", type=float, default=0.0,
                            help="Fix B: Sobel boundary loss on the silhouette edge band "
                                 "(sharpens piece outlines; anti-halo / anti-transparent-crown)")
        parser.add_argument("--l1_head_w", type=float, default=0.0,
                            help="Fix F: extra L1 weight on the TOP of each silhouette "
                                 "(crown/mitre/horse-head) -> anti-vanishing-head")
        parser.add_argument("--l1_shadow_w", type=float, default=0.0,
                            help="Fix F: extra L1 weight on the band just BELOW the piece "
                                 "base -> contact-shadow grounding")
        parser.add_argument("--head_px", type=int, default=20,
                            help="rows from the top of each silhouette column counted as head")
        parser.add_argument("--shadow_px", type=int, default=6,
                            help="rows below the piece base counted as contact-shadow band")
        parser.add_argument("--num_seg_classes", type=int, default=15)
        parser.add_argument("--num_geom_channels", type=int, default=3)
        parser.add_argument("--num_D", type=int, default=2)
        parser.add_argument("--vgg_max", type=int, default=256)
        parser.add_argument("--g_init_path", type=str, default="")
        parser.add_argument("--piece_cls_path", type=str, default="./checkpoints/square_eval.pth")
        # --- local piece-crop objective ---
        parser.add_argument("--lambda_piece_gan", type=float, default=1.0,
                            help="weight of the local class-conditional crop GAN (G side)")
        parser.add_argument("--lambda_piece_fm", type=float, default=10.0,
                            help="weight of local-crop discriminator feature matching")
        parser.add_argument("--lambda_piece_vgg", type=float, default=5.0,
                            help="weight of VGG perceptual on occupied-square crops")
        parser.add_argument("--piece_crop_size", type=int, default=96,
                            help="output resolution of crops fed to netDp / crop VGG")
        parser.add_argument("--piece_win", type=int, default=112,
                            help="pixel window around a cell centre (captures sheared body)")
        parser.add_argument("--piece_crops_max", type=int, default=64,
                            help="max occupied-square crops per optimisation step")
        parser.add_argument("--piece_class_balance", type=int, default=1,
                            help="1 = class-balanced + hard-class boosted crop sampling")
        parser.add_argument("--piece_ndf", type=int, default=64)
        # --- misalignment-robust Contextual Loss (Mechrez 2018), fake_B vs real_B ---
        parser.add_argument("--lambda_cx", type=float, default=0.0,
                            help="weight of Contextual Loss on VGG features (robust to shift)")
        parser.add_argument("--cx_level", type=int, default=2,
                            help="VGG slice index (0..4) for CX features; 2 ~= relu3")
        parser.add_argument("--cx_feat_size", type=int, default=48,
                            help="CX feature maps are pooled to this HxW to bound O((HW)^2) memory")
        parser.add_argument("--cx_band_h", type=float, default=0.5,
                            help="CX bandwidth h (smaller = stricter best-match)")
        parser.add_argument("--lambda_piece_cx", type=float, default=0.0,
                            help="crop-level Contextual Loss on the 96px piece crops: fine "
                                 "resolution where the global CX is too coarse to resolve an "
                                 "intra-piece double-lobe / ghost head (anti-double-head)")
        parser.add_argument("--lambda_piece_shape", type=float, default=0.0,
                            help="crop-level soft piece-mask loss. Extracts a differentiable "
                                 "foreground saliency mask from fake/real crops using contrast "
                                 "against the crop border, then matches fake mask to real mask. "
                                 "This targets internal blob/topology without pasting synthetic RGB.")
        parser.add_argument("--piece_shape_size", type=int, default=48,
                            help="spatial size for the soft piece-mask loss")
        parser.add_argument("--piece_shape_thresh", type=float, default=0.10,
                            help="foreground contrast threshold in normalized [-1,1] crop units")
        parser.add_argument("--piece_shape_temp", type=float, default=0.035,
                            help="softness of the foreground sigmoid for piece-shape loss")
        parser.add_argument("--piece_shape_out_w", type=float, default=1.5,
                            help="extra penalty weight for fake foreground outside real foreground")
        parser.add_argument("--lambda_piece_src_shape", type=float, default=0.0,
                            help="crop-level source-silhouette shape loss. Extracts a "
                                 "texture-agnostic foreground mask from fake crops and "
                                 "penalizes salient generated structure outside the rendered "
                                 "piece silhouette, with optional weak coverage inside it. "
                                 "This constrains topology without pasting synthetic RGB.")
        parser.add_argument("--piece_src_shape_in_w", type=float, default=0.25,
                            help="coverage penalty weight for missing fake foreground inside "
                                 "the rendered silhouette")
        parser.add_argument("--piece_src_shape_out_w", type=float, default=3.0,
                            help="penalty weight for fake foreground outside the rendered "
                                 "silhouette")
        parser.add_argument("--piece_src_shape_blur", type=int, default=1,
                            help="average-pool radius for softening the rendered silhouette "
                                 "target in source-shape crop loss")
        parser.add_argument("--cx_crop_size", type=int, default=16,
                            help="feature HxW for crop-level CX (bounds memory over many crops)")
        parser.add_argument("--geom_lock_alpha", type=float, default=0.0,
                            help="0=normal generator output. >0 blends piece regions toward a "
                                 "geometry-locked stylized synthetic render using the true "
                                 "synthetic silhouette; intended to make extra heads/lobes "
                                 "geometrically impossible while retaining generated board style.")
        parser.add_argument("--geom_lock_feather", type=int, default=0,
                            help="average-pool radius for softening the geometry-lock alpha mask")
        parser.add_argument("--geom_lock_blur", type=int, default=0,
                            help="average-pool radius for softening the locked synthetic piece")
        parser.add_argument("--geom_lock_contrast", type=float, default=0.45,
                            help="piece contrast retained after matching synthetic piece stats to "
                                 "the raw generator output stats")
        parser.add_argument("--piece_comp_alpha", type=float, default=0.0,
                            help="0=normal RGB generator. >0 makes G output two RGB streams "
                                 "(background, foreground piece) and composites them with the "
                                 "true synthetic silhouette as alpha. Unlike geom_lock, no "
                                 "synthetic RGB texture is pasted into the output.")
        parser.add_argument("--piece_comp_feather", type=int, default=1,
                            help="average-pool radius for softening the piece-composite alpha mask")
        parser.set_defaults(dataset_mode="geom_aligned", direction="AtoB",
                            netG="resnet_9blocks", ngf=64, normG="instance", no_dropout=True,
                            gan_mode="lsgan", pool_size=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.K = opt.num_seg_classes
        self.GEOM = opt.num_geom_channels
        self.n_piece = 12
        # Which optional objectives are active (lets one model serve every ablation).
        self.use_pieceD = self.isTrain and (opt.lambda_piece_gan > 0 or opt.lambda_piece_fm > 0)
        self.use_piece = self.isTrain and (self.use_pieceD or opt.lambda_piece_vgg > 0
                                           or opt.lambda_piece_cls > 0
                                           or opt.lambda_piece_cx > 0
                                           or opt.lambda_piece_shape > 0
                                           or opt.lambda_piece_src_shape > 0)
        self.loss_names = ["G_GAN", "G_FM", "G_VGG", "G_L1", "G_edge", "G_CX", "D_real", "D_fake"]
        if self.use_piece:
            self.loss_names += ["G_PGAN", "G_PFM", "G_PVGG", "G_PCLS", "G_PCX", "G_PSHAPE", "G_PSRC"]
        if self.use_pieceD:
            self.loss_names += ["Dp_real", "Dp_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if self.isTrain:
            self.model_names = ["G", "D"] + (["Dp"] if self.use_pieceD else [])
        else:
            self.model_names = ["G"]

        g_in = opt.input_nc + self.K + self.GEOM
        g_out = opt.output_nc * 2 if getattr(opt, "piece_comp_alpha", 0.0) > 0 else opt.output_nc
        self.netG = networks.define_G(
            g_in, g_out, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain and getattr(opt, "g_init_path", ""):
            self._load_g_init(opt.g_init_path)

        if self.isTrain:
            netD = MultiscaleDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, opt.num_D)
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionFeat = nn.L1Loss()
            # One VGG instance: max_size=256 leaves the 96px crops untouched, so the
            # same module serves both the whole-image and crop perceptual losses.
            self.criterionVGG = VGGPerceptualLoss(self.device, max_size=opt.vgg_max)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

            # Local class-conditional crop discriminator (only if its losses are on).
            if self.use_pieceD:
                netDp = NLayerDiscriminatorFeat(opt.output_nc + self.n_piece, opt.piece_ndf, n_layers=3)
                self.netDp = networks.init_net(netDp, opt.init_type, opt.init_gain, self.gpu_ids)
                self.optimizer_Dp = torch.optim.Adam(self.netDp.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers += [self.optimizer_Dp]

            self.criterionPiece = None
            if opt.lambda_piece_cls > 0:
                if not os.path.exists(opt.piece_cls_path):
                    raise FileNotFoundError(opt.piece_cls_path)
                self.criterionPiece = FrozenPieceClassifierLoss(opt.piece_cls_path, self.device)

            self.hard_boost = HARD_CLASS_BOOST.to(self.device)

    def _load_g_init(self, path):
        sd = torch.load(path, map_location=self.device)
        if hasattr(sd, "_metadata"):
            del sd._metadata
        net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        own = net.state_dict()
        filt = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        # If the experimental foreground/background composite head is enabled, netG
        # ends in 6 channels instead of 3. Warm-start both output streams from the
        # old RGB head so the initial composite is close to the baseline generator,
        # instead of starting the final layer from random noise.
        expanded = 0
        for k, v in sd.items():
            if k in filt or k not in own:
                continue
            target = own[k]
            if target.ndim >= 1 and v.ndim == target.ndim and target.shape[0] == v.shape[0] * 2:
                if tuple(target.shape[1:]) == tuple(v.shape[1:]):
                    filt[k] = torch.cat([v, v], dim=0)
                    expanded += 1
        missing, _ = net.load_state_dict(filt, strict=False)
        print(f"[paired_geom_hd] warm-started G from {path}: loaded {len(filt)}/{len(sd)} tensors; "
              f"expanded={expanded}; missing={len(missing)}")

    def data_dependent_initialize(self, data):
        return

    def parallelize(self):
        return

    def save_networks(self, epoch):
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            save_path = os.path.join(self.save_dir, "%s_net_%s.pth" % (epoch, name))
            net = getattr(self, "net" + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.geom = input["geom"].to(self.device)
        self.seg = input["seg"].to(self.device).long()
        onehot = F.one_hot(self.seg.clamp(0, self.K - 1), self.K).permute(0, 3, 1, 2).float()
        self.gen_input = torch.cat([self.real_A, onehot, self.geom], dim=1)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_raw_B = self.netG(self.gen_input)
        self.fake_B = self._piece_composite_output(self.fake_raw_B)
        self.fake_B = self._geometry_locked_output(self.fake_B)

    def _avg_blur(self, x, radius):
        if radius <= 0:
            return x
        k = radius * 2 + 1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius)

    def _masked_stats(self, x, mask):
        cnt = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        mean = (x * mask).sum(dim=(2, 3), keepdim=True) / cnt
        var = (((x - mean) * mask) ** 2).sum(dim=(2, 3), keepdim=True) / cnt
        return mean, torch.sqrt(var + 1e-4), cnt

    def _piece_composite_output(self, raw):
        alpha_w = getattr(self.opt, "piece_comp_alpha", 0.0)
        if alpha_w <= 0:
            return raw
        if raw.shape[1] < self.opt.output_nc * 2:
            return raw[:, :self.opt.output_nc]
        bg = raw[:, :self.opt.output_nc]
        fg = raw[:, self.opt.output_nc:self.opt.output_nc * 2]
        sil = (self.geom[:, 1:2] > 0).float()
        alpha = self._avg_blur(sil, int(getattr(self.opt, "piece_comp_feather", 1))).clamp(0.0, 1.0)
        alpha = (alpha * alpha_w).clamp(0.0, 1.0)
        return fg * alpha + bg * (1.0 - alpha)

    def _geometry_locked_output(self, raw):
        """Blend piece regions toward a source-geometry-preserving stylization.

        The generator still controls the board/background and piece color statistics, but
        the high-contrast structure inside each visible synthetic silhouette comes from
        real_A. This is a model-integrated version of the local geometry-locked composite
        probe that cut split-cue defects by more than half.
        """
        alpha_w = getattr(self.opt, "geom_lock_alpha", 0.0)
        if alpha_w <= 0:
            return raw
        sil = (self.geom[:, 1:2] > 0).float()
        if sil.sum().item() < 1:
            return raw

        alpha = self._avg_blur(sil, int(getattr(self.opt, "geom_lock_feather", 0))).clamp(0.0, 1.0)
        alpha = (alpha * alpha_w).clamp(0.0, 1.0)
        styled = raw
        # Match per-piece-class stats so black/white pieces do not share a palette.
        for sid in range(3, 15):
            m = ((self.seg == sid).unsqueeze(1).float() * sil)
            if m.sum().item() < 8:
                continue
            src_mean, src_std, _ = self._masked_stats(self.real_A, m)
            dst_mean, dst_std, _ = self._masked_stats(raw, m)
            cls_styled = (self.real_A - src_mean) / src_std
            cls_styled = cls_styled * (dst_std * self.opt.geom_lock_contrast) + dst_mean
            styled = cls_styled * m + styled * (1.0 - m)
        styled = self._avg_blur(styled, int(getattr(self.opt, "geom_lock_blur", 0))).clamp(-1.0, 1.0)
        return styled * alpha + raw * (1.0 - alpha)

    def _gan_over_scales(self, preds, target_is_real):
        loss = 0.0
        for scale in preds:
            loss = loss + self.criterionGAN(scale[-1], target_is_real).mean()
        return loss / len(preds)

    def _sobel_mag(self, x):
        """Luminance Sobel-gradient magnitude: [N,3,H,W] in [-1,1] -> [N,1,H,W]."""
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        kx = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        gx = F.conv2d(lum, kx, padding=1)
        gy = F.conv2d(lum, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    # ------------------------------------------------------------------ #
    # Occupied-square crop sampling (class-balanced, hard-class boosted)
    # ------------------------------------------------------------------ #
    def _occupied_cells(self):
        """List of (batch_idx, row, col, piece_class 0..11) for occupied cells."""
        n, h, w = self.seg.shape
        cells = []
        if h != 512 or w != 512:
            return cells
        # Under the silhouette-shaped seg (fen_silhouette) the cell-CENTRE pixel may be
        # empty or covered by a leaning neighbour's overhang -> the old centre-pixel test
        # dropped/misclassified pieces. Use the MOST COMMON piece id in the cell's BASE
        # region (bottom-centre, where the piece stands and where the FEN base-patch
        # guardrail writes the true id) for a robust per-cell class.
        for bi in range(n):
            for row in range(8):
                for col in range(8):
                    base = self.seg[bi, row * 64 + 38:row * 64 + 64, col * 64 + 16:col * 64 + 48]
                    vals = base[base >= 3]
                    if vals.numel() == 0:
                        continue
                    sid = int(torch.bincount(vals.flatten()).argmax().item())
                    cells.append((bi, row, col, sid - 3))
        return cells

    def _sample_cells(self, cells):
        """Class-balanced + hard-class-boosted subsample, capped at piece_crops_max."""
        if not cells:
            return cells
        if not self.opt.piece_class_balance:
            if len(cells) > self.opt.piece_crops_max:
                idx = torch.randperm(len(cells))[:self.opt.piece_crops_max].tolist()
                return [cells[i] for i in idx]
            return cells
        classes = torch.tensor([c[3] for c in cells])
        freq = torch.bincount(classes, minlength=12).float()
        # sampling prob ~ boost / freq  (oversample rare + confused classes)
        p = self.hard_boost.cpu()[classes] / freq[classes].clamp(min=1.0)
        n_take = min(self.opt.piece_crops_max, len(cells))
        idx = torch.multinomial(p, n_take, replacement=False).tolist()
        return [cells[i] for i in idx]

    def _crops(self, img, cells, out_size):
        """Per-class crop window around each cell centre -> (K,3,out,out).

        Tall pieces (B,Q,K) use a bigger window so the crown/mitre — which the old
        fixed 112px window clipped (causing the Stage-A queen/king regression) — is
        captured even though the rectified piece leans outward. Each crop is resized
        to out_size individually before stacking."""
        maxwin = max(WIN_BY_CLASS)
        half = maxwin // 2
        padded = F.pad(img, (half, half, half, half), mode="replicate")
        crops = []
        for (bi, row, col, cls) in cells:
            h = WIN_BY_CLASS[cls] // 2
            cy = row * 64 + 32 + half
            cx = col * 64 + 32 + half
            crop = padded[bi:bi + 1, :, cy - h:cy + h, cx - h:cx + h]
            if crop.shape[-1] != out_size:
                crop = F.interpolate(crop, size=(out_size, out_size), mode="bilinear", align_corners=False)
            crops.append(crop)
        return torch.cat(crops, dim=0)

    def _piece_cond(self, cells, out_size):
        classes = torch.tensor([c[3] for c in cells], device=self.device)
        onehot = F.one_hot(classes, self.n_piece).float()
        cond = onehot[:, :, None, None].expand(-1, -1, out_size, out_size)
        return classes, cond

    # ------------------------------------------------------------------ #
    # Discriminators
    # ------------------------------------------------------------------ #
    def compute_D_loss(self):
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self._gan_over_scales(pred_fake, False)
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self._gan_over_scales(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_Dp_loss(self):
        cells = self._cells_sub
        if not cells:
            self.loss_Dp_real = self.fake_B.new_tensor(0.0)
            self.loss_Dp_fake = self.fake_B.new_tensor(0.0)
            self.loss_Dp = self.fake_B.new_tensor(0.0)
            return self.loss_Dp
        sz = self.opt.piece_crop_size
        _, cond = self._piece_cond(cells, sz)
        real_c = self._crops(self.real_B, cells, sz)
        fake_c = self._crops(self.fake_B.detach(), cells, sz)
        logit_r = self.netDp(torch.cat([real_c, cond], dim=1))[-1]
        logit_f = self.netDp(torch.cat([fake_c, cond], dim=1))[-1]
        self.loss_Dp_real = ((logit_r - 1.0) ** 2).mean()
        self.loss_Dp_fake = (logit_f ** 2).mean()
        self.loss_Dp = 0.5 * (self.loss_Dp_real + self.loss_Dp_fake)
        return self.loss_Dp

    # ------------------------------------------------------------------ #
    # Generator
    # ------------------------------------------------------------------ #
    def compute_G_loss(self):
        pred_fake = self.netD(self.fake_B)
        with torch.no_grad():
            pred_real = self.netD(self.real_B)
        self.loss_G_GAN = self._gan_over_scales(pred_fake, True) * self.opt.lambda_GAN

        fm = 0.0
        feat_w = 4.0 / (self.opt.n_layers_D + 1)
        d_w = 1.0 / self.opt.num_D
        for i in range(len(pred_fake)):
            for j in range(len(pred_fake[i]) - 1):
                fm = fm + d_w * feat_w * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        self.loss_G_FM = fm * self.opt.lambda_feat

        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_VGG

        # Fix B: silhouette-masked L1. geom ch.1 is the rendered piece silhouette
        # (piece*2-1, so >0 == piece). Weighting the TRUE silhouette (not the full
        # 64px cell) lets us raise piece L1 (opaque heads) while the cell's empty
        # surround returns to full weight (kills the square halo / wood bleed) with
        # no full-cell blur. Decoupled from the semantic source.
        sil = (self.geom[:, 1:2] > 0).float()                       # [N,1,H,W]
        wmap = 1.0 + (self.opt.l1_piece_w - 1.0) * sil              # empty=1.0, piece=l1_piece_w
        # Fix F: extra L1 on the TOP of each piece (anti-vanishing-head). cumsum down each
        # column counts silhouette pixels seen; the first head_px of them = the crown/mitre.
        if self.opt.l1_head_w > 0:
            cum = torch.cumsum(sil, dim=2)
            head = sil * (cum <= self.opt.head_px).float()
            wmap = wmap + self.opt.l1_head_w * head
        # Fix F: extra L1 on the band just BELOW the piece base (contact-shadow grounding):
        # shift the silhouette DOWN by shadow_px and keep the part that is not piece.
        if self.opt.l1_shadow_w > 0:
            k = self.opt.shadow_px
            shifted = F.pad(sil, (0, 0, k, 0))[:, :, :sil.shape[2], :]   # silhouette moved down k rows
            shadow = (shifted > 0).float() * (1.0 - sil)
            wmap = wmap + self.opt.l1_shadow_w * shadow
        self.loss_G_L1 = (wmap * (self.fake_B - self.real_B).abs()).mean() * self.opt.lambda_L1

        # Fix B: Sobel boundary loss on the silhouette edge band -> sharper piece
        # outlines, less halo / transparent-crown.
        if self.opt.lambda_edge > 0:
            band = (self.geom[:, 2:3] > 0).float()                  # silhouette boundary
            band = F.max_pool2d(band, kernel_size=5, stride=1, padding=2)  # widen the band
            ediff = (self._sobel_mag(self.fake_B) - self._sobel_mag(self.real_B)).abs()
            self.loss_G_edge = (band * ediff).sum() / band.sum().clamp(min=1.0) * self.opt.lambda_edge
        else:
            self.loss_G_edge = self.fake_B.new_tensor(0.0)

        # ----- misalignment-robust Contextual Loss (fake_B vs real_B) -----
        if self.opt.lambda_cx > 0:
            self.loss_G_CX = self._contextual_loss(self.fake_B, self.real_B) * self.opt.lambda_cx
        else:
            self.loss_G_CX = self.fake_B.new_tensor(0.0)

        # ----- local piece-crop objective -----
        self._compute_piece_losses()

        self.loss_G = (self.loss_G_GAN + self.loss_G_FM + self.loss_G_VGG + self.loss_G_L1
                       + self.loss_G_edge + self.loss_G_CX + self.loss_G_PCLS + self.loss_G_PGAN
                       + self.loss_G_PFM + self.loss_G_PVGG + self.loss_G_PCX + self.loss_G_PSHAPE
                       + self.loss_G_PSRC)
        return self.loss_G

    # ------------------------------------------------------------------ #
    # Contextual Loss (Mechrez et al. 2018): treats VGG features as an
    # unaligned bag, so residual piece-level shift does not force blur.
    # ------------------------------------------------------------------ #
    def _vgg_feat(self, x, level):
        v = self.criterionVGG
        h = v._prep(x)
        for i in range(level + 1):
            h = v.slices[i](h)
        return h

    def _contextual_loss(self, fake, real, feat_size=None):
        fx = self._vgg_feat(fake, self.opt.cx_level)
        fy = self._vgg_feat(real, self.opt.cx_level).detach()
        s = feat_size if feat_size is not None else self.opt.cx_feat_size
        if fx.shape[-1] > s:
            fx = F.interpolate(fx, size=(s, s), mode="bilinear", align_corners=False)
            fy = F.interpolate(fy, size=(s, s), mode="bilinear", align_corners=False)
        n, c, h, w = fx.shape
        # centre by target mean, then L2-normalise channels -> cosine geometry
        mu = fy.mean(dim=(2, 3), keepdim=True)
        fx = F.normalize(fx - mu, dim=1).reshape(n, c, -1)
        fy = F.normalize(fy - mu, dim=1).reshape(n, c, -1)
        cos = torch.einsum("nca,ncb->nab", fx, fy)           # n, HWx, HWy in [-1,1]
        d = (1.0 - cos).clamp(min=0.0)
        d_tilde = d / (d.min(dim=2, keepdim=True)[0] + 1e-5)
        w_ij = torch.exp((1.0 - d_tilde) / self.opt.cx_band_h)
        cx = w_ij / (w_ij.sum(dim=2, keepdim=True) + 1e-5)   # affinity x->y
        cx = cx.max(dim=2)[0].mean(dim=1)                     # per-image CX score
        return -torch.log(cx + 1e-5).mean()

    def _compute_piece_losses(self):
        zero = self.fake_B.new_tensor(0.0)
        self.loss_G_PGAN = zero.clone()
        self.loss_G_PFM = zero.clone()
        self.loss_G_PVGG = zero.clone()
        self.loss_G_PCLS = zero.clone()
        self.loss_G_PCX = zero.clone()
        self.loss_G_PSHAPE = zero.clone()
        self.loss_G_PSRC = zero.clone()
        if not self.use_piece:
            return
        cells = self._cells_sub
        if not cells:
            return
        sz = self.opt.piece_crop_size
        classes, cond = self._piece_cond(cells, sz)
        real_c = self._crops(self.real_B, cells, sz)
        fake_c = self._crops(self.fake_B, cells, sz)

        if self.use_pieceD:
            feats_f = self.netDp(torch.cat([fake_c, cond], dim=1))
            with torch.no_grad():
                feats_r = self.netDp(torch.cat([real_c, cond], dim=1))
            # adversarial: fool netDp
            self.loss_G_PGAN = ((feats_f[-1] - 1.0) ** 2).mean() * self.opt.lambda_piece_gan
            # feature matching against real crop features (anti-blur structure term)
            fm = 0.0
            for j in range(len(feats_f) - 1):
                fm = fm + self.criterionFeat(feats_f[j], feats_r[j].detach())
            self.loss_G_PFM = fm * self.opt.lambda_piece_fm
        # crop perceptual (96px < vgg max_size, so no downscaling)
        if self.opt.lambda_piece_vgg > 0:
            self.loss_G_PVGG = self.criterionVGG(fake_c, real_c) * self.opt.lambda_piece_vgg
        # frozen-classifier identity anchor on the same balanced subset
        if self.criterionPiece is not None:
            fake_c64 = F.interpolate(fake_c, size=(64, 64), mode="bilinear", align_corners=False)
            self.loss_G_PCLS = self.criterionPiece(fake_c64, classes) * self.opt.lambda_piece_cls
        # crop-level Contextual Loss: fine resolution on the 96px crop, where the global CX
        # is too coarse to resolve an intra-piece double-lobe / ghost head (esp. pawns +
        # black officers). Misalignment-robust, so it does not blur.
        if self.opt.lambda_piece_cx > 0:
            self.loss_G_PCX = self._contextual_loss(
                fake_c, real_c, feat_size=self.opt.cx_crop_size) * self.opt.lambda_piece_cx
        # crop-level pseudo-mask shape loss. This is deliberately texture-agnostic:
        # it compares where fake/real crops have foreground contrast relative to
        # their own border-estimated board background, so it can penalise extra
        # lobes/blobs without forcing the synthetic render's RGB or shading.
        if self.opt.lambda_piece_shape > 0:
            fake_m = self._soft_piece_mask(fake_c, self.opt.piece_shape_size)
            with torch.no_grad():
                real_m = self._soft_piece_mask(real_c, self.opt.piece_shape_size)
            l1 = (fake_m - real_m).abs().mean()
            outside = (fake_m * (1.0 - real_m)).mean()
            self.loss_G_PSHAPE = (l1 + self.opt.piece_shape_out_w * outside) * self.opt.lambda_piece_shape
        # Source-silhouette shape guardrail. This does not copy source RGB; it only asks
        # the fake crop's own contrast mask not to grow extra salient lobes outside the
        # rendered piece silhouette. The inside term is intentionally weak by default so
        # the model can keep real-looking shading instead of recreating Blender texture.
        if self.opt.lambda_piece_src_shape > 0:
            fake_m = self._soft_piece_mask(fake_c, self.opt.piece_shape_size)
            with torch.no_grad():
                src_m = self._source_piece_mask_crops(cells, self.opt.piece_shape_size)
            outside = (fake_m * (1.0 - src_m)).mean()
            missing = ((1.0 - fake_m) * src_m).mean()
            self.loss_G_PSRC = (
                self.opt.piece_src_shape_out_w * outside
                + self.opt.piece_src_shape_in_w * missing
            ) * self.opt.lambda_piece_src_shape

    def _soft_piece_mask(self, crops, out_size):
        n, c, h, w = crops.shape
        b = max(4, min(h, w) // 12)
        border = crops.new_zeros((1, 1, h, w))
        border[:, :, :b, :] = 1.0
        border[:, :, -b:, :] = 1.0
        border[:, :, :, :b] = 1.0
        border[:, :, :, -b:] = 1.0
        cnt = border.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        bg = (crops * border).sum(dim=(2, 3), keepdim=True) / cnt
        contrast = (crops - bg).abs().mean(dim=1, keepdim=True)
        contrast = F.avg_pool2d(contrast, kernel_size=5, stride=1, padding=2)
        if contrast.shape[-1] != out_size:
            contrast = F.interpolate(contrast, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return torch.sigmoid((contrast - self.opt.piece_shape_thresh) / self.opt.piece_shape_temp)

    def _source_piece_mask_crops(self, cells, out_size):
        sil = (self.geom[:, 1:2] > 0).float()
        crops = self._crops(sil, cells, out_size).clamp(0.0, 1.0)
        radius = int(getattr(self.opt, "piece_src_shape_blur", 1))
        if radius > 0:
            crops = self._avg_blur(crops, radius)
        return crops.clamp(0.0, 1.0)

    def optimize_parameters(self):
        self.forward()
        # one balanced occupied-crop subset for this step (shared by Dp and G)
        self._cells_sub = self._sample_cells(self._occupied_cells()) if self.use_piece else []

        # global discriminator
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.compute_D_loss().backward()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)

        # local piece discriminator (optional)
        if self.use_pieceD:
            self.set_requires_grad(self.netDp, True)
            self.optimizer_Dp.zero_grad()
            self.compute_Dp_loss().backward()
            self.optimizer_Dp.step()
            self.set_requires_grad(self.netDp, False)

        # generator
        self.optimizer_G.zero_grad()
        self.compute_G_loss().backward()
        self.optimizer_G.step()
