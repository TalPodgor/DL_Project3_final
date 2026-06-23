"""
square_eval.py -- independent per-square correctness evaluator (Wave 5, M3).

Trains a ResNet-18 to classify a board-square crop into 13 classes
(empty + {P,N,B,R,Q,K} x {white,black}) using REAL photos (board-rectified canonical
512) with labels from the FEN. Then SCORES generated images: occupancy accuracy,
phantom rate, missing rate, piece-type accuracy, whole-board exact match, confusion.

This is decoupled from the generator -- it measures correctness objectively (priority #1).

Train:
  conda run -n pytorch python square_eval.py --mode train \
     --data ./datasets/chess_paired_v3 --out ./checkpoints/square_eval.pth
Score generated images (real_B-style filenames -> FEN via labels.json):
  conda run -n pytorch python square_eval.py --mode score \
     --data ./datasets/chess_paired_v3 --ckpt ./checkpoints/square_eval.pth \
     --images ./results/chess_segv3/test_latest/images/fake_B --out report.json
"""
import os, re, json, glob, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

CLASSES = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
CIDX = {c: i for i, c in enumerate(CLASSES)}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CROP = 112  # window around a cell centre (captures sheared piece body)

def cell_center(f, rank, vp, S=512):
    sq = S / 8.0
    if vp == 'white':
        return (f + 0.5) * sq, (8 - rank + 0.5) * sq
    return (7 - f + 0.5) * sq, (rank - 1 + 0.5) * sq

def fen_grid(fen):
    g = {}
    for ri, row in enumerate(fen.split()[0].split('/')):
        rank = 8 - ri; f = 0
        for ch in row:
            if ch.isdigit(): f += int(ch)
            else: g[(f, rank)] = ch; f += 1
    return g  # (file,rank)->char

def crop_square(img, f, rank, vp):
    cx, cy = cell_center(f, rank, vp, img.width)
    half = CROP // 2
    box = (cx - half, cy - half, cx + half, cy + half)
    c = img.crop(box).resize((64, 64), Image.BILINEAR)
    return c

def squares_of(img, vp):
    """yield (file,rank, 64x64 RGB crop) for all 64 squares."""
    for rank in range(1, 9):
        for f in range(8):
            yield f, rank, crop_square(img, f, rank, vp)

def to_tensor(pil):
    a = np.asarray(pil.convert('RGB'), np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)

# ---------------- data ----------------
def build_crops(data, split, labels, half='B'):
    """returns (X tensor N,3,64,64 ; y tensor N) from real B-halves of a split."""
    X, Y = [], []
    for p in sorted(glob.glob(os.path.join(data, split, '*.png'))):
        if p.endswith('_seg.png'): continue
        nm = os.path.basename(p)[:-4]
        if nm not in labels: continue
        vp = labels[nm]['viewpoint']; grid = fen_grid(labels[nm]['fen'])
        comb = Image.open(p).convert('RGB'); W = comb.width // 2
        img = comb.crop((W, 0, 2 * W, comb.height)) if half == 'B' else comb.crop((0, 0, W, comb.height))
        for f, rank, c in squares_of(img, vp):
            ch = grid.get((f, rank), None)
            Y.append(CIDX[ch] if ch else 0); X.append(to_tensor(c))
    return torch.stack(X), torch.tensor(Y, dtype=torch.long)

def net():
    try:
        m = torchvision.models.resnet18(weights=None, num_classes=len(CLASSES))
    except TypeError:                      # older torchvision (torch 1.10): no `weights=`
        m = torchvision.models.resnet18(pretrained=False, num_classes=len(CLASSES))
    return m.to(DEV)

# ---------------- train ----------------
def train(data, out, epochs=18):
    labels = json.load(open(os.path.join(data, 'labels.json')))
    Xtr, ytr = build_crops(data, 'train', labels)
    Xte, yte = build_crops(data, 'test', labels)
    print(f'[data] train crops {len(ytr)}  test crops {len(yte)}')
    freq = torch.bincount(ytr, minlength=len(CLASSES)).float()
    w = (freq.sum() / (freq + 1)).clamp(max=50.0).to(DEV)   # inverse-freq class weights
    model = net(); opt = torch.optim.Adam(model.parameters(), 3e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    n = len(ytr); bs = 256
    best = 0.0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb = Xtr[idx].to(DEV); yb = ytr[idx].to(DEV)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        # eval ceiling on real game2 test
        acc, occ, typ = evaluate(model, Xte, yte)
        print(f'[ep {ep}] real-test square_acc={acc:.3f} occ_acc={occ:.3f} type_acc(occupied)={typ:.3f}')
        if acc > best:
            best = acc; torch.save(model.state_dict(), out)
    print(f'[done] best real-test square_acc={best:.3f} -> {out}')

@torch.no_grad()
def evaluate(model, X, y):
    model.eval(); preds = []
    for i in range(0, len(y), 512):
        preds.append(model(X[i:i+512].to(DEV)).argmax(1).cpu())
    p = torch.cat(preds)
    acc = (p == y).float().mean().item()
    occ_p = (p > 0); occ_y = (y > 0)
    occ = (occ_p == occ_y).float().mean().item()
    m = occ_y
    typ = (p[m] == y[m]).float().mean().item() if m.any() else 0.0
    return acc, occ, typ

# ---------------- score generated ----------------
@torch.no_grad()
def score(data, ckpt, images, out):
    labels = json.load(open(os.path.join(data, 'labels.json')))
    model = net(); model.load_state_dict(torch.load(ckpt, map_location=DEV)); model.eval()
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    tot = dict(sq=0, sq_ok=0, occ_ok=0, phantom=0, empty=0, missing=0, occ=0,
               type_ok=0, color_ok=0, boards=0, boards_occ_exact=0, boards_full_exact=0)
    for p in sorted(glob.glob(os.path.join(images, '*.png'))) + sorted(glob.glob(os.path.join(images, '*.jpg'))):
        nm = re.sub(r'_(fake_B|real_B|real_A)$', '', os.path.basename(p)[:-4])
        if nm not in labels: continue
        vp = labels[nm]['viewpoint']; grid = fen_grid(labels[nm]['fen'])
        img = Image.open(p).convert('RGB')
        xs, keys = [], []
        for f, rank, c in squares_of(img, vp):
            xs.append(to_tensor(c)); keys.append((f, rank))
        pred = model(torch.stack(xs).to(DEV)).argmax(1).cpu().numpy()
        board_occ_ok = board_full_ok = True
        for (f, rank), pr in zip(keys, pred):
            ch = grid.get((f, rank)); ty = CIDX[ch] if ch else 0
            conf[ty, pr] += 1; tot['sq'] += 1
            tot['sq_ok'] += int(pr == ty)
            po, yo = pr > 0, ty > 0
            tot['occ_ok'] += int(po == yo)
            if po != yo: board_occ_ok = False
            if pr != ty: board_full_ok = False
            if yo: tot['occ'] += 1
            else: tot['empty'] += 1
            if (not yo) and po: tot['phantom'] += 1
            if yo and (not po): tot['missing'] += 1
            if yo and po:
                tot['type_ok'] += int(pr == ty)
                tot['color_ok'] += int((pr >= 7) == (ty >= 7))
        tot['boards'] += 1
        tot['boards_occ_exact'] += int(board_occ_ok)
        tot['boards_full_exact'] += int(board_full_ok)
    rep = {
        'n_boards': tot['boards'],
        'square_acc': tot['sq_ok'] / max(tot['sq'], 1),
        'occupancy_acc': tot['occ_ok'] / max(tot['sq'], 1),
        'phantom_rate(empty->piece)': tot['phantom'] / max(tot['empty'], 1),
        'missing_rate(piece->empty)': tot['missing'] / max(tot['occ'], 1),
        'type_acc(both occupied)': tot['type_ok'] / max(tot['occ'] - tot['missing'], 1),
        'color_acc(both occupied)': tot['color_ok'] / max(tot['occ'] - tot['missing'], 1),
        'whole_board_occupancy_exact': tot['boards_occ_exact'] / max(tot['boards'], 1),
        'whole_board_full_exact': tot['boards_full_exact'] / max(tot['boards'], 1),
    }
    # per-class recall/precision table (the honest type-legibility signal)
    per_class = {}
    print('\nclass  support  recall  precision  topConfusion')
    for i, c in enumerate(CLASSES):
        sup = int(conf[i].sum()); rec = conf[i, i] / max(sup, 1)
        colsum = int(conf[:, i].sum()); prec = conf[i, i] / max(colsum, 1)
        row = conf[i].copy(); row[i] = 0; j = int(row.argmax())
        extra = f'->{CLASSES[j]}({int(row[j])})' if row[j] > 0 else ''
        per_class[c] = {'support': sup, 'recall': float(rec), 'precision': float(prec)}
        print(f'{c:5s} {sup:7d}   {rec:5.3f}     {prec:5.3f}    {extra}')
    json.dump({'metrics': rep, 'per_class': per_class, 'classes': CLASSES,
               'confusion': conf.tolist()}, open(out, 'w'), indent=2)
    print('\n' + json.dumps(rep, indent=2)); print('[saved]', out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['train', 'score'])
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='report.json')
    ap.add_argument('--ckpt'); ap.add_argument('--images'); ap.add_argument('--epochs', type=int, default=18)
    a = ap.parse_args()
    if a.mode == 'train':
        train(a.data, a.out if a.out.endswith('.pth') else './checkpoints/square_eval.pth', a.epochs)
    else:
        score(a.data, a.ckpt, a.images, a.out)
