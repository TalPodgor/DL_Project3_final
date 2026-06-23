"""
V5 oblique Blender renderer.

Runs inside Blender with the original chess-set.blend asset and renders a single
camera-matched, board-rectified source candidate. This is intentionally separate
from earlier v3/v4 renderers so we can evaluate the geometry before training.

Outputs raw RGB / semantic-silhouette / depth-like renders plus board corners.
Crop them with pil_perspective_crop.py.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_chess_generator import (  # noqa: E402
    apply_fen,
    detect_starting_positions,
    fix_board_rotation,
    get_board_info,
    parse_fen,
    setup_render_settings,
    world_to_camera_view,
)


CLASS_COLORS = {
    "P": (0.92, 0.88, 0.72, 1.0),
    "N": (0.82, 0.82, 0.52, 1.0),
    "B": (0.92, 0.72, 0.44, 1.0),
    "R": (0.72, 0.92, 0.72, 1.0),
    "Q": (0.72, 0.82, 0.98, 1.0),
    "K": (0.90, 0.70, 0.98, 1.0),
    "p": (0.18, 0.12, 0.08, 1.0),
    "n": (0.32, 0.18, 0.08, 1.0),
    "b": (0.20, 0.32, 0.12, 1.0),
    "r": (0.10, 0.24, 0.34, 1.0),
    "q": (0.34, 0.12, 0.30, 1.0),
    "k": (0.34, 0.10, 0.10, 1.0),
}


def visible_meshes():
    return [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render]


def material_slots_snapshot():
    snap = {}
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            snap[obj.name] = [slot.material for slot in obj.material_slots]
    return snap


def restore_materials(snap):
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name not in snap:
            continue
        obj.data.materials.clear()
        for mat in snap[obj.name]:
            obj.data.materials.append(mat)


def emission_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    emit = nodes.new(type="ShaderNodeEmission")
    emit.inputs["Color"].default_value = color
    emit.inputs["Strength"].default_value = 1.0
    mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def depth_material(name, camera, board_info):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    emit = nodes.new(type="ShaderNodeEmission")
    cam_data = nodes.new(type="ShaderNodeCameraData")
    map_range = nodes.new(type="ShaderNodeMapRange")

    dists = [(obj.location - camera.location).length for obj in visible_meshes()]
    if dists:
        near = max(0.01, min(dists) - board_info["board_size"] * 0.25)
        far = max(dists) + board_info["board_size"] * 0.35
    else:
        near, far = 1.0, 100.0
    map_range.clamp = True
    map_range.inputs["From Min"].default_value = near
    map_range.inputs["From Max"].default_value = far
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0
    emit.inputs["Strength"].default_value = 1.0

    mat.node_tree.links.new(cam_data.outputs["View Z Depth"], map_range.inputs["Value"])
    mat.node_tree.links.new(map_range.outputs["Result"], emit.inputs["Color"])
    mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def assign_single_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def set_segmentation_materials(starting_pieces):
    board_mat = emission_material("v5_seg_board_black", (0.0, 0.0, 0.0, 1.0))
    frame_mat = emission_material("v5_seg_frame_gray", (0.04, 0.04, 0.04, 1.0))
    piece_mats = {k: emission_material(f"v5_seg_{k}", v) for k, v in CLASS_COLORS.items()}

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.name in starting_pieces and not obj.hide_render:
            assign_single_material(obj, piece_mats[starting_pieces[obj.name]["piece_type"]])
        elif obj.name == "Outer frame":
            assign_single_material(obj, frame_mat)
        else:
            assign_single_material(obj, board_mat)


def set_depth_materials(camera, board_info):
    mat = depth_material("v5_depth", camera, board_info)
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            assign_single_material(obj, mat)


def configure_color_management():
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0     # neutral baseline; only the RGB pass is brightened (see main)
    scene.view_settings.gamma = 1


def setup_lighting(board_info):
    center = board_info["center"]
    board = board_info["board_size"]
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="AREA", location=(center.x - board * 0.25, center.y - board * 0.55, center.z + board * 1.15))
    key = bpy.context.object
    key.name = "v5_key_area"
    key.data.energy = 1200   # E: brighter key (was 650), keep key>>fill for directional shadows
    key.data.size = board * 0.55

    bpy.ops.object.light_add(type="AREA", location=(center.x + board * 0.45, center.y + board * 0.35, center.z + board * 0.85))
    fill = bpy.context.object
    fill.name = "v5_fill_area"
    fill.data.energy = 150   # E: brighter fill (was 90)
    fill.data.size = board * 0.8


def create_oblique_camera(board_info, viewpoint, camera_y, camera_z, lens):
    center = board_info["center"]
    board = board_info["board_size"]
    side = 1.0 if viewpoint == "white" else -1.0
    loc = Vector((center.x, center.y + side * board * camera_y, center.z + board * camera_z))
    target = Vector((center.x, center.y, board_info["board_z"]))

    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    cam.name = "v5_oblique_camera"
    cam.data.type = "PERSP"
    cam.data.lens = lens
    cam.data.sensor_width = 32
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()
    return cam


def board_corners_2d(camera, board_info, resolution):
    scene = bpy.context.scene
    plane_min = board_info["plane_min"]
    plane_max = board_info["plane_max"]
    z = board_info["board_z"]
    corners = [
        Vector((plane_min.x, plane_max.y, z)),
        Vector((plane_max.x, plane_max.y, z)),
        Vector((plane_max.x, plane_min.y, z)),
        Vector((plane_min.x, plane_min.y, z)),
    ]
    pts = []
    for corner in corners:
        co = world_to_camera_view(scene, camera, corner)
        pts.append([float(co.x * resolution), float((1.0 - co.y) * resolution)])
    return order_corners(pts)


def order_corners(corners):
    pts = sorted(corners, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[1], bottom[0]]


def render_to(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--viewpoint", choices=["white", "black"], default="white")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="sample")
    ap.add_argument("--resolution", type=int, default=1100)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--camera-y", type=float, default=0.70, help="camera distance along board y in board-size units")
    ap.add_argument("--camera-z", type=float, default=0.92, help="camera height in board-size units")
    ap.add_argument("--lens", type=float, default=48.0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_color_management()
    setup_render_settings(args.resolution, args.samples)
    board_info = get_board_info()
    fix_board_rotation(board_info)
    board_info = get_board_info()
    starting_pieces = detect_starting_positions(board_info)
    apply_fen(args.fen, starting_pieces, board_info)
    setup_lighting(board_info)
    camera = create_oblique_camera(board_info, args.viewpoint, args.camera_y, args.camera_z, args.lens)
    corners = board_corners_2d(camera, board_info, args.resolution)

    snap = material_slots_snapshot()
    rgb_raw = out_dir / f"{args.name}_rgb_raw.png"
    seg_raw = out_dir / f"{args.name}_seg_raw.png"
    depth_raw = out_dir / f"{args.name}_depth_raw.png"

    # E: brighten ONLY the RGB pass (exposure is a global post-process, so seg/depth
    # — which use emission/depth materials and a fixed palette — must stay neutral).
    bpy.context.scene.view_settings.exposure = 0.8
    render_to(rgb_raw)
    bpy.context.scene.view_settings.exposure = 0.0
    set_segmentation_materials(starting_pieces)
    render_to(seg_raw)
    restore_materials(snap)
    set_depth_materials(camera, board_info)
    render_to(depth_raw)
    restore_materials(snap)

    metadata = {
        "name": args.name,
        "fen": args.fen,
        "viewpoint": args.viewpoint,
        "resolution": args.resolution,
        "camera_y": args.camera_y,
        "camera_z": args.camera_z,
        "lens": args.lens,
        "board_corners_2d": corners,
        "raw": {
            "rgb": str(rgb_raw),
            "seg": str(seg_raw),
            "depth": str(depth_raw),
        },
        "position": parse_fen(args.fen),
    }
    meta_path = out_dir / f"{args.name}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[v5] wrote {meta_path}")


if __name__ == "__main__":
    main()
