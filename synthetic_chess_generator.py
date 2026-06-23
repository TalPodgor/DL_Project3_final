"""
Synthetic Chess Data Generation Pipeline - Blender Script
==========================================================
Generates chessboard renders from FEN strings with 3 camera views.
Outputs raw renders + board corner coordinates for perfect cropping.

Usage:
    blender chess-set.blend --background --python synthetic_chess_generator.py -- \
        --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" \
        --output_dir "./synthetic_output" \
        --resolution 800 \
        --samples 64 \
        --view black
"""

import bpy
import math
import json
import os
import sys
import argparse
from mathutils import Vector, Matrix

# ==========================
# CONFIGURATION
# ==========================
REAL_BOARD_SIZE = 0.53
RENDER_MARGIN = 1.3  # Render 30% larger than needed for safe cropping
DEFAULT_SAMPLES = 64
DEFAULT_RESOLUTION = 1024  # Higher resolution for better quality crops


def get_board_info():
    """Get board dimensions and corners from the Blender scene."""
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")

    if not plane or not frame:
        raise RuntimeError("Could not find 'Black & white' or 'Outer frame' objects in scene")

    # Get plane (playing surface) bounds
    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8

    # Get frame (outer border) bounds
    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)

    # Board surface Z coordinate (top of the playing area)
    board_z = plane_max.z

    # Define the 4 corners of the playing surface (8x8 squares area)
    # Order: top-left, top-right, bottom-right, bottom-left (from camera's perspective)
    corners_3d = [
        Vector((plane_min.x, plane_max.y, board_z)),  # Top-left
        Vector((plane_max.x, plane_max.y, board_z)),  # Top-right
        Vector((plane_max.x, plane_min.y, board_z)),  # Bottom-right
        Vector((plane_min.x, plane_min.y, board_z)),  # Bottom-left
    ]

    return {
        'square_size': square_size,
        'plane_min': plane_min,
        'plane_max': plane_max,
        'center': center,
        'board_size': board_size,
        'board_z': board_z,
        'corners_3d': corners_3d,
        'scale_factor': board_size / REAL_BOARD_SIZE,
    }


def position_to_square(pos, board_info):
    """Convert 3D world position to chess square notation (e.g., 'e2')."""
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']

    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)

    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1

    return f"{file_letter}{rank_number}"


def detect_starting_positions(board_info):
    """Detect which piece is currently on which square."""
    print("\n" + "="*70)
    print("DETECTING STARTING POSITIONS")
    print("="*70)

    pieces = {}

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        name = obj.name
        piece_type = None

        # White pawns
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']:
            piece_type = 'P'
        # Black pawns
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']:
            piece_type = 'p'
        # Other pieces by name
        elif 'rook' in name.lower():
            piece_type = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower():
            piece_type = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower():
            piece_type = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower():
            piece_type = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower():
            piece_type = 'K' if 'white' in name.lower() else 'k'

        if piece_type:
            square = position_to_square(obj.location, board_info)
            pieces[name] = {
                'square': square,
                'piece_type': piece_type,
                'start_pos': obj.location.copy()
            }
            print(f"  {name:20s} -> {square:4s} ({piece_type})")

    print(f"\nDetected {len(pieces)} pieces")
    return pieces


def parse_fen(fen):
    """Parse FEN string into dict {square: piece_char}."""
    board_fen = fen.split()[0]
    ranks = board_fen.split('/')

    position = {}
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        board_rank = 8 - rank_idx

        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                file_letter = chr(ord('a') + file_idx)
                square = f"{file_letter}{board_rank}"
                position[square] = char
                file_idx += 1

    return position


def apply_fen(fen, starting_pieces, board_info):
    """Apply FEN position by moving pieces from their starting positions."""
    print("\n" + "="*70)
    print("APPLYING FEN POSITION")
    print("="*70)
    print(f"FEN: {fen}\n")

    target_position = parse_fen(fen)
    square_size = board_info['square_size']
    pieces_used = set()

    for target_square, piece_type in target_position.items():
        # Find closest available piece of this type
        candidates = []
        for piece_name, info in starting_pieces.items():
            if info['piece_type'] == piece_type and piece_name not in pieces_used:
                from_square = info['square']
                from_file = ord(from_square[0]) - ord('a')
                from_rank = int(from_square[1]) - 1
                to_file = ord(target_square[0]) - ord('a')
                to_rank = int(target_square[1]) - 1
                distance = abs(to_file - from_file) + abs(to_rank - from_rank)
                candidates.append((distance, piece_name, from_square))

        if not candidates:
            print(f"  Warning: No piece of type '{piece_type}' available for {target_square}")
            continue

        candidates.sort()
        _, piece_name, from_square = candidates[0]

        obj = bpy.data.objects.get(piece_name)
        if obj:
            from_file = ord(from_square[0]) - ord('a')
            from_rank = int(from_square[1]) - 1
            to_file = ord(target_square[0]) - ord('a')
            to_rank = int(target_square[1]) - 1

            file_diff = to_file - from_file
            rank_diff = to_rank - from_rank

            obj.location.x -= file_diff * square_size
            obj.location.y -= rank_diff * square_size
            obj.hide_render = False
            obj.hide_viewport = False

            pieces_used.add(piece_name)

            if from_square != target_square:
                print(f"  Moved {piece_name:20s}: {from_square} -> {target_square}")
            else:
                print(f"  Kept  {piece_name:20s}: {target_square}")

    # Hide captured pieces
    for piece_name in starting_pieces.keys():
        if piece_name not in pieces_used:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                obj.hide_render = True
                obj.hide_viewport = True
                print(f"  Hidden (captured): {piece_name}")

    print(f"\nPosition set: {len(pieces_used)} pieces visible")


def world_to_camera_view(scene, camera, coord):
    """Convert 3D world coordinate to 2D camera view coordinate (0-1 range)."""
    from bpy_extras.object_utils import world_to_camera_view as w2cv
    return w2cv(scene, camera, coord)


def setup_render_settings(resolution, samples):
    """Configure Blender render settings."""
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.cycles.use_denoising = True

    # Try to use GPU
    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        for device in prefs.devices:
            device.use = True
    except Exception as e:
        print(f"GPU setup failed, using CPU: {e}")
        scene.cycles.device = 'CPU'


def setup_lighting(board_info):
    """Setup scene lighting for consistent renders."""
    center = board_info['center']
    scale_factor = board_info['scale_factor']

    # Remove existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add main sun light
    light_height = center.z + 2.0 * scale_factor
    bpy.ops.object.light_add(type='SUN', location=(center.x, center.y, light_height))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.data.angle = math.radians(10)  # Soft shadows

    # Add fill light
    bpy.ops.object.light_add(type='AREA', location=(center.x + 1.0, center.y - 1.0, light_height * 0.7))
    fill = bpy.context.active_object
    fill.data.energy = 100.0
    fill.data.size = 2.0


def create_camera_for_view(view_name, board_info, view_side='black'):
    """
    Create and position camera for a specific view.

    Approach (from chess_position_api_v2.py):
    - All cameras look STRAIGHT DOWN (no tilt)
    - Middle: centered above board
    - Left/Right: positioned to the side but still pointing down
    - This gives different piece perspectives without extreme warping
    """
    center = board_info['center']
    scale_factor = board_info['scale_factor']
    board_size = board_info['board_size']

    # Clean existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Camera parameters
    camera_height = 2.0 * scale_factor
    angle_degrees = 12  # Small angle to keep board in frame with zoomed lens
    horizontal_offset = camera_height * math.tan(math.radians(angle_degrees))
    lens = 45  # Good zoom while keeping board in frame

    # Base Z rotation for view side (180 degrees flip for white view)
    base_z_rotation = math.radians(180) if view_side == 'white' else 0

    camera_z = center.z + camera_height

    if view_name == 'middle':
        # Center overhead view - points at board center
        bpy.ops.object.camera_add(location=(center.x, center.y, camera_z))
        cam = bpy.context.active_object
        cam.data.type = 'PERSP'
        cam.data.lens = lens

        # Point at center
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        cam.rotation_euler.z += base_z_rotation

    elif view_name == 'left':
        # Left (west) view - camera offset to left, but looking STRAIGHT DOWN
        if view_side == 'black':
            loc = (center.x - horizontal_offset, center.y, camera_z)
        else:
            loc = (center.x + horizontal_offset, center.y, camera_z)

        bpy.ops.object.camera_add(location=loc)
        cam = bpy.context.active_object
        cam.data.type = 'PERSP'
        cam.data.lens = lens

        # Look straight down (not at center!) - this is the key difference
        cam.rotation_euler = (0, 0, base_z_rotation)

    elif view_name == 'right':
        # Right (east) view - camera offset to right, but looking STRAIGHT DOWN
        if view_side == 'black':
            loc = (center.x + horizontal_offset, center.y, camera_z)
        else:
            loc = (center.x - horizontal_offset, center.y, camera_z)

        bpy.ops.object.camera_add(location=loc)
        cam = bpy.context.active_object
        cam.data.type = 'PERSP'
        cam.data.lens = lens

        # Look straight down (not at center!)
        cam.rotation_euler = (0, 0, base_z_rotation)

    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()

    return cam


def get_board_corners_2d(camera, board_info, resolution):
    """
    Project 3D board corners to 2D pixel coordinates.
    Returns corners in order: top-left, top-right, bottom-right, bottom-left
    (as they appear in the rendered image from the camera's perspective)
    """
    scene = bpy.context.scene
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']
    board_z = board_info['board_z']

    # Ensure scene is updated (important for constraint-based cameras)
    bpy.context.view_layer.update()

    # Define all 4 corners of the playing surface
    corners_3d = [
        Vector((plane_min.x, plane_max.y, board_z)),  # Corner A
        Vector((plane_max.x, plane_max.y, board_z)),  # Corner B
        Vector((plane_max.x, plane_min.y, board_z)),  # Corner C
        Vector((plane_min.x, plane_min.y, board_z)),  # Corner D
    ]

    print(f"    Camera location: {camera.location}")
    print(f"    Camera type: {camera.data.type}")
    if camera.data.type == 'PERSP':
        print(f"    Camera lens: {camera.data.lens}mm")

    # Project to 2D
    corners_2d = []
    for i, corner in enumerate(corners_3d):
        co_2d = world_to_camera_view(scene, camera, corner)
        # Convert from normalized (0-1) to pixel coordinates
        pixel_x = co_2d.x * resolution
        pixel_y = (1.0 - co_2d.y) * resolution  # Flip Y for image coordinates
        corners_2d.append([pixel_x, pixel_y])
        print(f"    Corner {i}: 3D{tuple(round(v, 3) for v in corner)} -> 2D({pixel_x:.1f}, {pixel_y:.1f}) [normalized: {co_2d.x:.3f}, {co_2d.y:.3f}]")

    # Validate corners are within image bounds (with some tolerance for edge cases)
    margin = resolution * 0.1  # Allow 10% outside for cropping headroom
    all_valid = True
    for i, (px, py) in enumerate(corners_2d):
        if px < -margin or px > resolution + margin or py < -margin or py > resolution + margin:
            print(f"    WARNING: Corner {i} is far outside image bounds!")
            all_valid = False

    if not all_valid:
        print("    WARNING: Some corners are outside image bounds - camera may need adjustment")

    # Sort corners to ensure correct order: TL, TR, BR, BL
    # Find center point
    cx = sum(c[0] for c in corners_2d) / 4
    cy = sum(c[1] for c in corners_2d) / 4

    # Categorize corners based on position relative to center
    top_corners = sorted([c for c in corners_2d if c[1] < cy], key=lambda c: c[0])
    bottom_corners = sorted([c for c in corners_2d if c[1] >= cy], key=lambda c: c[0])

    if len(top_corners) >= 2 and len(bottom_corners) >= 2:
        ordered_corners = [
            top_corners[0],   # Top-left
            top_corners[-1],  # Top-right
            bottom_corners[-1],  # Bottom-right
            bottom_corners[0],   # Bottom-left
        ]
    else:
        # Fallback - use as-is
        ordered_corners = corners_2d

    return ordered_corners


def render_view(view_name, board_info, output_dir, fen_id, resolution, view_side='black'):
    """Render a single view and save corner coordinates."""
    print(f"\n  Rendering {view_name} view...")

    # Create camera
    camera = create_camera_for_view(view_name, board_info, view_side)

    # Get corner coordinates before rendering
    corners_2d = get_board_corners_2d(camera, board_info, resolution)

    # Render - use absolute path for Blender
    output_path = os.path.join(output_dir, f"{fen_id}_{view_name}_raw.png")
    abs_output_path = os.path.abspath(output_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    bpy.context.scene.render.filepath = abs_output_path
    bpy.ops.render.render(write_still=True)

    print(f"    Saved: {abs_output_path}")

    # Verify file was created
    if os.path.exists(abs_output_path):
        print(f"    [OK] File verified")
    else:
        print(f"    [WARNING] File not found after render!")

    return {
        'view': view_name,
        'raw_image': abs_output_path,
        'corners_2d': corners_2d,
        'resolution': resolution
    }


def render_all_views(board_info, output_dir, fen_id, resolution, samples, view_side='black'):
    """Render all 3 views (left, middle, right) for a single FEN."""
    print("\n" + "="*70)
    print(f"RENDERING ALL VIEWS (FEN ID: {fen_id})")
    print("="*70)

    # Convert to absolute path
    output_dir = os.path.abspath(output_dir)
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    setup_render_settings(resolution, samples)
    setup_lighting(board_info)

    views_data = []
    for view_name in ['left', 'middle', 'right']:
        view_data = render_view(view_name, board_info, output_dir, fen_id, resolution, view_side)
        views_data.append(view_data)

    # Save metadata JSON
    metadata = {
        'fen_id': fen_id,
        'view_side': view_side,
        'resolution': resolution,
        'views': views_data
    }

    metadata_path = os.path.join(output_dir, f"{fen_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Metadata saved: {metadata_path}")
    print("\nRendering complete!")

    return metadata


def fix_board_rotation(board_info):
    """Fix inverted board by rotating checkerboard 90 degrees."""
    plane = bpy.data.objects.get("Black & white")
    if plane:
        center = board_info['center']
        original_pos = plane.location.copy()
        offset = original_pos - center
        plane.rotation_euler.z = math.radians(90)
        rot_matrix = Matrix.Rotation(math.radians(90), 3, 'Z')
        rotated_offset = rot_matrix @ offset
        plane.location = center + rotated_offset


def main():
    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description='Generate synthetic chessboard images')
    parser.add_argument('--fen', type=str,
                        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                        help='FEN string for board position')
    parser.add_argument('--fen_id', type=str, default=None,
                        help='Unique identifier for this FEN (default: auto-generated)')
    parser.add_argument('--output_dir', type=str, default='./synthetic_output',
                        help='Output directory for renders')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION,
                        help='Render resolution (will be cropped to this size)')
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                        help='Cycles render samples')
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'],
                        help='Render from white or black perspective')

    args = parser.parse_args(argv)

    # Generate FEN ID if not provided
    if args.fen_id is None:
        import hashlib
        args.fen_id = hashlib.md5(args.fen.encode()).hexdigest()[:8]

    print("\n" + "="*70)
    print("SYNTHETIC CHESS GENERATOR")
    print("="*70)
    print(f"FEN: {args.fen}")
    print(f"FEN ID: {args.fen_id}")
    print(f"Output: {args.output_dir}")
    print(f"Resolution: {args.resolution}")
    print(f"Samples: {args.samples}")
    print(f"View: {args.view}")

    # Get board info
    board_info = get_board_info()

    # Fix board rotation
    fix_board_rotation(board_info)

    # Detect and set piece positions
    starting_pieces = detect_starting_positions(board_info)
    apply_fen(args.fen, starting_pieces, board_info)

    # Render all views
    metadata = render_all_views(
        board_info,
        args.output_dir,
        args.fen_id,
        args.resolution,
        args.samples,
        args.view
    )

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
