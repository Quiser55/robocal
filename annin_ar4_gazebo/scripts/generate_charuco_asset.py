#!/usr/bin/env python3
"""Generate the simulated ChArUco board: texture, mesh, and geometry manifest.

Writes four files into ``--out-dir``:

- ``charuco.png``   the board texture, with a white quiet-zone margin
- ``charuco.obj``   a two-quad plane carrying that texture, in board-frame metres
- ``charuco.mtl``   its material (fully matte - a specular highlight blows out markers)
- ``board.yaml``    the geometry, so the detector params, the xacro and the
                    ground-truth node can all be derived from one source

Board frame (verified empirically against OpenCV 4.6, and load-bearing for both
the mesh UVs and the ground truth):

    origin = the board's top-left corner in the drawn image
    +X = image right,  +Y = image DOWN,  +Z = INTO the board

so the textured face is the **-Z** face. ``chessboardCorners`` then run from
``(square, square, 0)`` to ``(W-square, H-square, 0)``, which is exactly what
``estimatePoseCharucoBoard`` reports poses in - meaning the URDF link placed at
this frame *is* the ``calibration_target`` frame, and comparing measured against
ground truth is a plain TF lookup.

Why the margin is added here rather than passed to ``draw()``: OpenCV 4.6's
legacy ``CharucoBoard::draw(outSize, marginSize, ...)`` does not lay the board
out at a predictable scale when ``marginSize > 0`` - it rescales the board to
nearly fill the image (measured: a requested 40 px margin came out as 12 px,
with the pixels-per-metre scale changed by 8%). Drawing with ``marginSize=0``
and adding the border explicitly keeps the mapping exactly
``pixel = margin_px + coord_m * px_per_m``, which is what the OBJ UVs assume.

Usage:
    generate_charuco_asset.py [--squares-x 5] [--squares-y 7]
        [--square-length 0.035] [--marker-length 0.026]
        [--dictionary DICT_5X5_250] [--px-per-m 8000] [--margin-m 0.012]
        [--out-dir <pkg>/models/charuco_5x7_35mm]
"""
import argparse
import os

import cv2
import numpy as np
import yaml

#: text burned into the quiet-zone margin. A mirrored ArUco board is *silently*
#: undetectable - there is no error anywhere, detections simply never appear.
#: Mirrored text, by contrast, is obvious at a glance in rqt_image_view, which
#: turns a long debugging session into a two-second check. It lives outside the
#: board proper, so it cannot interfere with detection.
MIRROR_SENTINEL = 'AR4 >>'


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--squares-x', type=int, default=5)
    parser.add_argument('--squares-y', type=int, default=7)
    parser.add_argument('--square-length', type=float, default=0.035,
                        help='chessboard square side, metres')
    parser.add_argument('--marker-length', type=float, default=0.026,
                        help='ArUco marker side, metres')
    parser.add_argument('--dictionary', default='DICT_5X5_250')
    parser.add_argument('--px-per-m', type=float, default=8000.0,
                        help='texture resolution. 8000 keeps the texture well clear of '
                             'being the detection bottleneck at ~0.4 m viewing distance.')
    parser.add_argument('--margin-m', type=float, default=0.012,
                        help='white quiet zone around the board, metres. Not optional: '
                             'without it the outermost markers have only the '
                             '(square-marker)/2 inset and drop out intermittently.')
    parser.add_argument('--out-dir', default=os.path.join(
        here, os.pardir, 'models', 'charuco_5x7_35mm'))
    return parser.parse_args()


def _new_dictionary(name: str):
    """OpenCV 4.6 (Ubuntu 24.04 apt) uses the legacy aruco API; 4.7+ renamed
    everything. core/target_detection.py branches the same way."""
    key = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, 'Dictionary_get'):
        return cv2.aruco.Dictionary_get(key)
    return cv2.aruco.getPredefinedDictionary(key)


def _new_board(squares_x, squares_y, square_length, marker_length, dictionary):
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        return cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length, marker_length, dictionary)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary)


def _draw(board, size_px):
    if hasattr(board, 'draw'):
        return board.draw(size_px, 0, 1)
    return board.generateImage(size_px, marginSize=0, borderBits=1)


def _detect(image, dictionary):
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        params = cv2.aruco.DetectorParameters_create()
        return cv2.aruco.detectMarkers(image, dictionary, parameters=params)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    return detector.detectMarkers(image)


def render(args):
    """-> (image, board_w_m, board_h_m, margin_px)"""
    dictionary = _new_dictionary(args.dictionary)
    board = _new_board(args.squares_x, args.squares_y,
                       args.square_length, args.marker_length, dictionary)

    board_w = args.squares_x * args.square_length
    board_h = args.squares_y * args.square_length
    core_w = int(round(board_w * args.px_per_m))
    core_h = int(round(board_h * args.px_per_m))
    margin_px = int(round(args.margin_m * args.px_per_m))

    core = _draw(board, (core_w, core_h))
    image = cv2.copyMakeBorder(core, margin_px, margin_px, margin_px, margin_px,
                               cv2.BORDER_CONSTANT, value=255)

    # Sentinel goes in the bottom margin strip, below the board.
    scale = margin_px / 30.0
    cv2.putText(image, MIRROR_SENTINEL, (margin_px, image.shape[0] - int(margin_px * 0.25)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 0, max(1, int(scale * 2)), cv2.LINE_AA)

    # Self-check: if this passes, the texture is definitely fine, and any
    # later in-sim detection failure is a UV/winding/lighting problem instead.
    _, ids, _ = _detect(image, dictionary)
    expected = (args.squares_x * args.squares_y) // 2
    found = 0 if ids is None else len(ids)
    if found != expected:
        raise SystemExit(
            f'rendered board self-check failed: detected {found}/{expected} markers. '
            f'Refusing to write assets.')
    print(f'  self-check: detected {found}/{expected} markers in the rendered texture')
    return image, board_w, board_h, margin_px


def write_obj(path, board_w, board_h, margin_m, backing_thickness=0.002):
    """A textured quad on the -Z face plus an untextured backing quad.

    The quad extends to -margin_m so the drawn image maps corner-to-corner onto
    it; the board frame's origin stays at the board's own top-left corner, i.e.
    inset from the mesh corner by exactly the margin.

    The backing quad is not decoration: without it the board is invisible from
    behind, and "mounted facing the wrong way" is indistinguishable from "not
    spawned at all".
    """
    x0, y0 = -margin_m, -margin_m
    x1, y1 = board_w + margin_m, board_h + margin_m
    z1 = backing_thickness

    with open(path, 'w') as f:
        f.write('# Generated by generate_charuco_asset.py - do not edit by hand.\n')
        f.write('# Board frame: +X image right, +Y image DOWN, +Z INTO the board.\n')
        f.write('# The textured face is -Z; the backing face is +Z.\n')
        f.write('mtllib charuco.mtl\n')
        # 1-4: textured face, 5-8: backing face
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            f.write(f'v {x:.6f} {y:.6f} 0.000000\n')
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            f.write(f'v {x:.6f} {y:.6f} {z1:.6f}\n')
        # OBJ/Ogre texture origin is bottom-left, image row 0 is at v=1.
        f.write('vt 0.0 1.0\nvt 1.0 1.0\nvt 1.0 0.0\nvt 0.0 0.0\n')
        f.write('vn 0.0 0.0 -1.0\nvn 0.0 0.0 1.0\n')
        f.write('usemtl charuco\n')
        # Wound so the normal is -Z, i.e. the face is lit and visible from -Z.
        f.write('f 1/1/1 4/4/1 3/3/1 2/2/1\n')
        f.write('usemtl charuco_backing\n')
        f.write('f 5/1/2 6/2/2 7/3/2 8/4/2\n')


def write_mtl(path):
    with open(path, 'w') as f:
        f.write('# Generated by generate_charuco_asset.py - do not edit by hand.\n')
        # Ks 0: a specular highlight sweeping across the board blows out markers
        # and produces dropouts that look exactly like random measurement noise.
        f.write('newmtl charuco\nKa 0.35 0.35 0.35\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n'
                'Ns 1.0\nd 1.0\nillum 1\nmap_Kd charuco.png\n\n')
        f.write('newmtl charuco_backing\nKa 0.1 0.1 0.1\nKd 0.25 0.25 0.28\n'
                'Ks 0.0 0.0 0.0\nNs 1.0\nd 1.0\nillum 1\n')


def write_manifest(path, args, board_w, board_h):
    document = {
        'charuco_board': {
            'generated_by': 'generate_charuco_asset.py',
            'dictionary': args.dictionary,
            'squares_x': args.squares_x,
            'squares_y': args.squares_y,
            'square_length_m': args.square_length,
            'marker_length_m': args.marker_length,
            'margin_m': args.margin_m,
            'board_width_m': round(board_w, 6),
            'board_height_m': round(board_h, 6),
            'frame': ('origin at the board top-left corner; +X image right, '
                      '+Y image down, +Z into the board (textured face is -Z)'),
            'note': ("square_length_m/marker_length_m must match target_detector's "
                     "charuco.* parameters, or every pose is scaled."),
        }
    }
    with open(path, 'w') as f:
        yaml.safe_dump(document, f, sort_keys=False)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'generating {args.squares_x}x{args.squares_y} ChArUco board '
          f'({args.square_length * 1000:.1f} mm squares, {args.dictionary})')
    image, board_w, board_h, margin_px = render(args)

    png_path = os.path.join(out_dir, 'charuco.png')
    cv2.imwrite(png_path, image)
    write_obj(os.path.join(out_dir, 'charuco.obj'), board_w, board_h, args.margin_m)
    write_mtl(os.path.join(out_dir, 'charuco.mtl'))
    write_manifest(os.path.join(out_dir, 'board.yaml'), args, board_w, board_h)

    print(f'  board   : {board_w:.4f} x {board_h:.4f} m')
    print(f'  texture : {image.shape[1]} x {image.shape[0]} px '
          f'({margin_px} px margin) -> {png_path}')
    print(f'  wrote charuco.obj, charuco.mtl, board.yaml into {out_dir}')


if __name__ == '__main__':
    main()
