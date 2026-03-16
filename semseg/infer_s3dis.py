"""
Inference for S3DIS semantic segmentation (13 classes).

Loads a checkpoint trained by segmentation/train_semseg.py (models/semseg.py),
slides over each test room's blocks using the provided whole-scene loader, and
generates per-point predictions saved alongside simple text files.

Outputs: for each room file (e.g., Area_5_office_1.npy in the dataset root),
creates a corresponding prediction file under --save_dir with the same name
and suffix ".pred.txt" containing one integer label per original point.

Example:
  python segmentation/infer_s3dis.py \
    --ckpt ./segmentation/log/sem_seg/exp/checkpoints/best_model.pth \
    --root ../data/stanford_indoor3d/ \
    --test_area 5 \
    --gpu 0 \
    --save_dir ./segmentation/preds_s3dis
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch
import importlib
from tqdm import tqdm

# Local imports
SEG_DIR = Path(__file__).parent
sys.path.append(str(SEG_DIR))
sys.path.append(str(SEG_DIR / 'models'))
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene  # type: ignore
from typing import Optional


def robust_load_model(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        model.load_state_dict(state, strict=False)
    else:
        # Fallback to direct state_dict
        model.load_state_dict(ckpt, strict=False)


# Fixed color palette for 13 S3DIS classes (default)
S3DIS_COLORS_DEFAULT = np.array([
    [  0, 255, 255],  # ceiling - cyan
    [255,   0, 255],  # floor   - magenta
    [255, 255,   0],  # wall    - yellow
    [255, 128,   0],  # beam    - orange
    [128,   0, 255],  # column  - purple
    [  0, 128, 255],  # window  - sky blue
    [  0, 255,   0],  # door    - green
    [128, 128, 128],  # table   - gray
    [255,   0,   0],  # chair   - red
    [128,  64,   0],  # sofa    - brown
    [  0,  64, 128],  # bookcase- dark blue
    [ 64, 128,   0],  # board   - olive
    [  0,   0,   0],  # clutter - black
], dtype=np.uint8)


def load_palette(palette: str, palette_file: Optional[str]) -> np.ndarray:
    if palette_file:
        arr = []
        with open(palette_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) < 3:
                    continue
                r, g, b = map(int, parts[:3])
                arr.append([r, g, b])
        arr = np.array(arr, dtype=np.uint8)
        if arr.shape[0] < 13:
            pad = np.zeros((13 - arr.shape[0], 3), dtype=np.uint8)
            arr = np.vstack([arr, pad])
        return arr[:13]
    if palette == 'default':
        return S3DIS_COLORS_DEFAULT
    if palette == 'random':
        rng = np.random.RandomState(42)
        return rng.randint(0, 256, size=(13, 3), dtype=np.uint8)
    return S3DIS_COLORS_DEFAULT


def _sanitize_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray):
    n0 = xyz.shape[0]
    mask = np.isfinite(xyz).all(axis=1)
    if mask.sum() != n0:
        xyz = xyz[mask]
        rgb = rgb[mask]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return xyz.astype(np.float32), rgb


def write_ply_xyzrgb(xyz: np.ndarray, rgb: np.ndarray, out_path: Path, fmt: str = 'ascii'):
    xyz, rgb = _sanitize_xyz_rgb(xyz, rgb)
    n = xyz.shape[0]
    if fmt == 'binary':
        import struct
        with open(out_path, 'wb') as f:
            header = (
                'ply\n'
                'format binary_little_endian 1.0\n'
                f'element vertex {n}\n'
                'property float x\n'
                'property float y\n'
                'property float z\n'
                'property uchar red\n'
                'property uchar green\n'
                'property uchar blue\n'
                'end_header\n'
            )
            f.write(header.encode('ascii'))
            for i in range(n):
                f.write(struct.pack('<fffBBB', float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]), int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2])))
    else:
        with open(out_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {n}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(n):
                x, y, z = xyz[i]
                r, g, b = rgb[i]
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n')


def run_inference(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Build model
    MODEL = importlib.import_module('semseg')
    model = MODEL.get_model(cls_dim=13).to(device)
    model.eval()
    robust_load_model(model, Path(args.ckpt), device)

    # Ensure root ends with path separator because loader does np.load(root + file)
    root = args.root
    if not root.endswith(os.sep):
        root = root + os.sep

    # Dataset of whole scenes (rooms) for the target Area
    dataset = ScannetDatasetWholeScene(
        root=root,
        block_points=args.block_points,
        split='test',
        test_area=args.test_area,
        stride=args.stride,
        block_size=args.block_size,
        padding=0.001,
    )

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # Load palette for coloring consistency with GT exporter
    palette_arr = load_palette(args.palette, args.palette_file)

    # If a specific room is requested, filter indices
    indices = list(range(len(dataset)))
    if args.room is not None:
        target = args.room
        # accept either full filename or stem
        matches = []
        for i, fname in enumerate(dataset.file_list):
            stem = Path(fname).stem
            if fname == target or stem == target or target in fname:
                matches.append(i)
        if not matches:
            raise FileNotFoundError(f'Room "{target}" not found in Area_{args.test_area} under {root}. Available example: {dataset.file_list[:3]}')
        indices = matches

    for room_idx in indices:
        # Get all blocks for this room
        data_room, label_room, sample_weight, index_room = dataset[room_idx]
        # Shapes: [num_blocks, P, 9], [num_blocks, P], [num_blocks, P], [num_blocks, P]
        num_blocks, P, _ = data_room.shape
        # Original points count
        N = dataset.scene_points_list[room_idx].shape[0]
        # Accumulate per-point class scores
        scores = np.zeros((N, 13), dtype=np.float64)

        # Batched inference over blocks
        bs = args.batch_size
        for start in tqdm(range(0, num_blocks, bs), desc=f'Room {room_idx+1}/{len(dataset)}', leave=False):
            end = min(start + bs, num_blocks)
            batch = data_room[start:end]  # [b, P, 9]
            batch_idx = index_room[start:end]  # [b, P]
            # To torch: [b, 9, P]
            tensor = torch.from_numpy(batch.transpose(0, 2, 1)).float().to(device)
            with torch.no_grad():
                logp = model(tensor)  # [b, P, 13] (log softmax)
                prob = logp.exp().cpu().numpy()
            # Accumulate per original point index
            for i in range(end - start):
                idxs = batch_idx[i]
                scores[idxs, :] += prob[i]

        pred = scores.argmax(axis=1).astype(np.int32)  # [N]

        # Compose output file name using dataset's file list
        room_file = dataset.file_list[room_idx]
        out_file = save_root / (Path(room_file).stem + '.pred.txt')
        np.savetxt(out_file, pred, fmt='%d')
        print(f'Saved {out_file} (N={N})')

        if args.export_ply:
            # Build colored PLY with predicted labels palette
            xyzrgb = dataset.scene_points_list[room_idx]  # [N,6]
            xyz = xyzrgb[:, :3]
            # Use selected palette for consistency with GT
            color = palette_arr[np.clip(pred, 0, 12)]
            ply_path = save_root / (Path(room_file).stem + '.pred.ply')
            write_ply_xyzrgb(xyz, color, ply_path, fmt=args.ply_format)
            print(f'Saved {ply_path} (colored by prediction)')

        # Optional: export ground-truth labels for visualization/comparison
        if args.export_gt_txt:
            gt = dataset.semantic_labels_list[room_idx].astype(np.int32)
            gt_txt = save_root / (Path(room_file).stem + '.gt.txt')
            np.savetxt(gt_txt, gt, fmt='%d')
            print(f'Saved {gt_txt} (ground-truth labels)')

        if args.export_gt_ply:
            xyzrgb = dataset.scene_points_list[room_idx]
            xyz = xyzrgb[:, :3]
            gt = dataset.semantic_labels_list[room_idx].astype(np.int32)
            gt_color = S3DIS_COLORS[np.clip(gt, 0, 12)]
            gt_ply = save_root / (Path(room_file).stem + '.gt.ply')
            write_ply_xyzrgb(xyz, gt_color, gt_ply)
            print(f'Saved {gt_ply} (colored by ground-truth)')


def main():
    parser = argparse.ArgumentParser(description='S3DIS inference (whole-scene)')
    parser.add_argument('--ckpt', type=str, required=True, help='path to trained checkpoint .pth')
    parser.add_argument('--root', type=str, default='../data/stanford_indoor3d/', help='S3DIS data root')
    parser.add_argument('--test_area', type=int, default=5, help='Area id to test [1..6]')
    parser.add_argument('--block_points', type=int, default=4096, help='points per block')
    parser.add_argument('--block_size', type=float, default=1.0, help='block size in meters')
    parser.add_argument('--stride', type=float, default=0.5, help='sliding window stride in meters')
    parser.add_argument('--batch_size', type=int, default=8, help='inference batch size over blocks')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device id(s)')
    parser.add_argument('--save_dir', type=str, default=str(SEG_DIR / 'preds_s3dis'), help='output folder')
    parser.add_argument('--room', type=str, default=None, help='room file name or stem to run only that room (e.g., Area_5_office_1 or Area_5_office_1.npy)')
    parser.add_argument('--export_ply', action='store_true', help='export colored PLY per room using prediction palette')
    parser.add_argument('--export_gt_ply', action='store_true', help='export colored PLY per room using ground-truth labels')
    parser.add_argument('--export_gt_txt', action='store_true', help='export ground-truth labels as <room>.gt.txt')
    parser.add_argument('--ply_format', type=str, default='ascii', choices=['ascii', 'binary'], help='PLY output format')
    parser.add_argument('--palette', type=str, default='default', choices=['default', 'random'], help='which palette to use if not providing palette_file')
    parser.add_argument('--palette_file', type=str, default=None, help='custom palette file (13 lines of r g b or r,g,b)')
    args = parser.parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
