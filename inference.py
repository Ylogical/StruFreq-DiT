"""
MedSegDiT inference and evaluation.

Every test image is sampled K times with deterministic DDIM; the K soft
predictions are averaged into an MMSE estimate and then thresholded. Reports
mIoU, DSC, Sensitivity (Recall), Accuracy, Precision, HD95, GED and ECE, plus
the parameter count and GFLOPs.

Example:
  python inference.py --checkpoint checkpoints/<run>/best_model.pth \\
      --dataset glas --use_ema --K 25 --save_images
"""
import os
import time
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model_medsegdit import MedSegDiT_models
from dataset import get_dataloader
from diffusion_utils import DiffusionSchedule

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ----------------------------------------------
# Metrics
# ----------------------------------------------
def compute_metrics(pred_binary, gt_binary):
    """IoU / Dice / Precision / Recall / Accuracy for one (H, W) pair."""
    pred = pred_binary.astype(bool)
    gt   = gt_binary.astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()

    iou       = tp / (tp + fp + fn + 1e-8)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    acc       = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return iou, dice, precision, recall, acc


def _surface_distances(pred, gt):
    """Distance from every point of the pred surface to the gt surface."""
    conn = generate_binary_structure(pred.ndim, 1)
    pred_border = pred ^ binary_erosion(pred, structure=conn, iterations=1)
    gt_border   = gt   ^ binary_erosion(gt,   structure=conn, iterations=1)
    return distance_transform_edt(~gt_border)[pred_border]


def compute_hd95(pred, gt):
    """Symmetric 95th percentile Hausdorff distance in pixels (lower is better).

    Returns None without scipy. A degenerate case (pred or GT empty) counts as
    0, the convention used in the segmentation literature.
    """
    if not _SCIPY_OK:
        return None
    pred = np.asarray(pred).astype(bool)
    gt   = np.asarray(gt).astype(bool)
    if not pred.any() or not gt.any():
        return 0.0

    # Crop to the joint bounding box (1 pixel margin): same result, faster
    idx = np.nonzero(pred | gt)
    slicer = tuple(slice(max(int(i.min()) - 1, 0), min(int(i.max()) + 2, s))
                   for i, s in zip(idx, pred.shape))
    pred, gt = pred[slicer], gt[slicer]

    sds = np.hstack((_surface_distances(pred, gt), _surface_distances(gt, pred)))
    return float(np.percentile(sds, 95))


def _dice_dist(a, b):
    """Dice distance d = 1 - Dice(a, b), used inside the GED."""
    a, b = a.astype(bool), b.astype(bool)
    return 1.0 - 2.0 * (a & b).sum() / (a.sum() + b.sum() + 1e-8)


def compute_ged(k_preds, gt):
    """Generalized Energy Distance over K samples (lower is better).

        GED^2 = 2*E[d(S,G)] - E[d(S,S')] - E[d(G,G')]

    With a single GT annotation E[d(G,G')] = 0.
    """
    K = len(k_preds)
    sg = 2.0 * sum(_dice_dist(s, gt) for s in k_preds) / K
    ss = sum(_dice_dist(k_preds[i], k_preds[j])
             for i in range(K) for j in range(K)) / (K * K)
    return float(sg - ss)


def compute_ece(prob_flat, gt_flat, n_bins=15):
    """Expected Calibration Error over equal-width probability bins."""
    prob_flat = np.clip(prob_flat, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N = 0.0, len(prob_flat)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_flat >= lo) & (prob_flat < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / N * abs(prob_flat[mask].mean() - gt_flat[mask].mean())
    return float(ece)


# ----------------------------------------------
# Model cost
# ----------------------------------------------
def count_params(model):
    """Total parameters, in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_gflops(model, image_size, mask_channels, image_channels, device):
    """GFLOPs of one denoising step, from MACs counted with forward hooks
    (Conv2d / ConvTranspose2d / Linear); GFLOPs = 2 * MACs / 1e9."""
    import torch.nn as nn

    macs = [0]
    hooks = []

    def _hook_conv2d(m, inp, out):
        B, Co, Ho, Wo = out.shape
        kH, kW = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2
        macs[0] += B * Co * Ho * Wo * (m.in_channels // m.groups) * kH * kW

    def _hook_deconv2d(m, inp, out):
        B, Ci, Hi, Wi = inp[0].shape
        kH, kW = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2
        macs[0] += B * Ci * Hi * Wi * (m.out_channels // m.groups) * kH * kW

    def _hook_linear(m, inp, out):
        macs[0] += (inp[0].numel() // m.in_features) * m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(_hook_conv2d))
        elif isinstance(mod, nn.ConvTranspose2d):
            hooks.append(mod.register_forward_hook(_hook_deconv2d))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_hook_linear))

    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, mask_channels, image_size, image_size, device=device),
                  torch.zeros(1, dtype=torch.long, device=device),
                  torch.zeros(1, image_channels, image_size, image_size, device=device))
        gflops = 2.0 * macs[0] / 1e9
    except Exception:
        gflops = None
    finally:
        for h in hooks:
            h.remove()
    return gflops


def make_comparison(img_np, gt_np, pred_np):
    """Comparison strip [image | GT overlay | prediction overlay]."""
    def overlay(img, mask, color, alpha=0.4):
        out = img.copy().astype(np.float32)
        for c, v in enumerate(color):
            out[:, :, c] = np.where(mask > 0, out[:, :, c] * (1 - alpha) + v * alpha,
                                    out[:, :, c])
        return out.clip(0, 255).astype(np.uint8)

    return np.concatenate([img_np,
                           overlay(img_np, gt_np,   (0, 200, 0)),
                           overlay(img_np, pred_np, (255, 80, 0))], axis=1)


# ----------------------------------------------
# Inference
# ----------------------------------------------
@torch.no_grad()
def run_inference(model, diffusion, test_loader, device, args):
    model.eval()

    ious, dices, precs, recs, accs = [], [], [], [], []
    all_hd95, all_ged = [], []
    all_prob, all_gt_flat = [], []

    if args.save_images:
        img_dir = os.path.join(args.output_dir, args.dataset, 'images')
        os.makedirs(img_dir, exist_ok=True)

    sample_idx = 0
    t_start = time.perf_counter()
    hd95_overhead = 0.0

    for batch in tqdm(test_loader, desc='Running inference'):
        images = batch['image'].to(device)      # (B, 3, H, W) in [-1, 1]
        masks  = batch['mask'].to(device)       # (B, 1, H, W) in {-1, +1}
        B, _, H, W = images.shape

        # K independent DDIM trajectories
        k_soft = [diffusion.ddim_sample(model, (B, 1, H, W), images, device,
                                        ddim_steps=args.ddim_steps, eta=args.eta,
                                        progress=False).float()
                  for _ in range(args.K)]

        # MMSE estimate
        pred_mask = torch.stack(k_soft, dim=0).mean(dim=0)               # (B, 1, H, W)
        prob_map  = (pred_mask.cpu().numpy()[:, 0] + 1.0) / 2.0          # [0, 1] for the ECE
        pred_np   = (pred_mask.cpu().numpy()[:, 0] > args.threshold).astype(np.uint8)
        gt_np     = (masks.cpu().numpy()[:, 0] > args.threshold).astype(np.uint8)
        k_binary  = [(ks.cpu().numpy()[:, 0] > args.threshold).astype(np.uint8)
                     for ks in k_soft]

        for i in range(B):
            pred_b, gt_b = pred_np[i], gt_np[i]

            iou, dice, prec, rec, acc = compute_metrics(pred_b, gt_b)
            ious.append(iou); dices.append(dice)
            precs.append(prec); recs.append(rec); accs.append(acc)

            if not args.skip_hd95 and _SCIPY_OK:
                _t = time.perf_counter()
                h = compute_hd95(pred_b, gt_b)
                if h is not None:
                    all_hd95.append(h)
                hd95_overhead += time.perf_counter() - _t

            # The GED needs several samples
            if args.K > 1:
                all_ged.append(compute_ged([kb[i] for kb in k_binary], gt_b))

            all_prob.append(prob_map[i].flatten())
            all_gt_flat.append(gt_b.flatten().astype(np.float32))

            if args.save_images:
                img_vis = ((images[i].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
                           * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_vis).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_image.png'))
                Image.fromarray((gt_b * 255).astype(np.uint8)).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_gt.png'))
                Image.fromarray((pred_b * 255).astype(np.uint8)).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_pred.png'))
                Image.fromarray(make_comparison(img_vis, gt_b, pred_b)).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_compare.png'))

            sample_idx += 1

    total_time = time.perf_counter() - t_start - hd95_overhead

    return {
        'iou':  float(np.mean(ious)),
        'dice': float(np.mean(dices)),
        'prec': float(np.mean(precs)),
        'rec':  float(np.mean(recs)),
        'acc':  float(np.mean(accs)),
        'hd95': float(np.mean(all_hd95)) if all_hd95 else None,
        'ged':  float(np.mean(all_ged)) if all_ged else None,
        'ece':  compute_ece(np.concatenate(all_prob), np.concatenate(all_gt_flat)),
        'n_images':   sample_idx,
        'total_sec':  total_time,
        'per_image_ms': total_time / max(sample_idx, 1) * 1000,
    }


# ----------------------------------------------
# CLI
# ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='MedSegDiT inference')
    p.add_argument('--checkpoint',     type=str, required=True)
    p.add_argument('--dataset',        type=str, default='glas',
                   choices=['glas', 'monuseg', 'ph2', 'imid', 'tnbc'])
    p.add_argument('--model',          type=str, default=None,
                   choices=list(MedSegDiT_models.keys()),
                   help='Model variant (default: read from the checkpoint, else B/16)')
    p.add_argument('--image_size',     type=int, default=None,
                   help='Input size (default: glas/ph2/imid=256, monuseg/tnbc=512)')
    p.add_argument('--batch_size',     type=int, default=4)
    p.add_argument('--image_channels', type=int, default=3)
    p.add_argument('--num_timesteps',  type=int, default=200)
    p.add_argument('--beta_schedule',  type=str, default='cosine',
                   choices=['linear', 'cosine'])
    p.add_argument('--ddim_steps',     type=int, default=20,
                   help="Number of DDIM reverse steps T'")
    p.add_argument('--eta',            type=float, default=0.0,
                   help='DDIM stochasticity: 0 = deterministic')
    p.add_argument('--K',              type=int, default=1,
                   help='Number of DDIM samples averaged into the MMSE estimate '
                        '(K=25 is the setting used in the paper)')
    p.add_argument('--threshold',      type=float, default=0.0,
                   help='Binarization threshold for masks in [-1, 1]')
    p.add_argument('--use_ema',        action='store_true', default=False,
                   help='Evaluate the EMA weights stored in the checkpoint')
    p.add_argument('--skip_hd95',      action='store_true', default=False)
    p.add_argument('--save_images',    action='store_true', default=False)
    p.add_argument('--output_dir',     type=str, default='results')
    p.add_argument('--num_workers',    type=int, default=4)
    p.add_argument('--device',         type=str, default='cuda')
    p.add_argument('--gpu',            type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # Read the training config recorded in the checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('train_config', {})
    if cfg:
        print("\n[Checkpoint train config]")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print()

    if args.model is None:
        args.model = cfg.get('model', 'MedSegDiT-B/16')
    elif 'model' in cfg and cfg['model'] != args.model:
        raise RuntimeError(f"--model {args.model} but the checkpoint was trained with "
                           f"{cfg['model']}; the architectures do not match.")
    if args.image_size is None:
        args.image_size = cfg.get('image_size') or (
            512 if args.dataset in ('monuseg', 'tnbc') else 256)

    # Data
    print("Loading dataset...")
    _, test_loader = get_dataloader(args.dataset, batch_size=args.batch_size,
                                    image_size=args.image_size,
                                    num_workers=args.num_workers)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    print(f"Creating model {args.model} @ {args.image_size}x{args.image_size}...")
    model = MedSegDiT_models[args.model](
        input_size=args.image_size,
        mask_channels=1,
        image_channels=args.image_channels,
    ).to(device)

    if args.use_ema and 'ema_shadow' in ckpt:
        print("Loading EMA weights...")
        ema_shadow = ckpt['ema_shadow']
        model_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        if model_keys != set(ema_shadow.keys()):
            raise RuntimeError("The EMA weights do not match the model structure.")
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name])
    else:
        if args.use_ema:
            print("[WARNING] no ema_shadow in the checkpoint, using the raw weights")
        model.load_state_dict(ckpt['model_state_dict'])

    diffusion = DiffusionSchedule(num_timesteps=args.num_timesteps,
                                  schedule_type=args.beta_schedule)

    params = count_params(model)
    gflops = estimate_gflops(model, args.image_size, 1, args.image_channels, device)

    print("Running inference...")
    r = run_inference(model, diffusion, test_loader, device, args)

    # ---- Report ----
    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Dataset:    {args.dataset} ({r['n_images']} images @ {args.image_size})",
        f"Sampling:   DDIM T'={args.ddim_steps}, eta={args.eta}, K={args.K}"
        f"{' (EMA)' if args.use_ema else ''}",
        "",
        f"Params:    {params:.2f} M",
        f"GFLOPs:    {gflops:.2f}" if gflops is not None else "GFLOPs:    N/A",
        f"mIoU:      {r['iou']:.4f}",
        f"DSC:       {r['dice']:.4f}",
        f"Sen:       {r['rec']:.4f}",
        f"Acc:       {r['acc']:.4f}",
        f"Precision: {r['prec']:.4f}",
        (f"HD95:      {r['hd95']:.2f} px" if r['hd95'] is not None
         else "HD95:      N/A (needs scipy, or disabled with --skip_hd95)"),
        (f"GED:       {r['ged']:.4f}" if r['ged'] is not None
         else "GED:       N/A (needs --K > 1)"),
        f"ECE:       {r['ece']:.4f}",
        "",
        f"Total time: {r['total_sec']:.2f} s  ({r['per_image_ms']:.1f} ms/image)",
    ]
    print("\n" + "=" * 50)
    print("\n".join(lines))
    print("=" * 50)

    dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    metrics_path = os.path.join(dataset_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results saved to {metrics_path}")


if __name__ == '__main__':
    main()
