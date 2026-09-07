"""
MedSegDiT training entry point.

The model is trained from scratch (no pretrained weights) and relies on EMA,
early stopping, weight decay and dropout for regularization.

Per-dataset defaults (image size, model variant, epochs, lr, batch size, early
stopping, ...) are selected automatically from --dataset; see _DATASET_DEFAULTS
below. Run `python train.py --help` for the full list of flags.
"""
import os
import json
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_medsegdit import MedSegDiT_models
from dataset import get_dataloader_with_val
from diffusion_utils import DiffusionSchedule, compute_loss
from training_logger import TrainingLogger

# ============================================================
# Dataset-specific defaults (see Sec. IV-A of the paper)
# ============================================================
_DATASET_DEFAULTS = {
    # Small datasets: many epochs are needed for the diffusion model to
    # converge. Early stop after patience x metrics_eval_freq epochs without
    # improvement, but not before es_min_epochs.
    'glas':    {'image_size': 256, 'model': 'MedSegDiT-B/16', 'batch_size': 8,
                'lr': 2e-4, 'epochs': 20000, 'warmup_epochs': 50,
                'patience': 20, 'metrics_eval_freq': 30, 'es_min_epochs': 10000,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
    'ph2':     {'image_size': 256, 'model': 'MedSegDiT-B/16', 'batch_size': 8,
                'lr': 2e-4, 'epochs': 20000, 'warmup_epochs': 50,
                'patience': 20, 'metrics_eval_freq': 30, 'es_min_epochs': 10000,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
    'imid':    {'image_size': 256, 'model': 'MedSegDiT-B/16', 'batch_size': 4,
                'lr': 2e-4, 'epochs': 20000, 'warmup_epochs': 50,
                'patience': 20, 'metrics_eval_freq': 30, 'es_min_epochs': 10000,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
    # High-resolution nuclei dataset: 512x512 input with patch size 32.
    'monuseg': {'image_size': 512, 'model': 'MedSegDiT-B/32', 'batch_size': 4,
                'lr': 1e-4, 'epochs': 20000, 'warmup_epochs': 60,
                'patience': 20, 'metrics_eval_freq': 25, 'es_min_epochs': 10000,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
}


class EMA:
    """Exponential Moving Average of the model parameters."""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = ((1.0 - self.decay) * param.data
                                     + self.decay * self.shadow[name]).clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """Early stopping on the validation mIoU.

    patience counts IoU evaluations (not epochs), so the behaviour does not
    depend on --metrics_eval_freq; min_epochs is a protection period.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4, min_epochs: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.best_epoch = -1

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        if epoch < self.min_epochs:
            return False
        self.counter += 1
        return self.counter >= self.patience

    def status_str(self) -> str:
        return (f'[EarlyStop] counter={self.counter}/{self.patience}  '
                f'best_mIoU={self.best_score:.4f} @ epoch {self.best_epoch}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MedSegDiT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python train.py --dataset glas
  python train.py --dataset monuseg --gpu 0
  python train.py --dataset glas --resume checkpoints/<run>/checkpoint_epoch100.pth
''')
    # Data
    parser.add_argument('--dataset', type=str, default='glas',
                        choices=['glas', 'monuseg', 'ph2', 'imid'])
    parser.add_argument('--image_size', type=int, default=None,
                        help='Input size (default: glas/ph2/imid=256, monuseg=512)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: glas/ph2=8, monuseg/imid=4)')
    parser.add_argument('--num_workers', type=int, default=4)

    # Model
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MedSegDiT_models.keys()),
                        help='Model variant (default: B/16, monuseg=B/32)')
    parser.add_argument('--image_channels', type=int, default=3)

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=200,
                        help='Number of diffusion steps T')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'])

    # Optimization
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: glas/ph2/imid=2e-4, monuseg=1e-4)')
    parser.add_argument('--weight_decay', type=float, default=None, help='Default: 0.05')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--no_ema', action='store_true', default=False,
                        help='Disable EMA (enabled by default)')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--no_dice_loss', action='store_true', default=False,
                        help='Train with the MSE term only')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Weight of the Dice term (lambda_Dice)')
    parser.add_argument('--attn_drop', type=float, default=None, help='Default: 0.1')
    parser.add_argument('--proj_drop', type=float, default=None, help='Default: 0.1')

    # Logging / checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--eval_freq', type=int, default=None,
                        help='Validation-loss interval in epochs (default: metrics_eval_freq)')
    parser.add_argument('--metrics_eval_freq', type=int, default=None,
                        help='DDIM mIoU/DSC evaluation interval in epochs')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early-stopping patience, in IoU evaluations (0 disables it)')
    parser.add_argument('--es_delta', type=float, default=1e-4)
    parser.add_argument('--es_min_epochs', type=int, default=None,
                        help='No early stop before this epoch')
    parser.add_argument('--save_freq', type=int, default=0,
                        help='Save a periodic checkpoint every N epochs (0 = off)')

    # Misc
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """Cosine LR schedule with warmup (min lr = 0.05 x initial)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / denom
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch,
                    writer, args, ema=None):
    model.train()
    total_loss = 0.0
    loss_window = deque(maxlen=min(20, len(train_loader)))

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
    for step, batch in enumerate(pbar):
        loss, _, _ = compute_loss(
            model, diffusion, batch, device,
            use_dice=not args.no_dice_loss,
            dice_weight=args.dice_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update()

        loss_val = loss.item()
        total_loss += loss_val
        loss_window.append(loss_val)
        smooth_loss = sum(loss_window) / len(loss_window)
        pbar.set_postfix({'loss(smooth)': f'{smooth_loss:.4f}',
                          'loss(step)': f'{loss_val:.4f}'})

        global_step = epoch * len(train_loader) + step
        writer.add_scalar('train/loss_step', loss_val, global_step)
        writer.add_scalar('train/loss_smooth', smooth_loss, global_step)

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, diffusion, val_loader, device, epoch, writer, args):
    """Validation loss."""
    model.eval()
    total_loss = 0.0
    for batch in tqdm(val_loader, desc='Evaluating', leave=False):
        loss, _, _ = compute_loss(
            model, diffusion, batch, device,
            use_dice=not args.no_dice_loss,
            dice_weight=args.dice_weight,
        )
        total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    writer.add_scalar('eval/loss', avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def evaluate_metrics(model, diffusion, val_loader, device, epoch, writer, args):
    """Validation mIoU / DSC / Sensitivity / Accuracy via 20-step DDIM sampling."""
    model.eval()
    iou_list, dice_list, sen_list, acc_list = [], [], [], []

    for batch in tqdm(val_loader, desc=f'Metrics Eval (Epoch {epoch + 1})', leave=False):
        images   = batch['image'].to(device)
        gt_masks = batch['mask'].to(device)
        B, C, H, W = gt_masks.shape

        pred_masks = diffusion.ddim_sample(model, (B, C, H, W), images, device,
                                           ddim_steps=20, eta=0.0, progress=False)

        pred_bin = (pred_masks > 0.0).float()
        gt_bin   = (gt_masks   > 0.0).float()
        for i in range(B):
            p = pred_bin[i].reshape(-1)
            g = gt_bin[i].reshape(-1)
            tp = (p * g).sum().item()
            fp = (p * (1 - g)).sum().item()
            fn = ((1 - p) * g).sum().item()
            tn = ((1 - p) * (1 - g)).sum().item()
            iou_list.append(tp / (tp + fp + fn + 1e-6))
            dice_list.append(2 * tp / (2 * tp + fp + fn + 1e-6))
            sen_list.append(tp / (tp + fn + 1e-6))
            acc_list.append((tp + tn) / (tp + fp + fn + tn + 1e-6))

    mean_iou  = float(np.mean(iou_list))
    mean_dice = float(np.mean(dice_list))
    mean_sen  = float(np.mean(sen_list))
    mean_acc  = float(np.mean(acc_list))

    writer.add_scalar('eval/iou',  mean_iou,  epoch)
    writer.add_scalar('eval/dice', mean_dice, epoch)
    writer.add_scalar('eval/sen',  mean_sen,  epoch)
    writer.add_scalar('eval/acc',  mean_acc,  epoch)

    return mean_iou, mean_dice, mean_sen, mean_acc


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path,
                    ema=None, args=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    if args is not None:
        checkpoint['train_config'] = {
            'model': args.model,
            'image_size': args.image_size,
            'num_timesteps': args.num_timesteps,
            'beta_schedule': args.beta_schedule,
            'attn_drop': args.attn_drop,
            'proj_drop': args.proj_drop,
            'weight_decay': args.weight_decay,
        }
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, ema=None):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if ema is not None and 'ema_shadow' in checkpoint:
        ema.shadow = checkpoint['ema_shadow']
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {checkpoint['epoch']}, "
          f"loss {checkpoint['loss']:.4f}")
    return checkpoint['epoch']


def main():
    args = parse_args()

    # ---- Dataset-specific defaults for unspecified args ----
    cfg = _DATASET_DEFAULTS[args.dataset.lower()]
    for key in ('image_size', 'batch_size', 'model', 'epochs', 'lr', 'weight_decay',
                'warmup_epochs', 'attn_drop', 'proj_drop', 'metrics_eval_freq',
                'patience', 'es_min_epochs'):
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    if args.eval_freq is None:
        args.eval_freq = args.metrics_eval_freq

    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.dataset}_{args.model.replace('/', '_')}_{timestamp}"
    save_dir = os.path.join(args.save_dir, exp_name)
    log_dir  = os.path.join(args.log_dir,  exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    writer = SummaryWriter(log_dir)

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Dataset: {args.dataset} | Image: {args.image_size} | Batch: {args.batch_size}")
    print(f"Model: {args.model} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"WeightDecay: {args.weight_decay} | Dropout(attn/proj): "
          f"{args.attn_drop}/{args.proj_drop}")
    print(f"Warmup: {args.warmup_epochs} | Patience: {args.patience} "
          f"(early stop after {args.patience} x {args.metrics_eval_freq} = "
          f"{args.patience * args.metrics_eval_freq} epochs w/o improvement)")
    print(f"EMA: {'decay=' + str(args.ema_decay) if not args.no_ema else 'disabled'}"
          f" | Training from scratch (no pretrained weights)")
    print("=" * 60)

    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Data
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloader_with_val(
        args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    # Model
    print("Creating model...")
    model = MedSegDiT_models[args.model](
        input_size=args.image_size,
        mask_channels=1,
        image_channels=args.image_channels,
        attn_drop=args.attn_drop,
        proj_drop=args.proj_drop,
    ).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_p:,} ({total_p / 1e6:.2f}M)")

    # Diffusion
    diffusion = DiffusionSchedule(num_timesteps=args.num_timesteps,
                                  schedule_type=args.beta_schedule)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)

    ema = EMA(model, decay=args.ema_decay) if not args.no_ema else None

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, ema) + 1

    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.es_delta,
                                  min_epochs=args.es_min_epochs) if args.patience > 0 else None

    logger = TrainingLogger(log_dir=log_dir, exp_name=exp_name)
    logger.log(f"Training config:\n{json.dumps(vars(args), indent=2, default=str)}")
    logger.log(f"Dataset: {args.dataset}  Train={len(train_loader.dataset)}  "
               f"Val={len(val_loader.dataset)}  Test={len(test_loader.dataset)}")
    logger.log(f"Model params: {total_p:,} ({total_p / 1e6:.2f}M)")

    # ---- Training loop ----
    print("Start training...")
    best_iou = -1.0
    best_loss = float('inf')
    train_loss = float('inf')
    last_epoch = start_epoch

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        train_loss = train_one_epoch(model, diffusion, train_loader, optimizer,
                                     device, epoch, writer, args, ema)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        writer.add_scalar('train/loss_epoch', train_loss, epoch)

        # ---- Validation loss ----
        eval_loss = None
        if (epoch + 1) % args.eval_freq == 0:
            if ema is not None:
                ema.apply_shadow()
            try:
                eval_loss = evaluate(model, diffusion, val_loader, device, epoch, writer, args)
            finally:
                if ema is not None:
                    ema.restore()

            # Before the first IoU evaluation, keep the best model by loss
            if best_iou < 0 and eval_loss is not None and eval_loss < best_loss:
                best_loss = eval_loss
                if ema is not None:
                    ema.apply_shadow()
                save_checkpoint(model, optimizer, scheduler, epoch, eval_loss,
                                os.path.join(save_dir, 'best_model.pth'), ema, args)
                if ema is not None:
                    ema.restore()

        # ---- IoU / DSC metrics ----
        iou = dice = sen = acc = None
        if (epoch + 1) % args.metrics_eval_freq == 0:
            if ema is not None:
                ema.apply_shadow()
            try:
                iou, dice, sen, acc = evaluate_metrics(model, diffusion, val_loader,
                                                       device, epoch, writer, args)
                if iou > best_iou:
                    best_iou = iou
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        eval_loss if eval_loss is not None else train_loss,
                        os.path.join(save_dir, 'best_model.pth'), ema, args,
                    )
                    logger.log(f"  >> Best model (mIoU={best_iou:.4f} DSC={dice:.4f} "
                               f"Sen={sen:.4f} Acc={acc:.4f})")
            finally:
                if ema is not None:
                    ema.restore()
            logger.log(f"  >> Metrics @ {epoch + 1}: mIoU={iou:.4f} DSC={dice:.4f} "
                       f"Sen={sen:.4f} Acc={acc:.4f}  best_mIoU={best_iou:.4f}")

            if early_stopper is not None and early_stopper(iou, epoch):
                print(f'[Early Stop] epoch={epoch + 1}  '
                      f'best_mIoU={early_stopper.best_score:.4f} '
                      f'@ epoch {early_stopper.best_epoch + 1}')
                logger.log(f'[Early Stop] epoch={epoch + 1}  '
                           f'best_mIoU={early_stopper.best_score:.4f} '
                           f'@ epoch {early_stopper.best_epoch + 1}')
                break
            if early_stopper is not None:
                print(f'  {early_stopper.status_str()}')

        # ---- Periodic checkpoint ----
        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            if ema is not None:
                ema.apply_shadow()
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss,
                            os.path.join(save_dir, f'checkpoint_epoch{epoch + 1}.pth'),
                            ema, args)
            if ema is not None:
                ema.restore()

        logger.log_epoch(epoch, train_loss, eval_loss, current_lr, iou=iou, dice=dice)

    # ---- Save the final model ----
    if ema is not None:
        ema.apply_shadow()
    save_checkpoint(model, optimizer, scheduler, last_epoch, train_loss,
                    os.path.join(save_dir, 'final_model.pth'), ema, args)

    logger.close()
    writer.close()
    print("Training completed! Use best_model.pth (best validation mIoU) for testing.")


if __name__ == '__main__':
    main()
