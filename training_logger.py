"""
Training logger for StruFreqDiT: mirrors stdout to a log file, records train/eval losses and
IoU/Dice, and plots the curves. Resumes from an existing loss_history.json.
"""
import os
import sys
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # must be set before importing pyplot
import matplotlib.pyplot as plt
import json


class TrainingLogger:
    """Training logger."""

    def __init__(self, log_dir='logs', exp_name=None):
        """
        Args:
            log_dir: directory the logs are written to
            exp_name: experiment name (auto-generated from the time if None)
        """
        if exp_name is None:
            exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, 'training.log')
        self.loss_file = os.path.join(self.log_dir, 'loss_history.json')
        self.plot_file = os.path.join(self.log_dir, 'loss_curve.png')

        self.log_fp = open(self.log_file, 'a', encoding='utf-8')

        self.train_losses = []
        self.eval_losses = []
        self.eval_epochs = []
        self.epochs = []

        self.iou_list = []
        self.dice_list = []
        self.metric_epochs = []

        self.load_loss_history()

        self.terminal = sys.stdout
        sys.stdout = self

        self.start_time = time.time()
        self.log(f"{'='*60}")
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Log dir: {self.log_dir}")
        self.log(f"{'='*60}\n")

    def write(self, message):
        """stdout hook: write to both the terminal and the log file."""
        self.terminal.write(message)
        self.log_fp.write(message)
        self.log_fp.flush()

    def flush(self):
        self.terminal.flush()
        self.log_fp.flush()

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        self.write(log_message)

    def log_epoch(self, epoch, train_loss, eval_loss=None, lr=None, iou=None, dice=None):
        """
        Args:
            epoch: current epoch
            train_loss: mean training loss
            eval_loss: validation loss (optional)
            lr: current learning rate (optional)
            iou: mean IoU (optional, only computed every few epochs)
            dice: mean Dice (optional, only computed every few epochs)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if eval_loss is not None:
            self.eval_losses.append(eval_loss)
            self.eval_epochs.append(epoch)

        if iou is not None and dice is not None:
            self.iou_list.append(iou)
            self.dice_list.append(dice)
            self.metric_epochs.append(epoch)

        message = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f}"
        if eval_loss is not None:
            message += f" | Eval Loss: {eval_loss:.6f}"
        if lr is not None:
            message += f" | LR: {lr:.6e}"
        if iou is not None:
            message += f" | IoU: {iou:.4f}"
        if dice is not None:
            message += f" | Dice: {dice:.4f}"

        self.log(message)
        self.save_loss_history()

        if epoch % 10 == 0:
            self.plot_loss_curve()

    def save_loss_history(self):
        history = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'eval_epochs': self.eval_epochs,
            'iou_list': self.iou_list,
            'dice_list': self.dice_list,
            'metric_epochs': self.metric_epochs,
        }
        with open(self.loss_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def load_loss_history(self):
        if os.path.exists(self.loss_file):
            try:
                with open(self.loss_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                self.epochs = history.get('epochs', [])
                self.train_losses = history.get('train_losses', [])
                self.eval_losses = history.get('eval_losses', [])
                self.eval_epochs = history.get('eval_epochs', [])
                self.iou_list = history.get('iou_list', [])
                self.dice_list = history.get('dice_list', [])
                self.metric_epochs = history.get('metric_epochs', [])
                # Older histories have no eval_epochs field
                if not self.eval_epochs and self.eval_losses:
                    self.eval_epochs = [self.epochs[i] for i in range(len(self.eval_losses))]
                self.log(f"Loaded loss history: {len(self.epochs)} epochs")
            except Exception:
                pass

    def plot_loss_curve(self):
        """Plot the loss curves, plus IoU/Dice when metrics were recorded."""
        if len(self.epochs) == 0:
            return

        has_metrics = len(self.metric_epochs) > 0
        nrows = 2 if has_metrics else 1
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
        if nrows == 1:
            axes = [axes]

        for col, use_log in enumerate([False, True]):
            ax = axes[0][col]
            ax.plot(self.epochs, self.train_losses, 'b-', label='Train Loss',
                    linewidth=1.5, alpha=0.85)
            if len(self.eval_losses) > 0:
                ax.plot(self.eval_epochs, self.eval_losses, 'r-o',
                        label='Eval Loss', linewidth=2, markersize=4)
            if use_log:
                ax.set_yscale('log')
                ax.set_ylabel('Loss (log scale)', fontsize=12)
                ax.set_title('Training and Evaluation Loss (Log Scale)',
                             fontsize=13, fontweight='bold')
            else:
                ax.set_ylabel('Loss', fontsize=12)
                ax.set_title('Training and Evaluation Loss',
                             fontsize=13, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        if has_metrics:
            for col, (values, label, color) in enumerate([
                (self.iou_list,  'IoU',  'green'),
                (self.dice_list, 'Dice', 'orange'),
            ]):
                ax = axes[1][col]
                ax.plot(self.metric_epochs, values, color=color, marker='o',
                        linewidth=2, markersize=5, label=label)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(label, fontsize=12)
                ax.set_title(f'{label} over Training', fontsize=13, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def close(self):
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        self.log(f"\n{'='*60}")
        self.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Elapsed: {hours}h {minutes}m {seconds}s")
        self.log(f"{'='*60}")

        self.plot_loss_curve()

        sys.stdout = self.terminal
        self.log_fp.close()

        print(f"\nLogs saved to: {self.log_dir}")
        print(f"  - training log: {self.log_file}")
        print(f"  - loss history: {self.loss_file}")
        print(f"  - loss curve:   {self.plot_file}")
