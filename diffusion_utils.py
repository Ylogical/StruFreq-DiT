"""
Diffusion utilities for MedSegDiT.

Mask-space diffusion with the cosine noise schedule, direct clean-mask
prediction (x0) and deterministic DDIM sampling, as used in the paper.
"""
import torch
import torch.nn.functional as F
import numpy as np


class DiffusionSchedule:
    """Cosine noise schedule + DDIM sampler for x0 prediction.

    Masks live in [-1, 1]: the forward process noises the clean mask x0, and the
    network is trained to predict x0 directly at every timestep.
    """

    def __init__(self, num_timesteps=200, schedule_type='cosine'):
        """
        Args:
            num_timesteps: number of diffusion steps T
            schedule_type: 'cosine' (used in the paper) | 'linear'
        """
        self.num_timesteps = num_timesteps

        if schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule_type == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Diffusion coefficients
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """Cosine beta schedule (Nichol & Dhariwal, 2021)."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def _extract(a, t, x_shape):
        """Gather coefficients at t and reshape for broadcasting."""
        out = a.to(t.device).gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion q(x_t | x_0).

            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_start: clean mask (B, C, H, W) in [-1, 1]
            t:       timesteps (B,)
            noise:   optional noise tensor
        Returns:
            (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ac    = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1m_ac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_1m_ac * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, shape, image_cond, device,
                    ddim_steps=20, eta=0.0, progress=False):
        """DDIM sampling from pure noise to the clean mask.

        Args:
            model:      the denoiser, called as model(x_t, t, image_cond)
            shape:      output mask shape (B, C, H, W)
            image_cond: conditioning image (B, 3, H, W)
            device:     target device
            ddim_steps: number of reverse steps T' (20 in the paper)
            eta:        stochasticity, 0 = deterministic (used in the paper)
            progress:   show a progress bar
        Returns:
            the sampled mask (B, C, H, W) in [-1, 1]
        """
        batch_size = shape[0]

        # Uniformly sub-sampled timesteps
        c = self.num_timesteps // ddim_steps
        ddim_timesteps = np.asarray(list(range(0, self.num_timesteps, c)))

        x_t = torch.randn(shape, device=device)

        steps = reversed(ddim_timesteps)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc='DDIM Sampling', total=len(ddim_timesteps))

        for i, t in enumerate(steps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # The model predicts x_0 directly
            pred_x0 = torch.clamp(model(x_t, t_batch, image_cond), -1.0, 1.0)

            alpha_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
            if i < len(ddim_timesteps) - 1:
                t_prev = torch.full((batch_size,),
                                    ddim_timesteps[len(ddim_timesteps) - i - 2],
                                    device=device, dtype=torch.long)
                alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            # Noise implied by the x0 prediction
            pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1.0 - alpha_t)

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )
            noise = (torch.randn_like(x_t) if i < len(ddim_timesteps) - 1
                     else torch.zeros_like(x_t))
            dir_xt = torch.sqrt(1.0 - alpha_t_prev - sigma ** 2) * pred_noise
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise

        return x_t


def dice_loss(pred, target, smooth=1e-5):
    """Soft Dice loss.

    Args:
        pred:   predicted mask (B, C, H, W) in [0, 1]
        target: binary GT mask (B, C, H, W) in {0, 1}
    Returns:
        scalar loss
    """
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice.mean(dim=1)).mean()


def compute_loss(model, diffusion, batch, device, use_dice=True, dice_weight=1.0):
    """Training objective: MSE on the clean mask + soft Dice.

        L = || x0_hat - x0 ||^2 + lambda_dice * Dice(x0_hat, x0)

    Because the same target x0 is regressed at every timestep, the whole
    denoising trajectory receives consistent shape supervision.

    Returns:
        (loss, model_output, target)
    """
    image = batch['image'].to(device)
    mask  = batch['mask'].to(device)        # (B, 1, H, W) in {-1, +1}

    # Sample a timestep per image and add noise
    t = torch.randint(0, diffusion.num_timesteps, (mask.shape[0],), device=device).long()
    x_t, _ = diffusion.q_sample(mask, t)

    pred_x0 = model(x_t, t, image)          # tanh output in [-1, 1]

    loss = F.mse_loss(pred_x0, mask)

    if use_dice:
        # Linear map to [0, 1] instead of a sigmoid, which would squash +-1 into
        # [0.27, 0.73] and weaken the binary supervision.
        loss = loss + dice_weight * dice_loss((pred_x0 + 1.0) / 2.0,
                                              (mask > 0.0).float())

    return loss, pred_x0, mask
