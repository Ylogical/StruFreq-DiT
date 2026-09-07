# --------------------------------------------------------
# MedSegDiT: A Diffusion Transformer for Medical Image Segmentation.
#
# The diffusion process runs in the segmentation-mask space; the network is the
# denoiser that predicts the clean mask x0 from a noisy mask x_t conditioned on
# the medical image.
#
#   SSE encoder (image / noisy mask, dual stream, multi-scale)
#     -> DiT backbone over mask tokens (self-attention + 2D RoPE + adaLN-Zero)
#     -> DFCA + PBDF frequency-decoupled condition injection after every block
#     -> SSE decoder (symmetric U-shaped, dual-stream skips) -> x0_hat
#
# The image condition reaches the mask stream through DFCA only: the
# conditioning vector c carries the timestep, and image tokens never enter the
# backbone self-attention.
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Timestep embedder: maps a scalar timestep to a vector."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(q, k, v,
                                          dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


# ============================================================
# SSE encoding path: one multi-scale CNN branch
# (Spatial Structure Enhancement, Sec. III-C of the paper)
# ============================================================

class ConvBlock(nn.Module):
    """Double convolution block: (Conv-GN-GELU) x 2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleEncoder(nn.Module):
    """One SSE encoding branch (E_img or E_msk).

    Cascaded Conv-GroupNorm-GELU blocks with downsampling produce a feature
    pyramid {f1, f2, f3} used as decoder skips, and the last stage is pooled and
    projected into N = (H/p)^2 backbone tokens.
    """
    def __init__(self, in_chans=3, hidden_size=768, patch_size=16, input_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        D = hidden_size

        self.enc1 = ConvBlock(in_chans, 64)             # H x W, 64ch
        self.down1 = nn.Conv2d(64, 128, 2, 2)           # H/2 x W/2

        self.enc2 = ConvBlock(128, 256)                  # H/2 x W/2, 256ch
        self.down2 = nn.Conv2d(256, 512, 2, 2)          # H/4 x W/4

        self.enc3 = ConvBlock(512, D)                    # H/4 x W/4, D ch

        # enc1..down2 downsample by 4x, to_tokens by another patch_size//4.
        # AvgPool + 1x1 Conv instead of a large strided conv, so the parameter
        # count does not grow with the square of patch_size.
        stride = patch_size // 4
        self.to_tokens = nn.Sequential(
            nn.AvgPool2d(kernel_size=stride, stride=stride),
            nn.Conv2d(D, D, kernel_size=1),
        )

        self.skip_channels = [64, 256, D]                # [f1_ch, f2_ch, f3_ch]
        self.num_patches = (input_size // patch_size) ** 2

    def forward(self, x):
        f1 = self.enc1(x)                                # (B, 64,  H,   W)
        f2 = self.enc2(self.down1(f1))                   # (B, 256, H/2, W/2)
        f3 = self.enc3(self.down2(f2))                   # (B, D,   H/4, W/4)
        tokens_2d = self.to_tokens(f3)                   # (B, D,   H/p, W/p)
        tokens = tokens_2d.flatten(2).transpose(1, 2)    # (B, N, D)
        return tokens, [f1, f2, f3]


# ============================================================
# SSE decoding path: symmetric multi-scale decoder
# ============================================================

class UpBlock(nn.Module):
    """Bilinear upsample + skip concat + double conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class TimeModulatedSkip(nn.Module):
    """
    Time-Modulated joint skip (TM-Skip).

    At every decoder scale the two skip streams are first concatenated into a
    joint skip:
        S_i = [m_i ; n_i]   with m_i the mask-encoder skip (noisy mask state)
                            and  n_i the image-encoder skip (image detail)
    which is then modulated per timestep with FiLM driven by e_t:
        gamma_i, beta_i = MLP_i(e_t)                 # (B, C)
        F_i             = S_i * (1 + gamma_i) + beta_i

    Without the time modulation the joint skip is a pure conditioning bypass:
    one mapping fits all t, the network ignores x_t and degenerates into a
    discriminative model. FiLM scales and shifts the same skip differently at
    every noise level, so the skip becomes part of the reverse trajectory.

    Init: the last t_mlp layer uses small random weights (std=0.02) and a zero
    bias, so gamma/beta vary with t from the first step (a zero init brings the
    shortcut back) while the constant component starts at identity.
    """
    def __init__(self, skip_ch, cond_dim, t_proj_dim=64):
        super().__init__()
        # e_t -> (gamma, beta), one pair per channel
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, t_proj_dim),
            nn.SiLU(),
            nn.Linear(t_proj_dim, 2 * skip_ch),
        )
        # Small random weights + zero bias; the effective init lives in the
        # outer initialize_weights, this stays consistent with it.
        nn.init.normal_(self.t_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.t_mlp[-1].bias)

    def forward(self, skip, e_t):
        gamma, beta = self.t_mlp(e_t).chunk(2, dim=1)    # each (B, C)
        return skip * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class UNetDecoder(nn.Module):
    """Symmetric U-shaped decoder of SSE.

    The backbone tokens are adaLN-modulated by the timestep, reshaped to a
    feature map and upsampled stage by stage; at every scale the image-stream
    and mask-stream skips of the same resolution are concatenated (TM-Skip) and
    fused by a convolutional block. A 1x1 conv gives the clean-mask estimate.
    """
    def __init__(self, hidden_size, out_channels, mask_skip_channels, image_skip_channels):
        super().__init__()
        D = hidden_size

        self.norm   = RMSNorm(D)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D, 2 * D, bias=True)
        )

        self.stage0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(D, D // 2),
        )

        # Joint skip channels per scale = mask_skip + image_skip
        sc = [mask_skip_channels[i] + image_skip_channels[i] for i in range(3)]
        self.tm3 = TimeModulatedSkip(sc[2], cond_dim=D)   # deep, low resolution
        self.tm2 = TimeModulatedSkip(sc[1], cond_dim=D)
        self.tm1 = TimeModulatedSkip(sc[0], cond_dim=D)   # shallow, high resolution

        self.up1 = UpBlock(D // 2, sc[2], D // 4)
        self.up2 = UpBlock(D // 4, sc[1], D // 8)
        self.up3 = UpBlock(D // 8, sc[0], 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, c, mask_skips, image_skips, t_emb):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)

        B, N, D = x.shape
        hw = math.isqrt(N)
        assert hw * hw == N, f"number of mask tokens {N} is not a perfect square"
        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)

        x = self.stage0(x)

        def _joint(idx):
            return torch.cat([mask_skips[idx], image_skips[idx]], dim=1)

        x = self.up1(x, self.tm3(_joint(2), t_emb))
        x = self.up2(x, self.tm2(_joint(1), t_emb))
        x = self.up3(x, self.tm1(_joint(0), t_emb))

        return self.out_conv(x)


class DFCA(nn.Module):
    """
    Diffusion Frequency Cross-Attention (DFCA) with the Parametric
    Band-Decoupling Filter (PBDF), Sec. III-D of the paper.

    Design:
      1. PBDF frequency decomposition: a 2D FFT of image_tokens plus two
         learnable band parameters separate low frequencies
         (structure/semantics) from a mid-to-high band (detail/boundary).
      2. Two independent attention streams:
         - semantic stream: low-frequency K/V, large temperature tau_max
         - detail stream:   high-frequency K/V, small temperature tau_min
      3. alpha(t) diffusion-aware fusion of the two stream outputs only
         (K/V and tau are not coupled to t): high t favours the semantic
         stream, low t favours the detail stream.
    """
    def __init__(self, hidden_size: int, num_heads: int, tau_min: float = 0.5, tau_max: float = 1.5,
                 attn_drop: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.tau_min   = tau_min   # detail stream, sharp attention
        self.tau_max   = tau_max   # semantic stream, smooth attention
        self.attn_drop = attn_drop

        # PBDF bands, initialised to reproduce a fixed sigma = hw/8 Gaussian
        # low-pass (warm start).
        #   sigma_s = hw / exp(band_low_log); init ln(8) -> sigma_s = hw/8
        self.band_low_log = nn.Parameter(torch.tensor(math.log(8.0)))
        # Ordered bandwidth: sigma_n = sigma_s + hw*softplus(band_gap_raw).
        # The initial normalized gap gives sigma_n = hw/1.5.
        init_gap = 1.0 / 1.5 - 1.0 / 8.0
        self.band_gap_raw = nn.Parameter(torch.tensor(math.log(math.expm1(init_gap))))

        # Q side: AdaLN modulation + Q projection (Q comes from mask tokens)
        self.norm_q  = RMSNorm(hidden_size)
        self.norm_kv = RMSNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)

        # Two K/V projections, one per frequency band
        self.to_k_sem = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_sem = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k_det = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_det = nn.Linear(hidden_size, hidden_size, bias=False)

        # Detail refinement on the high-frequency features after the inverse
        # FFT, which adaptively amplifies boundary responses.
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1, groups=hidden_size, bias=False),
            nn.GroupNorm(min(32, hidden_size), hidden_size),
            nn.GELU(),
        )

        # alpha(t) fusion weight, warm-started to 0.5 by a zero init
        self.proj_alpha = nn.Linear(hidden_size, 1, bias=True)

        # Output projection, zero-initialised for a warm start
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def _freq_decompose(self, kv_in: torch.Tensor, hw: int):
        """PBDF decomposition on the centred 2D spectrum:
            low  (semantics) = low-pass:  m_low  = exp(-r2 / 2 sigma_s^2)
            high (detail)    = band-pass: m_high = (1-m_low)*exp(-r2 / 2 sigma_n^2)
        sigma_n suppresses the very top frequencies, keeping the mid-to-high
        boundary band. Forced to fp32 because FFT is unstable under autocast.
        """
        B, Ni, D = kv_in.shape
        feat     = kv_in.reshape(B, hw, hw, D).permute(0, 3, 1, 2).contiguous()  # (B, D, hw, hw)

        with torch.amp.autocast(device_type=feat.device.type, enabled=False):
            feat_f   = feat.float()
            # 2D FFT, then fftshift to centre the spectrum
            feat_fft = torch.fft.fft2(feat_f, dim=(-2, -1), norm='ortho')
            feat_fft = torch.fft.fftshift(feat_fft, dim=(-2, -1))

            # Squared radius from the spectrum centre
            coords = torch.arange(hw, device=feat.device, dtype=torch.float32) - hw // 2
            yy, xx = torch.meshgrid(coords, coords, indexing='ij')
            r2     = xx ** 2 + yy ** 2

            sigma_s = (hw / torch.exp(self.band_low_log)).clamp(min=1.0)
            sigma_n = sigma_s + hw * F.softplus(self.band_gap_raw)
            m_low   = torch.exp(-r2 / (2 * sigma_s ** 2))
            m_high  = (1.0 - m_low) * torch.exp(-r2 / (2 * sigma_n ** 2))

            m_low    = m_low.unsqueeze(0).unsqueeze(0)   # (1, 1, hw, hw)
            m_high   = m_high.unsqueeze(0).unsqueeze(0)

            fft_low  = feat_fft * m_low
            fft_high = feat_fft * m_high

            # Inverse shift + iFFT back to the spatial domain, keep the real part
            kv_low_spatial  = torch.fft.ifft2(torch.fft.ifftshift(fft_low,  dim=(-2, -1)),
                                              dim=(-2, -1), norm='ortho').real
            kv_high_spatial = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)),
                                              dim=(-2, -1), norm='ortho').real

        # Back to the original dtype; refine the high band
        kv_low_spatial  = kv_low_spatial.to(feat.dtype)
        kv_high_spatial = self.detail_enhance(kv_high_spatial.to(feat.dtype))

        kv_low  = kv_low_spatial.permute(0, 2, 3, 1).reshape(B, Ni, D)
        kv_high = kv_high_spatial.permute(0, 2, 3, 1).reshape(B, Ni, D)
        return kv_low, kv_high

    def forward(
        self,
        mask_tokens:  torch.Tensor,   # (B, N,  D)  Q side
        image_tokens: torch.Tensor,   # (B, Ni, D)  K/V side (static image features)
        c:            torch.Tensor,   # (B, D)       conditioning vector
        t_emb:        torch.Tensor,   # (B, D)       timestep embedding for alpha(t)
    ) -> torch.Tensor:
        B, N,  D  = mask_tokens.shape
        _,  Ni, _ = image_tokens.shape
        H,  Hd    = self.num_heads, self.head_dim

        # AdaLN modulation on the Q side
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        q_in  = modulate(self.norm_q(mask_tokens),  shift, scale)
        kv_in = self.norm_kv(image_tokens)

        # FFT decomposition into a low and a high band
        kv_hw = math.isqrt(Ni)
        assert kv_hw * kv_hw == Ni, (
            f"number of image tokens {Ni} is not a perfect square, a square "
            f"feature map is required; check the image_encoder output"
        )
        kv_low, kv_high = self._freq_decompose(kv_in, kv_hw)

        # Q / K / V projections
        q     = self.to_q(q_in).reshape(B, N, H, Hd).transpose(1, 2)              # (B, H, N,  Hd)
        k_sem = self.to_k_sem(kv_low).reshape(B, Ni, H, Hd).transpose(1, 2)        # (B, H, Ni, Hd)
        v_sem = self.to_v_sem(kv_low).reshape(B, Ni, H, Hd).transpose(1, 2)
        k_det = self.to_k_det(kv_high).reshape(B, Ni, H, Hd).transpose(1, 2)
        v_det = self.to_v_det(kv_high).reshape(B, Ni, H, Hd).transpose(1, 2)

        alpha = torch.sigmoid(self.proj_alpha(t_emb))          # (B, 1), depends on t

        # Large tau smooths the semantic stream, small tau sharpens the detail one
        scale_sem = 1.0 / (math.sqrt(Hd) * self.tau_max)
        scale_det = 1.0 / (math.sqrt(Hd) * self.tau_min)

        # Both streams in fp32 to avoid precision loss at the autocast boundary
        with torch.amp.autocast(device_type=mask_tokens.device.type, enabled=False):
            q_f, k_sem_f, v_sem_f = q.float(), k_sem.float(), v_sem.float()
            k_det_f, v_det_f      = k_det.float(), v_det.float()
            alpha_f               = alpha.float().unsqueeze(1).unsqueeze(-1)  # (B, 1, 1, 1)

            # Semantic stream (low band)
            attn_sem = (q_f @ k_sem_f.transpose(-2, -1)) * scale_sem
            attn_sem = attn_sem.softmax(dim=-1)
            if self.training and self.attn_drop > 0.0:
                attn_sem = F.dropout(attn_sem, p=self.attn_drop, training=True)
            out_sem = attn_sem @ v_sem_f                                   # (B, H, N, Hd)

            # Detail stream (high band)
            attn_det = (q_f @ k_det_f.transpose(-2, -1)) * scale_det
            attn_det = attn_det.softmax(dim=-1)
            if self.training and self.attn_drop > 0.0:
                attn_det = F.dropout(attn_det, p=self.attn_drop, training=True)
            out_det = attn_det @ v_det_f

            # alpha(t) fusion
            out = alpha_f * out_sem + (1.0 - alpha_f) * out_det            # (B, H, N, Hd)

        out = out.to(mask_tokens.dtype).transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        return self.proj(out)


class MedSegDiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MedSegDiT_Segmentation(nn.Module):

    def __init__(
        self,
        input_size=256,          # input image size
        patch_size=16,           # patch size
        mask_channels=1,         # mask channels (binary segmentation = 1)
        image_channels=3,        # image channels (RGB = 3)
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.1,           # attention dropout in the middle blocks
        proj_drop=0.1,           # projection dropout in the middle blocks
        dfca_attn_drop=0.1,      # attention dropout inside DFCA
    ):
        super().__init__()
        self.mask_channels = mask_channels
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size

        # === Embeddings ===
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.mask_encoder = MultiScaleEncoder(
            in_chans=mask_channels, hidden_size=hidden_size,
            patch_size=patch_size, input_size=input_size
        )
        self.image_encoder = MultiScaleEncoder(
            in_chans=image_channels, hidden_size=hidden_size,
            patch_size=patch_size, input_size=input_size
        )

        # === Modality embedding ===
        self.modality_embed_image = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.modality_embed_mask  = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.modality_embed_image, std=0.02)
        torch.nn.init.normal_(self.modality_embed_mask,  std=0.02)

        # === Positional encoding ===
        num_patches = self.mask_encoder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # === 2D rotary position encoding ===
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=hidden_size // num_heads // 2,
            pt_seq_len=input_size // patch_size,
        )

        # === Conditioning MLP ===
        # The image reaches the mask stream through DFCA only, so c is a
        # function of the timestep alone.
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # === Transformer blocks ===
        self.blocks = nn.ModuleList([
            MedSegDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                        proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # === DFCA, one module after every block ===
        self.dfca_layers = nn.ModuleList([
            DFCA(hidden_size, num_heads, attn_drop=dfca_attn_drop)
            for _ in range(depth)
        ])

        # === Output head ===
        self.final_layer = UNetDecoder(
            hidden_size=hidden_size,
            out_channels=mask_channels,
            mask_skip_channels=self.mask_encoder.skip_channels,
            image_skip_channels=self.image_encoder.skip_channels,
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Shared 2D sincos positional encoding, hw inferred from pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[1] ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # CNN encoders
        for encoder in (self.image_encoder, self.mask_encoder):
            for m in encoder.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Conditioning MLP
        nn.init.xavier_uniform_(self.cond_mlp[0].weight)
        nn.init.constant_(self.cond_mlp[0].bias, 0)
        nn.init.normal_(self.cond_mlp[2].weight, std=0.02)
        nn.init.constant_(self.cond_mlp[2].bias, 0)

        # Zero-out adaLN modulation, so each block starts as the identity
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Decoder adaLN
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # 1x1 out_conv: small random init instead of the default kaiming_uniform,
        # which keeps the starting prediction near the linear region of tanh
        nn.init.trunc_normal_(self.final_layer.out_conv.weight, std=0.02)
        nn.init.constant_(self.final_layer.out_conv.bias, 0)

        # TM-Skip: the last t_mlp layer gets small random weights (std=0.02) and
        # a zero bias, so gamma/beta depend on t from step 0 and the joint skip
        # cannot degenerate into a time-independent bypass. This must run after
        # _basic_init (Xavier) above, otherwise Xavier overwrites it.
        for tm in (self.final_layer.tm1, self.final_layer.tm2, self.final_layer.tm3):
            nn.init.normal_(tm.t_mlp[-1].weight, std=0.02)
            nn.init.zeros_(tm.t_mlp[-1].bias)

        # DFCA warm start: zero proj + adaLN, so the output starts near 0, and
        # zero proj_alpha -> sigmoid(0) = 0.5, an even mix of the two streams
        for dfca in self.dfca_layers:
            nn.init.zeros_(dfca.proj.weight)
            nn.init.zeros_(dfca.proj.bias)
            nn.init.constant_(dfca.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(dfca.adaLN_modulation[-1].bias, 0)
            nn.init.zeros_(dfca.proj_alpha.weight)
            nn.init.zeros_(dfca.proj_alpha.bias)

    def forward(self, x_mask_noisy, t, image_cond):
        # === 1. Image tokens and image skips (static, independent of t) ===
        image_tokens, image_skips = self.image_encoder(image_cond)
        image_tokens = image_tokens + self.pos_embed + self.modality_embed_image

        # === 2. Conditioning vector ===
        t_emb = self.t_embedder(t)
        c = self.cond_mlp(t_emb)

        # === 3. Mask tokens and mask skips (depend on x_t) ===
        mask_tokens, mask_skips = self.mask_encoder(x_mask_noisy)
        mask_tokens = mask_tokens + self.pos_embed + self.modality_embed_mask

        # === 4. Backbone: self-attention over mask tokens, DFCA injection ===
        for block, dfca in zip(self.blocks, self.dfca_layers):
            mask_tokens = block(mask_tokens, c, self.feat_rope)
            mask_tokens = mask_tokens + dfca(mask_tokens, image_tokens, c, t_emb)

        # === 5. Output head ===
        # tanh maps the clean-mask estimate into the [-1, 1] mask range
        return torch.tanh(self.final_layer(mask_tokens, c, mask_skips, image_skips, t_emb))


# ============================================
# Model variants
# ============================================

def MedSegDiT_S_16(**kwargs):
    """Small variant, patch_size=16 (4 blocks)."""
    return MedSegDiT_Segmentation(depth=4, hidden_size=512, num_heads=8, patch_size=16, **kwargs)

def MedSegDiT_S_32(**kwargs):
    """Small variant, patch_size=32."""
    return MedSegDiT_Segmentation(depth=4, hidden_size=512, num_heads=8, patch_size=32, **kwargs)

def MedSegDiT_B_16(**kwargs):
    """Base variant, patch_size=16 (6 blocks, tuned for small datasets)."""
    return MedSegDiT_Segmentation(depth=6, hidden_size=768, num_heads=12, patch_size=16, **kwargs)

def MedSegDiT_B_32(**kwargs):
    """Base variant, patch_size=32 (6 blocks, tuned for small datasets)."""
    return MedSegDiT_Segmentation(depth=6, hidden_size=768, num_heads=12, patch_size=32, **kwargs)

def MedSegDiT_L_16(**kwargs):
    """Large variant, patch_size=16 (8 blocks)."""
    return MedSegDiT_Segmentation(depth=8, hidden_size=1024, num_heads=16, patch_size=16, **kwargs)

def MedSegDiT_L_32(**kwargs):
    """Large variant, patch_size=32 (8 blocks)."""
    return MedSegDiT_Segmentation(depth=8, hidden_size=1024, num_heads=16, patch_size=32, **kwargs)

def MedSegDiT_H_16(**kwargs):
    """Huge variant, patch_size=16 (12 blocks)."""
    return MedSegDiT_Segmentation(depth=12, hidden_size=1280, num_heads=16, patch_size=16, **kwargs)

def MedSegDiT_H_32(**kwargs):
    """Huge variant, patch_size=32 (12 blocks)."""
    return MedSegDiT_Segmentation(depth=12, hidden_size=1280, num_heads=16, patch_size=32, **kwargs)


# Model registry
MedSegDiT_models = {
    'MedSegDiT-S/16': MedSegDiT_S_16,
    'MedSegDiT-S/32': MedSegDiT_S_32,
    'MedSegDiT-B/16': MedSegDiT_B_16,
    'MedSegDiT-B/32': MedSegDiT_B_32,
    'MedSegDiT-L/16': MedSegDiT_L_16,
    'MedSegDiT-L/32': MedSegDiT_L_32,
    'MedSegDiT-H/16': MedSegDiT_H_16,
    'MedSegDiT-H/32': MedSegDiT_H_32,
}
