import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from .vgg19 import VGG19
from .loss import StyleTransferLoss


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional

        # Guidance switches
        self.enable_gram_guidance = True
        self.use_style_ref = True
        self.use_hr_guidance = False
        self.guidance_scale = 0.8e-12
        self._style_cache = {}
        # Layer-wise weights (omega_l in paper Eq.10), can be edited from config/runtime.
        self.style_layer_weights = {
            'conv1_1': 0.10, 'conv1_2': 0.10,
            'conv2_1': 0.20, 'conv2_2': 0.20,
            'conv3_1': 0.30, 'conv3_2': 0.30, 'conv3_3': 0.30, 'conv3_4': 0.30,
            'conv4_1': 0.40, 'conv4_2': 0.40, 'conv4_3': 0.40, 'conv4_4': 0.40,
            'conv5_1': 0.50, 'conv5_2': 0.50, 'conv5_3': 0.50, 'conv5_4': 0.50,
        }
        self.style_criterion = StyleTransferLoss(
            alpha=1, beta=1, layer_weights=self.style_layer_weights)
        self.enable_style_train_loss = True
        self.style_train_weight = 0.1

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def _to_style_input(self, x):
        # Inputs are normalized to [-1, 1] in the dataset pipeline.
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    def _predict_x0_from_noise_continuous(self, x_t, continuous_sqrt_alpha_cumprod, noise):
        eps = 1e-8
        return (
            x_t - (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        ) / (continuous_sqrt_alpha_cumprod + eps)

    def _get_style_model(self, device):
        key = str(device)
        model = self._style_cache.get(key)
        if model is None:
            model = VGG19().to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._style_cache[key] = model
        return model

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.full(
            (batch_size, 1),
            float(self.sqrt_alphas_cumprod_prev[t + 1]),
            device=x.device,
            dtype=x.dtype
        )
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, style_ref=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        if self.enable_gram_guidance and style_ref is not None:
            model_mean = self.condition_mean(
                model_mean, model_log_variance, style_ref=style_ref, x_in=x, t=t)

        return model_mean + noise * (0.5 * model_log_variance).exp()

    def condition_mean(self, mean, log_var, style_ref, x_in, t):
        grad_loss = self.gram_func(style_ref=style_ref, x_in=x_in, t=t)
        variance = torch.exp(log_var).to(mean.dtype)
        # Follow paper Eq.(15): mu + omega * Sigma * grad(Ls).
        new_mean = mean.float() + self.guidance_scale * variance * grad_loss.float()
        return new_mean

    def gram_func(self, style_ref, x_in, t, noise=None):
        style_model = self._get_style_model(x_in.device)

        with torch.no_grad():
            b = x_in.shape[0]
            alpha_t = torch.full(
                (b, 1, 1, 1),
                float(self.sqrt_alphas_cumprod_prev[t + 1]),
                device=x_in.device,
                dtype=x_in.dtype
            )
            noise = default(noise, lambda: torch.randn_like(x_in))
            style_noised = self.q_sample(
                x_start=style_ref,
                continuous_sqrt_alpha_cumprod=alpha_t,
                noise=noise
            )
            style_features = style_model(self._to_style_input(style_noised))

        with torch.enable_grad():
            x_in = x_in.detach().requires_grad_(True)
            sr_features = style_model(self._to_style_input(x_in))
            gram_loss = self.style_criterion(style_features, sr_features)
            return torch.autograd.grad(gram_loss, x_in)[0]

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in['SR']
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x

            style_ref = x_in.get('STYLE_REF', None) if self.use_style_ref else None
            if style_ref is None and self.use_hr_guidance and ('HR' in x_in):
                style_ref = x_in['HR']

            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x, style_ref=style_ref)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gamma
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        # Histograms should use the real normalized range [-1, 1].
        sample_hist = torch.histc(x_in['HR'], bins=256, min=-1, max=1)
        abnormal_hist = torch.histc(x_in['SR'], bins=256, min=-1, max=1)

        gray_loss = nn.L1Loss(reduction='sum')
        loss_gray = gray_loss(sample_hist, abnormal_hist)

        loss_noise = self.loss_func(noise, x_recon)
        loss_style = x_start.new_tensor(0.0)
        if self.enable_style_train_loss and ('STYLE_REF' in x_in):
            style_ref = x_in['STYLE_REF']
            style_model = self._get_style_model(x_start.device)
            alpha = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)
            x0_pred = self._predict_x0_from_noise_continuous(x_noisy, alpha, x_recon)
            with torch.no_grad():
                ref_features = style_model(self._to_style_input(style_ref))
            pred_features = style_model(self._to_style_input(x0_pred))
            loss_style = self.style_criterion(ref_features, pred_features)

        return loss_noise + 5 * loss_gray + self.style_train_weight * loss_style

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
