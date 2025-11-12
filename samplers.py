import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur



def euler_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        **kwargs,
    ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y            
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            d_cur = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            ).to(torch.float64)
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and (i < num_steps - 1):
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    model_input = torch.cat([x_next] * 2)
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    model_input = x_next
                    y_cur = y
                kwargs = dict(y=y_cur)
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model.inference(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    ).to(torch.float64)
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        **kwargs,
    ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y            
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)            
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            ).to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    v_cur = model.inference(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
    ).to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
    return mean_x

def euler_cond_feats_sampler(
        model,
        latents,
        y,
        zs,
        v_feat_scale=1.0,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        start_time=1,
        mask=None
    ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype    
    t_steps = torch.linspace(start_time, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    align_loss, corr_dir = [], []

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y            
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        d_cur, d_feat, proj_loss = model.inference_cond_feats(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype),zs=zs,v_feat_scale=v_feat_scale,mask=mask, **kwargs
        )
        d_cur = d_cur.to(torch.float64) 
        d_feat = d_feat.to(torch.float64)
        d_cur = d_cur + d_feat
        corr_dir.append(torch.cosine_similarity(d_cur.flatten(1),d_feat.flatten(1)).mean().item())
        align_loss.append(proj_loss.item())
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
        x_next = x_cur + (t_next - t_cur) * d_cur
        if heun and (i < num_steps - 1):
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_next] * 2)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_next
                y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(
                device=model_input.device, dtype=torch.float64
                ) * t_next
            d_prime = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                ).to(torch.float64)
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next, corr_dir, align_loss

def euler_maruyama_cond_feats_sampler(
        model,
        latents,
        y,
        zs,
        v_feat_scale=1.0,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        start_time=1,
        mask=None
    ):
        # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(start_time, 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    align_loss, corr_dir, dirs_feat, dirs_diff = [], [], [], []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y            
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        diffusion = compute_diffusion(t_cur)            
        eps_i = torch.randn_like(x_cur).to(device)
        deps = eps_i * torch.sqrt(torch.abs(dt))

        # compute drift
        v_cur, d_feat, proj_loss = model.inference_cond_feats(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype),zs=zs,v_feat_scale=v_feat_scale,mask=mask, **kwargs)
        v_cur = v_cur.to(torch.float64)
        d_feat = d_feat.to(torch.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        corr_dir.append(torch.cosine_similarity(d_cur.flatten(1),d_feat.flatten(1)).mean().item())
        align_loss.append(proj_loss.item())
        dirs_feat.append(d_feat.squeeze().detach().cpu().numpy())
        dirs_diff.append(d_cur.squeeze().detach().cpu().numpy())
        d_cur = d_cur + d_feat*t_cur
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    v_cur = model.inference(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
    ).to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
    return mean_x, corr_dir, align_loss, dirs_feat, dirs_diff


def transport_to_time(
        model,
        gt_latent,
        target_time,
        y,
        num_steps=20,
    ):
    """Transport ground truth latent from t=0 to target_time using model's ODE
    
    Args:
        model: The velocity prediction model
        gt_latent: Ground truth latent at t=0
        target_time: Target time to transport to (between 0 and 1)
        y: Conditioning information
        num_steps: Number of integration steps
        path_type: Path type for the diffusion process
        
    Returns:
        Transported latent at target_time
    """
    _dtype = gt_latent.dtype
    device = gt_latent.device
    
    # Create time steps from 0 to target_time
    t_steps = torch.linspace(0, target_time, num_steps + 1, dtype=torch.float64)
    x_cur = gt_latent.to(torch.float64)
    
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            dt = t_next - t_cur
            
            # Prepare model inputs
            time_input = torch.ones(x_cur.size(0)).to(device=device, dtype=torch.float64) * t_cur
            kwargs = dict(y=y)
            
            # Get velocity prediction
            v_cur = model.inference(
                x_cur.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            ).to(torch.float64)
            
            # Forward Euler step
            x_cur = x_cur + dt * v_cur
    
    return x_cur.to(_dtype)


def pnp_flow_feature(x, feature, m_feat, m_latent, model, num_steps, v_feat_scale, alpha):

    def H(x, mask):
        return x * ~mask
    
    def denoiser(x, t, y):
        return x - t * model.inference(x, t, y)
    
    def feature_grad(x, feature, m_feat, t, y):
        x = x.detach().requires_grad_(True)  # Enable gradients
        x_feat = model.forward_feats(x, t, y)
        proj_loss = 0
        for i, (z, z_tilde) in enumerate(zip(feature, x_feat)):
            print(z.shape, z_tilde.shape, m_feat.shape)
            z_tilde = torch.nn.functional.normalize(z_tilde[m_feat[:,:]], dim=-1) 
            z = torch.nn.functional.normalize(z[m_feat[:,:]], dim=-1) 
            proj_loss += torch.mean(-(z * z_tilde).sum(dim=-1))
        proj_loss /= (len(feature))

        grad = torch.autograd.grad(proj_loss, x, retain_graph=False)[0]
        return grad

    x_ref = H(x, m_latent)
    x = torch.zeros_like(x_ref)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    
    t_steps = torch.linspace(1, 0, num_steps+1)

    for t in t_steps:

        t = torch.ones(x.size(0)).to(device=x.device) * t

        lr_t = t ** alpha

        grad = feature_grad(x, feature, m_feat, t, y)  # Get gradient
        z = x - lr_t * grad * v_feat_scale # feature gradient step

        with torch.no_grad():
            z = z - lr_t * H(H(z, m_latent) - x_ref, m_latent) # data fidelity gradient step

            z_tilde =  t * torch.randn_like(z) + (1 - t) * z # Interpolation step

            x = denoiser(z_tilde, t, y) # denoising step
    
    return x
