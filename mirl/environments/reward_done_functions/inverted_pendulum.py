import torch  
import numpy as np
import mirl.torch_modules.utils as ptu

def reward_function(obs, action, next_obs, input_type='torch'):
    raise NotImplementedError

def done_function(obs, action, next_obs, input_type='torch'):
    ang = next_obs[...,1]
    if input_type == "torch":
        finite = torch.isfinite(next_obs).all(-1)
    elif input_type == "numpy":
        finite = np.isfinite(next_obs).all(-1)
    else:
        raise NotImplementedError

    live = finite & (ang <= 0.2) & (ang >= -0.2)
    d = (~live)[...,None]

    if input_type == "torch":
        d = d.float()
    elif(input_type) == "numpy":
        d = d.astype(float)

    return d