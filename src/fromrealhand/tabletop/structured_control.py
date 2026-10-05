"""Data-fitted diagonal residual feedback in physical joint-error coordinates."""
import numpy as np
import torch


def fit_diagonal_residual(inputs, targets, device='cuda'):
    x=np.asarray(inputs); y=np.asarray(targets)
    if (x.ndim!=2 or x.shape[1]!=139 or y.shape!=(len(x),30) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Finite aligned training arrays required')
    error=torch.as_tensor(x[:,109:139],dtype=torch.float64,device=device)
    target=torch.as_tensor(y,dtype=torch.float64,device=device)
    energy=(error*error).sum(0)
    cross=(error*target).sum(0)
    gain=torch.where(energy>1e-16,cross/energy.clamp_min(1e-16),torch.zeros_like(cross))
    predicted=error*gain
    return gain.cpu().numpy(), dict(mse=float(((predicted-target)**2).mean().cpu()),
        max_error=float((predicted-target).abs().max().cpu()),
        excited_dimensions=(energy>1e-16).cpu().numpy().tolist())


class StructuredResidual:
    def __init__(self,gain):
        self.gain=np.asarray(gain,dtype=float)
        if self.gain.shape!=(30,) or not np.isfinite(self.gain).all():
            raise ValueError('Invalid learned diagonal gain')

    def predict(self,inputs):
        x=np.asarray(inputs)
        if x.shape!=(139,) or not np.isfinite(x).all(): raise ValueError('Invalid residual inputs')
        return x[109:139]*self.gain
