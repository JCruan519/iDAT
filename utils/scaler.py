import torch
from timm.utils import ApexScaler, NativeScaler
from timm.utils.clip_grad import dispatch_clip_grad
try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

class ApexScaler_SAM(ApexScaler):

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, step=0, rho=0.05):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if step==0 or step==2:
            if clip_grad is not None:
                dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
            optimizer.step()
        elif step==1:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), rho, norm_type=2.0)
            optimizer.step()


class KD_NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
