import torch
import torch.distributed as dist



class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input.contiguous())
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out



def entroy_loss(features, tau=1.0):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T / tau
    loss = sim.logsumexp(dim=1).mean()
    return loss

def spectral_loss_neg(features, tau=1.0):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T / tau
    loss = sim.pow(2).mean()
    return loss