import torch
import torch.distributed as dist


def uniformity_loss(features):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T 
    loss = sim.pow(2).mean()
    return loss


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




