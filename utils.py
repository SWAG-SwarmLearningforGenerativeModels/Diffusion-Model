from matplotlib import pyplot as plt
import torch

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def inverse_transform(inp_tensor):
    # Scale image from [-1,1] to [0,1]
    invTrans = (inp_tensor.clamp(-1, 1) + 1) / 2

    return invTrans
