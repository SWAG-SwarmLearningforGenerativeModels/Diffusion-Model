from matplotlib import pyplot as plt
from PIL import Image
import torchvision
import torch
import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(
                              config.optim.beta1, config.optim.beta2), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    if config.optim.optimizer == 'AdamW':
        return optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                           betas=(
                               config.optim.beta1, config.optim.beta2), amsgrad=config.optim.amsgrad,
                           eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, kwargs['nrow'])
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def inverse_transform(inp_tensor):
    invTrans = (inp_tensor.clamp(-1, 1) + 1) / 2
    invTrans = (invTrans * 255).type(torch.uint8)
    return invTrans
