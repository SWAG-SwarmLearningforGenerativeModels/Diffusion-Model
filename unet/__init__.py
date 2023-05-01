from unet.modules import UNetUnconditionalDeeper, UNetConditionalDeeper
import torch.optim as optim


def get_model(args, config, model_config):
    print('Loading Model ...')

    if args.conditional:
            model = UNetConditionalDeeper(in_channel=config.data.channels,
                                          out_channel=config.data.channels,
                                          num_classes=config.data.num_classes,
                                          channel=model_config.model.channel,
                                          channel_multiplier=model_config.model.channel_multiplier,
                                          n_res_blocks=model_config.model.n_res_blocks,
                                          attn_strides=model_config.model.attn_strides,
                                          attn_heads=model_config.model.attn_heads,
                                          use_affine_time=model_config.model.use_affine_time,
                                          dropout=model_config.model.dropout,
                                          fold=model_config.model.fold
                                          )

    else:
            model = UNetUnconditionalDeeper(in_channel=config.data.channels,
                                            out_channel=config.data.channels,
                                            channel=model_config.model.channel,
                                            channel_multiplier=model_config.model.channel_multiplier,
                                            n_res_blocks=model_config.model.n_res_blocks,
                                            attn_strides=model_config.model.attn_strides,
                                            attn_heads=model_config.model.attn_heads,
                                            use_affine_time=model_config.model.use_affine_time,
                                            dropout=model_config.model.dropout,
                                            fold=model_config.model.fold
                                            )

    return model


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
