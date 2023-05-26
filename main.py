import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runner.ddpm_conditional import ConditionDiffusion
from runner.ddpm import UnconditionDiffusion


def parse_args_and_config():

    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='rsna_pe.yml',
                        required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp',
                        help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True,
                        default='rsna_pe', help='A string for name of the log folder.')
    parser.add_argument('--sample', action='store_true',
                        help='Whether to produce samples from the model')
    parser.add_argument('--conditional', action='store_true',
                        help='Whether to train a conditional or unconditional model')
    parser.add_argument('--resume_training', action='store_true',
                        help='Whether to resume training')

    args = parser.parse_args()

    args.log_path = os.path.join(
        args.exp, 'logs', args.doc, f'conditional' if args.conditional else 'unconditional')

    # parse config file
    # Data config
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    new_config.logger = tb_path

    def setup_logging(args):
        os.makedirs(os.path.join(args.log_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(args.log_path, 'results'), exist_ok=True)
        os.makedirs(os.path.join(args.log_path, 'samples'), exist_ok=True)

        os.makedirs(new_config.logger, exist_ok=True)

    # Create experiment paths
    setup_logging(args)

    # add device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Training conditional model: {}".format(args.conditional))
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Config =")
    config_dict = copy.copy(vars(config))
    print(">" * 100)
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 100)

    print("Initializing Model ... ")
    try:
        if args.sample:
            if args.conditional:
                cond_runner = ConditionDiffusion(args, config)
                cond_runner.generate()
            else:
                uncond_runner = UnconditionDiffusion(
                    args, config)
                uncond_runner.generate()

        else:
            if args.conditional:
                cond_runner = ConditionDiffusion(args, config)
                cond_runner.train()
            else:
                uncond_runner = UnconditionDiffusion(
                    args, config)
                uncond_runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
