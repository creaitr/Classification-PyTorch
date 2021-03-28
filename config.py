import argparse
import sys
import yaml
from pathlib import Path

from utils import load_config


args = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Quantization Training")

    ####    Basic Configuration    ####
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to the logpath")
    parser.add_argument(
        "--idx", default=None, type=int, help="The index of experiment name to append to the logpath")
    # model architecture
    parser.add_argument("-a", "--arch", default="preactresnet", type=str, metavar="ARCH",
                        help="model architecture")
    parser.add_argument('--layers', default=18, type=int, metavar='N',
                        help='number of layers in the neural network (default: 18)')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier uniformly at each layer (default: 1.0)')
    parser.add_argument('--depth-mult', default=1.0, type=float, metavar='DM',
                         help='depth multiplier at the network (default: 1.0)')
    parser.add_argument('--model-mult', default=0., type=float, metavar='MM',
                        help='model multiplier at the network (default: 0)')
    # dataset
    parser.add_argument("--dataset", default="cifar100", type=str, metavar='DATA',
                        help="name of dataset (default: cifar100)")
    parser.add_argument('--datapath', default='../data', type=str, metavar='PATH',
                        help='path to load the dataset (default: ../data)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
    # training configuration
    parser.add_argument("-c", "--config", default=[], type=Path, nargs='+', metavar="CONFIG.yaml",
                        help="A yaml file or yaml files to load configuration")
    parser.add_argument("--run-type", default='train', type=str,
                        help="Which type to run the main {train or evaluate}")
    parser.add_argument("--init", default=None, type=Path, metavar="PATH/FILE.pth",
                        help="path of checkpoint to initialize (default: None)")
    parser.add_argument('--init-opt', action='store_true',
                        help="initialize the optimizer state (default: False)")
    parser.add_argument("--resume", default=None, type=Path, metavar="PATH",
                        help="log path to resume the training (default: None)")
    parser.add_argument("--eval", default=None, type=Path, metavar="PATH",
                        help="log path to evaludate the trained model (default: None)")
    parser.add_argument("-g", "--gpu", default=[0], type=int, nargs='+', metavar="0 1 2 3",
                        help="Which devices to use for cpu, single-gpu, or multi-gpu training")
    # Learning Policy
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run (default: 300)')
    parser.add_argument('--optimizer', default='SGD', type=str,
                        help='name of optimizer to train the model (default: SGD)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nest', '--nesterov', dest='nesterov', action='store_true',
                        help='use nesterov momentum?')
    parser.add_argument('--sched', '--scheduler', dest='scheduler', default='cosine', type=str, metavar='TYPE',
                        help='schedulers: {step multistep exp cosine} (default: multistep)')
    parser.add_argument('--step-size', dest='step_size', default=100, type=int, metavar='STEP',
                        help='period of learning rate decay / '
                             'maximum number of iterations for '
                             'cosine annealing scheduler (default: 100)')
    parser.add_argument('--milestones', metavar='EPOCH', default=[100,200], type=int, nargs='+',
                        help='list of epoch indices for multi step scheduler '
                             '(must be increasing) (default: 100 200)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--sched-batch', default='True', action='store_true',
                        help='update the learning rate for every batch (sched: cos)')

    args = parser.parse_args()
    
    # Allow for use from notebook without config file
    for i in range(len(args.config)):
        load_config(args, args.config[i])
    
    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


if __name__ == '__main__':
    run_args()
