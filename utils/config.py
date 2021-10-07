import argparse
import sys
import yaml
from pathlib import Path


def add_arguments(parser):
    ####    Basic Configuration    ####
    # model architecture
    parser.add_argument("-a", "--arch", default="resnet", type=str, metavar="ARCH",
                        help="model architecture (default: resnet")
    parser.add_argument('--layers', default=20, type=int, metavar='N',
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
    parser.add_argument("--yamls", default=[], type=Path, nargs='+', metavar="CONFIG.yaml",
                        help="A yaml file or yaml files to load the pre-defined configurations")
    parser.add_argument("--run-type", default='train', type=str, choices=['train', 'validate', 'test', 'analyze'],
                        help="Which type to run the main {train, validate, or test}")
    parser.add_argument("--load", default=None, type=Path,
                        help='Path of checkpoint file or dir to load, '
                              'e.g., PATH/FILE.pth or PATH/DIR (default: None)')
    parser.add_argument('--resume', action='store_true',
                        help="resume the training (default: False)")
    parser.add_argument("--device", default='gpu', type=str, choices=['cpu', 'gpu'],
                        help="Which devices to use {cpu, gpu}")
    parser.add_argument("-g", "--gpu", default=[0], type=int, nargs='+', metavar="0 1 2 3",
                        help="list of numbers of gpu ids for single or multi gpu training")
    # Learning Policy
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run (default: 300)')
    parser.add_argument('--optimizer', default='SGD', type=str,
                        help='name of optimizer to train the model (default: SGD)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nest', '--nesterov', dest='nesterov', action='store_true',
                        help='use nesterov momentum?')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, metavar='TYPE',
                        help='schedulers of the learning rate: {step multistep exp cosine warmup_cosine} (default: cosine)')
    parser.add_argument('--step-size', dest='step_size', default=100, type=int, metavar='STEP',
                        help='period of learning rate decay / '
                             'maximum number of iterations for '
                             'cosine annealing scheduler (default: 100)')
    parser.add_argument('--step-milestones', metavar='EPOCH', default=[100,200], type=int, nargs='+',
                        help='list of epoch indices for multi step scheduler '
                             '(must be increasing) (default: 100 200)')
    parser.add_argument('--step-gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--step-location', default='batch', type=str, choices=['epoch', 'batch'],
                        help='update the learning rate for every epoch or batch')
    parser.add_argument('--warmup', default=5, type=int,
                        help='the number of epochs for warmup training')
    # etc
    parser.add_argument("--name", default=None, type=str,
                        help="Experiment name to append to the logpath")
    parser.add_argument("--idx", default=None, type=int,
                        help="The index of experiment name to append to the logpath")
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')


def parse_arguments(funcs=[], description="Classification Implementation with PyTorch"):
    parser = argparse.ArgumentParser(description=description)

    # add arguments
    assert len(funcs) > 0, 'You should add at least one argument.'
    for func in funcs:
        func(parser)

    args = parser.parse_args()

    # load arguments from a yaml file
    if 'yamls' in args.__dict__.keys():
        for yaml_path in args.yamls:
            load_yaml(args, yaml_path)

    return args


def load_yaml(args, yaml_path):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = yaml_path.open().read()

    yaml.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', path_constructor)
    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print("Reading configurations from {}".format(str(yaml_path)))
    args.__dict__.update(loaded_yaml)


def path_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*seq)


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]
