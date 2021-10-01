import argparse
import sys
import yaml
from pathlib import Path


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
