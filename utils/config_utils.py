import sys
import yaml
from pathlib import Path


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def load_config(args, config):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = config.open().read()

    yaml.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', path_constructor)
    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print("=> Reading YAML config from {}".format(str(config)))
    args.__dict__.update(loaded_yaml)


def path_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*seq)