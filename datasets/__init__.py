avail_datasets = ['cifar10', 'cifar100', 'imagenet']

for dataset in avail_datasets:
    exec(f'from .{dataset} import set_dataset as {dataset}')

def set_dataset(cfg, image_size, train=True, val=True, test=False):
    assert cfg.dataset in avail_datasets, f'The dataset:{cfg.dataset} is not supported.'

    if cfg.run_type == 'train':
        train = True; val = True; test = False
    elif cfg.run_type == 'validate':
        train = False; val = True; test = False
    elif cfg.run_type == 'test':
        train = False; val = False; test = True
    elif cfg.run_type == 'analyze':
        train = False; val = True; test = False

    _local = locals()
    exec(f'loaders = {cfg.dataset}(cfg, image_size, train, val, test)', globals(), _local)
    return _local['loaders']
