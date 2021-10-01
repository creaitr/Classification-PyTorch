from .utils import *

avail_archs = ['resnet', 'preactresnet', 'wideresnet',
               'shufflenetv2', 'mobilenetv2', 'rexnet']

for arch in avail_archs:
    exec(f'from .{arch} import set_model as {arch}')

def set_model(cfg):
    assert cfg.arch in avail_archs, f'The architecture:{cfg.arch} is unimplemented.'
    _local = locals()
    exec(f'model, image_size = {cfg.arch}(cfg)', globals(), _local)
    return _local['model'], _local['image_size']
