# torch
import torch
import torch.nn as nn
from thop import profile #ptflops
# packages
import models
import datasets
import utils
from utils import parse_arguments, add_arguments
from utils import summarize_reports
from train import Trainer, set_optimizer, set_lr_scheduler
from train import load_init, load_resume, load_valid, save_train, save_pred
from train import step_lr_epoch, step_lr_batch


def main():
    # parse arguments
    cfg = parse_arguments(funcs=[add_arguments])
    
    # get the name of a model
    arch_name = models.utils.set_arch_name(cfg)

    # set a logger
    logger = utils.Logger(cfg, arch_name)

    # construct a model
    logger.print('Building a model ...')
    model, image_size = models.set_model(cfg)
    
    # profile the model
    input = torch.randn(1, 3, image_size, image_size)
    macs, params = profile(model, inputs=(input, ), verbose=False)
    logger.print(f'Name: {arch_name}    (Params: {int(params)}, FLOPs: {int(macs)})')
    
    # set other options
    criterion = nn.CrossEntropyLoss()
    optimizer = set_optimizer(model, cfg)
    lr_scheduler = set_lr_scheduler(optimizer, cfg)

    # load dataset
    loaders = datasets.set_dataset(cfg, image_size)

    # set a trainer
    trainer = Trainer(cfg=cfg,
                      model=model, criterion=criterion,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      loaders=loaders, logger=logger)

    # set device
    trainer.set_device()

    # run
    if cfg.run_type == 'train':
        # set hooks
        if cfg.load is not None:
            if not cfg.resume:
                trainer.register_hooks(loc='before_train', func=[load_init])
            else:
                trainer.register_hooks(loc='before_train', func=[load_resume])
        if cfg.step_location == 'epoch':
            trainer.register_hooks(loc='after_epoch', func=[step_lr_epoch])
        else:
            trainer.register_hooks(loc='after_batch', func=[step_lr_batch])
        trainer.register_hooks(loc='after_epoch', func=[save_train, summarize_reports])

        trainer.train()

    elif cfg.run_type == 'validate':
        # set hooks
        trainer.register_hooks(loc='before_epoch', func=[load_valid])
        trainer.register_hooks(loc='after_epoch', func=[summarize_reports])

        trainer.validate()
    
    elif cfg.run_type == 'test':
        # set hooks
        trainer.register_hooks(loc='before_epoch', func=[load_valid])
        trainer.register_hooks(loc='after_epoch', func=[save_pred])

        trainer.test()
        
    elif cfg.run_type == 'analyze':
        # set hooks
        trainer.register_hooks(loc='before_epoch', func=[load_valid])
        # extract features
        from utils import FeatureExtractor
        extractor = FeatureExtractor()
        trainer.register_hooks(loc='before_epoch', func=[extractor.initialize])
        trainer.register_hooks(loc='after_batch', func=[extractor.check_feature])
        trainer.register_hooks(loc='after_epoch', func=[extractor.save_feature])

        trainer.analyze()


if __name__ == '__main__':
    main()
