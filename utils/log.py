import os
import time
import shutil
import yaml
from pathlib import Path
import logging
import logging.config

from .config import load_yaml


class Logger:
    config_name = 'config.yaml'
    logging_conf_path = Path(__file__).parent / 'logging.conf'
    log_name = 'logs.log'
    summary_name = 'summary.yaml'
    ckpt_name = 'best.pth'

    def __init__(self, cfg, arch_name):
        self.arch_name = arch_name
        
        self.logger = None
        self.summary_logger = None

        # set logger
        if cfg.run_type == 'train' and not cfg.resume:
            # initialize a logger
            self.initialize(cfg, arch_name)
        else:
            assert cfg.load != None, f'The cfg.load:{cfg.load} should be setted.'
            assert cfg.load.exists(), f'The cfg.load:{cfg.load} is not exists.'

            if cfg.resume:
                assert cfg.load.is_dir(), f'The cfg.load:{cfg.load} should be a directory.'
                # load a logger
                self.load(cfg)
            
            elif cfg.run_type in ['validate', 'test', 'analyze']:
                if cfg.load.is_dir():
                    self.load(cfg)
                elif cfg.load.is_file():
                    self.initialize(cfg, arch_name)
                    shutil.copyfile(str(cfg.load), str(self.log_path / self.ckpt_name))

        # print initial logs
        if cfg.run_type == 'train':
            if not cfg.resume:
                self.print('Train a model')
                self.print(f'Logs are being written at:{self.log_path}')
            else:
                self.print('\n')
                self.print('Resume the training')
        else:
            self.print('\n')
            self.print('Evaluate the trained model')

    def initialize(self, cfg, arch_name):
        # set a log path
        self.log_path = Path('logs') / arch_name / cfg.dataset
        for subpath in cfg.savepath:
            self.log_path = self.log_path / subpath
        assert not self.log_path.exists(), f'PATH:{self.log_path} is already exists.'

        # make a log directory
        self.log_path.mkdir(exist_ok=True, parents=True)

        # save the configuration
        with (self.log_path / self.config_name).open('w') as f:
            yaml.dump(vars(cfg), f, sort_keys=False)

        # set python loggers
        self.set_logger(mode='w')
    
    def load(self, cfg):
        # set a log path
        self.log_path = cfg.load

        # load the yaml files for resume a training or test a model
        load_yaml(cfg, self.log_path / self.config_name)

        # set python loggers
        self.set_logger(mode='a')

    def set_logger(self, mode='w'):
        logging.config.fileConfig(self.logging_conf_path,
            defaults={'logfilename': str(self.log_path / self.log_name),
                      'logfilemode': mode,
                      'csvfilename': str(self.log_path / self.summary_name),
                      'csvfilemode': mode})

        self.logger = logging.getLogger('root')
        self.summary_logger = logging.getLogger('summary')

    def print(self, msg):
        self.logger.info(msg)

    def summarize(self, dic):
        assert isinstance(dic, dict), 'The type of object for summary should be dictionary.'
        self.summary_logger.info(f'- {str(dic)}')


def summarize_reports(trainer):
    trainer.logger.summarize(trainer.reports)