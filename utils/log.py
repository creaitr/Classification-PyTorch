import os
import time
import yaml
import pathlib
from pathlib import Path
import logging
import logging.config

from .config_utils import load_config


class Logger:
    keys = ['epoch',
            'loss_trn',
            'acc1_trn',
            'acc5_trn',
            'loss_val',
            'acc1_val',
            'acc5_val']

    def __init__(self, log_path, cfg):
        self.log_path = log_path
        
        # for the first training
        if cfg.resume is None and cfg.eval is None:
            # make a log directory
            log_path.mkdir(exist_ok=True, parents=True)

            # save the configuration
            with (log_path / 'config.yaml').open('w') as f:
                yaml.dump(vars(cfg), f, sort_keys=False)
        else:
            # load the configuration for resume
            load_config(cfg, log_path / 'config.yaml')
        
        self.logging_cfg = Path('logging.conf')
        self.logger, self.csv_logger = self.init_logger(self.log_path, self.logging_cfg, cfg)

    def init_logger(self, log_path, logging_cfg, cfg):
        logging.config.fileConfig(logging_cfg,
                                  defaults={'logfilename': str(log_path / 'logs.log'),
                                            'logfilemode': 'w' if cfg.resume is None and cfg.eval is None else 'a',
                                            'csvfilename': str(log_path / 'summary.csv'),
                                            'csvfilemode': 'w' if cfg.resume is None and cfg.eval is None else 'a'})
        # for logger
        logger = logging.getLogger('root')
        logger.info('Log file for this {}: {}'.format('run' if cfg.resume is None else 'resume',
                                                      str(log_path / 'logs.log')))
        
        # for csv logger
        csv_logger = logging.getLogger('summary')
        if cfg.resume is None:
            head = self.keys[0]
            for i in range(1, len(self.keys)):
                head += ',' + self.keys[i]
            csv_logger.info(head)
        return logger, csv_logger

    def print(self, msg):
        self.logger.info(msg)

    def summarize(self, results):
        row = str(results[self.keys[0]])
        for i in range(1, len(self.keys)):
            if self.keys[i] in results.keys():
                row += ',' + str(results[self.keys[i]])
            else:
                row += ',-'
        self.csv_logger.info(row)
        