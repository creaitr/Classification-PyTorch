from pathlib import Path
import numpy as np

import torch.nn as nn


class FeatureExtractor(object):
    def __init__(self):
        super(FeatureExtractor).__init__()
    
    def initialize(self, trainer):
        self.feature_path = trainer.logger.log_path / 'features'
        if not self.feature_path.exists():
            self.feature_path.mkdir()

        self.k = 0

        trainer.logger.print('Extract features after convolution and linear layers')
        trainer.logger.print(f'The features are saved at:{self.feature_path}')

        self.register_module_hook(trainer)

    def keep_feature(self, module, input, output):
        mat = output.cpu().numpy()
        if mat.ndim == 4:
            mat = np.mean(mat, axis=(2,3))
            
        if module.extracted is None:
            module.extracted = mat
        else:
            module.extracted = np.vstack((module.extracted, mat))

    def register_module_hook(self, trainer):
        self.first_module = None
        for name, module in trainer.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if self.first_module == None:
                    self.first_module = module
                
                module.extracted = None
                module.register_forward_hook(self.keep_feature)
                
                temp_path = self.feature_path / name
                if not temp_path.exists():
                    temp_path.mkdir()

    def save_feature(self, trainer):
        for name, module in trainer.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                np.save(self.feature_path / name / (str(self.k).zfill(1) + '.npy'), module.extracted)
                module.extracted = None
        self.k += 1

    def check_feature(self, trainer):
        if self.first_module.extracted.shape[0] > 10000:
            self.save_feature(trainer)