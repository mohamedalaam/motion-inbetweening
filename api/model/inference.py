import numpy as np
from torch.utils.data import Dataset

from src.features import utils
from src.features.LaFan import LaFan1
import yaml
from src.models.model import Model
from src.utils import get_project_path
import os
from src.features.extract import read_bvh

ROOT_PATH = get_project_path()


class InferModel:
    def __init__(self):
        config_file_path = os.path.join(ROOT_PATH, 'config', 'test-base.yaml')
        self.test_configrations = yaml.load(open(config_file_path, 'r').read())

    def infer(self, file_path):
        dataset = LaFan1(file_path)
        model = Model(load_pre_trained=True, results_path=os.path.join(ROOT_PATH,'api\\results'),calc_loss=False)

        model.predict(dataset)


#model = InferModel()
#model.infer('../uploaded_files/test_subject5.bvh')
