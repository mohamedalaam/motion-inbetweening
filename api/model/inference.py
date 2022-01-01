from src.features.LaFan import LaFan1
import yaml
from src.models.model import Model
from src.utils import get_project_path
ROOT_DIR=get_project_path()
class InferModel:
    def __init__(self,):
        self.test_configrations = yaml.load(open('../../config/test-base.yaml', 'r').read())

    def infer(self,file_path):
        lafan_data_test = LaFan1(file_path, \
                                 seq_len=self.test_configrations['model']['seq_length'], \
                                 offset=40, \
                                 train=False)

        lafan_data_test.cur_seq_length = self.test_configrations['model']['seq_length']

        model =Model(load_pre_trained=True,results_path='../results')
        model.predict(lafan_data_test)










