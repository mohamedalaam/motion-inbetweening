import os
import yaml
from model import  Model
from src.features.LaFan import LaFan1
from src.utils import get_project_path
ROOT_PATH=get_project_path()
if __name__ == '__main__':
    opt = yaml.load(open('../../config/test-base.yaml', 'r').read())
    lafan_data_train = LaFan1(os.path.join(ROOT_PATH,opt['data']['data_dir']), \
                          seq_len=opt['model']['seq_length'], \
                          offset=opt['data']['offset'], \
                          train=False, reprocess=True)
    model=Model(load_pre_trained=True,results_path=os.path.join(ROOT_PATH,'results3'))
    model.predict(lafan_data_train)

