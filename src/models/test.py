import os
import yaml
from model import  Model
from src.features.LaFan import LaFan1
from src.utils import get_project_path
ROOT_PATH=get_project_path()
if __name__ == '__main__':
    opt = yaml.load(open('../../config/train-base.yaml', 'r').read())
    lafan_data_train = LaFan1(opt['data']['proc_dir'], \
                          seq_len=opt['model']['seq_length'], \
                          offset=opt['data']['offset'], \
                          train=False, reprocess=False)
    model=Model(load_pre_trained=True,results_path=os.path.join(ROOT_PATH,'results3'))
    model.predict(lafan_data_train)

