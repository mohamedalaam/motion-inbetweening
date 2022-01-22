import yaml

from src.models.model import  Model
from src.features.LaFan import LaFan1
import os
from src.utils import get_project_path


ROOT_PATH=get_project_path()
if __name__ == '__main__':

    opt = yaml.load(open(os.path.join(ROOT_PATH,'config/train-base.yaml'), 'r').read())
    lafan_data_train = LaFan1(os.path.join(ROOT_PATH,opt['data']['data_dir']), \
                          seq_len=opt['model']['seq_length'], \
                          offset=opt['data']['offset'], \
                          train=True, reprocess=True)
    model=Model(load_pre_trained=False)
    model.train(lafan_data_train)

