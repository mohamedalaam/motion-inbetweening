import yaml

from model import  Model
from src.features.LaFan import LaFan1

if __name__ == '__main__':
    opt = yaml.load(open('../../config/train-base.yaml', 'r').read())
    lafan_data_train = LaFan1(opt['data']['proc_dir'], \
                          seq_len=opt['model']['seq_length'], \
                          offset=opt['data']['offset'], \
                          train=True, reprocess=False)
    model=Model(load_pre_trained=False)
    model.train(lafan_data_train)

