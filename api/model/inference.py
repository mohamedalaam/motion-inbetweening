import numpy as np

from src.features import utils
from src.features.LaFan import LaFan1
import yaml
from src.models.model import Model
from src.utils import get_project_path
import os
from src.features.extract import read_bvh
ROOT_PATH=get_project_path()
class InferModel:
    def __init__(self):
        config_file_path=os.path.join(ROOT_PATH,'config','test-base.yaml')
        self.test_configrations = yaml.load(open(config_file_path,'r').read())
    def load_file(self,file_path):
        anim=read_bvh(file_path)
        q, x = utils.quat_fk(anim.quats[:], anim.pos[:], anim.parents)
        # Extract contacts
        c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
        return anim.pos, anim.quats, anim.parents, c_l, c_r


    def get_data_dict(self,file_path):
        X, Q, parents, contacts_l, contacts_r=self.load_file(file_path)
        q_glbl, x_glbl = utils.quat_fk(Q, X, parents)
        input_ = {}
        # The following features are inputs:
        # 1. local quaternion vector (J * 4d)
        input_['local_q'] = Q

        # 2. global root velocity vector (3d)
        input_['root_v'] = x_glbl[ 1:, 0, :] - x_glbl[ :-1, 0, :]

        # 3. contact information vector (4d)
        input_['contact'] = np.concatenate([contacts_l, contacts_r], -1)

        # 4. global root position offset (?d)
        input_['root_p_offset'] = x_glbl[ -1, 0, :]

        # 5. local quaternion offset (?d)
        input_['local_q_offset'] = Q[ -1, :, :]

        # 6. target
        input_['target'] = Q[ -1, :, :]

        # 7. root pos
        input_['root_p'] = x_glbl[ :, 0, :]

        # 8. X
        input_['X'] = x_glbl[ :, :, :]

        print('Nb of sequences : {}\n'.format(X.shape[0]))

        return input_



    def infer(self,file_path):
        dataset=self.get_data_dict(file_path)
        model =Model(load_pre_trained=True,results_path='../results')
        model.predict(dataset)





model=InferModel()
model.infer('../uploaded_files/test_subject5.bvh')




