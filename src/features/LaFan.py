import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append("..")
import h5py
import numpy as np
import extract
from src.features import utils


class LaFan1(Dataset):
    def __init__(self, bvh_path, train = False, seq_len = 50, offset = 10,reprocess=True):
        """
        Args:
            bvh_path (string): Path to the bvh files.
            seq_len (int): The max len of the sequence for interpolation.
        """
        if train:
            self.actors = ['subject1', 'subject2', 'subject3', 'subject4']
        else:
            self.actors = ['subject5']
        self.train = train
        self.seq_len = seq_len
        self.offset = offset
        self.reprocess=reprocess
        self.data = self.load_data(bvh_path)
        self.cur_seq_length = 5

        
    def load_h5f(self,path):
        parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
        with h5py.File(path, 'r') as hf:
            X = hf['X'][:]
            Q = hf['Q'][:]
            contacts_l = hf['contacts_l'][:]
            contacts_r = hf['contacts_r'][:]
        return X,Q,contacts_l,contacts_r,parents




    def load_single_file(self,file_path):
        anim = extract.read_bvh(file_path)
        q, x = utils.quat_fk(anim.quats[:], anim.pos[:], anim.parents)
        # Extract contacts
        c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
        seq_no=len(anim.pos)//2
        anim.pos=anim.pos.reshape(seq_no,2,-1,3)
        anim.quats=anim.quats.reshape(seq_no,2,-1,4)
        c_l=c_l.reshape(seq_no,2,2)
        c_r=c_r.reshape(seq_no,2,2)
        return anim.pos, anim.quats, anim.parents, c_l, c_r


    def load_train_data(self, bvh_path):
        if self.reprocess:
            X, Q, parents, contacts_l, contacts_r = extract.get_lafan1_set( \
                bvh_path, self.actors, window=self.seq_len, offset=self.offset)
        else:
            X, Q, contacts_l, contacts_r, parents = self.load_h5f(bvh_path)

        return  X, Q, contacts_l, contacts_r, parents

    def load_data(self, bvh_path):
        # Get test-set for windows of 65 frames, offset by 40 frames
        print('Building the data set...')
        #for inference
        if bvh_path.endswith('.bvh'):
            X, Q, parents, contacts_l, contacts_r = self.load_single_file(bvh_path)
        else:
            X, Q, contacts_l, contacts_r, parents= self.load_train_data(bvh_path)


        # Global representation:
        q_glbl, x_glbl = utils.quat_fk(Q, X, parents)

        # if self.train:
        # Global positions stats:
        x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        self.x_mean = torch.from_numpy(x_mean)
        self.x_std = torch.from_numpy(x_std)

        input_ = {}
        # The following features are inputs:
        # 1. local quaternion vector (J * 4d)
        
        input_['local_q'] = Q
        
        # 2. global root velocity vector (3d)
        input_['root_v'] = x_glbl[:,1:,0,:] - x_glbl[:,:-1,0,:]

        # 3. contact information vector (4d)
        input_['contact'] = np.concatenate([contacts_l, contacts_r], -1)
        
        # 4. global root position offset (?d)
        input_['root_p_offset'] = x_glbl[:,-1,0,:]

        # 5. local quaternion offset (?d)
        input_['local_q_offset'] = Q[:,-1,:,:]

        # 6. target 
        input_['target'] = Q[:,-1,:,:]

        # 7. root pos
        input_['root_p'] = x_glbl[:,:,0,:]

        # 8. X
        input_['X'] = x_glbl[:,:,:,:]

        print('Nb of sequences : {}\n'.format(X.shape[0]))

        return input_

    def __len__(self):
        return len(self.data['local_q'])

    def __getitem__(self, idx):

        idx_ = idx
        sample = {}
        sample['local_q'] = self.data['local_q'][idx_].astype(np.float32)
        sample['root_v'] = self.data['root_v'][idx_].astype(np.float32)
        sample['contact'] = self.data['contact'][idx_].astype(np.float32)
        sample['root_p_offset'] = self.data['root_p_offset'][idx_].astype(np.float32)
        sample['local_q_offset'] = self.data['local_q_offset'][idx_].astype(np.float32)
        sample['target'] = self.data['target'][idx_].astype(np.float32)
        sample['root_p'] = self.data['root_p'][idx_].astype(np.float32)
        sample['X'] = self.data['X'][idx_].astype(np.float32)
        
        # sample['local_q_aug'] = self.data['local_q'][idx_].astype(np.float32)
        # sample['root_v_aug'] = self.data['root_v'][idx_].astype(np.float32)
        # sample['contact_aug'] = self.data['contact'][idx_].astype(np.float32)
        # ## data aug ##
        # sample['root_p_offset'] = self.data['root_p_offset'][idx_].astype(np.float32)
        # sample['local_q_offset'] = self.data['local_q_offset'][idx_].astype(np.float32)
        # sample['target'] = self.data['target'][idx_].astype(np.float32)
        # sample['root_p'] = self.data['root_p'][idx_].astype(np.float32)
        # sample['X'] = self.data['X'][idx_].astype(np.float32)

        return sample



# if __name__=="__main__":
#     lafan_data = LaFan1("../../data/raw/lafan1")
#     print(lafan_data.data['X'].shape, lafan_data.data['local_q'].shape)
#     save to

