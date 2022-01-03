import os
import torch
import yaml
from torch.utils.data import DataLoader

from src.features.remove_fs import remove_fs
from src.models.components import StateEncoder, OffsetEncoder, TargetEncoder, LSTM, Decoder
from src.models.functions import gen_ztta, write_to_bvhfile


class Model:
    def __init__(self, load_pre_trained=True,results_path=None,calc_loss=True):
        self.calc_loss = calc_loss
        self.test_configrations = yaml.load(open('../../config/test-base.yaml', 'r').read())
        self.train_configrations = yaml.load(open('../../config/train-base.yaml', 'r').read())
        self.results_path=results_path
        self.load_components()
        if load_pre_trained:
            self.load_pre_trained()
        self.x_std=None

        self.skeleton_mocap = Skeleton(offsets=self.test_configrations['data']['offsets'],
                                       parents=self.test_configrations['data']['parents'])
        self.skeleton_mocap.cuda()
        self.skeleton_mocap.remove_joints(self.test_configrations['data']['joints_to_remove'])

    def set_train_mode(self):
        self.state_encoder.train()
        self.offset_encoder.train()
        self.target_encoder.train()
        self.lstm.train()
        self.decoder.train()

    def set_eval_mode(self):
        self.state_encoder.eval()
        self.offset_encoder.eval()
        self.target_encoder.eval()
        self.lstm.eval()
        self.decoder.eval()

    def load_components(self):
        state_encoder = StateEncoder(in_dim=self.test_configrations['model']['state_input_dim'])
        self.state_encoder = state_encoder.cuda()
        offset_encoder = OffsetEncoder(in_dim=self.test_configrations['model']['offset_input_dim'])
        self.offset_encoder = offset_encoder.cuda()
        target_encoder = TargetEncoder(in_dim=self.test_configrations['model']['target_input_dim'])
        self.target_encoder = target_encoder.cuda()
        lstm = LSTM(in_dim=self.test_configrations['model']['lstm_dim'],
                    hidden_dim=self.test_configrations['model']['lstm_dim'] * 2)
        self.lstm = lstm.cuda()
        decoder = Decoder(in_dim=self.test_configrations['model']['lstm_dim'] * 2,
                          out_dim=self.test_configrations['model']['state_input_dim'])
        self.decoder = decoder.cuda()
        self.ztta = gen_ztta().cuda()

    def load_pre_trained(self):
        self.state_encoder.load_state_dict(
            torch.load(os.path.join(self.test_configrations['test']['model_dir'], 'state_encoder.pkl')))
        self.offset_encoder.load_state_dict(
            torch.load(os.path.join(self.test_configrations['test']['model_dir'], 'offset_encoder.pkl')))
        self.target_encoder.load_state_dict(
            torch.load(os.path.join(self.test_configrations['test']['model_dir'], 'target_encoder.pkl')))
        self.decoder.load_state_dict(
            torch.load(os.path.join(self.test_configrations['test']['model_dir'], 'decoder.pkl')))

    def train(self,lafan_dataset):
        self.set_train_mode()
        lafan_loader_train=self.create_dataloader(lafan_dataset)





    def create_dataloader(self,dataset):
        x_mean = dataset.x_mean.cuda()
        x_std = dataset.x_std.cuda().view(1, 1, self.train_configrations['model']['num_joints'], 3)
        lafan_loader_train = DataLoader(dataset, \
                                        batch_size=self.train_configrations['train']['batch_size'], \
                                        shuffle=True, num_workers=self.train_configrations['data']['num_workers'])
        self.x_std=x_std
        return lafan_loader_train


    def predict(self,lafan_dataset):
        lafan_dataloader= self.create_dataloader(lafan_dataset)
        self.set_eval_mode()
        for i_batch, sampled_batch in enumerate(lafan_dataloader):
            with torch.no_grad():
                (pred_list,bvh_list, contact_list), (loss_pos, loss_quat, loss_contact, loss_root) =self.generate_seq(sampled_batch)
                self.save_results(contact_list=contact_list,pred_list=pred_list,bvh_list=bvh_list,i_batch=i_batch)






    def generate_seq(self, batch_dict, seq_length=50):
        loss_pos, loss_quat, loss_contact, loss_root = 0, 0, 0, 0
        pred_list, contact_list = [], []
        local_q, root_v, contact, root_p_offset, local_q_offset, target, root_p, X = self.get_data_from_dict(batch_dict)
        bvh_list = []
        bvh_list.append(torch.cat([X[:, 0, 0], local_q[:, 0, ].view(local_q.size(0), -1)], -1))
        for t in range(seq_length):
            # root pos
            if t == 0:
                root_p_t = root_p[:, t]
                local_q_t = local_q[:, t]
                local_q_t = local_q_t.view(local_q_t.size(0), -1)
                contact_t = contact[:, t]
                root_v_t = root_v[:, t]
            else:
                root_p_t = root_pred[0]
                local_q_t = local_q_pred[0]
                contact_t = contact_pred[0]
                root_v_t = root_v_pred[0]

            # state input
            state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)
            root_p_offset_t = root_p_offset - root_p_t
            local_q_offset_t = local_q_offset - local_q_t

            offset_input = torch.cat([root_p_offset_t, local_q_offset_t], -1)
            # target input
            target_input = target

            # print('state_input:',state_input.size())
            h_state = self.state_encoder(state_input)
            h_offset = self.offset_encoder(offset_input)
            h_target = self.target_encoder(target_input)
            h_state += self.ztta[:, t]
            h_offset += self.ztta[:, t]
            h_target += self.ztta[:, t]
            lambda_target = self._get_lambda(t)
            h_offset += 0.5 * lambda_target * torch.cuda.FloatTensor(h_offset.size()).normal_()
            h_target += 0.5 * lambda_target * torch.cuda.FloatTensor(h_target.size()).normal_()

            h_in = torch.cat([h_state, h_offset, h_target], -1).unsqueeze(0)
            h_out = self.lstm(h_in)
            # print('h_out:', h_out.size())

            h_pred, contact_pred = self.decoder(h_out)
            local_q_v_pred = h_pred[:, :, :self.test_configrations['model']['target_input_dim']]
            local_q_pred = local_q_v_pred + local_q_t
            # print('q_pred:', q_pred.size())
            local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
            local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim=-1, keepdim=True)

            root_v_pred = h_pred[:, :, self.test_configrations['model']['target_input_dim']:]
            root_pred = root_v_pred + root_p_t
            # print(''contact:'', contact_pred.size())
            # print('root_pred:', root_pred.size())
            pos_pred = self.skeleton_mocap.forward_kinematics(local_q_pred_, root_pred)
            bvh_list.append(torch.cat([root_pred[0], local_q_pred_[0].view(local_q_pred_.size(1), -1)], -1))
            if self.calc_loss:
                pos_next = X[:, t + 1]
                local_q_next = local_q[:, t + 1]
                local_q_next = local_q_next.view(local_q_next.size(0), -1)
                root_p_next = root_p[:, t + 1]
                contact_next = contact[:, t + 1]
                # print(pos_pred.size(), x_std.size())
                loss_pos += torch.mean(torch.abs(
                    pos_pred[0] - pos_next) / self.x_std) / seq_length  # opt['model']['seq_length']
                loss_quat += torch.mean(torch.abs(
                    local_q_pred[0] - local_q_next)) / seq_length  # opt['model']['seq_length']
                loss_root += torch.mean(torch.abs(root_pred[0] - root_p_next) / self.x_std[:, :,
                                                                                0]) / seq_length  # opt['model']['seq_length']
                loss_contact += torch.mean(torch.abs(
                    contact_pred[0] - contact_next)) / seq_length  # opt['model']['seq_length']
            pred_list.append(pos_pred[0])
            contact_list.append(contact_pred[0])

        return (pred_list,bvh_list, contact_list), (loss_pos, loss_quat, loss_contact, loss_root)


    def _get_lambda(self, t):
        tta = self.test_configrations['model']['seq_length'] - 2 - t
        if tta < 5:
            lambda_target = 0.0
        elif tta >= 5 and tta < 30:
            lambda_target = (tta - 5) / 25.0
        else:
            lambda_target = 1.0
        return lambda_target

    def get_data_from_dict(self, data_dict):
        local_q = data_dict['local_q'].cuda()
        root_v = data_dict['root_v'].cuda()
        contact = data_dict['contact'].cuda()
        # offset input
        root_p_offset = data_dict['root_p_offset'].cuda()
        local_q_offset = data_dict['local_q_offset'].cuda()
        local_q_offset = local_q_offset.view(local_q_offset.size(0), -1)
        # target input
        target = data_dict['target'].cuda()
        target = target.view(target.size(0), -1)
        # root pos
        root_p = data_dict['root_p'].cuda()
        # X
        X = data_dict['X'].cuda()

        return local_q, root_v, contact, root_p_offset, local_q_offset, target, root_p, X


    '''
    input: pred_list for the generated seq (returned from generate_seq function)
    output: image_list contains image for each frame in the sequence
    we could use it to build our gif
    '''
    def save_gif(self, pred_list,path_to_save):
        pass

    def save_bvh(self,contact_list, bvh_list,i_batch,path_to_save):
        bs=6
        bvh_data = torch.cat([x[bs].unsqueeze(0) for x in bvh_list], 0).detach().cpu().numpy()
        write_to_bvhfile(bvh_data,
                         (path_to_save+'/test_%03d.bvh' % i_batch),
                         self.test_configrations['data']['joints_to_remove'])

        contact_data = torch.cat([x[bs].unsqueeze(0) for x in contact_list], 0).detach().cpu().numpy()
        foot = contact_data.transpose(1, 0)
        foot[foot > 0.5] = 1.0
        foot[foot <= 0.5] = 0.0

        glb = remove_fs((path_to_save+'/test_%03d.bvh' % i_batch), \
                        foot, \
                        fid_l=(3, 4), \
                        fid_r=(7, 8), \
                        output_path=(
                                    path_to_save+'/test_%03d.bvh' % i_batch))

    def save_results(self,contact_list=None, pred_list=None, bvh_list=None,i_batch=1):
        os.mkdir(self.results_path+'/bvh')
        os.mkdir(self.results_path+'/gif')
        if self.test_configrations['test']['save_gif']:
            self.save_gif(pred_list,self.results_path+'/gif')
        if self.test_configrations['test']['save_bvh']:
            self.save_bvh(contact_list,bvh_list,i_batch,self.results_path+'/bvh')

