import os

import imageio
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.features.LaFan import LaFan1
from src.features.remove_fs import remove_fs
from src.models.components import StateEncoder, OffsetEncoder, TargetEncoder, LSTM, Decoder, ShortMotionDiscriminator, \
    LongMotionDiscriminator
from src.models.functions import gen_ztta, write_to_bvhfile
from src.skeleton import Skeleton
from src.utils import get_project_path
ROOT_PATH=get_project_path()

class Model:
    def __init__(self, load_pre_trained=True,results_path=None,calc_loss=True):
        self.calc_loss = calc_loss
        self.test_configrations = yaml.load(open(os.path.join(ROOT_PATH,'config\\test-base.yaml'), 'r').read())
        self.train_configrations = yaml.load(open(os.path.join(ROOT_PATH,'config\\train-base.yaml'), 'r').read())
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
        self.ztta = gen_ztta(length=self.train_configrations['model']['seq_length']).cuda()
        self.train_mode = True
        self.state_encoder.train()
        self.offset_encoder.train()
        self.target_encoder.train()
        self.lstm.train()
        self.decoder.train()

    def set_eval_mode(self):
        self.ztta = gen_ztta(length=self.test_configrations['model']['seq_length']).cuda()
        self.train_mode = False
        self.state_encoder.eval()
        self.offset_encoder.eval()
        self.target_encoder.eval()
        self.lstm.eval()
        self.decoder.eval()

    def load_components(self):
        state_encoder = StateEncoder(in_dim=self.train_configrations['model']['state_input_dim'])
        self.state_encoder = state_encoder.cuda()
        offset_encoder = OffsetEncoder(in_dim=self.train_configrations['model']['offset_input_dim'])
        self.offset_encoder = offset_encoder.cuda()
        target_encoder = TargetEncoder(in_dim=self.train_configrations['model']['target_input_dim'])
        self.target_encoder = target_encoder.cuda()
        lstm = LSTM(in_dim=self.train_configrations['model']['lstm_dim'],
                    hidden_dim=self.train_configrations['model']['lstm_dim'] * 2)
        self.lstm = lstm.cuda()
        decoder = Decoder(in_dim=self.train_configrations['model']['lstm_dim'] * 2,
                          out_dim=self.train_configrations['model']['state_input_dim'])
        self.decoder = decoder.cuda()

        short_discriminator = ShortMotionDiscriminator(in_dim=(self.train_configrations['model']['num_joints'] * 3 * 2))
        self.short_discriminator = short_discriminator.cuda()
        long_discriminator = LongMotionDiscriminator(in_dim=(self.train_configrations['model']['num_joints'] * 3 * 2))
        self.long_discriminator = long_discriminator.cuda()
        self.optimizer_g = optim.Adam(lr=self.train_configrations['train']['lr'], params=list(state_encoder.parameters()) + \
                                                               list(offset_encoder.parameters()) + \
                                                               list(target_encoder.parameters()) + \
                                                               list(lstm.parameters()) + \
                                                               list(decoder.parameters()), \
                                 betas=(self.train_configrations['train']['beta1'], self.train_configrations['train']['beta2']), \
                                 weight_decay=self.train_configrations['train']['weight_decay'])

        self.optimizer_d = optim.Adam(lr=self.train_configrations['train']['lr'] * 0.1, params=list(short_discriminator.parameters()) + \
                                                                         list(long_discriminator.parameters()), \
                                     betas=(self.train_configrations['train']['beta1'], self.train_configrations['train']['beta2']), \
                                     weight_decay=self.train_configrations['train']['weight_decay'])

    def load_pre_trained(self):
        self.state_encoder.load_state_dict(
            torch.load(os.path.join(ROOT_PATH, self.test_configrations['test']['model_dir'], 'state_encoder.pkl')))
        self.offset_encoder.load_state_dict(
            torch.load(os.path.join(ROOT_PATH,self.test_configrations['test']['model_dir'], 'offset_encoder.pkl')))
        self.target_encoder.load_state_dict(
            torch.load(os.path.join(ROOT_PATH,self.test_configrations['test']['model_dir'], 'target_encoder.pkl')))
        self.lstm.load_state_dict(
            torch.load(os.path.join(ROOT_PATH, self.test_configrations['test']['model_dir'], 'lstm.pkl')))
        self.decoder.load_state_dict(
            torch.load(os.path.join(ROOT_PATH,self.test_configrations['test']['model_dir'], 'decoder.pkl')))


    def train(self,lafan_dataset):
        self.set_train_mode()
        lafan_loader_train=self.create_dataloader(lafan_dataset)
        model_dir = os.path.join(ROOT_PATH, 'models')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)


        loss_total_min = 10000000.0
        for epoch in range(self.train_configrations['train']['num_epoch']):
            loss_total_list = []

            if self.train_configrations['train']['progressive_training']:
                ## get positional code ##
                if self.train_configrations['train']['use_ztta']:
                    self.ztta = gen_ztta(length=lafan_dataset.cur_seq_length).cuda()
                    if (10 + (epoch // 2)) < self.train_configrations['model']['seq_length']:
                        lafan_dataset.cur_seq_length = 10 + (epoch // 2)
                    else:
                        lafan_dataset.cur_seq_length = self.train_configrations['model']['seq_length']
            else:
                ## get positional code ##
                if self.train_configrations['train']['use_ztta']:
                    lafan_dataset.cur_seq_length = self.train_configrations['model']['seq_length']
                    self.ztta = gen_ztta(length=self.train_configrations['model']['seq_length']).cuda()
            for i_batch, sampled_batch in tqdm(enumerate(lafan_loader_train)):
                (pred_list, bvh_list, contact_list), (loss_pos, loss_quat, loss_contact, loss_root) = self.generate_seq(sampled_batch)
                X=sampled_batch['X'].cuda()
                contact = sampled_batch['contact'].cuda()





                fake_input = torch.cat([x.reshape(x.size(0), -1).unsqueeze(-1) for x in pred_list], -1)
                fake_v_input = torch.cat(
                    [fake_input[:, :, 1:] - fake_input[:, :, :-1], torch.zeros_like(fake_input[:, :, 0:1]).cuda()],
                    -1)
                fake_input = torch.cat([fake_input, fake_v_input], 1)

                real_input = torch.cat(
                    [X[:, i].view(X.size(0), -1).unsqueeze(-1) for i in range(lafan_dataset.cur_seq_length)], -1)
                real_v_input = torch.cat(
                    [real_input[:, :, 1:] - real_input[:, :, :-1], torch.zeros_like(real_input[:, :, 0:1]).cuda()],
                    -1)
                real_input = torch.cat([real_input, real_v_input], 1)

                self.optimizer_d.zero_grad()
                short_fake_logits = torch.mean(self.short_discriminator(fake_input.detach())[:, 0], 1)
                short_real_logits = torch.mean(self.short_discriminator(real_input)[:, 0], 1)
                short_d_fake_loss = torch.mean((short_fake_logits) ** 2)
                short_d_real_loss = torch.mean((short_real_logits - 1) ** 2)
                short_d_loss = (short_d_fake_loss + short_d_real_loss) / 2.0

                long_fake_logits = torch.mean(self.long_discriminator(fake_input.detach())[:, 0], 1)
                long_real_logits = torch.mean(self.long_discriminator(real_input)[:, 0], 1)
                long_d_fake_loss = torch.mean((long_fake_logits) ** 2)
                long_d_real_loss = torch.mean((long_real_logits - 1) ** 2)
                long_d_loss = (long_d_fake_loss + long_d_real_loss) / 2.0
                total_d_loss = self.train_configrations['train']['loss_adv_weight'] * long_d_loss + \
                               self.train_configrations['train']['loss_adv_weight'] * short_d_loss
                total_d_loss.backward()
                self.optimizer_d.step()

                self.optimizer_g.zero_grad()
                pred_pos = torch.cat([x.reshape(x.size(0), -1).unsqueeze(-1) for x in pred_list], -1)
                pred_vel = (pred_pos[:, self.train_configrations['data']['foot_index'], 1:] - pred_pos[:, self.train_configrations['data']['foot_index'], :-1])
                pred_vel = pred_vel.view(pred_vel.size(0), 4, 3, pred_vel.size(-1))
                loss_slide = torch.mean(torch.abs(pred_vel * contact[:, :-1].permute(0, 2, 1).unsqueeze(2)))
                loss_total = self.train_configrations['train']['loss_pos_weight'] * loss_pos + \
                             self.train_configrations['train']['loss_quat_weight'] * loss_quat + \
                             self.train_configrations['train']['loss_root_weight'] * loss_root + \
                             self.train_configrations['train']['loss_slide_weight'] * loss_slide + \
                             self.train_configrations['train']['loss_contact_weight'] * loss_contact


                short_fake_logits = torch.mean(self.short_discriminator(fake_input)[:, 0], 1)
                short_g_loss = torch.mean((short_fake_logits - 1) ** 2)
                long_fake_logits = torch.mean(self.long_discriminator(fake_input)[:, 0], 1)
                long_g_loss = torch.mean((long_fake_logits - 1) ** 2)
                total_g_loss = self.train_configrations['train']['loss_adv_weight'] * long_g_loss + \
                               self.train_configrations['train']['loss_adv_weight'] * short_g_loss
                loss_total += total_g_loss

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.offset_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.target_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.5)
                self.optimizer_g.step()
                loss_total_list.append(loss_total.item())

            loss_total_cur = np.mean(loss_total_list)
            if loss_total_cur < loss_total_min:
                loss_total_min = loss_total_cur
                torch.save(self.state_encoder.state_dict(), model_dir + '/state_encoder.pkl')
                torch.save(self.target_encoder.state_dict(), model_dir + '/target_encoder.pkl')
                torch.save(self.offset_encoder.state_dict(), model_dir + '/offset_encoder.pkl')
                torch.save(self.lstm.state_dict(), model_dir + '/lstm.pkl')
                torch.save(self.decoder.state_dict(), model_dir + '/decoder.pkl')
                torch.save(self.optimizer_g.state_dict(), model_dir + '/optimizer_g.pkl')
                torch.save(self.short_discriminator.state_dict(), model_dir + '/short_discriminator.pkl')
                torch.save(self.long_discriminator.state_dict(), model_dir + '/long_discriminator.pkl')
                torch.save(self.optimizer_d.state_dict(), model_dir + '/optimizer_d.pkl')
            print(
                "train epoch: %03d, cur total loss:%.3f, cur best loss:%.3f" % (epoch, loss_total_cur, loss_total_min))




    def create_dataloader(self,dataset):
        x_mean = dataset.x_mean.cuda()
        self.x_std = dataset.x_std.cuda().view(1, 1, self.train_configrations['model']['num_joints'], 3)
        lafan_loader_train = DataLoader(dataset, \
                                        batch_size=self.train_configrations['train']['batch_size'], \
                                        shuffle=False, num_workers=self.train_configrations['data']['num_workers'])

        return lafan_loader_train


    def predict(self,lafan_dataset):
        lafan_dataloader= self.create_dataloader(lafan_dataset)
        self.set_eval_mode()
        c=[]
        p=[]
        b=[]
        for i_batch, sampled_batch in enumerate(lafan_dataloader):
            with torch.no_grad():


                (pred_list,bvh_list, contact_list), (loss_pos, loss_quat, loss_contact, loss_root) =self.generate_seq(sampled_batch,self.test_configrations['model']['seq_length'])
                if len(b) != 0:
                    bvh_list[0]=b[-1]
                b+=bvh_list
                p+=pred_list
                c+=contact_list

        self.save_results(contact_list=c,pred_list=p,bvh_list=b,i_batch=i_batch)






    def generate_seq(self, batch_dict, seq_length=50):
        loss_pos, loss_quat, loss_contact, loss_root = 0, 0, 0, 0
        pred_list, contact_list = [], []
        local_q, root_v, contact, root_p_offset, local_q_offset, target, root_p, X = self.get_data_from_dict(batch_dict)
        self.lstm.init_hidden(local_q.size(0))
        bvh_list = []
        bvh_list.append(torch.cat([X[:, 0, 0], local_q[:, 0, ].view(local_q.size(0), -1)], -1))
        contact_list.append(contact[:, 0])
        if self.train_mode:
            pred_list.append(X[:, 0])


        for t in range(seq_length-1):
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
            if self.train_mode:
                pred_list.append(pos_pred[0])

            else:

                pred_list.append(np.concatenate([X[0, 0].view(22, 3).detach().cpu().numpy(), \
                                          pos_pred[0,0].view(22, 3).detach().cpu().numpy(), \
                                          X[0, -1].view(22, 3).detach().cpu().numpy()], 0))

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
        img_list=[]
        for frame in pred_list:
            self.plot_pose(frame,path_to_save)
            pred_img = Image.open( path_to_save+'image.png', 'r')
            img_list.append(np.array(pred_img))
        imageio.mimsave(path_to_save+'animation.gif', img_list, duration=0.1)

    def save_bvh(self,contact_list, bvh_list,i_batch,path_to_save):
        bs=0
        bvh_data = torch.cat([x[bs].unsqueeze(0) for x in bvh_list], 0).detach().cpu().numpy()
        write_to_bvhfile(bvh_data,
                         (path_to_save+'/test_%03d.bvh' % i_batch),
                         self.test_configrations['data']['joints_to_remove'])

        contact_data = torch.cat([x[bs].unsqueeze(0) for x in contact_list], 0).detach().cpu().numpy()
        foot = contact_data.transpose(1, 0)
        foot[foot > 0.5] = 1.0
        foot[foot <= 0.5] = 0.0

        # glb = remove_fs((path_to_save+'/test_%03d.bvh' % i_batch), \
        #                 foot, \
        #                 fid_l=(3, 4), \
        #                 fid_r=(7, 8), \
        #                 output_path=(
        #                             path_to_save+'/test_%03d.bvh' % i_batch))

    def plot_pose(self,pose, prefix):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
        ax.cla()
        num_joint = pose.shape[0] // 3
        for i, p in enumerate(parents):
            if i > 0:
                ax.plot([pose[i, 0], pose[p, 0]], \
                        [pose[i, 2], pose[p, 2]], \
                        [pose[i, 1], pose[p, 1]], c='r')
                ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]], \
                        [pose[i + num_joint, 2], pose[p + num_joint, 2]], \
                        [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='b')
                ax.plot([pose[i + num_joint * 2, 0], pose[p + num_joint * 2, 0]], \
                        [pose[i + num_joint * 2, 2], pose[p + num_joint * 2, 2]], \
                        [pose[i + num_joint * 2, 1], pose[p + num_joint * 2, 1]], c='g')
        # ax.scatter(pose[:num_joint, 0], pose[:num_joint, 2], pose[:num_joint, 1],c='b')
        # ax.scatter(pose[num_joint:num_joint*2, 0], pose[num_joint:num_joint*2, 2], pose[num_joint:num_joint*2, 1],c='b')
        # ax.scatter(pose[num_joint*2:num_joint*3, 0], pose[num_joint*2:num_joint*3, 2], pose[num_joint*2:num_joint*3, 1],c='g')
        xmin = np.min(pose[:, 0])
        ymin = np.min(pose[:, 2])
        zmin = np.min(pose[:, 1])
        xmax = np.max(pose[:, 0])
        ymax = np.max(pose[:, 2])
        zmax = np.max(pose[:, 1])
        scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
        xmid = (xmax + xmin) // 2
        ymid = (ymax + ymin) // 2
        zmid = (zmax + zmin) // 2
        ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
        ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
        ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

        plt.draw()
        plt.savefig(prefix + 'image' + '.png', dpi=200, bbox_inches='tight')
        plt.close()

    def save_results(self,contact_list=None, pred_list=None, bvh_list=None,i_batch=1):
        if not os.path.exists(self.results_path+'/bvh'):
            os.mkdir(self.results_path+'/bvh')
            os.mkdir(self.results_path+'/gif')
        if self.test_configrations['test']['save_gif']:
            self.save_gif(pred_list,self.results_path+'/gif'+'/')
        if self.test_configrations['test']['save_bvh']:
            self.save_bvh(contact_list,bvh_list,i_batch,self.results_path+'/bvh')

