from __future__ import division
from os.path import join, split, exists
import torch
import torch.optim as optim
from torch.nn import functional as F

import time
import trimesh
import pickle as pkl
import numpy as np

from models.volumetric_SMPL import VolumetricSMPL
from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.torch_functions import batch_gather, chamfer_distance
from models.trainer import Trainer

from psbody.mesh import Mesh
from os.path import join, exists

NUM_POINTS = 30000

class SMPL_register(Trainer):

    def __init__(self, device, gender, opt_dict={'iter_per_step': {1: 200, 2: 101, 3: 1}}, optimizer='Adam'):
        self.device = device
        self.opt_dict = self.parse_opt_dict(opt_dict)
        self.optimizer_type = optimizer
        self.gender = gender
        # Load vsmpl
        self.vsmpl = VolumetricSMPL('assets/volumetric_smpl_function_64_{}'.format(self.gender), device,
                                    '{}'.format(self.gender)).to(self.device)
        sp = SmplPaths(gender=self.gender)
        self.ref_smpl = sp.get_smpl()
        self.template_points = torch.tensor(
            trimesh.Trimesh(vertices=self.ref_smpl.r, faces=self.ref_smpl.f).sample(NUM_POINTS).astype('float32'),
            requires_grad=False).unsqueeze(0)
        self.pose_prior = get_prior('{}'.format(self.gender), precomputed=True)

    @staticmethod
    def init_optimizer(optimizer, params, learning_rate=1e-4):
        if optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
        if optimizer == 'Adadelta':
            optimizer = optim.Adadelta(params)
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, momentum=0.9)
        return optimizer
    @staticmethod
    def parse_opt_dict(opt_dict):
        timestamp = int(time.time())
        parsed_dict = {'iter_per_step': {1: 200, 2: 200, 3: 1}, 'cache_folder': join('cache', str(timestamp)),
                       'epochs_phase_01': 0, 'epochs_phase_02': 0}
        """ 
        Phase_01: Initialised SMPL are far off from the solution. Optimize SMPL based on correspondences.
        Phase_02: SMPL models are close to solution. Fit SMPL based on ICP.
        Phase_03: Jointly update SMPL and correspondences.
        """
        for k in parsed_dict:
            if k in opt_dict:
                if k == 'cache_folder':
                    parsed_dict[k] = join(opt_dict[k], str(timestamp))
                else:
                    parsed_dict[k] = opt_dict[k]
        print('Cache folder: ', parsed_dict['cache_folder'])
        return parsed_dict

    @staticmethod
    def get_optimization_weights(phase):
        """
        Phase_01: Initialised SMPL are far off from the solution. Optimize SMPL based on correspondences.
        Phase_02: SMPL models are close to solution. Fit SMPL based on ICP.
        Phase_03: Jointly update SMPL and correspondences.
        """
        if phase == 1:
            return {'corr': 2 * 10. ** 2, 'templ': 2 * 10. ** 2, 's2m': 10. ** 1, 'm2s': 10. ** 1, 'pose_pr': 10. ** -2,
                    'shape_pr': 10. ** -1}
        elif phase == 2:
            return {'corr': 3* 10. ** 2, 'templ': 2 * 10. ** 2, 's2m': 2 * 10. ** 3, 'm2s': 10. ** 3, 'pose_pr': 10. ** -4,
                    'shape_pr': 10. ** -1}
        else:
            return {'corr': 2 * 10. ** 2, 'templ': 2 * 10. ** 2, 's2m': 10. ** 4, 'm2s': 10. ** 4, 'pose_pr': 10. ** -4,
                    'shape_pr': 10. ** -1}


    def iteration_step(self, naked, instance_params, weight_dict={}):
        """
        Computes losses for a single step of optimization.
        Entries in loss/weight dict should have the following entries (always edit loss_keys to modify loss terms):
        corr, templ, s2m, m2s, pose_pr, shape_pr
        """
        loss_keys = ['corr', 'templ', 's2m', 'm2s', 'pose_pr', 'shape_pr']
        for k in loss_keys:
            if k not in weight_dict.keys():
                weight_dict[k] = 1.

        device = self.device
        loss = {}
        batch_sz = naked.shape[0]
        # predict initial correspondences

        poses, betas, trans = instance_params['pose'], instance_params['betas'], instance_params['trans']
        corr = instance_params['corr']


        # Offset optimization should be implemented here
        if 'offsets' in instance_params:
            offsets = instance_params['offsets']
        else:
            offsets = None

        # Filter out bad correspondences
        # mask = self.filter_correspondences(corr, part_label)


        # get posed smpl points
        template_points = torch.cat([self.template_points] * batch_sz, axis=0).to(device)
        posed_smpl = self.vsmpl(template_points, poses, betas, trans)

        # get posed scan corresponding points
        posed_scan_correspondences = self.vsmpl(corr, poses, betas, trans)

        # correspondence loss
        loss['corr'] = F.l1_loss(naked, posed_scan_correspondences) * weight_dict['corr']

        # bring scan correspondences in R^3 closer to ref_smpl surface
        ''' Experiment to see if this should be bi-directional or not '''
        loss['templ'] = chamfer_distance(corr, template_points) * weight_dict['templ']

        # chamfer loss
        loss['s2m'] = chamfer_distance(naked, posed_smpl, w2=0) * weight_dict['s2m']
        loss['m2s'] = chamfer_distance(naked, posed_smpl, w1=0) * weight_dict['m2s']

        # pose prior
        loss['pose_pr'] = self.pose_prior(poses).mean() * weight_dict['pose_pr']

        # shape prior
        loss['shape_pr'] = (betas ** 2).mean() * weight_dict['shape_pr']

        return loss

    def main_register(self,corr_init,naked):
        pose = torch.zeros((72,)).unsqueeze(0).to(self.device).requires_grad_(True)
        betas = torch.zeros((10,)).unsqueeze(0).to(self.device).requires_grad_(True)
        trans = torch.zeros((3,)).unsqueeze(0).to(self.device).requires_grad_(True)
        corr = corr_init.to(self.device).clone().requires_grad_(True)

        naked = naked.to(self.device)

        instance_params = {'pose': pose, 'betas': betas, 'trans': trans}
        # initialize optimizer for instance specific SMPL params
        smpl_optimizer = optim.Adam(instance_params.values(), lr=0.04)
        instance_params['corr'] = corr
        corr_optimizer = optim.Adam([corr], lr=0.04)

        for it in range(self.opt_dict['iter_per_step'][1] + self.opt_dict['iter_per_step'][2]):
            # print(it)
            smpl_optimizer.zero_grad()
            corr_optimizer.zero_grad()

            if it == 0:
                phase = 1
                wts = self.get_optimization_weights(phase=1)
                print('Optimizing phase 1')
            elif it == self.opt_dict['iter_per_step'][1]:
                phase = 2
                wts = self.get_optimization_weights(phase=2)
                print('Optimizing phase 2')

            loss_ = self.iteration_step(naked, instance_params, weight_dict=wts)
            loss = self.sum_dict(loss_)

            if it % 50 == 0:
                l_str = 'Iter: {}'.format(it)
                for l in loss_:
                    l_str += ', {}: {:0.5f}'.format(l, loss_[l].item())
                print(l_str)

            # back propagate
            loss.backward()
            if phase == 1:
                smpl_optimizer.step()
            elif phase == 2:
                smpl_optimizer.step()
                corr_optimizer.step()

            if it % 300 == 0:
                pose_ = pose.detach().cpu().numpy()
                betas_ = betas.detach().cpu().numpy()
                trans_ = trans.detach().cpu().numpy()
                corr_ = corr.detach().cpu().numpy()
        return pose_,betas_,trans_,corr_


if __name__ == "__main__":
    filename = 'noise'
    gender = 'male'
    start_time = time.time()

    from models.UBSP_Net_v2 import UBSP_Net_repro
    net = UBSP_Net_repro(c=3, k=14).cuda()
    checkpoint = torch.load(
        'experiments/ubsp_net/naked_exp_id_CAPE_Dataset_Sampled_10000/checkpoints/checkpoint_epoch_283.tar')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    input_scan = trimesh.load_mesh('test/{}.ply'.format(filename))
    center = np.mean(input_scan.vertices,axis=0)
    input_scan.vertices -= center
    height_range = np.max(input_scan.vertices[:, 1]) - np.min(input_scan.vertices[:, 1])
    scale_factor = 1.5 / height_range
    input_scan.vertices *= scale_factor
    temp = trimesh.Trimesh(vertices=input_scan.vertices, faces=input_scan.faces)
    points = temp.sample(10000)
    points = torch.from_numpy(points).unsqueeze(0).to(torch.float32).cuda()


    out = net(points)
    part_labels = out['part_labels']
    _, part_label = torch.max(part_labels.data, 1)
    part_label = np.array(part_label.cpu())

    correspondences = (out['correspondences'].detach().permute(0, 2, 1).cpu().numpy()[0] /scale_factor) + center
    t = Mesh(correspondences , [])
    t.set_vertex_colors_from_weights(part_label[0])
    t.write_ply('test/correspondences.ply')

    inner_points = (out['inner_points'].detach().permute(0, 2, 1).cpu().numpy()[0]/scale_factor) + center
    t = Mesh(inner_points , [])
    t.set_vertex_colors_from_weights(part_label[0])
    t.write_ply('test/inner_points.ply')

    ###############
    inner_points = torch.from_numpy(inner_points).unsqueeze(0).to(torch.float32)
    corr_init = torch.from_numpy(correspondences).unsqueeze(0).to(torch.float32)

    register = SMPL_register('cuda', gender=gender)

    pose, betas, trans, corr = register.main_register(corr_init=corr_init , naked=inner_points)
    vcs = np.load("assets/vitruvian_cols.npy")
    sp = SmplPaths(gender=gender)
    smpl = sp.get_smpl()

    smpl.pose[:] = pose
    smpl.betas[:10] = betas
    smpl.trans[:] = trans
    # save registration
    # Mesh(smpl.r, smpl.f).set_vertex_colors(vcs).write_ply('test/shortlong_hips.000176_reg.ply')
    Mesh(smpl.r, smpl.f).write_ply('test/{}_reg.ply'.format(filename))

    with open('test/{}_reg.pkl'.format(filename), 'wb') as f:
        pkl.dump({'pose': smpl.pose, 'betas': smpl.betas, 'trans': smpl.trans}, f)

    print('SMPL cost time: ',time.time()-start_time)

