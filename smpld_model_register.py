
import os
from os.path import split, join, exists
from glob import glob
import torch
import trimesh
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import point_to_surface, laplacian_loss
from lib.smpl_layer import SMPL_Layer
from tqdm import tqdm
import pickle as pkl
import numpy as np
from lib.torch_functions import batch_gather, chamfer_distance
from lib.smpl_paths import SmplPaths
from lib.th_SMPL import th_batch_SMPL
from psbody.mesh import Mesh

hand_ind = np.loadtxt('assets/hand_ind.txt')
head_ind = np.loadtxt('assets/head_ind.txt')
def backward_step(loss_dict, weight_dict, it):
    w_loss = dict()
    for k in loss_dict:
        w_loss[k] = weight_dict[k](loss_dict[k], it)

    tot_loss = list(w_loss.values())
    tot_loss = torch.stack(tot_loss).sum()
    return tot_loss


def get_loss_weights():
    """Set loss weights"""

    loss_weight = {
                   's2m': lambda cst, it: 10. ** 3 * cst /(1 + it),
                   'm2s': lambda cst, it: 10. ** 3 * cst / (1 + it),
                   'lap': lambda cst, it:50** 2 * cst / (1 + it),
                   'offsets': lambda cst, it:20** 1 * cst / (1 + it),
                   'offsets_hand':lambda cst, it: 18. ** 3 * cst / (1 + it),
                   'offsets_head': lambda cst, it: 10 ** 3 * cst / (1 + it)
                   }
    return loss_weight


def forward_step(th_scan_points, smpl, init_smpl_meshes):
    """
    Performs a forward step, given smpl and scan meshes.
    Then computes the losses.
    """

    # forward
    verts, _, _, _ = smpl()
    th_smpl_meshes = [tm.from_tensors(vertices=v,
                                      faces=smpl.faces) for v in verts]

    # losses
    loss = dict()
    # chamfer loss
    loss['s2m'] = chamfer_distance(verts, th_scan_points, w2=0).unsqueeze(0)
    loss['m2s'] = chamfer_distance(th_scan_points, verts, w1=0).unsqueeze(0)
    loss['lap'] = torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(init_smpl_meshes, th_smpl_meshes)])
    loss['offsets'] = torch.mean(torch.mean(smpl.offsets**2, axis=1), axis=1)
    loss['offsets_hand'] = torch.mean(torch.mean(smpl.offsets[:, hand_ind, :] ** 2, axis=1), axis=1)
    loss['offsets_head'] = torch.mean(torch.mean(smpl.offsets[:, head_ind, :] ** 2, axis=1), axis=1)
    return loss


def optimize_offsets(th_scan_meshes, smpl, init_smpl_meshes, iterations, steps_per_iter):
    # Optimizer
    optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.trans, smpl.betas], 0.005, betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights()

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing SMPL+D')
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step(th_scan_meshes, smpl, init_smpl_meshes)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Lx100. Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item()*100)
            loop.set_description(l_str)


def fit_SMPLD(scan_path, smpl_pkl=None, gender='male', save_path=None):
    # Get SMPL faces
    sp = SmplPaths(gender=gender)
    temp = SMPL_Layer(gender=gender,model_root='assets/smpl')
    smpl_faces = temp.th_faces.numpy()
    # smpl_faces = sp.get_faces()
    th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).cuda()

    # Batch size
    batch_sz = 1

    # Init SMPL
    pose, betas, trans = [], [], []

    # load smpl_pkl
    smpl_dict = pkl.load(open(smpl_pkl, 'rb'), encoding='latin-1')
    p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
    pose.append(p)
    betas.append(b)
    trans.append(t)
    pose, betas, trans = np.array(pose), np.array(betas), np.array(trans)
    betas, pose, trans = torch.tensor(betas).to(torch.float32), torch.tensor(pose).to(torch.float32), torch.tensor(trans).to(torch.float32)

    smpl = th_batch_SMPL(batch_sz, betas, pose, trans, faces=th_faces,gender=gender).cuda()

    verts, _, _, _ = smpl()
    init_smpl_meshes = [tm.from_tensors(vertices=v.clone().detach(),
                                        faces=smpl.faces) for v in verts]

    # Load scans

    th_scan = trimesh.load(scan_path,process=False)
    # th_scan.vertices = th_scan.vertices.cuda()
    # th_scan.vertices.requires_grad = False
    th_scan_points = torch.from_numpy(th_scan.sample(10000)).unsqueeze(0).to(torch.float32).cuda()

    # Optimize
    optimize_offsets(th_scan_points, smpl, init_smpl_meshes, 5, 10)
    print('Done')

    verts, _, _, _ = smpl()
    # trimesh.Trimesh(vertices=verts.detach().cpu().numpy()[0],faces = smpl_faces ).export(join(save_path, split(scan_path)[1].replace('.ply', '_smpld.ply')))

    # Mesh(verts.detach().cpu().numpy()[0], smpl_faces).set_vertex_colors(vcs).write_ply(join(save_path, split(scan_path)[1].replace('.ply', '_smpld.ply')))
    Mesh(verts.detach().cpu().numpy()[0], smpl_faces).write_ply(
        join(save_path, split(scan_path)[1].replace('.ply', '_smpld.ply')))
    print( join(save_path, split(scan_path)[1].replace('.ply', '_smpld.ply')))


    # Save params
    for p, b, t, d,n in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
                             smpl.trans.cpu().detach().numpy(), smpl.offsets.cpu().detach().numpy(),[split(scan_path)[1]]):
        smpl_dict = {'pose': p, 'betas': b, 'trans': t, 'offsets': d}
        pkl.dump(smpl_dict, open(join(save_path, n.replace( '.ply','_smpld.pkl')), 'wb'))

    return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), \
           smpl.trans.cpu().detach().numpy(), smpl.offsets.cpu().detach().numpy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('--scan_path',default='test/noise.ply', type=str)
    parser.add_argument('--save_path',default='test', type=str)
    parser.add_argument('--smpl_pkl', type=str, default='test/noise_reg.pkl')  # In case SMPL fit is already available
    parser.add_argument('--gender', type=str, default='male')  # can be female/ male/ neutral
    args = parser.parse_args()
    import time
    start_time = time.time()
    _, _, _, _ = fit_SMPLD(args.scan_path, smpl_pkl=args.smpl_pkl, save_path=args.save_path,
                           gender=args.gender)
    print('SMPL cost time: ', time.time() - start_time)
