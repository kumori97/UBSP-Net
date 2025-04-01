import argparse
import os
import datetime
import random
import numpy as np
import torch
import torch.optim as Optim
import trimesh
from torch.utils.data.dataloader import DataLoader
from psbody.mesh import Mesh
from os.path import join, split, exists

if __name__ =="__main__":
    input_scan = trimesh.load_mesh('noise.ply')
    center = np.mean(input_scan.vertices,axis=0)
    input_scan.vertices -= center
    height_range = np.max(input_scan.vertices[:, 1]) - np.min(input_scan.vertices[:, 1])
    scale_factor = 1.5 / height_range
    input_scan.vertices *= scale_factor

    np.random.seed(666)
    rand_indices = np.random.choice(input_scan.vertices.shape[0],10000, replace=False)
    scan = input_scan.vertices[rand_indices]
    temp = trimesh.Trimesh(vertices=(scan/scale_factor)+center , faces=[])
    temp.export('noise_scan.ply')

    points =  torch.from_numpy(scan).unsqueeze(0).to(torch.float32).cuda()

    from models.UBSP_Net_v2 import UBSP_Net_repro
    net = UBSP_Net_repro(k=14, c=3).cuda()
    checkpoint = torch.load(
        'experiments/ubsp_net/naked_exp_id_CAPE_Dataset_Sampled_10000/checkpoints/checkpoint_epoch_283.tar')

    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    out = net(points)
    part_labels = out['part_labels']
    _, part_label = torch.max(part_labels.data, 1)
    part_label = np.array(part_label.cpu())

    correspondences = (out['correspondences'].detach().permute(0, 2, 1).cpu().numpy()[0] /scale_factor) + center
    t = Mesh(correspondences , [])
    t.set_vertex_colors_from_weights(part_label[0])
    t.write_ply('correspondences.ply')

    inner_points = (out['inner_points'].detach().permute(0, 2, 1).cpu().numpy()[0]/scale_factor) + center
    t = Mesh(inner_points , [])
    t.set_vertex_colors_from_weights(part_label[0])
    t.write_ply('inner_points.ply')

    print(part_labels)