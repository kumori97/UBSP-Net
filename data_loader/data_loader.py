
import torch
import os
from os.path import join, split, exists
import pickle as pkl
import numpy as np
import codecs
from glob import glob
from torch.utils.data import Dataset, DataLoader
import trimesh
# Number of points to sample from the scan

class MyDataLoader(Dataset):
    def __init__(self, mode, batch_sz,
                 split_file='assets/dataset_split.pkl', num_workers=12,num_samples=10000
                 ):
        self.mode = mode
        with open(split_file, "rb") as f:
            self.split = pkl.load(f)

        self.data = self.split[mode]
        self.batch_size = batch_sz
        self.num_workers = num_workers
        self.num_samples = num_samples


    def __len__(self):
        return len(self.data)

    def get_loader(self, shuffle=True):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    @staticmethod
    def worker_init_fn(worker_id):
        """
        Worker init function to ensure true randomness.
        """
        base_seed = int(codecs.encode(os.urandom(4), 'hex'), 16)
        np.random.seed(base_seed + worker_id)

    def __getitem__(self, idx):
        path = self.data[idx]
        scan = np.loadtxt(join(path,'scan.txt'))

        # 坐标中心化
        center = np.mean(scan, axis=0)
        scan -= center
        height_range = np.max(scan[:, 1]) - np.min(scan[:, 1])
        # 计算缩放系数
        scale_factor = 1.5 / height_range
        scan *= scale_factor

        posed_scan_correspondences = np.loadtxt(join(path,'posed_scan_correspondences.txt'))
        posed_scan_correspondences -= center
        posed_scan_correspondences *= scale_factor


        correspondences = np.loadtxt(join(path, 'correspondences.txt'))
        correspondences -= center
        correspondences *= scale_factor


        part_labels = np.loadtxt(join(path, 'part_labels.txt'))
        vc = np.loadtxt(join(path, 'vc.txt'))
        smpl_dict = pkl.load(open(join(path, "parameters.pkl"), 'rb'), encoding='latin-1')


        #设置采样，最大10000，默认10000
        if self.num_samples !=10000:
            np.random.seed(666)
            rand_indices = np.random.choice(len(scan), self.num_samples, replace=False)
            scan = scan[rand_indices]
            posed_scan_correspondences = posed_scan_correspondences[rand_indices]
            correspondences = correspondences[rand_indices]
            part_labels = part_labels[rand_indices]
            vc = vc[rand_indices]


        #SMPL vertices
        SMPL_vertices = np.asarray(trimesh.load(join(path, split(path)[-1]+'_naked.obj')).vertices)
        SMPL_vertices -= center
        SMPL_vertices *= scale_factor

        return {'scan': scan.astype('float32'),
                'correspondences': correspondences.astype('float32'),
                'posed_scan_correspondences': posed_scan_correspondences.astype('float32'),
                'part_labels': part_labels.astype('float32'),
                'scan_vc': vc,
                'smpl_dict': smpl_dict,
                'path': path,
                'center':center,
                'scale_factor':scale_factor,
                'SMPL_vertices': SMPL_vertices.astype('float32')
                }



if __name__ =="__main__":

    #从文件调用不报错
    train_dataset = MyDataLoader('train', 1, num_workers=0,split_file='../assets/CAPE_Dataset_Sampled_10000.pkl', num_samples=10000)
    train_dataloader = train_dataset.get_loader()
    for dic in train_dataloader:
        print(dic['path'])