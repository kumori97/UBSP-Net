
from __future__ import division
from os.path import join, split, exists
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import time
import trimesh
import pickle as pkl
import numpy as np
from collections import Counter
from tqdm import tqdm
# from kaolin.metrics.point import SidedDistance
from models.volumetric_SMPL import VolumetricSMPL
from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.torch_functions import batch_gather, chamfer_distance

NUM_POINTS = 30000


class Trainer(object):
    '''
    Trainer for predicting scan to SMPL correspondences from a pointcloud.
    This trainer does not optimise the correspondences.
    '''

    def __init__(self, model, device, train_loader, val_loader, exp_name, optimizer='Adam'):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_data_loader = train_loader
        self.val_data_loader = val_loader

        self.exp_path = join(os.path.dirname(__file__), '../experiments/{}'.format(exp_name))
        self.checkpoint_path = join(self.exp_path, 'checkpoints/')
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(join(self.exp_path, 'summary'.format(exp_name)))
        self.val_min = None

    @staticmethod
    def sum_dict(los):
        temp = 0
        for l in los:
            temp += los[l]
        return temp

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss_ = self.compute_loss(batch)
        loss = self.sum_dict(loss_)
        loss.backward()
        self.optimizer.step()

        return {k: loss_[k].item() for k in loss_}

    def compute_loss(self, batch):
        device = self.device
        scan = batch.get('scan').to(device)
        smpl = batch.get('smpl').to(device)
        logits = self.model(scan)

        # MSE loss on vertices
        mse = F.mse_loss(logits.view(-1, 6890, 3), smpl)

        loss = {'mse': mse}
        return loss

    def train_model(self, epochs):
        start = self.load_checkpoint()
        for epoch in range(start, epochs):
            print('Start epoch {}'.format(epoch))

            if epoch % 1 == 0:
                self.save_checkpoint(epoch)
                '''Add validation loss here if over-fitting is suspected'''

            sum_loss = None
            loop = tqdm(self.train_data_loader)
            for n, batch in enumerate(loop):
                loss = self.train_step(batch)
                # print(" Epoch: {}, Current loss: {}".format(epoch, loss))
                if sum_loss is None:
                    sum_loss = Counter(loss)
                else:
                    sum_loss += Counter(loss)
                l_str = 'Ep: {}'.format(epoch)
                for l in sum_loss:
                    l_str += ', {}: {:0.5f}'.format(l, sum_loss[l] / (1 + n))
                loop.set_description(l_str)

            for l in sum_loss:
                self.writer.add_scalar(l, sum_loss[l] / len(self.train_data_loader), epoch)

    def save_checkpoint(self, epoch):
        path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(epoch))
        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self, number=-1):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        if number == -1:
            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)

            if checkpoints[-1] == 0:
                print('Not loading model as this is the first epoch')
                return 0

            path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(checkpoints[-1]))
        else:
            path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(number))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

class UBSP_Net_Train(Trainer):
    def compute_loss(self, batch):
        device = self.device
        scan = batch.get('scan').to(device)
        part_label = batch.get('part_labels').to(device).long()
        correspondences = batch.get('correspondences').to(device)
        posed_scan_correspondences = batch.get('posed_scan_correspondences').to(device)

        logits = self.model(scan)
        ce = F.cross_entropy(logits['part_labels'], part_label.squeeze(-1))
        mse = F.mse_loss(logits['correspondences'].permute(0, 2, 1), correspondences)

        mse_inner = F.mse_loss(logits['inner_points'].permute(0, 2, 1), posed_scan_correspondences)*100

        loss = {'cross_entropy': ce, 'correspondences': mse,'inner': mse_inner}
        return loss

    def eval_model(self, mode):
        """
        average accuracy and distance
        """
        epoch = self.load_checkpoint()
        print('Evaluating on {} set with epoch {}'.format(mode, epoch))

        if mode == 'train':
            data_loader = self.train_data_loader
        else:
            data_loader = self.val_data_loader

        correct, total, error = 0, 0, 0
        for batch in tqdm(data_loader):
            scan = batch.get('scan').to(self.device)
            label_gt = batch.get('part_labels').view(-1, )
            corr_gt = batch.get('correspondences')
            with torch.no_grad():
                pred = self.model(scan)

                _, predicted = torch.max(pred['part_labels'].data, 1)
                correct += (predicted.cpu().view(-1, ) == label_gt).sum().item()
                total += label_gt.shape[0]

                # ToDo: map correspondences in R^3 to SMPL surface

                error += (((corr_gt - pred['correspondences'].permute(0, 2, 1).cpu())**2).sum(dim=-1)**0.5).mean()

        print('Part Accuracy: {}, Corr. Dist: {}'.format(correct * 1. / total, error/len(data_loader)))

    def pred_model(self, save_name, num_saves=None):
        """
        :param save_name: folder name to save the results to inside the experiment folder
        :param num_saves: number of examples to save, None implies save all
        """
        from psbody.mesh import Mesh
        from os.path import join, exists
        import os

        self.model.train(False)

        epoch = self.load_checkpoint()
        print('Testing with epoch {}'.format(epoch))
        if not exists(join(self.exp_path, save_name + '_ep_{}'.format(epoch))):
            os.makedirs(join(self.exp_path, save_name + '_ep_{}'.format(epoch)))

        count = 0
        for batch in tqdm(self.val_data_loader):
            names = batch.get('name')
            scan = batch.get('scan')
            vcs = batch.get('scan_vc').numpy()
            with torch.no_grad():
                out = self.model(scan.to(self.device))
                pred = out['correspondences'].detach().permute(0, 2, 1).cpu().numpy()   #(B,30000,3)
                part_label = out['part_labels'].detach()
                _,part_label  = torch.max(part_label.data, 1)
                part_label = np.array(part_label.cpu())

            for v, name, sc, vc, pl in zip(pred, names, scan, vcs, part_label):
                name = split(name)[1]
                # save scan with vc
                t = Mesh(sc, [])
                t.set_vertex_colors(vc)
                t.write_ply(join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_scan.ply'))

                # save raw correspondences
                t = Mesh(v, [])
                t.set_vertex_colors(vc)
                t.write_ply(join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_corr_raw.ply'))

                # save part_labels
                t.set_vertex_colors_from_weights(pl)
                t.write_ply(join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_part.ply'))
                count += 1
            if (num_saves is not None) and (count > num_saves):
                break
