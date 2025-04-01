import os
from os.path import split, join, exists
from data_loader.data_loader import MyDataLoader
import torch
from models.UBSP_Net_v2 import UBSP_Net_repro
from models.trainer import UBSP_Net_Train

def main(mode, exp_id, optimizer, batch_size, epochs, save_name=None, num_saves=None, naked=False,
         split_file=None):
    if split_file is None:
        split_file = 'assets/dataset_split.pkl'

    net = UBSP_Net_repro(c=3,k=14)


    if naked:
        exp_name = 'ubsp_net/naked_exp_id_{}'.format(exp_id)
    else:
        exp_name = 'ubsp_net/exp_id_{}'.format(exp_id)

    if mode == 'train':
        dataset = MyDataLoader('train', batch_size, num_workers=8,
                                 split_file=split_file).get_loader()
        trainer = UBSP_Net_Train(net, torch.device("cuda"), dataset, None, exp_name,
                                         optimizer=optimizer)
        trainer.train_model(epochs)
    elif mode == 'val':
        dataset = MyDataLoader('val', batch_size, num_workers=8,
                                 split_file=split_file).get_loader(shuffle=False)
        trainer = UBSP_Net_Train(net, torch.device("cuda"), None, dataset, exp_name,
                                         optimizer=optimizer)
        trainer.pred_model(save_name=save_name, num_saves=num_saves)
    elif mode == 'eval':
        dataset = MyDataLoader('val', batch_size, num_workers=16,
                                 split_file=split_file).get_loader(shuffle=False)
        trainer = UBSP_Net_Train(net, torch.device("cuda"), None, dataset, exp_name,
                                         optimizer=optimizer)
        trainer.eval_model('val')
    else:
        print('Invalid mode. should be either train, val or eval.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('-exp_id',default=10000, type=str)
    parser.add_argument('-batch_size', default=24, type=int)
    parser.add_argument('-optimizer', default='Adam', type=str)
    parser.add_argument('-epochs', default=4000, type=int)
    parser.add_argument('-augment', default=False, action='store_true')
    # Train network for dressed or undressed scans
    parser.add_argument('-naked', default=True, action='store_true')
    parser.add_argument('-split_file', type=str, default='assets/dataset_split_female.pkl')
    # Validation specific arguments
    parser.add_argument('-mode', default='train', choices=['train', 'val', 'eval'])
    parser.add_argument('-save_name', default='', type=str)
    parser.add_argument('-num_saves', default=1, type=int)
    args = parser.parse_args()

    if args.mode == 'val':
        # assert len(args.save_name) > 0
        main('val', args.exp_id, args.optimizer, args.batch_size, args.epochs,
             save_name=args.save_name, num_saves=args.num_saves, naked=args.naked, split_file=args.split_file)
    elif args.mode == 'train':
        main('train', args.exp_id, args.optimizer, args.batch_size, args.epochs, naked=args.naked,
             split_file=args.split_file)

    elif args.mode == 'eval':
        main('eval', args.exp_id, args.optimizer, args.batch_size, args.epochs, naked=args.naked,
             split_file=args.split_file)
