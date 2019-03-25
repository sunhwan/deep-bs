import os
from options.test_options import TestOptions
from data.base_dataset import get_transform
from data.pdbbind_dataset import GridPDB
from data.base_dataset import BaseDataset
from data.base_data_loader import BaseDataLoader
from models.models import create_model
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

opt = TestOptions().parse()
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.batch_size = 1
opt.rotate = 0

model = create_model(opt)

protein_file = 'jak2-test/jak2_docked/input/3LPB_A.pdbqt'
ligand_file = '/home/sunhwan/az5.pdbqt'

class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.transform = get_transform(opt)
    
    def __len__(self):
        return 1

    def __getitem__(self, index):
        pocket_pdbqt_file = protein_file
        if os.path.exists(pocket_pdbqt_file):
            pocket = GridPDB(pocket_pdbqt_file)
        else:
            raise Error("Please preprocess PDB files")
            #pocket = GridPDB(pocketfile)

        ligand_pdbqt_file = ligand_file
        if os.path.exists(ligand_pdbqt_file):
            ligand = GridPDB(ligand_pdbqt_file)
        else:
            raise Error("Please preprocess ligand files")
            #ligand = GridPDB(ligandfile)

        sample = {
            'code': '',
            'pdbfile': '',
            'pocket': GridPDB(pocket_pdbqt_file),
            'ligand': GridPDB(ligand_pdbqt_file),
            'channels': [],
            'pose': 1,
            'affinity': 1
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def name(self):
        return 'SingleDataset'

class SingleDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = SingleDataset()
        self.dataset.initialize(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=opt.batch_size,
            shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size: break
            yield data

    def name(self): return 'CustomDatasetDataLoader'
    def load_data(self): return self
    def __len__(self): return min(len(self.dataset), self.opt.max_dataset_size)

data_loader = SingleDataLoader()
data_loader.initialize(opt)
dataset = data_loader.load_data()

for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()

    print(model.preds_pose)
    print(model.preds_affinity)
