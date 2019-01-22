from .base_dataset import BaseDataset, get_transform

import sys
from .griddata import save
from .griddata.grid import Grid
from math import exp, sqrt, cos, sin
import numpy as np
import pandas as pd

class PdbBindDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.df = pd.read_csv(opt.csvfile)
        self.dataroot = opt.dataroot
        self.transform = get_transform(opt)
        self.df.affinity = -np.log(self.df.affinity)
        if opt.filter_kd:
            self.df = self.df[self.df.afftype == 'Kd']
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pdbfile = '{}/{}/{}_protein.pdb'.format(self.dataroot, row.code, row.code)
        pocketfile = '{}/{}/{}_pocket.pdb'.format(self.dataroot, row.code, row.code)
        ligandfile = '{}/{}/{}_ligand.mol2'.format(self.dataroot, row.code, row.code)
        sample = {
            'code': row.code,
            'pdbfile': pdbfile,
            'pocket': GridPDB(pocketfile),
            'ligand': GridPDB(ligandfile),
            'channels': [],
            'affinity': row.affinity
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def name(self):
        return 'PdbBindDataset'


class GridPDB:
    def __init__(self, file):
        if file.endswith('pdb'):
            self.pdbfile = file
            self.parse_pdb()
        if file.endswith('mol2'):
            self.mol2file = file
            self.parse_mol2()
        
    def parse_mol2(self):
        self.atoms = []
        self.atomtypes = []
        self.coords = []
        flag = False
        for line in open(self.mol2file):
            if line.startswith("@<TRIPOS>ATOM"):
                flag = True
                continue
            if line.startswith("@<TRIPOS>BOND"):
                break
            if flag:
                name = line[8:16].strip()
                if name[0] == 'H': continue
                    
                x = line[16:26]
                y = line[26:36]
                z = line[36:46]
                self.atoms.append(name)
                self.atomtypes.append(name[0])
                self.coords.append(list(map(float, (x, y, z))))
                
        self.atoms = np.array(self.atoms)
        self.atomtypes = np.array(self.atomtypes)
        self.coords = np.array(self.coords, dtype=np.float32)
        self.center = np.average(self.coords, axis=0)
    
    def parse_pdb(self):
        self.atoms = []
        self.atomtypes = []
        self.coords = []
        for line in open(self.pdbfile):
            if line.startswith("ATOM"):
                name = line[11:17].strip()
                if name[0] == 'H': continue
                if name[0].isdigit(): continue
                    
                x = line[30:38]
                y = line[38:46]
                z = line[46:54]
                self.atoms.append(name)
                self.atomtypes.append(name[0])
                self.coords.append(list(map(float, (x, y, z))))
                
        self.atoms = np.array(self.atoms)
        self.atomtypes = np.array(self.atomtypes)
        self.coords = np.array(self.coords, dtype=np.float32)
        self.center = np.average(self.coords, axis=0)
    
    def save_grid(self, filename):
        g = Grid()
        g.n_elements = np.cumprod(self.elements.shape)
        g.center = list(self.center)
        g.shape = self.elements.shape
        g.spacing = (self.spacing, self.spacing, self.spacing)
        g.set_elements(self.ndelements.flatten())
        griddata.save(g, open(filename, 'w'), format='dx')


