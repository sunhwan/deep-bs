from .base_dataset import BaseDataset, get_transform
from .atoms import *

import sys, os
from .griddata import save
from .griddata.grid import Grid
from math import exp, sqrt, cos, sin
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py

class Jak2Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.df = pd.read_csv(opt.csvfile)
        self.dataroot = opt.dataroot
        self.transform = get_transform(opt)
        self.df.affinity = -np.log10(self.df.affinity)
        if opt.filter_kd:
            self.df = self.df[self.df.afftype == 'Kd']
    
    def __len__(self):
        #return len(self.df) * 10
        return len(self.df)

    def __getitem__(self, index):
        #row = self.df.iloc[index % len(self.df)]
        row = self.df.iloc[index]
        pdbfile = '{}/input/3LPB_A.pdbqt'.format(self.dataroot)
        ligandfile = row.pose

        pocket_pdbqt_file = pdbfile
        if os.path.exists(pocket_pdbqt_file):
            pocket = GridPDB(pocket_pdbqt_file)
        else:
            raise Error("Please preprocess PDB files")
            #pocket = GridPDB(pocketfile)

        ligand_pdbqt_file = ligandfile
        if os.path.exists(ligand_pdbqt_file):
            ligand = GridPDB(ligand_pdbqt_file)
        else:
            raise Error("Please preprocess ligand files")
            #ligand = GridPDB(ligandfile)

        sample = {
            'code': '',
            'pdbfile': pdbfile,
            'pocket': GridPDB(pocket_pdbqt_file),
            'ligand': GridPDB(ligand_pdbqt_file),
            'channels': [],
            'pose': 1,
            'affinity': row.affinity
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def name(self):
        return 'PdbBindDataset'


class GridPDB:
    def __init__(self, file):
        self.filename = file
        if file.endswith('pdb'):
            self.pdbfile = file
            self.parse_pdb()
        if file.endswith('mol2'):
            self.mol2file = file
            self.parse_mol2()
        if file.endswith('pdbqt'):
            self.pdbqtfile = file
            h5file = file[:-5] + 'h5'
            if os.path.exists(h5file):
                self.from_h5(h5file)
            else:
                self.parse_pdbqt()
        
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

    @property
    def center(self):
        return np.average(self.coords, axis=0)
    
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
    
    def parse_pdbqt(self):
        self.atoms = []
        self.atomtypes = []
        self.coords = []
        self.atomdata = []
        atom_data = AtomData()
        for line in open(self.pdbqtfile):
            if line.startswith("ATOM"):
                name = line[11:17].strip()
                x = line[30:38]
                y = line[38:46]
                z = line[46:54]
                atomtype = line[77:].strip()
                resname = line[17:20].strip()
                if resname == 'HOH':
                    continue
                self.atoms.append(name)
                self.atomtypes.append(atomtype)
                self.coords.append(list(map(float, (x, y, z))))
                self.atomdata.append(atom_data.query_adtype(atomtype))
                
        self.natom = len(self.atoms)
        self.atoms = np.array(self.atoms, dtype='S')
        self.atomtypes = np.array(self.atomtypes, dtype='S')
        self.coords = np.array(self.coords, dtype=np.float32)
        self.atomdata = np.array(self.atomdata, dtype=np.int)
        self.assign_bonds(atom_data)
        self.adjust_atom_type(atom_data)

    def assign_bonds(self, atom_data):
        natom = len(self.atoms)
        bonds = [set([]) for _ in range(natom)]
        threshold = 4
        allowance_factor = 1.2
        atom_data = AtomData()
        for i in range(natom):
            d = np.sum((self.coords - self.coords[i])**2, axis=1) # sqr distance
            nearby_indices = np.argwhere(d < threshold).flatten()
            data_i = atom_data[self.atomdata[i]]
            bond_radius_i = data_i['bond_radius']
            for j in nearby_indices:
                if j == i: continue
                data_j = atom_data[self.atomdata[j]]
                bond_radius_j = data_j['bond_radius']
                optimal_bond_distance = bond_radius_i + bond_radius_j
                bond_distance = sqrt(d[j])
                if bond_distance < allowance_factor * optimal_bond_distance:
                    bonds[i].add(j)
                    bonds[j].add(i)
        self.bonds = bonds
    
    def _is_atom_bonded_to_hetero(self, atom_i, atom_data):
        neighbors = self.bonds[atom_i]
        for atom_j in neighbors:
            data_j = atom_data[self.atomdata[atom_j]]
            if data_j['xs_heteroatom']: return True
        return False

    def _is_atom_bonded_to_H(self, atom_i, atom_data):
        neighbors = self.bonds[atom_i]
        for atom_j in neighbors:
            data_j = atom_data[self.atomdata[atom_j]]
            if data_j['smina_type'] == SminaAtomType.PolarHydrogen: return True
        return False
    
    def adjust_atom_type(self, atom_data):
        for i in range(self.natom):
            data_i = atom_data[self.atomdata[i]]
            if data_i['smina_type'] in (SminaAtomType.AliphaticCarbonXSHydrophobe, SminaAtomType.AliphaticCarbonXSNonHydrophobe):
                if self._is_atom_bonded_to_hetero(i, atom_data):
                    self.atomdata[i] = atom_data.query_sminatype('AliphaticCarbonXSNonHydrophobe')
                else:
                    self.atomdata[i] = atom_data.query_sminatype('AliphaticCarbonXSHydrophobe')
            
            elif data_i['smina_type'] in (SminaAtomType.AromaticCarbonXSHydrophobe, SminaAtomType.AromaticCarbonXSNonHydrophobe):
                if self._is_atom_bonded_to_hetero(i, atom_data):
                    self.atomdata[i] = atom_data.query_sminatype('AromaticCarbonXSNonHydrophobe')
                else:
                    self.atomdata[i] = atom_data.query_sminatype('AromaticCarbonXSHydrophobe')

            elif data_i['smina_type'] in (SminaAtomType.NitrogenXSDonor, SminaAtomType.Nitrogen):
                if self._is_atom_bonded_to_H(i, atom_data):
                    self.atomdata[i] = atom_data.query_sminatype('NitrogenXSDonor')
                else:
                    self.atomdata[i] = atom_data.query_sminatype('Nitrogen')

            elif data_i['smina_type'] in (SminaAtomType.NitrogenXSDonorAcceptor, SminaAtomType.NitrogenXSAcceptor):
                if self._is_atom_bonded_to_H(i, atom_data):
                    self.atomdata[i] = atom_data.query_sminatype('NitrogenXSDonorAcceptor')
                else:
                    self.atomdata[i] = atom_data.query_sminatype('NitrogenXSAcceptor')

            elif data_i['smina_type'] in (SminaAtomType.OxygenXSDonor, SminaAtomType.Oxygen):
                if self._is_atom_bonded_to_H(i, atom_data):
                    self.atomdata[i] = atom_data.query_sminatype('OxygenXSDonor')
                else:
                    self.atomdata[i] = atom_data.query_sminatype('Oxygen')

            elif data_i['smina_type'] in (SminaAtomType.OxygenXSDonorAcceptor, SminaAtomType.OxygenXSAcceptor):
                if self._is_atom_bonded_to_H(i, atom_data):
                    self.atomdata[i] = atom_data.query_sminatype('OxygenXSDonorAcceptor')
                else:
                    self.atomdata[i] = atom_data.query_sminatype('OxygenXSAcceptor')

    def save_grid(self, filename):
        g = Grid()
        g.n_elements = np.cumprod(self.elements.shape)
        g.center = list(self.center)
        g.shape = self.elements.shape
        g.spacing = (self.spacing, self.spacing, self.spacing)
        g.set_elements(self.ndelements.flatten())
        griddata.save(g, open(filename, 'w'), format='dx')

    def to_h5(self, filename):
        with h5py.File(filename, "w") as f:
            natoms = len(self.atoms)
            coords = f.create_dataset("coords", (natoms,3), dtype='f4')
            atomdata = f.create_dataset("atomdata", (natoms,), dtype='i')
            coords[:] = self.coords
            atomdata[:] = self.atomdata

    def from_h5(self, filename):
        with h5py.File(filename, "r") as f:
            self.coords = f['coords'][:]
            self.atomdata = f['atomdata'][:]

