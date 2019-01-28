from .base_dataset import BaseDataset, get_transform

import sys
from .griddata import save
from .griddata.grid import Grid
from math import exp, sqrt, cos, sin
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py

_smina_atom_data = """{Hydrogen, EL_TYPE_H, AD_TYPE_H, XS_TYPE_SIZE,"Hydrogen", "H",	1.000000,	0.020000,	0.000510,	0.000000,	0.370000,	0.000000,	false,	false,	false,	false},
{PolarHydrogen, EL_TYPE_H, AD_TYPE_HD, XS_TYPE_SIZE,"PolarHydrogen", "HD",	1.000000,	0.020000,	0.000510,	0.000000,	0.370000,	0.000000,	false,	false,	false,	false},
{AliphaticCarbonXSHydrophobe, EL_TYPE_C, AD_TYPE_C, XS_TYPE_C_H,"AliphaticCarbonXSHydrophobe", "C",	2.000000,	0.150000,	-0.001430,	33.510300,	0.770000,	1.900000,	true,	false,	false,	false},
{AliphaticCarbonXSNonHydrophobe, EL_TYPE_C, AD_TYPE_C, XS_TYPE_C_P,"AliphaticCarbonXSNonHydrophobe", "C",	2.000000,	0.150000,	-0.001430,	33.510300,	0.770000,	1.900000,	false,	false,	false,	false},
{AromaticCarbonXSHydrophobe, EL_TYPE_C, AD_TYPE_A, XS_TYPE_C_H,"AromaticCarbonXSHydrophobe", "A",	2.000000,	0.150000,	-0.000520,	33.510300,	0.770000,	1.900000,	true,	false,	false,	false},
{AromaticCarbonXSNonHydrophobe, EL_TYPE_C, AD_TYPE_A, XS_TYPE_C_P,"AromaticCarbonXSNonHydrophobe", "A",	2.000000,	0.150000,	-0.000520,	33.510300,	0.770000,	1.900000,	false,	false,	false,	false},
{Nitrogen, EL_TYPE_N, AD_TYPE_N, XS_TYPE_N_P,"Nitrogen", "N",	1.750000,	0.160000,	-0.001620,	22.449300,	0.750000,	1.800000,	false,	false,	false,	true},
{NitrogenXSDonor, EL_TYPE_N, AD_TYPE_N, XS_TYPE_N_D,"NitrogenXSDonor", "N",	1.750000,	0.160000,	-0.001620,	22.449300,	0.750000,	1.800000,	false,	true,	false,	true},
{NitrogenXSDonorAcceptor, EL_TYPE_N, AD_TYPE_NA, XS_TYPE_N_DA,"NitrogenXSDonorAcceptor", "NA",	1.750000,	0.160000,	-0.001620,	22.449300,	0.750000,	1.800000,	false,	true,	true,	true},
{NitrogenXSAcceptor, EL_TYPE_N, AD_TYPE_NA, XS_TYPE_N_A,"NitrogenXSAcceptor", "NA",	1.750000,	0.160000,	-0.001620,	22.449300,	0.750000,	1.800000,	false,	false,	true,	true},
{Oxygen, EL_TYPE_O, AD_TYPE_O, XS_TYPE_O_P,"Oxygen", "O",	1.600000,	0.200000,	-0.002510,	17.157300,	0.730000,	1.700000,	false,	false,	false,	true},
{OxygenXSDonor, EL_TYPE_O, AD_TYPE_O, XS_TYPE_O_D,"OxygenXSDonor", "O",	1.600000,	0.200000,	-0.002510,	17.157300,	0.730000,	1.700000,	false,	true,	false,	true},
{OxygenXSDonorAcceptor, EL_TYPE_O, AD_TYPE_OA, XS_TYPE_O_DA,"OxygenXSDonorAcceptor", "OA",	1.600000,	0.200000,	-0.002510,	17.157300,	0.730000,	1.700000,	false,	true,	true,	true},
{OxygenXSAcceptor, EL_TYPE_O, AD_TYPE_OA, XS_TYPE_O_A,"OxygenXSAcceptor", "OA",	1.600000,	0.200000,	-0.002510,	17.157300,	0.730000,	1.700000,	false,	false,	true,	true},
{Sulfur, EL_TYPE_S, AD_TYPE_S, XS_TYPE_S_P,"Sulfur", "S",	2.000000,	0.200000,	-0.002140,	33.510300,	1.020000,	2.000000,	false,	false,	false,	true},
{SulfurAcceptor, EL_TYPE_S, AD_TYPE_SA, XS_TYPE_S_P,"SulfurAcceptor", "SA",	2.000000,	0.200000,	-0.002140,	33.510300,	1.020000,	2.000000,	false,	false,	false,	true},
{Phosphorus, EL_TYPE_P, AD_TYPE_P, XS_TYPE_P_P,"Phosphorus", "P",	2.100000,	0.200000,	-0.001100,	38.792400,	1.060000,	2.100000,	false,	false,	false,	true},
{Fluorine, EL_TYPE_F, AD_TYPE_F, XS_TYPE_F_H,"Fluorine", "F",	1.545000,	0.080000,	-0.001100,	15.448000,	0.710000,	1.500000,	true,	false,	false,	true},
{Chlorine, EL_TYPE_Cl, AD_TYPE_Cl, XS_TYPE_Cl_H,"Chlorine", "Cl",	2.045000,	0.276000,	-0.001100,	35.823500,	0.990000,	1.800000,	true,	false,	false,	true},
{Bromine, EL_TYPE_Br, AD_TYPE_Br, XS_TYPE_Br_H,"Bromine", "Br",	2.165000,	0.389000,	-0.001100,	42.566100,	1.140000,	2.000000,	true,	false,	false,	true},
{Iodine, EL_TYPE_I, AD_TYPE_I, XS_TYPE_I_H,"Iodine", "I",	2.360000,	0.550000,	-0.001100,	55.058500,	1.330000,	2.200000,	true,	false,	false,	true},
{Magnesium, EL_TYPE_Met, AD_TYPE_Mg, XS_TYPE_Met_D,"Magnesium", "Mg",	0.650000,	0.875000,	-0.001100,	1.560000,	1.300000,	1.200000,	false,	true,	false,	true},
{Manganese, EL_TYPE_Met, AD_TYPE_Mn, XS_TYPE_Met_D,"Manganese", "Mn",	0.650000,	0.875000,	-0.001100,	2.140000,	1.390000,	1.200000,	false,	true,	false,	true},
{Zinc, EL_TYPE_Met, AD_TYPE_Zn, XS_TYPE_Met_D,"Zinc", "Zn",	0.740000,	0.550000,	-0.001100,	1.700000,	1.310000,	1.200000,	false,	true,	false,	true},
{Calcium, EL_TYPE_Met, AD_TYPE_Ca, XS_TYPE_Met_D,"Calcium", "Ca",	0.990000,	0.550000,	-0.001100,	2.770000,	1.740000,	1.200000,	false,	true,	false,	true},
{Iron, EL_TYPE_Met, AD_TYPE_Fe, XS_TYPE_Met_D,"Iron", "Fe",	0.650000,	0.010000,	-0.001100,	1.840000,	1.250000,	1.200000,	false,	true,	false,	true},
{GenericMetal, EL_TYPE_Met, AD_TYPE_METAL, XS_TYPE_Met_D,"GenericMetal", "M",	1.200000,	0.000000,	-0.001100,	22.449300,	1.750000,	1.200000,	false,	true,	false,	true}
"""

class SminaAtomType:
    Hydrogen = 1
    PolarHydrogen = 2
    AliphaticCarbonXSHydrophobe = 3
    AliphaticCarbonXSNonHydrophobe = 4
    AromaticCarbonXSHydrophobe = 5
    AromaticCarbonXSNonHydrophobe = 6
    Nitrogen = 7
    NitrogenXSDonor = 8
    NitrogenXSDonorAcceptor = 9
    NitrogenXSAcceptor = 10
    Oxygen = 11
    OxygenXSDonor = 12
    OxygenXSDonorAcceptor = 13
    OxygenXSAcceptor = 14
    Sulfur = 15
    SulfurAcceptor = 16
    Phosphorus = 17
    Fluorine = 18
    Chlorine = 19
    Bromine = 20
    Iodine = 21
    Magnesium = 22
    Manganese = 23
    Zinc = 24
    Calcium = 25
    Iron = 26
    GenericMetal = 27

class Element:
    H    =  0;
    C    =  1;
    N    =  2;
    O    =  3;
    S    =  4;
    P    =  5;
    F    =  6;
    Cl   =  7;
    Br   =  8;
    I    =  9;
    Met  = 10;
    SIZE = 11;

class AutoDockType:
    C    =  0;
    A    =  1;
    N    =  2;
    O    =  3;
    P    =  4;
    S    =  5;
    H    =  6;
    F    =  7;
    I    =  8;
    NA   =  9;
    OA   = 10;
    SA   = 11;
    HD   = 12;
    Mg   = 13;
    Mn   = 14;
    Zn   = 15;
    Ca   = 16;
    Fe   = 17;
    Cl   = 18;
    Br   = 19;
    METAL = 20;
    
class XSType:
    C_H   =  0;
    C_P   =  1;
    N_P   =  2;
    N_D   =  3;
    N_A   =  4;
    N_DA  =  5;
    O_P   =  6;
    O_D   =  7;
    O_A   =  8;
    O_DA  =  9;
    S_P   = 10;
    P_P   = 11;
    F_H   = 12;
    Cl_H  = 13;
    Br_H  = 14;
    I_H   = 15;
    Met_D = 16;
    SIZE  = 17


class AtomData:
    def __init__(self):
        self.data = []
        self.adtype_map = {}
        self.xstype_map = {}
        self.sminatype_map = {}
        for line in _smina_atom_data.splitlines():
            token = [_.strip() for _ in line.strip()[1:-2].split(',')]
            entry = {
                'element': getattr(Element, token[1].split('_')[-1]),
                'smina_type': getattr(SminaAtomType, token[0]),
                'autodock_type': getattr(AutoDockType, token[2].split('_')[-1]),
                'xs_type': getattr(XSType, token[3][8:]),
                'smina_name': token[4][1:-1],
                'autodock_name': token[5][1:-1],
                'autodock_radius': float(token[6]),
                'autodock_depth': float(token[7]),
                'autodock_solvation': float(token[8]),
                'autodock_volume': float(token[9]),
                'bond_radius': float(token[10]),
                'xs_radius': float(token[11]),
                'xs_hydrophobe': token[12] == 'true',
                'xs_donor': token[13] == 'true',
                'xs_acceptor': token[14] == 'true',
                'xs_heteroatom': token[15] == 'true',
            }
            self.adtype_map[entry['autodock_type']] = len(self.data)
            self.sminatype_map[entry['smina_type']] = len(self.data)
            # this should be at the end
            self.data.append(entry)
    
    def __getitem__(self, index):
        return self.data[index]

    def query_adtype(self, autodock_type):
        return self.adtype_map[getattr(AutoDockType, autodock_type)]

    def query_xstype(self, xs_type):
        return self.xstype_map[getattr(XSType, xs_type)]

    def query_sminatype(self, smina_type):
        return self.sminatype_map[getattr(SminaAtomType, smina_type)]

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
        return len(self.df) * 10

    def __getitem__(self, index):
        row = self.df.iloc[index % len(self.df)]
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
        if file.endswith('pdbqt'):
            self.pdbqtfile = file
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
            self.coordss = f['coords']
            self.atomdata = f['atomdata']

