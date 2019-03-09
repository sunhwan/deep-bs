import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from math import exp, sqrt, cos, sin, e
import numpy as np
import numba
from .atoms import *

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = [Center()]
    if opt.rotate > 0:
        transform_list.append(Rotate(opt.rotate))
    if opt.channels == 'cno':
        transform_list += [ProteinChannel('element', Element.C, opt.grid_size, opt.grid_spacing, opt.grid_method, rvdw=opt.rvdw),
                           ProteinChannel('element', Element.N, opt.grid_size, opt.grid_spacing, opt.grid_method, rvdw=opt.rvdw),
                           ProteinChannel('element', Element.O, opt.grid_size, opt.grid_spacing, opt.grid_method, rvdw=opt.rvdw),
                           LigandChannel('element', Element.C, opt.grid_size, opt.grid_spacing, opt.grid_method, rvdw=opt.rvdw),
                           LigandChannel('element', Element.N, opt.grid_size, opt.grid_spacing, opt.grid_method, rvdw=opt.rvdw),
                           LigandChannel('element', Element.O, opt.grid_size, opt.grid_spacing, opt.grid_method, rvdw=opt.rvdw)]
    if opt.channels == 'gnina':
        protein_atom_types = ['AliphaticCarbonXSHydrophobe',
                              'AliphaticCarbonXSNonHydrophobe',
                              'AromaticCarbonXSHydrophobe',
                              'AromaticCarbonXSNonHydrophobe',
                              'Calcium',
                              'Iron',
                              'Magnesium',
                              'Nitrogen',
                              'NitrogenXSAcceptor',
                              'NitrogenXSDonor',
                              'NitrogenXSDonorAcceptor',
                              'OxygenXSAcceptor',
                              'OxygenXSDonorAcceptor',
                              'Phosphorus',
                              'Sulfur',
                              'Zinc']
        ligand_atom_types = ['AliphaticCarbonXSHydrophobe',
                             'AliphaticCarbonXSNonHydrophobe',
                             'AromaticCarbonXSHydrophobe',
                             'AromaticCarbonXSNonHydrophobe',
                             'Bromine',
                             'Chlorine',
                             'Fluorine',
                             'Nitrogen',
                             'NitrogenXSAcceptor',
                             'NitrogenXSDonor',
                             'NitrogenXSDonorAcceptor',
                             'OxygenXSAcceptor',
                             'OxygenXSDonorAcceptor',
                             'Phosphorus',
                             'Sulfur',
                             'SulfurAcceptor',
                             'Iodine',
                             'Boron']
        for atom_type in protein_atom_types:
            sm_type = getattr(SminaAtomType, atom_type)
            transform_list.append(ProteinChannel('smina_type', sm_type, opt.grid_size, opt.grid_spacing, opt.grid_method))
        for atom_type in ligand_atom_types:
            sm_type = getattr(SminaAtomType, atom_type)
            transform_list.append(LigandChannel('smina_type', sm_type, opt.grid_size, opt.grid_spacing, opt.grid_method))

    elif opt.channels == 'test':
        transform_list += [Empty(opt.grid_size, opt.grid_spacing, opt.rvdw)]

    transform_list += [ToTensor()]
    # TODO: Try normalize?
    # TODO: Try flip?
    return transforms.Compose(transform_list)


def coords_to_grid_numpy(coords, grid, nx, ny, nz, xmin, ymin, zmin, spacing, rvdw):
    assert grid.shape == (nx, ny, nz)
    ncoords = len(coords)
    X,Y,Z = np.mgrid[xmin:xmin+nx*spacing:spacing, 
                     ymin:ymin+ny*spacing:spacing,
                     zmin:zmin+nz*spacing:spacing]

    xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    for i in range(ncoords):
        r = np.linalg.norm(xyz - (coords[i]), axis=1).reshape((nx, ny, nz))
        grid += 1 - np.exp(-(rvdw/r)**12)
    return grid

@numba.jit('f4[:,:,:](f4[:,:], f4[:,:,:], i8, i8, i8, f8, f8, f8, f8, f8)', nopython=True)
def coords_to_grid_numba(coords, grid, nx, ny, nz, xmin, ymin, zmin, spacing, rvdw):
    exps = 0.001
    rmax = 30
    expt = np.exp(-(rvdw/np.arange(0,rmax,exps))**12)
    nc = len(coords)
    for i in range(nx):
        ix = xmin + i*spacing
        for j in range(ny):
            iy = ymin + j*spacing
            for k in range(nz):
                iz = zmin + k*spacing
                for l in range(nc):
                    dx = ix - coords[l,0]
                    dy = iy - coords[l,1]
                    dz = iz - coords[l,2]
                    r = sqrt(dx*dx + dy*dy + dz*dz)
                    #grid[i,j,k] += 1 - exp(-(rvdw/r)**12)
                    if r > rmax: continue
                    grid[i,j,k] += 1 - expt[int(r/exps)]
    return grid

@numba.jit('f4[:,:,:](f4[:,:], f4[:,:,:], i8, i8, i8, f8, f8, f8, f8, f8)', nopython=True)
def coords_to_grid_gnina(coords, grid, nx, ny, nz, xmin, ymin, zmin, spacing, rvdw):
    exps = 0.001
    rmax = 1.5*rvdw
    expt = [0.]
    for r in np.arange(exps, rvdw, exps):
        expt.append(exp(-2*rvdw**2/r**2))
    for r in np.arange(rvdw, rmax, exps):
        expt.append(4/(e**2*rvdw**2)*r**2 -12/(e**2*rvdw)*r + 9/e**2)
    nc = len(coords)
    for i in range(nx):
        ix = xmin + i*spacing
        for j in range(ny):
            iy = ymin + j*spacing
            for k in range(nz):
                iz = zmin + k*spacing
                for l in range(nc):
                    dx = ix - coords[l,0]
                    dy = iy - coords[l,1]
                    dz = iz - coords[l,2]
                    r = sqrt(dx*dx + dy*dy + dz*dz)
                    #grid[i,j,k] += 1 - exp(-(rvdw/r)**12)
                    if r > rmax: continue
                    if r < rvdw:
                        grid[i,j,k] += expt[int(r/exps)]
    return grid


class ProteinChannel:
    """Convert atomic coordinates into grid (channel)
    
    Args:
        atomtypes: list of atom types to convert into grid
        size: size of grid in angstrom
        spacing: grid spacing in angstrom
        rvdw: r_vdw parameter in grid
    """
    def __init__(self, atomtype_key, atomtype_filter, size, spacing, method='gnina', rvdw=None):
        self.atomtype_key = atomtype_key
        self.atomtype_filter = atomtype_filter
        self.size = size
        self.spacing = spacing
        self.rvdw = rvdw
        self.method = method
        self.atom_data = AtomData()
    
    def __call__(self, sample):
        size = float(self.size)
        spacing = float(self.spacing)
        data = self.atom_data.query(self.atomtype_key, self.atomtype_filter)
        rvdw = data['autodock_radius']
        if self.rvdw is not None:
            rvdw = float(self.rvdw)
        nx, ny, nz = [int(size/spacing)+1 for _ in range(3)]
        xmin, ymin, zmin = [_-size/2 for _ in sample['ligand'].center]

        idx = [i for i, data_i in enumerate(sample['pocket'].atomdata) if self.atom_data[data_i][self.atomtype_key] == self.atomtype_filter]
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        if len(idx) == 0:
            sample['channels'].append(grid)
            return sample

        if self.method == 'gnina':
            grid = coords_to_grid_gnina(sample['pocket'].coords[idx], grid, 
                                        nx, ny, nz, xmin, ymin, zmin, spacing, rvdw)
        elif self.method == 'kdeep':
            grid = coords_to_grid_numba(sample['pocket'].coords[idx], grid, 
                                        nx, ny, nz, xmin, ymin, zmin, spacing, rvdw)
        sample['channels'].append(grid)
        return sample

class LigandChannel:
    """Convert atomic coordinates into grid (channel)
    
    Args:
        atomtypes: list of atom types to convert into grid
        size: size of grid in angstrom
        spacing: grid spacing in angstrom
        rvdw: r_vdw parameter in grid
    """
    def __init__(self, atomtype_key, atomtype_filter, size, spacing, method='gnina', rvdw=None):
        self.atomtype_key = atomtype_key
        self.atomtype_filter = atomtype_filter
        self.size = size
        self.spacing = spacing
        self.rvdw = rvdw
        self.method = method
        self.atom_data = AtomData()
    
    def __call__(self, sample):
        size = float(self.size)
        spacing = float(self.spacing)
        data = self.atom_data.query(self.atomtype_key, self.atomtype_filter)
        rvdw = data['autodock_radius']
        if self.rvdw is not None:
            rvdw = float(self.rvdw)
        nx, ny, nz = [int(size/spacing)+1 for _ in range(3)]
        xmin, ymin, zmin = [_-size/2 for _ in sample['ligand'].center]

        idx = [i for i, data_i in enumerate(sample['ligand'].atomdata) if self.atom_data[data_i][self.atomtype_key] == self.atomtype_filter]
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        if len(idx) == 0:
            sample['channels'].append(grid)
            return sample

        if self.method == 'gnina':
            grid = coords_to_grid_gnina(sample['ligand'].coords[idx], grid, 
                                        nx, ny, nz, xmin, ymin, zmin, spacing, rvdw)
        elif self.method == 'kdeep':
            grid = coords_to_grid_numba(sample['ligand'].coords[idx], grid, 
                                        nx, ny, nz, xmin, ymin, zmin, spacing, rvdw)
        sample['channels'].append(grid)
        return sample

class Empty:
    """Empty channel for testing purpose
    
    Args:
        size: size of grid in angstrom
        spacing: grid spacing in angstrom
        rvdw: r_vdw parameter in grid
    """
    def __init__(self, size, spacing, rvdw):
        self.size = size
        self.spacing = spacing
        self.rvdw = rvdw
    
    def __call__(self, sample):
        size = float(self.size)
        spacing = float(self.spacing)
        rvdw = float(self.rvdw)
        nx, ny, nz = [int(size/spacing)+1 for _ in range(3)]
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        sample['channels'].append(grid)
        return sample


class Rotate:
    """Rotate input structure
    
    Args:
        degree: maximum degree to rotate (+/-)
    """
    def __init__(self, degree):
        self.degree = degree
    
    def __call__(self, sample):
        theta = (np.random.random_sample(3,) - 0.5)*self.degree/180*np.pi
        rx = np.matrix((( 1,             0,              0),
                        ( 0, cos(theta[0]), -sin(theta[0])),
                        ( 0, sin(theta[0]),  cos(theta[0]))))
        ry = np.matrix((( cos(theta[1]), 0, sin(theta[1])),
                        (             0, 1,             0),
                        (-sin(theta[1]), 0, cos(theta[1]))))
        rz = np.matrix((( cos(theta[2]), -sin(theta[2]), 0),
                        ( sin(theta[2]),  cos(theta[2]), 0),
                        (             0,              0, 1)))
        r = rx * ry * rz
        sample['pocket'].coords = np.array(np.dot(r, (sample['pocket'].coords).T).T, dtype=np.float32)
        sample['ligand'].coords = np.array(np.dot(r, (sample['ligand'].coords).T).T, dtype=np.float32)
        return sample
    
class Center:
    """Center input structure"""
    def __call__(self, sample):
        com = sample['ligand'].center
        sample['pocket'].coords = sample['pocket'].coords - com
        sample['ligand'].coords = sample['ligand'].coords - com
        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grids = np.vstack([c[np.newaxis,:] for c in sample['channels']])
        return {
            'grids': torch.from_numpy(grids),
            'affinity': torch.from_numpy(np.array([sample['affinity']], dtype=np.float32))
        }
