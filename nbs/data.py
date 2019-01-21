import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import griddata
from griddata.grid import Grid
import numba
from math import exp, sqrt, cos, sin
import pandas as pd
import numpy as np

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
    
    def compute_grid(self, size=20, spacing=1.0):
        nx, ny, nz = [int(size/spacing)+1 for _ in range(3)]
        xmin, ymin, zmin = [_-int(size/2) for _ in pdb.center]
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        self.ndelements = coords_to_grid_numba(self.coords, grid, nx, ny, nz, xmin, ymin, zmin, spacing)
    
    def save_grid(self, filename):
        g = Grid()
        g.n_elements = np.cumprod(self.elements.shape)
        g.center = list(self.center)
        g.shape = self.elements.shape
        g.spacing = (self.spacing, self.spacing, self.spacing)
        g.set_elements(self.ndelements.flatten())
        griddata.save(g, open(filename, 'w'), format='dx')

def coords_to_grid_np(coords, grid, nx, ny, nz, xmin, ymin, zmin, spacing, rvdw):
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

@numba.jit('f4[:,:,:](f4[:,:], f4[:,:,:], i8, i8, i8, f8, f8, f8, f8, f8)', nopython=True, parallel=True)
def coords_to_grid_numba(coords, grid, nx, ny, nz, xmin, ymin, zmin, spacing, rvdw):
    exps = 0.1
    rmax = 30
    expt = np.exp(-(rvdw/np.arange(0,rmax,exps))**12)
    nc = len(coords)
    for i in numba.prange(nx):
        ix = xmin + i*xmin
        for j in numba.prange(ny):
            iy = ymin + i*ymin
            for k in numba.prange(nz):
                iz = zmin + i*zmin
                for l in numba.prange(nc):
                    dx = ix - coords[l,0]
                    dy = iy - coords[l,1]
                    dz = iz - coords[l,2]
                    r = sqrt(dx*dx + dy*dy + dz*dz)
                    #grid[i,j,k] += 1 - exp(-(rvdw/r)**12)
                    if r > rmax: continue
                    grid[i,j,k] += 1 - expt[int(r/exps)]
    return grid

class PdbBindDataset(Dataset):
    def __init__(self, csvfile, rootdir, transform=None, filter_kd=False):
        self.df = pd.read_csv(csvfile)
        self.rootdir = rootdir
        self.transform = transform
        if filter_kd:
            self.df = self.df[self.df.afftype == 'Kd']
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pdbfile = '{}/{}/{}_protein.pdb'.format(self.rootdir, row.code, row.code)
        pocketfile = '{}/{}/{}_pocket.pdb'.format(self.rootdir, row.code, row.code)
        ligandfile = '{}/{}/{}_ligand.mol2'.format(self.rootdir, row.code, row.code)
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

class Channel:
    """Convert atomic coordinates into grid (channel)
    
    Args:
        atomtypes: list of atom types to convert into grid
        size: size of grid in angstrom
        spacing: grid spacing in angstrom
        rvdw: r_vdw parameter in grid
    """
    def __init__(self, atomtypes, size, spacing, rvdw):
        self.atomtypes = atomtypes
        self.size = size
        self.spacing = spacing
        self.rvdw = rvdw
    
    def __call__(self, sample):
        size = float(self.size)
        spacing = float(self.spacing)
        rvdw = float(self.rvdw)
        nx, ny, nz = [int(size/spacing)+1 for _ in range(3)]
        xmin, ymin, zmin = [_-size/2 for _ in sample['pocket'].center]
        idx = [_ in self.atomtypes for _ in sample['pocket'].atomtypes]
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        grid = coords_to_grid_numba(sample['pocket'].coords[idx], grid, 
                                    nx, ny, nz, xmin, ymin, zmin, spacing, rvdw)
        sample['channels'].append(grid)
        
        idx = [_ in self.atomtypes for _ in sample['ligand'].atomtypes]
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        grid = coords_to_grid_numba(sample['ligand'].coords[idx], grid, 
                                    nx, ny, nz, xmin, ymin, zmin, self.spacing, self.rvdw)
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
        com = sample['pocket'].center
        sample['pocket'].coords = sample['pocket'].coords - com
        sample['ligand'].coords = sample['ligand'].coords - com
        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grids = np.vstack([c[np.newaxis,:] for c in sample['channels']])
        return {
            'grids': torch.from_numpy(grids),
            'affinity': torch.from_numpy(np.array([sample['affinity']]))
        }