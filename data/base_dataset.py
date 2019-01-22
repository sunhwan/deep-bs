import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from math import exp, sqrt, cos, sin
import numpy as np
import numba

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
        transform_list += [Channel(('C'), opt.grid_size, opt.grid_spacing, opt.rvdw),
                           Channel(('N'), opt.grid_size, opt.grid_spacing, opt.rvdw),
                           Channel(('O'), opt.grid_size, opt.grid_spacing, opt.rvdw)]

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
            'affinity': torch.from_numpy(np.array([sample['affinity']], dtype=np.float32))
        }
