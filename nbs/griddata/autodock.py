"""AutoDock Grid Map read and write"""
from __future__ import print_function
import numpy as np

from .grid import Grid

_MAP_HEADER_TMPL = """GRID_PARAMETER_FILE {paramfile}
GRID_DATA_FILE {datafile}
MACROMOLECULE {molecule}
SPACING {spacing:4.3f}
NELEMENTS {npts[0]} {npts[1]} {npts[2]}
CENTER {center[0]:5.3f} {center[1]:5.3f} {center[2]:5.3f}
"""

class AutoDockMap(object):
    """Class for handling AutoDock Map files"""

    def load(self, file):
        """Load grid format file

        Args:
            file (:obj:`file`): File object.
        """

        header = []
        for _ in range(6):
            line = file.readline()
            header.append(line.strip())

        paramfile = header[0]
        datafile = header[1]
        molecule = header[2]
        spacing = float(header[3].split()[1])
        n_points = [int(_) for _ in header[4].split()[1:]]
        center = [float(_) for _ in header[5].split()[1:]]
        shape = [n_points[i]+1 for i in range(3)]
        n_elements = shape[0] * shape[1] * shape[2]

        self.paramfile = ''
        self.molecule = ''
        self.datafile = ''
        self.spacing = spacing
        self.npts = n_points
        self.center = center

        elements = []
        for _ in range(n_elements):
            elements.append(float(file.readline()))

        grid = Grid()
        grid.n_elements = n_elements
        grid.center = center
        grid.shape = shape
        grid.spacing = (spacing, spacing, spacing)

        # autodock grid map is ordered in Fortran
        # we are keeping the internal as C order.
        grid.set_elements(elements, order='F')

        return grid

    def meta(self):
        return _MAP_HEADER_TMPL.format(**self.__dict__)

    def save(self, file):
        """Writes to a file.

        Args:
            grid (:obj:`Grid`): Grid object.
            file (:obj:`file`): File object.
        """

        if hasattr(self, 'meta'):
            file.write(self.meta())
        else:
            meta = {
                'paramfile': '',
                'datafile': '',
                'molecule': '',
                'spacing': self.spacing[0],
                'npts': (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1),
                'center': self.center
            }
            file.write(_MAP_HEADER_TMPL.format(**meta))

        # bring back to Fortran order
        for value in self.get_elements(order='F'):
            file.write("%.3f\n" % value)
