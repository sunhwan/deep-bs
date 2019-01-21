"""OpenDX formatter"""

from __future__ import print_function
import numpy as np

from .grid import Grid

_DX_HEADER_TMPL = """object 1 class gridpositions counts {shape[0]} {shape[1]} {shape[2]}
origin {origin[0]:12.5e} {origin[1]:12.5e} {origin[2]:12.5e}
delta {spacing[0]:12.5e} 0 0
delta 0 {spacing[1]:12.5e} 0
delta 0 0 {spacing[2]:12.5e}
object 2 class gridconnections counts {shape[0]} {shape[1]} {shape[2]}
object 3 class array type double rank 0 items {n_elements} data follows
"""

_DX_FOOTER_TMPL = """attribute "dep" string "positions"
object "regular positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3
"""

class OpenDX(object):
    """OpenDX formatter"""

    def load(self, file):
        """Load grid format file

        Args:
            file (:obj:`file`): File object.
        """

        header = []
        for _ in range(7):
            line = file.readline()
            header.append(line.strip())

        n_points = tuple(int(_) for _ in header[0].split()[-3:])
        origin = tuple(float(_) for _ in header[1].split()[1:])
        spacing = (float(header[2].split()[1]), float(header[3].split()[2]), float(header[4].split()[3]))
        paramfile = header[0]
        datafile = header[1]
        molecule = header[2]
        shape = [n_points[i] for i in range(3)]
        n_elements = shape[0] * shape[1] * shape[2]

        self.spacing = spacing
        self.npts = n_points
        self.origin = origin

        elements = []
        for _ in range(n_elements):
            line = file.readline()
            if line.startswith(b'attribute'):
                break
            for e in line.strip().split():
                elements.append(float(e))

        grid = Grid()
        grid.n_elements = n_elements
        grid.origin = origin
        grid.shape = shape
        grid.spacing = spacing
        grid.elements = elements

        return grid

    @staticmethod
    def save(grid, file):
        """Writes to a file.

        Args:
            grid (:obj:`Grid`): Grid object.
            file (:obj:`file`): File object.
        """

        file.write(_DX_HEADER_TMPL.format( \
            shape=grid.shape, \
            origin=grid.origin, \
            spacing=grid.spacing, \
            n_elements=grid.n_elements \
        ))
        for i in range(grid.shape[0]):
            col = 0
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    idx = k*grid.shape[0]*grid.shape[1] + j*grid.shape[0] + i
                    file.write(" %12.5E" % grid.elements[idx])
                    col += 1
                    if col == 3:
                        file.write("\n")
                        col = 0
        if col != 0:
            file.write("\n")
        file.write(_DX_FOOTER_TMPL)
        file.close()
