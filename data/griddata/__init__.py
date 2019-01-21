"""This module provides a package that can read/write grid data format"""

from .autodock import AutoDockMap
from .opendx import OpenDX
from .ccp4 import CCP4

def load(file, format):
    """Load grid-format data

    Args:
        file (:obj:`file`): File object to the file containing grid-format data.
        format (str): Grid-format data format.
    """

    reader = None

    if format == 'map':
        reader = AutoDockMap()

    if format == 'dx':
        reader = OpenDX()

    if format == 'ccp4':
        reader = CCP4()

    if reader is None:
        raise NotSupportedError

    grid = reader.load(file)
    return grid

def save(grid, file, format):
    """Writes grid data to a file

    Args:
        grid (:obj:`Grid`): Grid object.
        file (:obj:`file`): File object.
        format (str): Grid-format data format.
    """

    if format == 'map':
        AutoDockMap.save(grid, file)

    if format == 'dx':
        OpenDX.save(grid, file)
