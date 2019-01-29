import argparse
import glob
import os

import luigi
from data.pdbbind_dataset import GridPDB

class Babel(luigi.Task):
    fmt_in = luigi.Parameter()
    fmt_out = luigi.Parameter()
    file_in = luigi.Parameter()
    file_out = luigi.Parameter(default='')

    def run(self):
        os.system(f'babel -i{self.fmt_in} {self.file_in} -o{self.fmt_out} {self.output().path}')

    def output(self):
        if self.file_out == '':
            file_out = '.'.join(self.file_in.split('.')[:-1]) + '.pdbqt'
        else:
            file_out = self.file_out
        return luigi.LocalTarget(file_out)


class ParsePDB(luigi.Task):
    pdbfile = luigi.Parameter()
    code = luigi.Parameter()

    def requires(self):
        return Babel(fmt_in='pdb', file_in=self.pdbfile, fmt_out='pdbqt')

    def run(self):
        pdb = GridPDB(self.input().path)
        pdb.to_h5(self.output().path)

    def output(self):
        return luigi.LocalTarget('.'.join(self.pdbfile.split('.')[:-1]) + '.h5')


class ParseMol2(luigi.Task):
    ligfile = luigi.Parameter()
    code = luigi.Parameter()

    def requires(self):
        return Babel(fmt_in='mol2', file_in=self.ligfile, fmt_out='pdbqt')

    def run(self):
        lig = GridPDB(self.input().path)
        lig.to_h5(self.output().path)

    def output(self):
        return luigi.LocalTarget('.'.join(self.ligfile.split('.')[:-1]) + '.h5')


class ExtractCoordinates(luigi.WrapperTask):
    dataroot = luigi.Parameter()
    def requires(self):
        for pdbdir in glob.glob(f'{self.dataroot}/*'):
            code = os.path.basename(pdbdir)
            pdbfile = os.path.join(pdbdir, f'{code}_pocket.pdb')
            ligfile = os.path.join(pdbdir, f'{code}_ligand.mol2')
            if os.path.exists(pdbfile):
                yield ParsePDB(pdbfile=pdbfile, code=code)
                yield ParseMol2(ligfile=ligfile, code=code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, default=1, help='number of workers')
    parser.add_argument('--dataroot', required=True, default='/home/sunhwan/work/pdbbind/2018/refined-set', help='data directory')
    args = parser.parse_args()

    luigi.build([ExtractCoordinates(dataroot=args.dataroot)], workers=args.worker, local_scheduler=True, log_level='INFO')
