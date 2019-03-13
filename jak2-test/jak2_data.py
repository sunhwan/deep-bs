import pandas as pd
import numpy as np

dg = np.load('/home/han/jak2_docking/bfp2_lesstype/dGs.npy')
aff = np.exp(dg/0.6)
n = len(dg)
arr = []
for i in range(n):
    fn = 'jak2_docked/output/mol_file_{}_out.pdbqt'.format(i)
    arr.append((fn, aff[i]))

pd.DataFrame(arr, columns=('pose', 'affinity')).to_csv('test.csv', index=0)

