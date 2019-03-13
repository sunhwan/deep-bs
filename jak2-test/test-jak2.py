import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
import pandas as pd
from tqdm import tqdm

opt = TestOptions().parse()
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

def correlation(Measure, Fit):
    """Calculates the correlation coefficient R^2 between the two sets
       of Y data provided. Logically, in order for the result to have a sense
       you want both Y arrays to have been created from the same X array."""

    Mean = np.mean(Measure)
    s1 = 0
    s2 = 0
    Size = np.size(Measure) # identical to np.size(Fit)

    for i in range(0, Size):
        s1 += (Measure[i] - Fit[i]) ** 2
        s2 += (Measure[i] - Mean) ** 2
    Rsquare = 1 - s1/s2
    return Rsquare

# test
preds = np.zeros(len(dataset))
trues = np.zeros(len(dataset))
with tqdm(total=int(len(dataset)/opt.batch_size)+1) as pbar:
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        offset = i * opt.batch_size
        preds[offset:offset+opt.batch_size] = model.preds.cpu().detach().numpy().flatten()
        trues[offset:offset+opt.batch_size] = data['affinity'].flatten()
        pbar.update()

from sklearn.metrics import r2_score
print("corr coef:", np.corrcoef(preds, trues)[0,1])
print("R2:", r2_score(trues, preds))

pd.DataFrame(np.vstack((trues, preds)).T, columns=('true', 'pred')).to_csv('output.csv', index=False)

