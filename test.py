import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np

opt = TestOptions().parse()
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many: break
    model.set_input(data)
    model.test()
    preds = model.preds.cpu().detach().numpy()
    print(np.hstack([preds, data['affinity']]))

