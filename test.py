import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
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
    print(model.preds)
    print(data['affinity'])

