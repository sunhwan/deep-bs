import torch.utils.data
from .base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'pdbbind':
        from .pdbbind_dataset import PdbBindDataset
        dataset = PdbBindDataset()
    elif opt.dataset_mode == 'pdbbind_docked':
        from .pdbbind_docked_dataset import PdbBindDockedDataset
        dataset = PdbBindDockedDataset()
    elif opt.dataset_mode == 'jak2':
        from .jak2_dataset import Jak2Dataset
        dataset = Jak2Dataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=opt.batch_size,
            shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size: break
            yield data

    def name(self): return 'CustomDatasetDataLoader'
    def load_data(self): return self
    def __len__(self): return min(len(self.dataset), self.opt.max_dataset_size)

