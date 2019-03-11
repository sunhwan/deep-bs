import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import sys
sys.path.append('..')
from util import util
from .base_model import BaseModel
from . import networks


class KDeepModel(BaseModel):
    def name(self):
        return 'KDeepModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.net = networks.define_kdeep_net(input_nc=16, model=opt.model, gpu_ids=opt.gpu_ids, init_type=opt.init_type)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.net, 'KDNet', which_epoch)

        if self.isTrain:
            self.criterion = torch.nn.MSELoss()
            if opt.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=opt.lr, momentum=opt.momentum)
            elif opt.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                 raise ValueError
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.net, opt)
        print('-----------------------------------------------')

    def set_input(self, input):
        grids = input['grids']
        affinities = input['affinity']
        if len(self.gpu_ids) > 0:
            grids = grids.cuda(self.gpu_ids[0], async=True)
            affinities = affinities.cuda(self.gpu_ids[0], async=True)
        self.input = grids
        self.affinities = affinities
    
    def forward(self):
        self.preds = self.net(self.input)
    
    def backward(self):
        loss = self.criterion(self.preds, self.affinities)
        loss.backward()
        self.loss = loss
    
    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def test(self):
        self.forward()

    def save(self, label):
        self.save_network(self.net, 'KDNet', label, self.gpu_ids)

    def load(self, label):
        self.load_network(self.net, 'KDNet', label)
