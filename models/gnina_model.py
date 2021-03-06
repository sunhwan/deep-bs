import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import sys
sys.path.append('..')
from util import util
from .base_model import BaseModel
from . import networks


class GninaModel(BaseModel):
    def name(self):
        return 'GninaModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.net = networks.define_gnina_net(input_nc=opt.input_nc, model=opt.model, gpu_ids=opt.gpu_ids, init_type=opt.init_type)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.net, 'GNINA', which_epoch)

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
            grids = grids.cuda(self.gpu_ids[0], non_blocking=True)
            affinities = affinities.cuda(self.gpu_ids[0], non_blocking=True)
        self.input = grids
        self.affinities = affinities
    
    def forward(self):
        self.preds = self.net(self.input)
    
    def backward(self):
        loss = self.criterion(self.preds, self.affinities)
        loss.backward()
        self.loss = loss
    
    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def test(self):
        self.forward()

    def save(self, label):
        self.save_network(self.net, 'GNINA', label, self.gpu_ids)

    def load(self, label):
        self.load_network(self.net, 'GNINA', label)


class GninaPoseModel(BaseModel):
    def name(self):
        return 'GninaPoseModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.net = networks.define_gnina_net(input_nc=opt.input_nc, model=opt.model, gpu_ids=opt.gpu_ids, init_type=opt.init_type)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.net, 'GNINA_POSE', which_epoch)

        if self.isTrain:
            self.affinity_criterion = torch.nn.SmoothL1Loss()
            self.pose_criterion = torch.nn.BCELoss()
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
        #networks.print_network(self.net, opt)
        print('-----------------------------------------------')

    def set_input(self, input):
        grids = input['grids']
        affinities = input['affinity']
        poses = input['pose']
        if len(self.gpu_ids) > 0:
            grids = grids.cuda(self.gpu_ids[0], non_blocking=True)
            affinities = affinities.cuda(self.gpu_ids[0], non_blocking=True)
            poses = poses.cuda(self.gpu_ids[0], non_blocking=True)
        self.input = grids
        self.poses = poses
        self.affinities = affinities

    def forward(self):
        self.embeds = self.net.body(self.input)
        self.preds_pose = self.net.pose_head(self.embeds)[:, 1:]
        self.preds_affinity = self.net.affinity_head(self.embeds)

    def backward(self):
        loss_pose = self.pose_criterion(self.preds_pose, self.poses)
        loss_affinity = self.affinity_criterion(self.preds_affinity, self.affinities)
        loss = loss_pose + loss_affinity
        loss.backward()
        self.loss = loss
    
    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def test(self):
        self.forward()

    def save(self, label):
        self.save_network(self.net, 'GNINA_POSE', label, self.gpu_ids)

    def load(self, label):
        self.load_network(self.net, 'GNINA_POSE', label)
