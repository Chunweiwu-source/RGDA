from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import data_loader
from model import RGDA
from torch.utils import model_zoo
import numpy as np
from IPython import embed
import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import feature_vis


import sys

# Training settings
parser = argparse.ArgumentParser(description='PyTorch RGDA')
parser.add_argument('--txt_datasets', type=int, default=0, metavar='txt_datasets',
                    help='use text datasets')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='log',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2333, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--root_path', type=str, default="...//data/image_CLEF/",
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="p",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="i",
                    help='the name of the test dir')
parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--num_class', default=12, type=int,
                    help='the number of classes')
parser.add_argument('--groups', default=32, type=int,
                    help='the number of groups')
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

def load_data():
    source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, args.txt_datasets, kwargs)
    target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, args.txt_datasets, kwargs)
    target_test_loader  = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, args.txt_datasets, kwargs)
    source_test_loader  = data_loader.load_testing(args.root_path, args.source_dir, args.batch_size, args.txt_datasets, kwargs)
    return source_train_loader, target_train_loader, target_test_loader, source_test_loader

def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k == 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)


def train(epoch, model, source_loader, target_loader):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)
    if args.diff_lr:
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters()},
            {'params': model.localattens.parameters()},
            {'params': model.bottleneck.parameters()},
            {'params': model.domain_classifier.parameters()},
            {'params': model.fc1.parameters()},
            {'params': model.fc2.parameters()},
            {'params': model.source_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)

    print_learning_rate(optimizer)


    model.train()
    len_dataloader = len(source_loader)
    DEV = DEVICE

    len_train_target = len(target_loader) - 1

    for batch_idx, (source_data, source_label) in tqdm.tqdm(enumerate(source_loader),
                                    file=sys.stdout,
                                    total=len_dataloader,
                                    mininterval=10,
                                    desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
        p = float(batch_idx+1 + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        optimizer.zero_grad()
        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        if batch_idx % len_train_target == 0:
            iter_target = iter(target_loader)

        target_data, target_label = iter_target.next()
        target_data, target_label = Variable(target_data).to(DEVICE), Variable(target_label).to(DEVICE)

        out = model(source_data, target_data, DEV, alpha)
        s_output, t_output = out[0], out[1]
        s_domain_output, t_domain_output = out[2], out[3]
        s_g_mi, t_g_mi = out[4], out[5]
        feature = out[6]


        sdomain_label = torch.zeros(args.batch_size).long().to(DEV)
        tdomain_label = torch.ones(args.batch_size).long().to(DEV)

        s_domain = F.softmax(s_output, dim=1)
        t_domain = F.softmax(t_output, dim=1)
        s_Calibrate_loss = - torch.sum(torch.sum(s_domain * torch.log(s_domain), dim=1, keepdim=True) * s_g_mi) / args.batch_size
        t_Calibrate_loss = - torch.sum(torch.sum(t_domain * torch.log(t_domain), dim=1, keepdim=True) * t_g_mi) / args.batch_size
        Calibrate_loss = s_Calibrate_loss + t_Calibrate_loss

        err_s_domain = F.nll_loss(F.log_softmax(s_domain_output, dim=1), sdomain_label)
        err_t_domain = F.nll_loss(F.log_softmax(t_domain_output, dim=1), tdomain_label)

        global_loss = err_s_domain + err_t_domain

        soft_loss = F.nll_loss(F.log_softmax(s_output, dim=1), source_label) + 0.1 * Calibrate_loss
       
        loss = soft_loss + global_loss
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('\nLoss: {:.6f},  label_Loss: {:.6f}, global_Loss:{:.4f}, Calibrate_loss:{:.4f}'.format(
                loss.item(), soft_loss.item(), global_loss.item(), Calibrate_loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data, data, DEVICE)
            s_output = out[0]
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, reduction='sum').item()
            pred = s_output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


def reduce_dimensions(all_preds, num_dimensions):
    num_dimensions = num_dimensions

    tsne = TSNE(n_components=num_dimensions, init='pca', random_state=0)
    all_preds = tsne.fit_transform(all_preds)

    return all_preds


def t_sne(model, test_loader):
    model.eval()
    all_feature = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data, data, DEVICE)
            output = out[6]
            
            all_feature.extend(output.cpu().data.numpy())
            
    return all_feature


if __name__ == '__main__':

    model = RGDA.RGDA(num_classes=args.num_class, base_net='ResNet50', groups=args.groups).to(DEVICE)
    train_loader, unsuptrain_loader, test_loader, source_test_loader = load_data()
    correct = 0
    num_dimensions = 2

    sys.stdout = open('...//RGDA/log/'+ args.source_dir + '-->' + args.test_dir +'.log', 'a')
    
    for epoch in range(1, args.epochs + 1):
        train_loader, unsuptrain_loader, test_loader, source_test_loader = load_data()
        train(epoch, model, train_loader, unsuptrain_loader)
        t_correct = test(model, test_loader)
        if t_correct > correct:
            correct = t_correct

            

        print("%s max correct:" % args.test_dir, correct)
        print(args.source_dir, "to", args.test_dir)

    torch.save(model, '...//RGDA/'+ args.source_dir + '-->' + args.test_dir +'modelpara.pth')


    


