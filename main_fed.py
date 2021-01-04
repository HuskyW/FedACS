#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
import openpyxl
import torchvision

from utils.sampling import mnist_iid, cifar_iid, complex_skewness_mnist, complex_skewness_cifar, nclass_skewness_cifar, nclass_skewness_mnist
from utils.options import args_parser
from models.Update import LocalUpdate, SingleBgdUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvgWithCmfl, FedAvgWithL2
from models.test import test_img

from utils.evaluate import l2NormEvaluate, FA_round
from models.Bound import estimateBounds
from models.Bandit import UcbqrBandit, Rexp3Bandit, MoveAvgBandit
from models.dla_simple import SimpleDLA


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # seed
    np.random.seed(args.seed)

    # restore args
    if args.dataset == 'mnist':
        args.num_channels = 1
        numsamples = int(60000/args.num_users)
    if args.dataset == 'cifar':
        args.num_channels = 3
        numsamples = int(50000/args.num_users)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid == 0:
            dict_users,dominance = mnist_iid(dataset_train, args.num_users, numsamples)
        elif args.iid == 1:
            dict_users, dominance = nclass_skewness_mnist(dataset_train, args.num_users, numsamples)
        elif args.iid == 2:
            dict_users, dominance = complex_skewness_mnist(dataset_train, args.num_users, numsamples)
        else:
            exit("Bad argument: iid")
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid == 0:
            dict_users, dominance = cifar_iid(dataset_train, args.num_users, numsamples)
        elif args.iid == 1:
            dict_users, dominance = nclass_skewness_cifar(dataset_train, args.num_users, numsamples)
        elif args.iid == 2:
            dict_users, dominance = complex_skewness_cifar(dataset_train,args.num_users, numsamples)
        else:
            exit("Bad argument: iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'resnet50' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet50(pretrained=False)
    elif args.model == 'vgg16' and args.dataset == 'cifar':
        net_glob = torchvision.models.vgg16(pretrained=False)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet18(pretrained=False)
    elif args.model == 'sdla' and args.dataset == 'cifar':
        net_glob = SimpleDLA().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # client selection scheme
    if args.client_sel == 0:
        pass
    elif args.client_sel == 1:
        bandit = UcbqrBandit(args.num_users)
    elif args.client_sel == 2:
        bandit = MoveAvgBandit(args.num_users)
    else:
        print("Bad Argument: client_sel")
        exit(-1)

    # copy weights
    w_glob = net_glob.state_dict()
    w_old = copy.deepcopy(w_glob)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    locallr = args.lr

    if args.testing > 0:
        testacc = []
    
    domilog = []
    rewardlog = []

    l2eval = np.ones((args.epochs-1,args.num_users))
    l2eval = l2eval*-1

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(1,args.epochs+1):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        eva_locals = []


        # client selection
        if args.client_sel != 0:
            idxs_users = bandit.requireArms(m)
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],locallr=locallr)
            w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))


            if FA_round(args,iter) is True:
                eva = SingleBgdUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                evaw, loss = eva.train(net=copy.deepcopy(net_glob).to(args.device))
                eva_locals.append(copy.deepcopy(w))

        # update global weights
        w_glob = FedAvg(w_locals)

        # Evaluate L2 norm and update rewards
        
        if FA_round(args,iter) is True:
            l2n = l2NormEvaluate(copy.deepcopy(w_old),copy.deepcopy(eva_locals))
            rewards = {}
            for i in range(len(l2n)):
                clientidx = idxs_users[i]
                rewards[clientidx] = l2n[i] * -1 * numsamples

                # write log
                l2eval[iter-2][clientidx] = l2n[i]
            
            if args.client_sel != 0:
                bandit.updateWithRewards(rewards)
            
            if iter > 0 and iter % args.testing == 0 and args.testing > 0:
                avgl2n = sum(l2n)/len(l2n)
                rewardlog.append(-1*numsamples*avgl2n)
        

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.cuda()
        
        # Log domi
        if iter > 0 and iter % args.testing == 0 and args.testing > 0:
            domi = []
            for client in idxs_users:
                domi.append(dominance[client])
            domilog.append(float(sum(domi)/len(domi)))


            # test and log accuracy
            
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:3d}, Testing accuracy: {:.2f}".format(iter, acc_test))
            net_glob.train()

            testacc.append(float(acc_test))

        # Learning rate decay
        locallr = newlr
        
    # =====Rounds terminals=====
    logidx = str(args.log_idx)

    if args.log_idx < 0:
        filepath = "./save/acc.log"
    else:
        filepath = "./save/acc/"+logidx+'.log'
    
    with open(filepath,'w') as fp:
        for i in range(len(testacc)):
            content = str(testacc[i]) + '\n'
            fp.write(content)
        print('Acc log written')

    if args.log_idx < 0:
        filepath = "./save/reward.log"
    else:
        filepath = "./save/reward/"+logidx+'.log'

    with open(filepath,'w') as fp:
        for i in range(len(rewardlog)):
            content = str(rewardlog[i]) + '\n'
            fp.write(content)
        print('Reward log written')

    if args.log_idx < 0:
        filepath = "./save/domi.log"
    else:
        filepath = "./save/domi/"+logidx+'.log'

    with open(filepath,'w') as fp:
        for i in range(len(domilog)):
            content = str(domilog[i]) + '\n'
            fp.write(content)
        print('dominance log written')

    '''
    dominance = np.expand_dims(dominance,0)
    l2eval = np.concatenate((dominance,l2eval),axis=0)
    writer = pd.ExcelWriter('l2eval.xlsx')
    l2pandas = pd.DataFrame(l2eval)
    l2pandas.to_excel(writer, 'sheet_1', float_format='%f')
    writer.save()
    writer.close()
    print('L2 record written')
    '''
    '''
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    '''
