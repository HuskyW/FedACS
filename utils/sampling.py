#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms
import math

def mnist_iid(dataset, num_users=200,num_samples=300):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    data_num = 60000
    class_num = 10
    idxs = np.arange(data_num)
    labels = dataset.train_labels.numpy()
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,dominance=0)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances


def cifar_iid(dataset, num_users=200, num_samples=250):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    data_num = 50000
    class_num = 10
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,dominance=0)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances

def add_data(data,idxs,head,overall,group,num,counts):
    remaining = overall[group+1] - head[group]
    counts[group] += num
    if remaining > num:
        data = np.concatenate((data,idxs[head[group] : head[group]+num]))
        head[group] += num

        return data
    
    data = np.concatenate((data,idxs[head[group] : head[group]+remaining]))
    head[group] = overall[group]

    filling = num - remaining
    data = np.concatenate((data,idxs[head[group] : head[group]+filling]))
    head[group] += filling
    return data

def dominance_client(heads,overalldist,idxs,counts,dominance=None,dClass=None,sampleNum=300,classNum=10):
    if dominance is None:
        dominance = random.uniform(0,1.0)
    if dClass is None:
        sortcounts = sorted(counts.items(),key=lambda x:x[1],reverse=False)
        dClass = sortcounts[0][0]

    dominance = float(dominance)
    
    iidClassSize = math.floor(sampleNum *(1 - dominance) / classNum)
    nonClassSize = sampleNum - classNum * iidClassSize
    result = np.array([], dtype='int64')
    result = add_data(result,idxs,heads,overalldist,dClass,nonClassSize,counts)

    for i in range(classNum):
        result = add_data(result,idxs,heads,overalldist,i,iidClassSize,counts)
    
    return result, dominance

def complex_skewness_mnist(dataset, num_users=100, num_samples=300):
    data_num = 60000
    class_num = 10
    idxs = np.arange(data_num)
    labels = dataset.train_labels.numpy()
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances

def complex_skewness_cifar(dataset, num_users=200, num_samples=250, class_num=10):
    data_num = 50000
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances

def nclass_client(heads,overalldist,idxs,counts,n=None,sampleNum=300,classNum=10):
    if n is None:
        draw = math.pow(random.uniform(0,1.0),4)
        n = math.ceil(classNum*draw)

    sortcounts = sorted(counts.items(),key=lambda x:x[1],reverse=False)

    selectedClass = []
    for i in range(n):
        selectedClass.append(sortcounts[i][0])

    dataPclass = math.floor(sampleNum/n)
    remaining = sampleNum - dataPclass * n
    remainingclass = np.random.choice(selectedClass,1)[0]

    result = np.array([], dtype='int64')
    for c in selectedClass:
        if c == remainingclass:
            result = add_data(result,idxs,heads,overalldist,c,remaining+dataPclass,counts)
        else:
            result = add_data(result,idxs,heads,overalldist,c,dataPclass,counts)
    
    return result, n

def nclass_skewness_cifar(dataset, num_users=200, num_samples=250, class_num=10):
    data_num = 50000
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = nclass_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances

def nclass_skewness_mnist(dataset, num_users=200, num_samples=300, class_num=10):
    data_num = 60000
    class_num = 10
    idxs = np.arange(data_num)
    labels = dataset.train_labels.numpy()
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = nclass_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)
    
    return dict_users, dominances



def spell_partition(partition, labels,dominence=None):

    for nodeid, dataid in partition.items():
        record = [0] * 10
        for data in dataid:
            label = labels[data]
            record[label] += 1
        
        print("Client %d"% nodeid)
        if dominence is not None:
            print("Dominence: %.2f"% dominence[nodeid])

        for classid in range(len(record)):
            print("Class %d: %d" %(classid,record[classid]))
        print("\n\n")


if __name__ == '__main__':
    
    dataset_train = datasets.MNIST('../../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 200
    d,domi = nclass_skewness_mnist(dataset_train, num)
    spell_partition(d,dataset_train.train_labels.numpy(),domi)
    
    '''
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../../data/cifar', train=True, download=True, transform=trans_cifar)
    num = 200
    d,domi = nclass_skewness_cifar(dataset_train, num)
    spell_partition(d,np.array(dataset_train.targets),domi)
    '''
