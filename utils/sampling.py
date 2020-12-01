#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_merged_iid(dataset, num_users):
    data_per_node = 300
    data_num = 60000
    class_num = 10
    class_size = int(data_num/class_num)
    data_per_nc = int(data_per_node / class_num)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(data_num)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    iidRatio = 0.7
    iid_users = int(num_users * iidRatio)
    noniid_users = num_users - iid_users

    # iid nodes
    for i in range(iid_users):
        for c in range(class_num):
            dict_users[i] = np.concatenate((dict_users[i], idxs[class_size*c+data_per_nc*i:class_size*c+data_per_nc*(i+1)]), axis=0)

    # non-iid nodes
    node_per_class = int(noniid_users / 3)
    for c in range(3):
        for i in range(node_per_class):
            nodeid = iid_users + c * node_per_class + i
            rand = np.random.randint(class_size - data_per_node)
            datastart = c*class_size + rand
            dict_users[nodeid] = np.concatenate((dict_users[nodeid], idxs[datastart:datastart+data_per_node]), axis=0)
    return dict_users

def spell_partition_mnist(partition, labels):
    for nodeid, dataid in partition.items():
        record = [0] * 10
        for data in dataid:
            label = labels[data]
            record[label] += 1
        
        print("Client %d", nodeid)
        for classid in range(len(record)):
            print("Class %d: %d" %(classid,record[classid]))
        print("\n\n")


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_merged_iid(dataset_train, num)
    spell_partition_mnist(d,dataset_train.train_labels.numpy())
