import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys, os
import torch
import re
import string
import torch
import torch.nn.functional as F
from path_zh import *
import csv
from config import *
import random
from random import shuffle
# import os


def split_data(size,y, train, val, test, shuffle=True):
    idx = list(range(size))
    # label_dict = {}
    idx_pos = []
    idx_neg = []
    for idx_temp in idx:
        # label_dict[idx_temp] = y[idx_temp].item()
        if y[idx_temp].item() == 1:
            idx_neg.append(idx_temp)
        elif y[idx_temp].item() == 0:
            idx_pos.append(idx_temp)
    print('数据正负例分布：','pos:',len(idx_pos),'neg',len(idx_neg))
    if shuffle:
        np.random.shuffle(idx_neg)
        np.random.shuffle(idx_pos)
    split_idx_pos = np.split(idx_pos, [int(train * len(idx_pos)), int((train + val) * len(idx_pos))])
    train_idx_pos, val_idx_pos, test_idx_pos = split_idx_pos[0], split_idx_pos[1], split_idx_pos[2]

    split_idx_neg = np.split(idx_neg, [int(train * len(idx_neg)), int((train + val) * len(idx_neg))])
    train_idx_neg, val_idx_neg, test_idx_neg = split_idx_neg[0], split_idx_neg[1], split_idx_neg[2]

    train_idx = np.concatenate((train_idx_pos, train_idx_neg),axis=0)
    val_idx = np.concatenate((val_idx_pos,val_idx_neg),axis=0)
    test_idx = np.concatenate((test_idx_pos,test_idx_neg),axis=0)
    # print('val idx:',val_idx)
    print('train数据正负例分布：', 'pos:', len(train_idx_pos), 'neg', len(train_idx_neg))
    print('val数据正负例分布：', 'pos:', len(val_idx_pos), 'neg', len(val_idx_neg))
    print('test数据正负例分布：', 'pos:', len(test_idx_pos), 'neg', len(test_idx_neg))
    return train_idx, val_idx, test_idx

def split_data_early(size,y, train, val, shuffle=True):
    idx = list(range(size))
    # label_dict = {}
    idx_pos = []
    idx_neg = []
    for idx_temp in idx:
        # label_dict[idx_temp] = y[idx_temp].item()
        if y[idx_temp].item() == 1:
            idx_neg.append(idx_temp)
        elif y[idx_temp].item() == 0:
            idx_pos.append(idx_temp)
    # print('数据正负例分布：','pos:',len(idx_pos),'neg',len(idx_neg))
    if shuffle:
        np.random.shuffle(idx_neg)
        np.random.shuffle(idx_pos)
    split_idx_pos = np.split(idx_pos, [int(train * len(idx_pos))])
    train_idx_pos, val_idx_pos = split_idx_pos[0], split_idx_pos[1]

    split_idx_neg = np.split(idx_neg, [int(train * len(idx_neg))])
    train_idx_neg, val_idx_neg = split_idx_neg[0], split_idx_neg[1]

    train_idx = np.concatenate((train_idx_pos, train_idx_neg),axis=0)
    val_idx = np.concatenate((val_idx_pos,val_idx_neg),axis=0)
    # print('val idx:',val_idx)
    # print('train数据正负例分布：', 'pos:', len(train_idx_pos), 'neg', len(train_idx_neg))
    # print('val数据正负例分布：', 'pos:', len(val_idx_pos), 'neg', len(val_idx_neg))
    # print('test数据正负例分布：', 'pos:', len(test_idx_pos), 'neg', len(test_idx_neg))
    return train_idx, val_idx

def split_data_5fold(size, y, train, val, test, shuffle=True):
    """Return five stratified 70/10/20 folds with complete test coverage."""
    if size != len(y):
        raise ValueError('size and label count differ')
    if not np.isclose(train + val + test, 1.0):
        raise ValueError('train, val and test must sum to one')
    if not np.isclose(test, 0.2):
        raise ValueError('five-fold cross validation requires test=0.2')
    by_label = {0: [], 1: []}
    for sample_idx in range(size):
        label = int(y[sample_idx].item())
        if label not in by_label:
            raise ValueError('DDGCN expects binary labels 0 and 1')
        by_label[label].append(sample_idx)
    print('数据正负例分布：', 'pos:', len(by_label[0]), 'neg', len(by_label[1]))
    if shuffle:
        for indices in by_label.values():
            random.shuffle(indices)
    test_folds = {
        label: [part.tolist() for part in np.array_split(indices, 5)]
        for label, indices in by_label.items()
    }
    folds = []
    all_test = []
    for fold_idx in range(5):
        fold_test, fold_val, fold_train = [], [], []
        for label, indices in by_label.items():
            label_test = test_folds[label][fold_idx]
            test_set = set(label_test)
            remaining = [index for index in indices if index not in test_set]
            val_count = int(round(val * len(indices)))
            fold_test.extend(label_test)
            fold_val.extend(remaining[:val_count])
            fold_train.extend(remaining[val_count:])
        random.shuffle(fold_test)
        random.shuffle(fold_val)
        random.shuffle(fold_train)
        test_set, val_set, train_set = map(set, (fold_test, fold_val, fold_train))
        assert not test_set & val_set
        assert not test_set & train_set
        assert not val_set & train_set
        assert test_set | val_set | train_set == set(range(size))
        all_test.extend(fold_test)
        folds.extend((fold_test, fold_val, fold_train))
    assert sorted(all_test) == list(range(size))
    return tuple(folds)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print('---------adj----------',type(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)))
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # return adj.tocoo()


def normalize(mx):
    """Row-normalize sparse matrix  (normalize feature)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # print('sparse_mx.row:',type(sparse_mx.row))
    # print('sparse_mx.col:',type(sparse_mx.col))
    # print('indices:',type(indices))
    # print('indices:',indices.size()) #tensor
    # print('values:',type(values)) #tensor
    # print('shape:',type(shape))
    # print('shape:',shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    # return indices,values
    # print('sparse_mx.row')
    # return torch.LongTensor(sparse_mx)

def sparse_mx_to_torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
        print('data bug')
        print('sparse_mx.data',sparse_mx.data)
        print('sparse_mx.shape',sparse_mx.shape)
    # print('--------row col-------:',type(sparse_mx.row),type(sparse_mx.col)) dp.ndarray
    if np.NAN in sparse_mx.data:
        print('有NaN数据')
    # with open('test_matraix_data.txt','a',encoding='utf-8')as f:
    #     v_list = []
    #     for v in sparse_mx.data:
    #         v_list.append(str(v))
    #     f.writelines(v_list)
    # assert sparse_mx.data.sum() == np.float32(len(sparse_mx.row))
    # print('data sparse_mx.data.sum',sparse_mx.data.sum(),type(sparse_mx.data.sum()))
    # print('data len(sparse_mx.row)',len(sparse_mx.row),type(len(sparse_mx.row)))
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices,values,shape

def sparse_mx_to_list(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
    indices = [sparse_mx.row.tolist(), sparse_mx.col.tolist()]
    values = sparse_mx.data.tolist()
    # shape = torch.Size(sparse_mx.shape)
    return indices,values

def sparse_mx_to_geometric(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
    indices = [list(sparse_mx.row),list(sparse_mx.col)]
    values = list(sparse_mx.data)
    # shape = torch.Size(sparse_mx.shape)
    return indices,values

def to_dense_matrix(sparse_mx):
    # shape_0,shape_1 = sparse_mx.shape
    dense_mx = np.zeros((sparse_mx.shape[0],sparse_mx.shape[1]),dtype=np.float32)
    for i in range(sparse_mx.shape[0]):
        dense_mx[sparse_mx.row[i]][sparse_mx.col[i]] = sparse_mx.data[i]
    return dense_mx


def sparse_mx_to_dense_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
    # print(type(sparse_mx.row),type(sparse_mx.col),type(sparse_mx.data))
    # print(sparse_mx.shape)
    dense_mx = to_dense_matrix(sparse_mx)
    # print(dense_mx)
    # print('-------sparse_mx------:',type(sparse_mx))
    # dense_mx = sparse_mx.to_dense()
    dense_mx = torch.from_numpy(dense_mx).long()
    return dense_mx
    # return indices,values
    # print('sparse_mx.row')
    # return torch.LongTensor(sparse_mx)

def dense_tensor_to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def check_exist(outf):
    return os.path.isfile(outf)


def load_w2v_emb(file):
    print('load_w2v_emb', file)
    with open(file, 'rb') as f:
        emb = np.load(f)
    return emb  # np.narray type

def text_length_pad(text_lst, max_length,config):
    if len(text_lst) > max_length:
        text_lst = text_lst[:max_length]
    elif len(text_lst) < max_length:
        pad_index = config.token_size
        for i in range(max_length-len(text_lst)):
            text_lst.append(pad_index)
    assert len(text_lst) == max_length
    return text_lst

def load_sparse_temporal_data(text_max_length, train, val, test,config):
    names = ['propagation_node_idx.npy', 'propagation_node.npy', 'knowledge_node_idx.npy', 'knowledge_node.npy','label.npy']
    objects = []
    for i in range(len(names)):
        with open(path_temporal+"/{}".format(names[i]), 'rb') as f:
            objects.append(np.load(f,encoding='latin1',allow_pickle=True))
    p_idx, p_node, k_idx, k_node, y= tuple(objects)
    ## train
    for xx in p_node:
        for i in range(len(xx)):
            xx[i] = sparse_mx_to_torch_sparse_tensor(normalize_adj(xx[i]))

    y = torch.from_numpy(y).float()
    y = torch.unsqueeze(y, 1)
    # ==================================================================================
    emb_mapped_dict = {}  # entity\concept\token的映射
    mid2token_dict = {}  # mid2token
    node2idx_dict = {}  # idx-->node
    node2emb_dict = {}
    with open(path_node2idx_mid, 'r', encoding='utf-8') as f:
        node2idx_mid = f.readlines()
    for line in node2idx_mid:
        node = line.strip('\n').split('\t')[0]
        idx = line.strip('\n').split('\t')[1]
        node2idx_dict[idx] = node
    r = csv.reader(open(path_mid2token, 'r', encoding='utf-8'))
    for line in r:
        mid = line[0].strip('\t')
        token_list = [token.strip(' ') for token in line[1].strip('\t').strip('[').strip(']').split(',')]
        mid2token_dict[mid] = [token_list]
    r = csv.reader(open(path_entity_concept_2id, 'r', encoding='utf-8'))
    for line in r:
        idx = line[0]
        ent = line[1]
        emb_mapped_dict[ent] = idx

    emb_idx_all_p = []
    for i in range(len(p_idx)):
        emb_idx = []
        for idx in p_idx[i]:
            node = node2idx_dict[idx]
            if node in emb_mapped_dict:
                new_idx = int(emb_mapped_dict[node])
            elif node in mid2token_dict:
                new_idx = mid2token_dict[node]
                new_idx = text_length_pad(new_idx, text_max_length,config)
                new_idx_array = np.array(new_idx)
                new_idx = [_idx.astype(int) for _idx in new_idx_array[:]]
            emb_idx.append(new_idx)
        emb_idx = np.array(emb_idx)
        emb_idx_all_p.append(emb_idx)
    # emb_idx_all_p = [_idx.astype(int) for _idx in emb_idx_all_p[:]]
    emb_idx_p = [torch.from_numpy(_idx).long() for _idx in emb_idx_all_p[:]]

    emb_idx_all_k = []
    for i in range(len(k_idx)):
        emb_idx = []
        for idx in k_idx[i]:
            node = node2idx_dict[idx]
            if node in emb_mapped_dict:
                new_idx = emb_mapped_dict[node]
            elif node in mid2token_dict:
                new_idx = mid2token_dict[node]
                new_idx = text_length_pad(new_idx, text_max_length,config)
            emb_idx.append(new_idx)
        emb_idx = np.array(emb_idx)
        emb_idx_all_k.append(emb_idx)
    emb_idx_all_k = [_idx.astype(int) for _idx in emb_idx_all_k[:]]
    emb_idx_k = [torch.from_numpy(_idx).long() for _idx in emb_idx_all_k[:]]
    # ==============================================================================

    p_idx = [_idx.astype(int) for _idx in p_idx[:]]
    p_idx = [torch.from_numpy(_idx).long() for _idx in p_idx[:]]

    for xx in k_node:
        for i in range(len(xx)):
            xx[i] = sparse_mx_to_torch_sparse_tensor(normalize_adj(xx[i]))
    k_idx = [_idx.astype(int) for _idx in k_idx[:]]
    k_idx = [torch.from_numpy(_idx).long() for _idx in k_idx[:]]


    train_idx, val_idx, test_idx  = split_data(len(p_idx), train, val, test, shuffle=False)

    train_dict, val_dict, test_dict = {}, {}, {}

    names_dict = {'x_p':p_node,'x_k':k_node, 'y':y, 'idx_p':p_idx,'idx_k':k_idx,
                  'idx_p_emb':emb_idx_p,'idx_k_emb':emb_idx_k}
    for name in names_dict:
        train_dict[name] = [names_dict[name][i] for i in train_idx]
        val_dict[name] = [names_dict[name][i] for i in val_idx]
        test_dict[name] = [names_dict[name][i] for i in test_idx]

    #-----------------------------------------------
    #embedding部分替换为将文本补充为定长
    # print("train_dict graph",train_dict['x_p'][1])
    # print("train_dict graph shape", type(train_dict['x_p'][1]))
    print('train_dict idx_p_emb',train_dict['idx_p_emb'][1])
    print('train_dict idx_p_emb size', train_dict['idx_p_emb'][1].size())
    return train_dict, val_dict, test_dict