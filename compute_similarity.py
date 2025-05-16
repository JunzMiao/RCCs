# -*- coding: utf-8 -*-

import torch.nn.functional as F
import torch

def compute_euc_dis(data):
    dis_matrix = torch.norm(data[:, None] - data, dim=2, p=2)
    T = 1
    p_matrix = F.softmax((dis_matrix) / T, dim=1)

    return p_matrix

def compute_manhattan_dis(data):
    dis_matrix = torch.norm(data[:, None] - data, dim=2, p=1)
    p_matrix = F.softmax(dis_matrix, dim=1)

    return p_matrix

def compute_cos_similarity(data):
    cosine_similarity = F.cosine_similarity(data[None,:,:], data[:,None,:], dim=-1)
    p_matrix = cosine_similarity

    return p_matrix