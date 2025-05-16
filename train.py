# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm, trange

from compute_similarity import compute_euc_dis, compute_manhattan_dis, compute_cos_similarity
from data_loader import Data_Loader
from fgsm import FGSM
from bim import BIM
from mim import MIM
from pgd import PGD
from models import model_loader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_path", type=str, default="None", required=True,
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="[Net_MNIST, Resnet_CIFAR10, Resnet_CIFAR100]",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="[mnist, fashionmnist, cifar10, cifar100]",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
    )
    parser.add_argument(
        "--class_num", type=int, required=True,
    )

    parser.add_argument(
        "--optimizer", type=str, default="SGD", required=True,
        help="[SGD, Adam]",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, required=True,
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, required=True,
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, required=True,
    )

    parser.add_argument(
        "--lr_scheduler", type=int, default=False, required=True,
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=20, required=True,
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, required=True,
    )

    parser.add_argument(
        "--max_iter", type=int, default=20, required=True,
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=128, required=True,
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1000, required=True,
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", required=True,
    )

    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="root of save the log and trained model."
    )

    parser.add_argument(
        "--attacker", type=str, default="fgsm", required=True,
        help="[fgsm, bim, pgd, mim]",
    )
    parser.add_argument(
        "--norm", type=float, default=np.inf, required=True,
        help="[np.inf, 1, 2]",
    )
    parser.add_argument(
        "--eps", type=float, default=0.01, required=True,
    )
    parser.add_argument(
        "--loss", type=str, default="ce", required=True,
    )

    parser.add_argument(
        "--step_size", type=float, default=0.0078, required=True,
    )
    parser.add_argument(
        "--steps", type=int, default=10, required=True,
    )
    parser.add_argument(
        "--decay_factor", type=float, default=0.9, required=True,
    )
    parser.add_argument(
        "--resize_rate", type=float, default=0.85, required=True,
    )
    parser.add_argument(
        "--diversity_prob", type=float, default=0.7, required=True,
    )

    parser.add_argument(
        "--coeff_adv_loss", type=float, default=0.5, required=True,
    )
    parser.add_argument(
        "--coeff_local_constraint", type=float, default=15, required=True,
    )
    parser.add_argument(
        "--dis_metric", type=str, default="euc", required=True,
        help="[euc, manhattan, cos]"
    )

    args = parser.parse_args()

    return args

def local_constraint_metric(clean_output, adv_output, dis_metric, target, epoch):
    if dis_metric == "euc":
        clean_output_matrix = compute_euc_dis(clean_output)
        adv_output_matrix = compute_euc_dis(adv_output)
    elif dis_metric == "manhattan":
        clean_output_matrix = compute_manhattan_dis(clean_output)
        adv_output_matrix = compute_manhattan_dis(adv_output)
    elif dis_metric == "cos":
        clean_output_matrix = compute_cos_similarity(clean_output)
        adv_output_matrix = compute_cos_similarity(adv_output)

    clean_matrix_local_constraint = clean_output_matrix * (target.unsqueeze(1) == target).float() * (1 - torch.eye(len(target)).to(target.device))
    adv_matrix_local_constraint = adv_output_matrix * (target.unsqueeze(1) == target).float() * (1 - torch.eye(len(target)).to(target.device))

    cos_diff = 1 - torch.cosine_similarity(clean_matrix_local_constraint, adv_matrix_local_constraint, dim=1, eps=1e-08)
    sim_loss = cos_diff.sum() / len(clean_output)

    return sim_loss

def train_epoch(model, device, attack_cls, coeff_adv_loss, coeff_local_constraint, dis_metric, train_loader, optimizer, epoch):
    model.train()

    clean_train_loss = 0
    clean_train_correct = 0

    adv_train_loss = 0
    adv_train_correct = 0

    local_constraint_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        attack_cls.model = model
        adv_data = attack_cls(data, target).to(device)

        optimizer.zero_grad()

        clean_output, cf1 = model(data)
        adv_output, af1 = model(adv_data)

        clean_loss = F.cross_entropy(clean_output, target)
        adv_loss = F.cross_entropy(adv_output, target)
        local_constraint_loss = local_constraint_metric(cf1, af1, dis_metric, target, epoch)

        loss = (1 - coeff_adv_loss) * clean_loss + coeff_adv_loss * adv_loss + coeff_local_constraint * local_constraint_loss

        loss.backward()

        optimizer.step()

        clean_pred = clean_output.argmax(dim=1, keepdim=True)
        adv_pred = adv_output.argmax(dim=1, keepdim=True)

        clean_train_loss += clean_loss.item()
        adv_train_loss += adv_loss.item()
        local_constraint_train_loss += local_constraint_loss.item()

        clean_train_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()
        adv_train_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
    
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        torch.cuda.empty_cache()

    clean_train_loss /= len(train_loader)
    adv_train_loss /= len(train_loader)
    local_constraint_train_loss /= len(train_loader)

    clean_train_acc = clean_train_correct / len(train_loader.dataset)
    adv_train_acc = adv_train_correct / len(train_loader.dataset)
    
    return clean_train_loss, adv_train_loss, local_constraint_train_loss, clean_train_acc, adv_train_acc

def test_epoch(model, device, attack_cls, coeff_adv_loss, coeff_local_constraint, dis_metric, test_loader, epoch):
    model.eval()

    clean_test_loss = 0
    clean_test_correct = 0

    adv_test_loss = 0
    adv_test_correct = 0

    local_constraint_test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        attack_cls.model = model
        adv_data = attack_cls(data, target)
        with torch.no_grad():
            clean_output, cf1 = model(data)
            adv_output, af1 = model(adv_data)

            clean_loss = F.cross_entropy(clean_output, target)
            adv_loss = F.cross_entropy(adv_output, target)
            local_constraint_loss = local_constraint_metric(cf1, af1, dis_metric, target, epoch)

            loss = (1 - coeff_adv_loss) * clean_loss + coeff_adv_loss * adv_loss + coeff_local_constraint * local_constraint_loss

            clean_pred = clean_output.argmax(dim=1, keepdim=True)
            adv_pred = adv_output.argmax(dim=1, keepdim=True)

            clean_test_loss += clean_loss.item()
            adv_test_loss += adv_loss.item()
            local_constraint_test_loss += local_constraint_loss.item()

            clean_test_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()
            adv_test_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
        
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))
        torch.cuda.empty_cache()

    clean_test_loss /= len(test_loader)
    adv_test_loss /= len(test_loader)
    local_constraint_test_loss /= len(test_loader)

    clean_test_acc = clean_test_correct / len(test_loader.dataset)
    adv_test_acc = adv_test_correct / len(test_loader.dataset)
    
    return clean_test_loss, adv_test_loss, local_constraint_test_loss, clean_test_acc, adv_test_acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    Timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    train_loader, test_loader = Data_Loader(args.dataset_name, args.dataset_path, args.train_batch_size, args.test_batch_size).load_data()
    
    train_net = model_loader(args.model_name, args.pretrained_model_path).to(args.device)

    if args.norm == 0:
        args.norm = np.inf

    if args.attacker == "fgsm":
        attack_cls = FGSM(train_net, args.device, args.norm, args.eps, args.loss)
    elif args.attacker == "bim":
        attack_cls = BIM(train_net, args.device, args.norm, args.eps, args.step_size, args.steps, args.loss)
    elif args.attacker == "pgd":
        attack_cls = PGD(train_net, args.device, args.norm, args.eps, args.step_size, args.steps, args.loss)
    elif args.attacker == "mim":
        attack_cls = MIM(train_net, args.device, args.norm, args.eps, args.step_size, args.steps, args.decay_factor)

    if args.model_name in ["resnet18", "resnet50"] and args.pretrained_model_path != "None":
        params_pre = [param for name, param in train_net.named_parameters() if name not in ["conv1.weight", "fc.weight", "fc.bias"]]
        params_new = [param for name, param in train_net.named_parameters() if name in ["conv1.weight", "fc.weight", "fc.bias"]]
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD([{'params': params_pre}, {'params': params_new, 'lr': args.learning_rate * 10}], lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam([{'params': params_pre}, {'params': params_new, 'lr': args.learning_rate * 10}], lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        if args.optimizer == "SGD":
            optimizer = optim.SGD(train_net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(train_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    res_path = os.path.join(args.output_dir, args.model_name + "_" + args.dataset_name + "_local_constraint_adv_train_log.csv")
    column_param = pd.DataFrame(data=[["pretrained_model_path", "model_name", "dataset_name", "dataset_path", "class_num",
                                       "optimizer", "learning_rate", "momentum", "weight_decay",
                                       "lr_scheduler", "lr_step_size", "gamma",
                                       "max_iter", "train_batch_size", "test_batch_size", "device", 
                                       "attacker", "norm", "eps", "loss",
                                       "step_size", "steps", "decay_factor", "resize_rate", "diversity_prob", "coeff_adv_loss", "coeff_local_constraint", "dis_metric",
                                       "Timestamp"]])
    if not os.path.exists(res_path):
        column_param.to_csv(res_path, mode='w', header=False, index=False)
    else:
        column_param.to_csv(res_path, mode='a', header=False, index=False)
    param = pd.DataFrame(data=[[args.pretrained_model_path, args.model_name, args.dataset_name, args.dataset_path, args.class_num,
                                args.optimizer, args.learning_rate, args.momentum, args.weight_decay,
                                args.lr_scheduler, args.lr_step_size, args.gamma,
                                args.max_iter, args.train_batch_size, args.test_batch_size, args.device, 
                                args.attacker, args.norm, args.eps, args.loss,
                                args.step_size, args.steps, args.decay_factor, args.resize_rate, args.diversity_prob, args.coeff_adv_loss, args.coeff_local_constraint, args.dis_metric, Timestamp]])
    param.to_csv(res_path, mode='a', header=False, index=False)

    res_column = pd.DataFrame(data=[['epoch', 
                                    'clean_train_loss', 'adv_train_loss',
                                    'clean_test_loss', 'adv_test_loss', 
                                    'local_constraint_train_loss', 'local_constraint_test_loss', 
                                    'clean_train_acc', 
                                    'adv_train_acc',
                                    'clean_test_acc', 
                                    'adv_test_acc',
                                    'best_clean_test_acc', 'best_adv_test_acc']])
    res_column.to_csv(res_path, mode='a', header=False, index=False)

    clean_best_acc = 0
    adv_best_acc = 0

    for epoch in trange(1, args.max_iter + 1):
        clean_train_loss, adv_train_loss, local_constraint_train_loss, clean_train_acc, adv_train_acc = train_epoch(train_net, args.device, attack_cls, args.coeff_adv_loss, args.coeff_local_constraint, args.dis_metric, train_loader, optimizer, epoch)
        clean_test_loss, adv_test_loss, local_constraint_test_loss, clean_test_acc, adv_test_acc = test_epoch(train_net, args.device, attack_cls, args.coeff_adv_loss, args.coeff_local_constraint, args.dis_metric, test_loader, epoch)

        if args.lr_scheduler:
            scheduler.step()

        if clean_test_acc > clean_best_acc:
            clean_best_acc = clean_test_acc
            torch.save(train_net.state_dict(), os.path.join(args.output_dir, "adv_train" + "_" + args.model_name + "_" + args.dataset_name + "_" + "clean" + "_" + Timestamp + ".pth"))
        if adv_test_acc > adv_best_acc:
            adv_best_acc = adv_test_acc
            torch.save(train_net.state_dict(), os.path.join(args.output_dir, "adv_train" + "_" + args.model_name + "_" + args.dataset_name + "_" + "adv" + "_" + Timestamp + ".pth"))

        res = pd.DataFrame(data=[[epoch,
                                clean_train_loss, adv_train_loss,
                                clean_test_loss, adv_test_loss,
                                local_constraint_train_loss, local_constraint_test_loss, 
                                clean_train_acc, adv_train_acc,
                                clean_test_acc, adv_test_acc,
                                clean_best_acc, adv_best_acc]])
        res.to_csv(res_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
