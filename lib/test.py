import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'lib/deep-person-reid')
import numpy as np
np.set_printoptions(suppress=True)
import sklearn, pickle, random, time, datetime, cv2, os

import gurobipy as gp
from lib.qpthlocal.qp import QPFunction, QPSolvers, make_gurobi_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
torch.set_printoptions(sci_mode=False)
from lib.data_utils.preprocessing import *
from lib.data_utils.data_track import MOT16
from lib.gnn import ReID
from lib.training import build_constraint, build_constraint_torch, make_gurobi_model_tracking
#from lib.training import _remove_redundant_rows
from lib.inference import forwardLP

import matplotlib.pyplot as plt
plt.style.use('bmh')
#%matplotlib inline

torch.random.manual_seed(123)
np.random.seed(123)

whole_train_data_list = []
whole_val_data_list = []
for file in os.listdir('data/train_data/'):
    file_name = 'data/train_data/' + file
    with open(file_name, 'rb') as f:
        data_list = pickle.load(f)
        
    if file.startswith('MOT16-09') or file.startswith('MOT16-13'):
        whole_val_data_list = whole_val_data_list + data_list
    else:
        whole_train_data_list = whole_train_data_list + data_list
        
print('{} samples in training set, {} in validation set'.format(
    len(whole_train_data_list),len(whole_val_data_list)))
train_data_list = []
val_data_list = []
for ind in range(len(whole_train_data_list)):
    if whole_train_data_list[ind].x.shape[0] < 200:
        train_data_list.append(whole_train_data_list[ind])
    else:
        continue
        
for ind in range(len(whole_val_data_list)):
    if whole_val_data_list[ind].x.shape[0] < 200:
        val_data_list.append(whole_val_data_list[ind])
    else:
        continue
        
print('{} samples in training set, {} in validation set'.format(len(train_data_list), len(val_data_list)))

class Model(nn.Module): 
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Sequential(nn.Linear(6,6), nn.ReLU(), nn.Linear(6,1))
    def forward(self, data):
        x = self.fc(data.edge_attr)
        x = nn.Sigmoid()(x)
        return x
    
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
train_list, val_list = [], []
gamma = 0.1

for epoch in range(1, 2):
    #Shuffle data list
    np.random.shuffle(train_data_list)
    for iteration in range(len(train_data_list)):
        train_data = train_data_list[iteration]
        
        A_eq, b_eq, A_ub, b_ub, x_gt, tran_indicator = build_constraint(train_data, 1)
        A, b, G, h = build_constraint_torch(A_eq, b_eq, A_ub, b_ub)
        num_nodes = int(A_eq.shape[0] / 2)
        Q = gamma*torch.eye(G.shape[1])
    
        prob = model(train_data)
        prob = torch.clamp(prob, min=1e-7, max=1-1e-7)
        prob_numpy = prob.detach().squeeze().numpy()
        auc = sklearn.metrics.roc_auc_score(x_gt[num_nodes*3: ].squeeze(), prob_numpy)

        c_entry = torch.ones(num_nodes)
        c_exit = torch.ones(num_nodes)
        c_det = -1 * torch.ones(num_nodes)
        c_pred = -1 * torch.log(prob).squeeze()
        c_pred = torch.cat([c_det, c_entry, c_exit, c_pred])

        model_params = make_gurobi_model_tracking(G.numpy(), h.numpy(), A.numpy(), b.numpy(), Q.numpy())
        x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, maxIter=50, 
                       model_params=model_params)(Q,c_pred,G,h,A,b)

        loss = nn.MSELoss()(x, torch.from_numpy(x_gt).float().t())
        loss_edge = nn.MSELoss()(x[:, num_nodes*3:], torch.from_numpy(x_gt[num_nodes*3:]).float().t())
        
        const_cost = 1
        c_ = 0 * x_gt[num_nodes*3:] + const_cost * (1 - x_gt[num_nodes*3:])
        c_ = c_.squeeze()
        c_gt = torch.cat([c_pred[:num_nodes*3], torch.from_numpy(c_).float()]) #this is the ground truth cost
        
        obj_gt = c_gt @ torch.from_numpy(x_gt.squeeze()).float()
        obj_pred = c_pred @ x.squeeze()

        bce = nn.BCELoss()(prob, torch.from_numpy(x_gt[num_nodes*3:]).float())
        x_sol = forwardLP(c_pred.detach().numpy(), A_eq, b_eq, A_ub, b_ub)
        ham_dist = sklearn.metrics.hamming_loss(x_gt, x_sol)
        train_list.append((loss.item(), loss_edge.item(), auc, bce.item()))
        print('Train Epoch {} iter {}/{}, Obj {:.2f}/{:.2f}, mse {:.4f} mse edge {:.4f} ce {:.3f} auc {:.3f} hamming {:.3f}'.format(
                    epoch, iteration, len(train_data_list), obj_pred.item(), obj_gt.item(), loss.item(), loss_edge.item(), bce.item(), auc, ham_dist))
        
        optimizer.zero_grad()
        loss_edge.backward()
        #bce.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'ckpt/epoch_{}.pth'.format(epoch))
    
    np.random.shuffle(val_data_list)
    for iteration in range(len(val_data_list)):
        val_data = val_data_list[iteration]
        A_eq, b_eq, A_ub, b_ub, x_gt, tran_indicator = build_constraint(val_data, 1)
        A, b, G, h = build_constraint_torch(A_eq, b_eq, A_ub, b_ub)
        Q = gamma*torch.eye(G.shape[1])
        num_nodes = int(A_eq.shape[0] / 2)
        
        with torch.no_grad():
            prob = model(val_data)
            prob = torch.clamp(prob, min=1e-7, max=1-1e-7)
        prob_numpy = prob.detach().squeeze().numpy()
        auc = sklearn.metrics.roc_auc_score(x_gt[num_nodes*3: ].squeeze(), prob_numpy)

        c_entry = torch.ones(num_nodes)
        c_exit = torch.ones(num_nodes)
        c_det = -1 * torch.ones(num_nodes)
        c_pred = -1 * torch.log(prob).squeeze()
        c_pred = torch.cat([c_det, c_entry, c_exit, c_pred])

        model_params_quad = make_gurobi_model_tracking(G.numpy(), h.numpy(), A.numpy(), b.numpy(), Q.numpy())
        x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, maxIter=50,
                       model_params=model_params_quad)(Q, c_pred, G, h, A, b)

        loss = nn.MSELoss()(x, torch.from_numpy(x_gt).float().t())
        loss_edge = nn.MSELoss()(x[:, num_nodes*3:], torch.from_numpy(x_gt[num_nodes*3:]).float().t())
        
        const_cost = 1
        c_ = 0 * x_gt[num_nodes*3:] + const_cost * (1 - x_gt[num_nodes*3:])
        c_ = c_.squeeze()
        c_gt = torch.cat([c_pred[:num_nodes*3], torch.from_numpy(c_).float()]) #this is the ground truth cost
        
        obj_gt = c_gt @ torch.from_numpy(x_gt.squeeze()).float()
        obj_pred = c_pred @ x.squeeze()

        bce = nn.BCELoss()(prob, torch.from_numpy(x_gt[num_nodes*3:]).float())
        x_sol = forwardLP(c_pred.detach().numpy(), A_eq, b_eq, A_ub, b_ub)
        ham_dist = sklearn.metrics.hamming_loss(x_gt, x_sol)
        val_list.append((loss.item(), loss_edge.item(), auc, bce.item()))
        print('Val Epoch {}, iter {}/{}, Obj {:.2f}/{:.2f}, mse {:.3f} mse edge {:.3f} ce {:.3f} auc {:.3f} hamming {:.3f}'.format(
                    epoch, iteration, len(val_data_list), obj_pred.item(), obj_gt.item(), loss.item(), loss_edge.item(), 
                    bce.item(), auc, ham_dist))