import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import cv2
import numpy as np

def getIou(bbox1, bbox2):
    """
    bbox1, bbox2 in xmin, ymin, xmax, ymax format
    """
    ixmin = max(bbox1[0], bbox2[0])
    ixmax = min(bbox1[2], bbox2[2])
    iymin = max(bbox1[1], bbox2[1])
    iymax = min(bbox1[3], bbox2[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)
    inters = iw*ih
    
    uni = ((bbox1[2]-bbox1[0]+1.) * (bbox1[3]-bbox1[1]+1.)+(bbox2[2]-bbox2[0]+1.) * (bbox2[3]-bbox2[1]+1.)-inters)
    iou = inters / uni
    return iou

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(loss)
        else:
            return loss

class optimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(512, 256)
        self.conv2 = GCNConv(256, 128)
        self.mlp1 = nn.Sequential(nn.Linear(512, 1), nn.ReLU())
        
    def similarity1(self, node_embedding, edge_index):
        edge_attr = []
        for i in range(edge_index.shape[1]):
            x1 = self.mlp1(torch.cat((node_embedding[edge_index[0][i]],
                                      node_embedding[edge_index[1][i]]), 0))
            edge_attr.append(x1.reshape(1))
        edge_attr = torch.stack(edge_attr)
        return edge_attr
    
    def forward(self, node_attr, edge_index, edge_attr):
        node_embedding= node_attr
        out = self.conv1(node_embedding, edge_index, edge_attr.reshape(-1))
        out = F.relu(out)
        edge_attr = self.similarity1(out, edge_index)
        out = self.conv2(out, edge_index, edge_attr.reshape(-1))
        return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.affinity_geom_net = nn.Sequential(nn.Linear(8, 1), nn.ReLU())
        self.affinity_appearance_net = nn.Sequential(nn.Linear(1024, 1), nn.ReLU())
        self.affinity_net = nn.Sequential(nn.Linear(2, 1), nn.ReLU())
        #self.affinity_final_net = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
#         self.affinity_final_net = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), 
#                                                 nn.Linear(64, 1))
        self.affinity_final_net = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        #self.affinity_final_net = nn.Sequential(nn.Linear(256, 1))
        
        self.optim_net = optimNet()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, data):
        img_file = '/home/lishuai/Experiment/MOT/MOT16/train/{}/img1/000001.jpg'.format(data.sequence)
        img = cv2.imread(img_file)
        height, width, _ = img.shape
        ground_truth = np.concatenate([data.ground_truth[:, :6], 
                                       np.arange(data.ground_truth.shape[0])[:, None]], 1)
        coords = ground_truth[:, 2:6].copy()
        coords[:, 2:4] = coords[:,2:4] - coords[:, 0:2] #x1,y1,x2,y2 to x1,y1,w,h
        coords = coords / np.array([width, height, width, height])
        coords = torch.Tensor(coords)
        
        distance_limit = 250
        edges_first_row = []
        edges_second_row = []
        frames = np.unique(ground_truth[:, 0]).astype(np.int)
        for frame in frames[:-1]:
            #print('frame %d'%frame)
            for node_i in ground_truth[ground_truth[:, 0] == frame]:
                src_id = int(node_i[-1])
                xmin, ymin, xmax, ymax = node_i[2:6]
                for node_j in ground_truth[ground_truth[:, 0] == frame+1]:
                    dst_id = int(node_j[-1])
                    xmin1, ymin1, xmax1, ymax1 = node_j[2:6]
                    distance= ((xmin-xmin1)**2 + (ymin-ymin1)**2)**0.5
                    if distance < distance_limit:
                        #print('Frame {}, distance of  {} and {} is {:.3f}'.format(node_i[0], src_id, dst_id, distance))
                        edges_first_row.append(src_id)
                        edges_second_row.append(dst_id)
                        
        edges = torch.tensor([edges_first_row, edges_second_row])
        sym_pruned_edges = torch.cat([edges[1][None], edges[0][None]], dim=0)
        edge_index = torch.cat([edges, sym_pruned_edges], dim=1) #pruned symmetric edge index used in GCN!
        #edge_index = double_edge(data)
        
        node_embedding = data.x
        edge_embedding = []
        edge_mlp = []
        edge_mlp1 = []
        for i in range(edge_index.shape[1]):
            x1 = self.affinity_appearance_net(torch.cat([node_embedding[edge_index[0][i]], 
                                                         node_embedding[edge_index[1][i]]]))
            x2 = self.affinity_geom_net(torch.cat([coords[edge_index[0][i]], 
                                                   coords[edge_index[1][i]]], 0))
            iou= getIou(ground_truth[edge_index[0][i], 2:6], 
                        ground_truth[edge_index[1][i], 2:6])
            edge_mlp.append(iou) 
            inputs = torch.cat([x1, x2])
            edge_embedding.append(self.affinity_net(inputs))

        edge_embedding= torch.stack(edge_embedding)
        output = self.optim_net(node_embedding, edge_index, edge_embedding)
        
        #Node level concatenation for regression
        #inp = torch.cat([output[data.edge_index[0]], output[data.edge_index[1]]], dim=1)
        #logits = self.affinity_final_net(inp)
        #prob = F.sigmoid(logits)
        
        #Node level difference for regression
        inp = output[data.edge_index[0]] - output[data.edge_index[1]]
        logits = self.affinity_final_net(inp)
        prob = F.sigmoid(logits) 
        ################################################
        
        
#         nodes_difference = nn.CosineSimilarity(dim=1)(output[data.edge_index[0]],
#                                                       output[data.edge_index[1]])
#         nodes_difference = nodes_difference.unsqueeze(1)
        
#         edge_mlp = torch.tensor(edge_mlp[:data.edge_index.shape[1]]).unsqueeze(1)
#         inp = torch.cat([nodes_difference, edge_mlp], dim=1)
#         inp = inp.float()
#         prob = self.affinity_final_net(inp)
        
        #inp = torch.cat([nodes_difference, torch.tensor(edge_mlp).unsqueeze(1)], dim=1)
        #inp = inp.float()
        #prob = self.affinity_final_net(inp)
        return prob
