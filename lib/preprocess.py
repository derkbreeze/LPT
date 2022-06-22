import pickle 
import argparse
from pathlib import Path 

import torch
import numpy as np 
from tqdm import tqdm 
from skimage.io import imsave
from skimage.transform import resize 
from sklearn.decomposition import PCA 
from torchreid.models.osnet import osnet_x0_5
from lib.dataset import MOT16

def build_graph1(nodes, edges, maxFrameGap, out):
    """
    nodes: shape: n x 4 np.array
    maxFrameGap: maximum number of skip connections between nodes. 
    out: shape: edges x 1 torch.tensor
    
    returns: costs, A_eq, b_eq, A_ub, b_ub used for LP inference, and gt binary x for supervision
    """
    entryOffset = nodes.shape[0]
    exitOffset = nodes.shape[0] * 2
    linkOffset = nodes.shape[0] * 3
    print('entry offset: {}, exit offset: {}, link offset: \
          {}, # edges: {}'.format(entryOffset, exitOffset, linkOffset, e_count))
    
    linkGraph = -1 * np.ones([entryOffset, entryOffset])
    linkIndexGraph = np.zeros([entryOffset, entryOffset])
    gtlinkIndexGraph = np.zeros([entryOffset, entryOffset])
    edge_idx = 1
    for i in range(linkGraph.shape[0]):
        for j in range(linkGraph.shape[0]):
            if timestamps[j] - timestamps[i] >= 1 and timestamps[j] - timestamps[i] <= maxFrameGap:
                indicator = np.logical_and(edges[:, 0] == i, edges[:, 1] == j)
                if np.any(indicator):
                    linkGraph[i, j] = out[indicator].item() #cost of linking node i and j
                    linkIndexGraph[i, j] = edge_idx
                    gtlinkIndexGraph[i, j] = int(gt[indicator]) #supervision whether two nodes should connect
                    edge_idx += 1        

    nConstraints = linkOffset + edge_idx
    costs = np.zeros((nConstraints, 1))
    gt_x = np.zeros((nConstraints, 1)) # currently only supervise edges. detection and s/t edges are not trained
    for i in range(nodes.shape[0]):
        costs[i] = -1.0
        costs[entryOffset + i] = 1.0
        costs[exitOffset + i] = 1.0
        
    #Initialize the constraint matrices
    A_ub = np.zeros([entryOffset * 2, nConstraints])
    b_ub = np.zeros([entryOffset * 2, 1])
    lEqIndex = 0
    A_eq = np.zeros([entryOffset * 2, nConstraints])
    b_eq = np.zeros([entryOffset * 2, 1])
    eqIndex = 0
    
    for node in range(linkGraph.shape[0]):
        out_nodes = np.where(linkGraph[node, :] != -1)[0]
        in_nodes = np.where(linkGraph[:, node] != -1)[0]

        if out_nodes.shape[0] != 0:
            for out_node in out_nodes:
                linkIndex = linkIndexGraph[node, out_node]
                costs[int(linkOffset + linkIndex)] = linkGraph[node, out_node]
                gt_x[int(linkOffset + linkIndex)] = gtlinkIndexGraph[node, out_node]

        if in_nodes.shape[0] != 0:
            for in_node in in_nodes:
                linkIndex = linkIndexGraph[in_node, node]
                costs[int(linkOffset + linkIndex)] = linkGraph[in_node, node]
                gt_x[int(linkOffset + linkIndex)] = gtlinkIndexGraph[in_node, node]

        if out_nodes.shape[0] != 0 and in_nodes.shape[0] != 0:
            
            # Flow coming in == detection edge
            constraint = np.zeros(nConstraints)
            constraint[entryOffset + node] = 1
            for d1 in in_nodes:
                linkIndex = linkIndexGraph[d1, node]
                constraint[int(linkOffset + linkIndex)] = 1
            constraint[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1

            # Flow coming out == detection edge
            constraint = np.zeros(nConstraints)
            constraint[exitOffset + node] = 1
            for d2 in out_nodes:
                linkIndex = linkIndexGraph[node, d2]
                constraint[int(linkOffset + linkIndex)] = 1
            constraint[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1
        elif out_nodes.shape[0] != 0 and in_nodes.shape[0] == 0:
            
            # Flow coming out == detection edge
            constraint = np.zeros(nConstraints)
            constraint[exitOffset + node] = 1
            for d2 in out_nodes:
                linkIndex = linkIndexGraph[node, d2]
                constraint[int(linkOffset + linkIndex)] = 1
            constraint[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1

             # Flow coming in == detection edge
            constraint = np.zeros(nConstraints)
            constraint[entryOffset + node] = 1
            constraint[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1

        elif out_nodes.shape[0] == 0 and in_nodes.shape[0] != 0:
            
            # Flow coming in == detection edge
            constraint = np.zeros(nConstraints)
            constraint[entryOffset + node] = 1
            for d1 in in_nodes:
                linkIndex = linkIndexGraph[d1, node]
                constraint[int(linkOffset + linkIndex)] = 1
            constraint[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1

             # Flow coming out == detection edge
            constraint = np.zeros(nConstraints)
            constraint[exitOffset + node] = 1
            constraint[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1
        elif out_nodes.shape[0] == 0 and in_nodes.shape[0] == 0:
            
            constraint = np.zeros(nConstraints)
            constraint[entryOffset + node] = 1
            constriant[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1

            constraint = np.zeros(nConstraints)
            constraint[exitOffset + node] = 1
            constriant[node] = -1
            A_eq[eqIndex, :] = constraint
            b_eq[eqIndex] = 0
            eqIndex += 1
            
    A_ub = np.concatenate([np.eye(A_eq.shape[1]), -1*np.eye(A_eq.shape[1])], axis=0)
    b_ub = np.concatenate([np.ones((A_eq.shape[1], 1)), np.zeros((A_eq.shape[1], 1))], axis=0)
    
#     A_eq = scipy.sparse.csr_matrix(A_eq)
#     A_ub = scipy.sparse.csr_matrix(A_ub)
    
    return costs, A_ub, b_ub, A_eq, b_eq, gt_x

def compute_box_features(box_1, box_2):
    top_1, left_1 = (box_1[0], box_1[3])
    top_2, left_2 = (box_2[0], box_2[3])

    width_1 = box_1[2] - box_1[0]
    width_2 = box_2[2] - box_2[0]

    height_1 = box_1[3] - box_1[1]
    height_2 = box_2[3] - box_2[1]

    y_rel_dist = 2 * (top_1 - top_2) / (height_1 + height_2)
    x_rel_dist = 2 * (left_1 - left_2) / (height_1 + height_2)
    rel_size_y = np.log(height_1 / height_2)
    rel_size_x = np.log(width_1 / width_2)

    return [x_rel_dist, y_rel_dist, rel_size_y, rel_size_x]

def get_top_k_nodes(cur_node, existing_nodes, k=50):
    cur_node_feat = cur_node['vis_feat']
    scores = []
    for ex in existing_nodes:
        scores.append(
            np.dot(cur_node_feat,ex['vis_feat'])/(np.linalg.norm(cur_node_feat)*np.linalg.norm(ex['vis_feat']))
            )
    sorted_nodes = [node for (score,node) in sorted(zip(scores, existing_nodes), reverse=True,
        key=lambda x:x[0])]

    try:
        return sorted_nodes[:k]
    except IndexError:
        return sorted_nodes

def fit_pca(save_path: str, dataset_path: str, re_id_net):

    dataset = MOT16(dataset_path, 'train')
    instances = []
    for sequence in dataset:

        for i in tqdm(range(0, len(sequence),50)):
            item = sequence[i]
            gt = item['gt']
            cropped = item['cropped_imgs']

            for gt_id, box in gt.items():
                with torch.no_grad():
                    try:
                        img = resize(cropped[gt_id].numpy().transpose(1,2,0),(256,128))
                        feat = re_id_net(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda().float()).cpu().squeeze().numpy()
                        instances.append(feat)
                    except Exception as e:
                        tqdm.write('Error when processing image: {}'.format(str(e)))
                        continue

    print("Number of instances: ".format(len(instances)))
    pca_transform = PCA(n_components=32)
    pca_transform.fit(np.stack(instances))
    pickle.dump(pca_transform, open(save_path, "wb"))


def preprocess(out_dir, re_id_net, mot_dataset, pca_transform, save_imgs=False, device="cuda"):

    for sequence in mot_dataset:
        tqdm.write('processing "{}"'.format(str(sequence)))
        seq_out = os.path.join(out_dir, str(sequence))

        if not os.path.exists(seq_out):
            os.makedirs(seq_out)

        for i in tqdm(range(len(sequence) - 15)):
            subseq_out = os.path.join(seq_out, "subseq_{}".format(i))

            try:
                os.makedirs(subseq_out)
            except FileExistsError:
                continue

            edges = [] #(2, num_edges) with pairs of connected node ids
            edge_features = [] # (num_edges, num_feat_edges) edge_id with features
            gt_edges = [] #(num_edges) with 0/1 depending on edge is gt

            existing_nodes = []
            node_id = 0

            for t,j in enumerate(range(i, i+15)):
                item = sequence[j]
                gt = item['gt']
                cropped = item['cropped_imgs']

                cur_ndoes = []
                for gt_id, box in gt.items():

                    with torch.no_grad():
                        try:
                            img = resize(cropped[gt_id].numpy().transpose(1, 2, 0),
                                         (256, 128))
                            feat = re_id_net(
                                torch.from_numpy(img)
                                     .permute(2, 0, 1)
                                     .unsqueeze(0)
                                     .to(device)
                                     .float()
                            ).cpu().numpy()
                            feat = pca_transform.transform(feat).squeeze()
                        except Exception as e:
                            tqdm.write(
                                'Error when processing image: {}'.format(str(e)))
                            continue

                    cur_nodes.append({'box': box,
                                      'gt_id': gt_id,
                                      'img': img,
                                      'node_id': node_id,
                                      'time': t,
                                      'vis_feat': feat})
                    node_id += 1

                for cur in cur_nodes:
                    best_nodes = get_top_k_nodes(cur, existing_nodes)
                    for ex in best_nodes:
                        ex_id, cur_id = ex['node_id'], cur['node_id']
                        edges.append([ex_id, cur_id])

                        gt_edges.append(0 if ex['gt_id'] != cur['gt_id'] else 1)

                        box_feats = compute_box_features(ex['box'], cur['box'])
                        rel_appearance = np.linalg.norm(
                            cur['vis_feat'] - ex['vis_feat'], ord=2)
                        box_feats.append(cur['time'] - ex['time'])
                        box_feats.append(rel_appearance)
                        edge_features.append(box_feats) 

                existing_nodes.extend(cur_nodes)

            all_nodes = sorted(existing_nodes, key=lambda n: n['node_id'])

            edges = torch.tensor(edges)
            gt_edges = torch.tensor(gt_edges)
            edge_features = torch.tensor(edge_features)
            node_features = torch.tensor([node['vis_feat'] for node in all_nodes])
            node_timestamps = torch.tensor([n['time'] for n in all_nodes])
            node_boxes = torch.tensor([n['box'] for n in all_nodes])

            
            torch.save(edges, os.path.join(subseq_out, 'edges.pth'))
            torch.save(gt_edges, os.path.join(subseq_out, 'gt.pth'))

            torch.save(node_timestamps, os.path.join(subseq_out,'node_timestamps.pth'))
            torch.save(edge_features, os.path.join(subseq_out,'edge_features.pth'))
            torch.save(node_features, os.path.join(subseq_out,'node_features.pth'))
            torch.save(node_boxes, os.path.join(subseq_out,'node_boxes.pth'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed',
                        help='Outout directory for the preprocessed sequences')

    parser.add_argument('--pca_path', type=str, default='pca.sklearn',
                        help='Path to the PCA model for reducing '
                             'dimensionality of the ReID network')

    parser.add_argument('--dataset_path', type=str, default='../MOT16',
                        help='Path to the root directory of MOT dataset')

    parser.add_argument('--mode', type=str, default='train',
                        help='Use train or test sequences (for test additional '
                             'work necessary)')

    parser.add_argument('--threshold', type=float, default=.1,
                        help='Visibility threshold for detection to be '
                             'considered a node')

    parser.add_argument('--save_imgs', action='store_true',
                        help='Save image crops according to bounding boxes for '
                             'training the CNN (only required if this is '
                             'wanted)')

    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to run the '
                                                      'preprocessing on.')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = osnet_x0_5(pretrained=True).to(args.device)
    net.eval()

    dataset = MOT16(args.dataset_path, args.mode, vis_threshold=args.threshold)
    pca = pickle.load(open(args.pca_path, "rb"))
    preprocess(output_dir, net, dataset, args.save_imgs, device=args.device)
