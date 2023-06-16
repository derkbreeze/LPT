import numpy as np
import gurobipy as gp
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
from lib.utils import interpolateTrack, interpolateTracks

class Tracker(nn.Module):
    def __init__(self, net):
        super(Tracker, self).__init__()
        self.net = net
        
    def linprog(self, c, A_eq, b_eq, A_ub, b_ub):
        """
        Performs LP during inference stage
        c: LP learned cost function, numpy.ndarray, shape: (n, ) where n is the problem size
        A_eq, b_eq: equality constraint(flow conservation), numpy.ndarray, shape: (num_equality_constraints, n)
        A_ub, b_ub: inequality constraint, numpy.ndarray, shape: (num_inequality_constraints, n)
        returns: LP solution.
        """
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape = A_eq.shape[1], vtype = gp.GRB.BINARY, name = 'x')
        model.setObjective(c @ x, gp.GRB.MINIMIZE)
        model.addConstr(A_eq @ x == b_eq.squeeze(), name = 'eq')
        model.addConstr(A_ub @ x <= b_ub.squeeze(), name = 'ineq')
        model.optimize()
        if model.status == 2:
            sol = x.X
        else:
            print('error in solver')
            sol = None
        return sol
    
    def build_constraint_torch(self, A_eq, b_eq, A_ub, b_ub):
        A, b = torch.from_numpy(A_eq).float(), torch.from_numpy(b_eq).float().squeeze() #Convert from size nx1 to n
        G, h = torch.from_numpy(A_ub).float(), torch.from_numpy(b_ub).float().squeeze() #Convert from size nx1 to n
        return A, b, G, h
    
    def build_constraint(self, data, max_frame_gap):
        """
        Build LP constraint used during training stage.
        Args: data: an instance of torch_geometric.data.Data
              max_frame_gap: maximal frame gap used to connect two detections across frames. Set to 1 during training
        returns:
        A_eq, b_eq, A_ub, b_ub: equality and in-equality constraints in LP
        x_gt: ground truth data association
        tran_indicator: indicating which parts of the edges(among temporal fully-connected edges) are selected
        """
        edge_ind = 0
        num_nodes = data.x.shape[0]
        edges = data.edge_index.t().numpy()
        timestamps = data.ground_truth[:, 0].astype(int)
        entry_offset, exit_offset, link_offset = num_nodes, num_nodes * 2, num_nodes * 3
        entry_indicator = np.zeros((num_nodes, 1), dtype=np.int32)  #Indicating which detections are selected as gt start 
        exit_indicator = np.zeros((num_nodes, 1), dtype=np.int32)   #Indicating which detections are selected as gt terminal
        tran_indicator = np.zeros((edges.shape[0], 1), dtype=np.int32)

        linkIndexGraph = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        gtlinkIndexGraph = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        for i in range(linkIndexGraph.shape[0]):
            for j in range(linkIndexGraph.shape[0]):
                frame_gap = timestamps[j] - timestamps[i]
                if frame_gap >= 1 and frame_gap <= max_frame_gap:        
                    inds = np.logical_and(edges[:, 0] == i, edges[:, 1] == j) #bool np.array, array([False, True, False...])
                    if inds.sum() != 0:
                        edge_ind += 1
                        ind = np.where(inds == True)[0][0] #Something like np.array([1]), so add one [0]
                        tran_indicator[ind] = 1 #Indicating this transition edge has been selected
                        linkIndexGraph[i, j] = edge_ind
                        gtlinkIndexGraph[i, j] = data.y.numpy()[ind]

        assert (linkIndexGraph.flatten() != 0).sum() == edge_ind, 'Shape Mismatch'
        assert tran_indicator.sum() == edge_ind, 'Shape Mismatch'
        num_constraints = link_offset + edge_ind
        for i in range(linkIndexGraph.shape[0]):
            incoming_flows = edges[:, 1] == i
            if not np.any(incoming_flows):
                start_node = i
                entry_indicator[i] = 1
                while np.any(gtlinkIndexGraph[i, :]):
                    i = np.where(gtlinkIndexGraph[i, :] == 1)[0][0]
                terminate_node = i
                if terminate_node != start_node:
                    exit_indicator[terminate_node] = 1
                else:
                    entry_indicator[i] = 0
        assert entry_indicator.shape == exit_indicator.shape, "Shape mismatch"
        assert entry_indicator.sum() ==  exit_indicator.sum(), "GT flow in != flow out" 

        #Initialize the constraint matrices
        x_gt = np.zeros((edge_ind, 1), dtype=np.float32)
        A_eq, b_eq = np.zeros((num_nodes * 2, num_constraints), dtype=np.float32), np.zeros((num_nodes * 2, 1), dtype=np.float32)
        A_ub, b_ub = np.zeros((num_nodes * 2, num_constraints), dtype=np.float32), np.ones((num_nodes * 2, 1), dtype=np.float32)
        eq_ind, leq_ind = 0, 0
        for node in range(linkIndexGraph.shape[0]):
            out_nodes = np.where(linkIndexGraph[node, :] != 0)[0]
            in_nodes = np.where(linkIndexGraph[:, node] != 0)[0] 
            if out_nodes.shape[0] != 0: # Note that linkIndex starts at 1, so minus 1 at certain indexes
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node]
                    x_gt[link_ind - 1] = gtlinkIndexGraph[node, out_node]

            if in_nodes.shape[0] != 0:
                for in_node in in_nodes:
                    link_ind = linkIndexGraph[in_node, node]
                    x_gt[link_ind - 1] = gtlinkIndexGraph[in_node, node]

            if out_nodes.shape[0] != 0 and in_nodes.shape[0] != 0:
                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in <= 1    
                for in_node in in_nodes:
                    link_ind = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind -1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in == detection edge
                for in_node in in_nodes:
                    link_index = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind - 1] = 1
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow coming out <= 1
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node]
                    constraint[link_offset + link_ind - 1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow out == detection edge
                for out_node in out_nodes:
                    link_index = linkIndexGraph[node, out_node]
                    constraint[link_offset + link_ind - 1] = 1
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1
            elif out_nodes.shape[0] != 0 and in_nodes.shape[0] == 0:
                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow coming out <= 1
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node]
                    constraint[link_offset + link_ind - 1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow coming in <= 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow out == detection edge
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node] #link_index starts from 1, so.
                    constraint[link_offset + link_ind - 1] = 1 #flow transition to next node
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint = np.zeros(num_constraints)                    #flow in == detection edge
                constraint[entry_offset + node], constraint[node] = 1, -1 #flow coming from start node
                A_eq[eq_ind, :] = constraint
                eq_ind += 1
            elif out_nodes.shape[0] == 0 and in_nodes.shape[0] != 0:
                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in <= 1
                for in_node in in_nodes:
                    link_index = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind - 1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow out <= 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in == detection edge
                for in_node in in_nodes:
                    link_ind = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind - 1] = 1
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint = np.zeros(num_constraints) # flow coming out == detection edge
                constraint[exit_offset + node], constraint[node] = 1, -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1
            elif out_nodes.shape[0] == 0 and in_nodes.shape[0] == 0:
                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint = np.zeros(num_constraints)
                constraint[entry_offset + node], constraint[node] = 1, -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint = np.zeros(num_constraints)
                constraint[exit_offset + node], constraint[node] = 1, -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1
        det_indicator = np.ones((entry_offset, 1))
        x_gt = np.concatenate([det_indicator, entry_indicator, exit_indicator, x_gt], axis=0)
        return A_eq, b_eq, A_ub, b_ub, x_gt, tran_indicator
    
    def make_gurobi_model_tracking(self, G, h, A, b, Q, test=False):
        '''
        Convert to Gurobi model. Copied from 
        https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/gurobi_.py
        '''
        vtype = gp.GRB.CONTINUOUS #gp.GRB.BINARY if test else gp.GRB.CONTINUOUS
        n = A.shape[1] if A is not None else G.shape[1]
        model = gp.Model()
        model.params.OutputFlag = 0
        x = [model.addVar(vtype= vtype, name='x_%d'%i, lb = 0, ub = 1) for i in range(n)]
        model.update()   # integrate new variables

        inequality_constraints = [] #subject to G * x <= h
        if G is not None:
            for i in range(G.shape[0]):
                row = np.where(G[i] != 0)[0]
                inequality_constraints.append(model.addConstr(gp.quicksum(G[i,j] * x[j] for j in row) <= h[i]))
        equality_constraints = []   #subject to A * x == b
        if A is not None:
            for i in range(A.shape[0]):
                row = np.where(A[i] != 0)[0]
                equality_constraints.append(model.addConstr(gp.quicksum(A[i,j] * x[j] for j in row) == b[i]))

        obj = gp.QuadExpr()
        if Q is not None:
            rows, cols = Q.nonzero()
            for i, j in zip(rows, cols):
                obj += x[i] * Q[i, j] * x[j]
        return model, x, inequality_constraints, equality_constraints, obj
    
    def buildConstraint(self, linkIndexGraph):
        """
        Build constraint for Network Flow inference stage.
        linkIndexGraph: Adjacency matrix, where non-zero element indicates the index of transition edge, begins from 1.
        Returns: A_eq, b_eq, A_ub, b_ub, constraints used in LP
        """
        num_nodes = linkIndexGraph.shape[0]
        entry_offset, exit_offset, link_offset = num_nodes, num_nodes * 2, num_nodes * 3

        edge_ind = linkIndexGraph.max() #Number of transition edges in the flow graph
        num_constraints = link_offset + edge_ind
        A_eq = np.zeros((num_nodes * 2, num_constraints), dtype=np.int8) #Reduce memory cost. e.g. MOT20 dataset.
        b_eq = np.zeros((num_nodes * 2, 1), dtype=np.int8)
        A_ub = np.zeros((num_nodes * 2, num_constraints), dtype=np.int8)
        b_ub = np.ones((num_nodes * 2, 1), dtype=np.int8)
        eq_ind, leq_ind = 0, 0
        for node in range(linkIndexGraph.shape[0]):

            out_nodes = np.where(linkIndexGraph[node, :] != 0)[0]
            in_nodes = np.where(linkIndexGraph[:, node] != 0)[0]
            if out_nodes.shape[0] != 0 and in_nodes.shape[0] != 0:
                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in <= 1
                for in_node in in_nodes:
                    link_ind = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind -1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in == detection edge
                for in_node in in_nodes:
                    link_ind = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind - 1] = 1
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow coming out <= 1
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node]
                    constraint[link_offset + link_ind - 1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow out == detection edge
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node]
                    constraint[link_offset + link_ind - 1] = 1
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1
            elif out_nodes.shape[0] != 0 and in_nodes.shape[0] == 0:
                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow out <= 1
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node]
                    constraint[link_offset + link_ind - 1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in <= 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow out == detection edge
                for out_node in out_nodes:
                    link_ind = linkIndexGraph[node, out_node] #link_ind starts from 1
                    constraint[link_offset + link_ind - 1] = 1 #flow transition to next node
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint = np.zeros(num_constraints)        #flow in == detection edge
                constraint[entry_offset + node], constraint[node] = 1, -1 #flow coming from start node
                A_eq[eq_ind, :] = constraint
                eq_ind += 1
            elif out_nodes.shape[0] == 0 and in_nodes.shape[0] != 0:
                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in <= 1
                for in_node in in_nodes:
                    link_ind = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind - 1] = 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1 #flow out <= 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1 #flow in == detection edge
                for in_node in in_nodes:
                    link_ind = linkIndexGraph[in_node, node]
                    constraint[link_offset + link_ind - 1] = 1
                constraint[node] = -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint = np.zeros(num_constraints) #flow out == detection edge
                constraint[exit_offset + node], constraint[node] = 1, -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1
            elif out_nodes.shape[0] == 0 and in_nodes.shape[0] == 0:
                constraint, constraint[entry_offset + node] = np.zeros(num_constraints), 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint, constraint[exit_offset + node] = np.zeros(num_constraints), 1
                A_ub[leq_ind, :] = constraint
                leq_ind += 1

                constraint = np.zeros(num_constraints)
                constraint[entry_offset + node], constraint[node] = 1, -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1

                constraint = np.zeros(num_constraints)
                constraint[exit_offset + node], constraint[node] = 1, -1
                A_eq[eq_ind, :] = constraint
                eq_ind += 1    
        return A_eq, b_eq, A_ub, b_ub
    
    def recoverTracklets(self, curr_dets, sol, linkIndexGraph, prune_len=3):
        """
        recover tracklets based on first stage LP's output.
        sol: solution produced by LP solver(Gurobi) of dimension nx1, where top 3*num_dets are unary terms.
        linkIndexGraph: graph indicating which of the two detections have a connection.
        """
        num_nodes = linkIndexGraph.shape[0]
        start_points = np.where(sol[num_nodes:num_nodes * 2] == 1)[0] # detection, entry/exit link nodes.
        tracklets, tracklet_id = [], 0 
        for d in start_points:
            curr_tracklet = [d]
            curr_node = d
            out_nodes = np.where(linkIndexGraph[curr_node, :] != 0)[0]
            out_edge_inds = linkIndexGraph[curr_node, :][out_nodes]
            while len(out_edge_inds) != 0:
                make_link = False
                for edge_ind in out_edge_inds:
                    edge_ind = int(edge_ind)
                    #If this linke is active, then proceed to next node.
                    if sol[num_nodes*3:][edge_ind-1]:
                        make_link = True
                        next_node = np.where(linkIndexGraph[curr_node, :] == edge_ind)[0].item()
                        curr_tracklet.append(next_node)
                        break

                if make_link:
                    curr_node = next_node
                    out_nodes = np.where(linkIndexGraph[curr_node, :] != 0)[0]
                    out_edge_inds = linkIndexGraph[curr_node,:][out_nodes]
                else:
                    out_edge_inds = []

            if len(curr_tracklet) <= prune_len:
                #print('{}-th tracklet has length {}, skip this one'.format(d, len(curr_tracklet)) )
                continue
            else:  
                tracklet = []
                for i in curr_tracklet:
                    tracklet.append(curr_dets[i, 0:7])
                tracklet = np.array(tracklet)
                tracklet = np.concatenate([tracklet_id * np.ones((tracklet.shape[0], 1)),tracklet],axis=1)
                tracklets.append(tracklet) #tracklet:local_tracklet_id,frame,x1,y1,x2,y2,score,local_det_index
                tracklet_id += 1
        tracklets = np.concatenate(tracklets).astype(np.int)
        tracklets[:, [0, 1]] = tracklets[:, [1, 0]]
        tracklets = np.delete(tracklets, -2, axis=1) #frame,local_tracklet_id,x1,y1,x2,y2,local det index
        return tracklets
    
    def recoverClusteredTracklets(self, tracklets, assignment_list):
        """
        assignment_list: second stage tracklet clustering result. e.g. [[1,2,5],[3,4],[6]]
        tracklets: tracklets of the first stage tracking result.
        """
        tracks, track_id = [], 0
        for i in assignment_list:
            if len(i) == 1:
                tracks.append(np.concatenate([tracklets[tracklets[:, 1] == i, 0][:, None], 
                                              track_id * np.ones([(tracklets[:, 1] == i).sum(), 1]), 
                                              tracklets[tracklets[:, 1] == i, 2:6]], axis=1)) #fr,id,x1,y1,x2,y2                                 
            else:
                uninterp_tracklets = []
                for j in i:
                    uninterp_tracklets.append(np.concatenate([tracklets[tracklets[:, 1] == j, 0][:, None], 
                                                              tracklets[tracklets[:, 1] == j, 2:6]], axis=1))
                                                              #append clustered tracklets together
                uninterp_tracklets = np.concatenate(uninterp_tracklets) #tracklets without interpolation yet
                interp_list = [] #if list stores array， then can only np.concatenate
                for ind in range(uninterp_tracklets.shape[0]-1): 
                    #uninterp_tracklets format: frame,x1,y1,x2,y2
                    if uninterp_tracklets[ind+1][0] - uninterp_tracklets[ind][0] > 1:
                        start_node, end_node = uninterp_tracklets[ind], uninterp_tracklets[ind+1]
                        start_frame, end_frame = int(start_node[0]), int(end_node[0])
                        for interp_frame in range(start_frame+1, end_frame):
                            tmp = start_node + ((end_node-start_node)/(end_frame-start_frame))*(interp_frame-start_frame) 
                            interp_list.append(tmp) 
                if len(interp_list) != 0:
                    #interp_tracklets is the interpolated tracklet, since a track can fragmented into multiple tracklets
                    interp_tracklets = np.concatenate([uninterp_tracklets, np.array(interp_list)], axis=0)
                else:
                    interp_tracklets = uninterp_tracklets
                interp_tracklets = interp_tracklets[np.argsort(interp_tracklets[:, 0])]
                interp_tracklets = interp_tracklets.astype(np.int)
                interp_tracklets = np.concatenate([track_id*np.ones([interp_tracklets.shape[0], 1]), interp_tracklets], axis=1)
                interp_tracklets[:, [0, 1]] = interp_tracklets[:, [1, 0]] #frame,id,xmin,ymin,xmax,ymax
                tracks.append(interp_tracklets)
            track_id += 1
        tracks = np.concatenate(tracks).astype(np.int) #tracks in the current batch
        return tracks
    
    def buildConstraintTracklet(self, tracklets):
        """
        build constraint for tracklet stitching, for handling long-term occlusion.
        tracklets: frame, id, x, y, w, h format.
        """
        num_tracklets = np.unique(tracklets[:, 1]).shape[0]
        linkIndexGraphTracklet = np.zeros((num_tracklets, num_tracklets), dtype=np.int32)
        edge_ind = 0
        for src_id in range(num_tracklets):
            for dst_id in range(num_tracklets):
                #print('tracklet %d and %d'%(src_id,dst_id))
                src_tracklet = tracklets[tracklets[:, 1] == src_id, :]
                dst_tracklet = tracklets[tracklets[:, 1] == dst_id, :]
                if src_tracklet[:, 0].max() < dst_tracklet[:, 0].min():
                    #print('tracklet %d ends at %d and %d starts at %d'%(src_id,src_tracklet[:, 0].max(),dst_id,dst_tracklet[:, 0].min()))
                    edge_ind += 1
                    linkIndexGraphTracklet[src_id, dst_id] = edge_ind

        A_eq, b_eq, A_ub, b_ub = self.buildConstraint(linkIndexGraphTracklet)  
        return linkIndexGraphTracklet, A_eq, b_eq, A_ub, b_ub
    
    def clusterTracklets(self, tracklets, curr_app_feat_norm, dist_thresh, app_thresh):
        mean_app_feats = []
        for i in np.unique(tracklets[:, 1]):
            tracklet = tracklets[tracklets[:, 1] == i]
            inds = tracklet[:, -1] # retrieving relavent detections.
            mean_app_feat = curr_app_feat_norm[inds].mean(axis=0)
            mean_app_feats.append(mean_app_feat)    
        mean_app_feats = np.array(mean_app_feats)
        mean_app_feats /= np.linalg.norm(mean_app_feats, axis=1, keepdims=True)
        linkIndexGraphTracklet, A_eq, b_eq, A_ub, b_ub = self.buildConstraintTracklet(tracklets)
        edge_cost, num_tracklets = [], linkIndexGraphTracklet.shape[0]

        for i in range(num_tracklets):
            for j in range(num_tracklets):
                if linkIndexGraphTracklet[i,j] != 0:
                    src_tracklet, dst_tracklet = tracklets[tracklets[:, 1] == i, :], tracklets[tracklets[:, 1] == j, :]
                    src_centers = ((src_tracklet[:, 2:4] + src_tracklet[:, 4:6]) / 2).astype(int)
                    dst_centers = ((dst_tracklet[:, 2:4] + dst_tracklet[:, 4:6]) / 2).astype(int)
                    src_vel = (src_centers[1:src_centers.shape[0]] - src_centers[0:-1]).mean(axis=0)
                    dst_vel = (dst_centers[1:dst_centers.shape[0]] - dst_centers[0:-1]).mean(axis=0)
                    vel_diff = np.linalg.norm(src_vel - dst_vel)

                    frame_gap = dst_tracklet[0, 0] - src_tracklet[-1, 0] #always assume dst_tracklet appears after src_tracklet
                    estimated_pos = src_centers[-1] + frame_gap * src_vel

                    dist = np.linalg.norm(estimated_pos - dst_centers[0])
                    if dist > dist_thresh: #The two tracklets are unlikely to be joined
                        edge_cost.append(50)
                    else:
                        if np.dot(mean_app_feats[i], mean_app_feats[j]) > app_thresh:
                            edge_cost.append(1 - np.dot(mean_app_feats[i], mean_app_feats[j]))
                        else:
                            edge_cost.append(50)

        trackletCost = np.concatenate([-5 * np.ones(num_tracklets), np.ones(num_tracklets), 
                                       np.ones(num_tracklets), np.array(edge_cost)])

        tracklet_sol = self.linprog(trackletCost, A_eq, b_eq, A_ub, b_ub)
        feature_list, assignment_list = [], [] #cluster tracklets in current batch
        num_nodes = int(A_eq.shape[0] / 2)
        start_points = np.where(tracklet_sol[num_tracklets:num_tracklets * 2] == 1)[0]
        for d in start_points:
            curr_tracklet = [d]
            curr_node = d
            out_nodes = np.where(linkIndexGraphTracklet[curr_node, :] != 0)[0]
            out_edge_inds = linkIndexGraphTracklet[curr_node,:][out_nodes]
            while len(out_edge_inds) != 0:
                make_link = False
                for edge_ind in out_edge_inds:
                    edge_ind = int(edge_ind)
                    if tracklet_sol[num_nodes*3:][edge_ind-1]:
                        make_link = True
                        next_node = np.where(linkIndexGraphTracklet[curr_node, :] == edge_ind)[0].item()
                        curr_tracklet.append(next_node)
                        break
                if make_link:
                    curr_node = next_node
                    out_nodes = np.where(linkIndexGraphTracklet[curr_node, :] != 0)[0]
                    out_edge_inds = linkIndexGraphTracklet[curr_node,:][out_nodes]
                else:
                    out_edge_inds = []
            tracklet = []
            for i in curr_tracklet:
                tracklet.append(i)
            assignment_list.append(tracklet)
            feature_list.append(mean_app_feats[np.array(tracklet)].mean(axis=0))

        return assignment_list, feature_list
    
    def clusterSkipTracklets(self, tracklets, curr_app_feats_norm, dist_thresh, app_thresh):
        
        tracklets_ = np.delete(tracklets, -1, axis=1)
        interp_tracklets = interpolateTracks(tracklets_)
        mean_app_feats = []
        for i in np.unique(tracklets[:, 1]):
            tracklet = tracklets[tracklets[:, 1] == i]
            det_inds = tracklet[:, -1] # retrieving relavent detections.
            mean_app_feat = curr_app_feats_norm[det_inds].mean(axis=0)
            mean_app_feats.append(mean_app_feat)    
        mean_app_feats = np.array(mean_app_feats)
        mean_app_feats /= np.linalg.norm(mean_app_feats, axis=1, keepdims=True)
        linkIndexGraphTracklet, A_eq, b_eq, A_ub, b_ub = self.buildConstraintTracklet(tracklets)
        num_tracklets = linkIndexGraphTracklet.shape[0]
        edge_cost = []
        for i in range(num_tracklets):
            for j in range(num_tracklets):
                if linkIndexGraphTracklet[i,j] != 0:
                    src_tracklet = interp_tracklets[interp_tracklets[:, 1] == i, :]
                    dst_tracklet = interp_tracklets[interp_tracklets[:, 1] == j, :]
                    src_centers = ((src_tracklet[:, 2:4] + src_tracklet[:, 4:6])/2).astype(int) 
                    dst_centers = ((dst_tracklet[:, 2:4] + dst_tracklet[:, 4:6])/2).astype(int)
                    src_vel = (src_centers[1:src_centers.shape[0]] - src_centers[0:-1]).mean(axis=0)
                    dst_vel = (dst_centers[1:dst_centers.shape[0]] - dst_centers[0:-1]).mean(axis=0)
                    vel_diff = np.linalg.norm(src_vel - dst_vel)

                    frame_gap = dst_tracklet[0, 0] - src_tracklet[-1, 0] #always assume dst_tracklet appears after src_tracklet
                    estimated_pos = src_centers[-1] + frame_gap * src_vel

                    dist = np.linalg.norm(estimated_pos - dst_centers[0])
                    if dist > dist_thresh: #The two tracklets are unlikely to be joined
                        edge_cost.append(50)
                    else:
                        if np.dot(mean_app_feats[i], mean_app_feats[j]) > app_thresh:
                            edge_cost.append(1 - np.dot(mean_app_feats[i], mean_app_feats[j]))
                        else:
                            edge_cost.append(50)

        trackletCost = np.concatenate([-5 * np.ones(num_tracklets), np.ones(num_tracklets), 
                                           np.ones(num_tracklets), np.array(edge_cost)])

        tracklet_sol = self.linprog(trackletCost, A_eq, b_eq, A_ub, b_ub)
        feature_list, assignment_list = [], [] #cluster tracklets in current batch
        
        num_nodes = int(A_eq.shape[0]/2)
        start_points = np.where(tracklet_sol[num_tracklets:num_tracklets * 2] == 1)[0]
        for d in start_points:
            curr_tracklet = [d]
            curr_node = d
            out_nodes = np.where(linkIndexGraphTracklet[curr_node, :] != 0)[0]
            out_edge_inds = linkIndexGraphTracklet[curr_node,:][out_nodes]
            while len(out_edge_inds) != 0:
                make_link = False
                for edge_ind in out_edge_inds:
                    edge_ind = int(edge_ind)
                    if tracklet_sol[num_nodes*3:][edge_ind-1]:
                        make_link = True
                        next_node = np.where(linkIndexGraphTracklet[curr_node, :] == edge_ind)[0].item()
                        curr_tracklet.append(next_node)
                        break
                if make_link:
                    curr_node = next_node
                    out_nodes = np.where(linkIndexGraphTracklet[curr_node, :] != 0)[0]
                    out_edge_inds = linkIndexGraphTracklet[curr_node,:][out_nodes]
                else:
                    out_edge_inds = []
            tracklet = []
            for i in curr_tracklet:
                tracklet.append(i)
            assignment_list.append(tracklet)
            feature_list.append(mean_app_feats[np.array(tracklet)].mean(axis=0))
        return assignment_list, feature_list
    
    def mergeTracklets(self, tracks_list, features_list):
        global_assignment = dict() #This saves the tracklet(node) labeling in each batch.
        thresh = 6
        for batch_ind in range(len(tracks_list)-1):
            #print('Batch {} and {}'.format(batch_ind, batch_ind+1))
            src_tracklets, dst_tracklets = tracks_list[batch_ind], tracks_list[batch_ind+1]
            num_src_tracklets = np.unique(src_tracklets[:, 1]).shape[0]
            num_dst_tracklets = np.unique(dst_tracklets[:, 1]).shape[0]
            if batch_ind == 0:
                maxID = num_src_tracklets
                IDSOld = np.unique(src_tracklets[:, 1])
                global_assignment[batch_ind] = IDSOld

            avg_spatial_dist = np.zeros([num_src_tracklets, num_dst_tracklets])
            avg_app_dist = np.zeros([num_src_tracklets, num_dst_tracklets])
            vel_spatial_dist = np.zeros([num_src_tracklets, num_dst_tracklets])
            IDSNew = -1 * np.ones(num_dst_tracklets)
            for src_ind in range(num_src_tracklets):
                for dst_ind in range(num_dst_tracklets):
                    src_tracklet = src_tracklets[src_tracklets[:, 1] == src_ind, :] #frame,id,x1,y1,x2,y2
                    dst_tracklet = dst_tracklets[dst_tracklets[:, 1] == dst_ind, :]
                    src_frames, dst_frames = src_tracklet[:, 0], dst_tracklet[:, 0]

                    src_centers = ((src_tracklet[:, 2:4] + src_tracklet[:, 4:6])/2).astype(np.int)
                    dst_centers = ((dst_tracklet[:, 2:4] + dst_tracklet[:, 4:6])/2).astype(np.int)
                    src_vel = (src_centers[1:src_centers.shape[0]] - src_centers[0:-1]).mean(axis=0)
                    frame_gap = dst_tracklet[0, 0] - src_tracklet[-1, 0]
                    estimated_pos = src_centers[-1] + frame_gap * src_vel
                    dist = np.linalg.norm(estimated_pos - dst_centers[0])
                    vel_spatial_dist[src_ind,dst_ind] = dist

                    intersect_frames = np.intersect1d(src_frames, dst_frames) #Retrive tracklet coordinates in overlapping frames 
                    if max(src_frames) < min(dst_frames) or intersect_frames.shape[0] == 0:
                        avg_spatial_dist[src_ind, dst_ind] = 300
                    else:
                        src_ind_ = np.logical_and(src_tracklet[:, 0] >= intersect_frames.min(), 
                                                  src_tracklet[:, 0] <= intersect_frames.max())
                        dst_ind_ = np.logical_and(dst_tracklet[:, 0] >= intersect_frames.min(), 
                                                  dst_tracklet[:, 0] <= intersect_frames.max())

                        coorOld = src_tracklet[src_ind_, 2:4] + src_tracklet[src_ind_, 4:6] / 2
                        coorNew = dst_tracklet[dst_ind_, 2:4] + dst_tracklet[dst_ind_, 4:6] / 2
                        overlap = np.linalg.norm(coorOld - coorNew) / intersect_frames.shape[0]
                        avg_spatial_dist[src_ind, dst_ind] = overlap
                    avg_app_dist[src_ind,dst_ind]=1-np.dot(features_list[batch_ind][src_ind],
                                                           features_list[batch_ind+1][dst_ind])

            inds = np.argmin(avg_spatial_dist, axis=0)
            matched_old = []
            for col_ind in range(inds.shape[0]):
                row_ind = inds[col_ind]
                if avg_spatial_dist[row_ind, col_ind] < thresh:
                    IDSNew[col_ind] = global_assignment[batch_ind][row_ind] #Stitch two tracklets
                    #print('track {} and {} combined with dist {}'.format(row_ind, col_ind, dist))
                    matched_old.append(row_ind)
                else:
                    ind = np.argmin(avg_app_dist[:, col_ind])
                    app_dist = avg_app_dist[ind][col_ind]
                    #print('track {} and {} combined with appear cost {:.3f}'.format(tmp, col_ind, app_dist))
                    if vel_spatial_dist[ind][col_ind] < 100 and app_dist < 0.1 and ind not in matched_old:
                        #print('seg {} track {} and {} track {}, app {}'.format(segmentInd, index, segmentInd+1, col_ind, app_dist))
                        IDSNew[col_ind] = global_assignment[batch_ind][ind]
                        matched_old.append(ind)
                    else:
                        IDSNew[col_ind] = maxID #Global track id
                        maxID += 1
            global_assignment[batch_ind+1] = IDSNew

        tracks_dict = dict()
        for batch_ind in range(len(tracks_list)):
            curr_tracks = tracks_list[batch_ind]
            for ind in range(global_assignment[batch_ind].shape[0]):
                track_id = global_assignment[batch_ind][ind]
                track_id = int(track_id)
                if not track_id in tracks_dict.keys():
                    tracks_dict[track_id] = [];
                tracks_dict[track_id].append(curr_tracks[curr_tracks[:, 1] == ind, :])

        final_tracks = []
        for k in tracks_dict.keys():
            curr_track = np.concatenate(tracks_dict[k])
            curr_track = np.concatenate([curr_track[:, 0][:,None], curr_track[:, 2:6]], axis=1)
            curr_track = curr_track[np.argsort(curr_track[:, 0])]

            interpTrack, Track = [], []
            for ind in range(curr_track.shape[0]-1):
                curr_frame, next_frame = curr_track[ind][0], curr_track[ind+1][0]        
                bbox = curr_track[curr_track[:, 0] == curr_frame, 0:5]

                if bbox.shape[0] == 1 and next_frame - curr_frame > 1:
                    Track.append(bbox.squeeze())
                    start_node, end_node = curr_track[ind], curr_track[ind+1] 
                    start_frame, end_frame = start_node[0], end_node[0]
                    for frame in range(start_frame+1, end_frame):
                        tmp = start_node + ((end_node-start_node)/(end_frame-start_frame))*(frame-start_frame) 
                        interpTrack.append(tmp)
                elif bbox.shape[0] == 1 and next_frame - curr_frame == 1:
                    Track.append(bbox.squeeze())
                    continue
                else:
                    if curr_frame == curr_track[ind-1][0]:
                        continue
                    else:
                        Track.append(np.mean(bbox, axis=0).squeeze())
            if not np.all(curr_track[ind] == curr_track[ind+1]):
                Track.append(curr_track[-1, 0:5])

            if len(interpTrack) != 0:
                Track = np.concatenate([np.array(Track), np.array(interpTrack)], axis=0).astype(np.int)
            else:
                Track = np.array(Track).astype(np.int)
            Track = Track[np.argsort(Track[:, 0])]
            Track = np.concatenate([k*np.ones(Track.shape[0])[:,None], Track], axis=1)
            final_tracks.append(Track)

        final_tracks = np.concatenate(final_tracks)
        final_tracks[:, 4:6] = final_tracks[:, 4:6] - final_tracks[:, 2:4] # xmin,ymin,xmax,ymax to xmin,ymin,w,h
        final_tracks[:, [0, 1]] = final_tracks[:, [1, 0]] #track_id, frame to frame, track_id
        final_tracks[:, 1] += 1                           #since frame index is 1 based
        final_tracks = final_tracks[np.argsort(final_tracks[:, 0]), :] #Convert to matlab evaluation format
        final_tracks = np.concatenate([final_tracks,-1*np.ones((final_tracks.shape[0], 4))],axis=1).astype(np.int)
        return final_tracks
