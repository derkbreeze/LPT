import numpy as np
import gurobipy as gp
from numba import jit

def forwardLP(c, A_eq, b_eq, A_ub, b_ub):
    """
    Perform Linear Program inference step
    c: LP learned cost function, numpy.ndarray, shape: (n, ) where n is the problem size
    A_eq, b_eq: equality constraint(flow conservation), numpy.ndarray, shape: (n_equality_constraints, n)
    A_ub, b_ub: inequality constraint, numpy.ndarray, shape: (n_inequality_constraints, n)
    return: LP solution.
    """
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    x = model.addMVar(shape=A_eq.shape[1], vtype=gp.GRB.BINARY, name='x')
    model.setObjective(c @ x, gp.GRB.MINIMIZE)
    model.addConstr(A_eq @ x==b_eq.squeeze(), name='eq')
    model.addConstr(A_ub @ x<=b_ub.squeeze(), name='ineq')
    model.optimize()
    if model.status==2:
        sol = x.X
    else:
        print('error in solver')
        sol = None
    return sol

@jit
def generateGraph(curr_dets, maxFrameGap=1): 
    """
    curr_det, detections in current batch. format: frame, xmin,ymin,w,h
    """
    edge_index = 0
    linkIndexGraph = np.zeros((curr_dets.shape[0], curr_dets.shape[0]), dtype=np.int32)
    for i in range(linkIndexGraph.shape[0]):
        for j in range(linkIndexGraph.shape[0]):
            frameGap = curr_dets[j][0] - curr_dets[i][0]
            if frameGap == maxFrameGap:
                edge_index += 1
                linkIndexGraph[i, j] = edge_index
    return linkIndexGraph

def buildConstraint(linkIndexGraph):
    """
    build constraint for network flow linear program.
    linkIndexGraph: Adjacency matrix, where non-zero element indicates the index of transition edge, starts from 1.
    """
    num_nodes = linkIndexGraph.shape[0]
    entry_offset, exit_offset, link_offset = num_nodes, num_nodes * 2, num_nodes * 3
    
    edge_index = linkIndexGraph.max() #Number of transition edges in the flow graph
    num_constraints = link_offset + edge_index   
    A_eq = np.zeros([num_nodes * 2, num_constraints], dtype=np.int8) #Reduce memory consumption for large problems. e.g. MOT20.
    b_eq = np.zeros([num_nodes * 2, 1], dtype=np.int8)
    eq_index = 0
       
    A_ub = np.zeros([num_nodes * 2, num_constraints], dtype=np.int8)
    b_ub = np.zeros([num_nodes * 2, 1], dtype=np.int8)
    leq_index = 0

    for node in range(linkIndexGraph.shape[0]):

        out_nodes = np.where(linkIndexGraph[node, :] != 0)[0]
        in_nodes = np.where(linkIndexGraph[:, node] != 0)[0]
        if out_nodes.shape[0] != 0 and in_nodes.shape[0] != 0:
            ## Flow coming in <= 1
            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node] = 1
            for in_node in in_nodes:
                link_index = linkIndexGraph[in_node, node]
                constraint[link_offset + link_index -1] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

            # Flow coming in == detection edge
            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node] = 1
            for in_node in in_nodes:
                link_index = linkIndexGraph[in_node, node]
                constraint[link_offset + link_index - 1] = 1
            constraint[node] = -1
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1

            ## Flow coming out <= 1
            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node] = 1
            for out_node in out_nodes:
                link_index = linkIndexGraph[node, out_node]
                constraint[link_offset + link_index - 1] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

             # Flow coming out == detection edge
            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node] = 1
            for out_node in out_nodes:
                link_index = linkIndexGraph[node, out_node]
                constraint[link_offset + link_index - 1] = 1
            constraint[node] = -1
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1
        elif out_nodes.shape[0] != 0 and in_nodes.shape[0] == 0:
            ## Flow coming out <= 1
            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node] = 1
            for out_node in out_nodes:
                link_index = linkIndexGraph[node, out_node]
                constraint[link_offset + link_index - 1] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

            ## Flow coming in <= 1
            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

            # Flow coming out == detection edge
            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node] = 1 #flow going to terminal node
            for out_node in out_nodes:
                link_index = linkIndexGraph[node, out_node] #link_index starts from 1, so.
                constraint[link_offset + link_index - 1] = 1 #flow transition to next node
            constraint[node] = -1
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1

             # Flow coming in == detection edge
            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node], constraint[node] = 1, -1 #flow coming from start node
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1
        elif out_nodes.shape[0] == 0 and in_nodes.shape[0] != 0:
            ## Flow coming in <= 1
            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node] = 1
            for in_node in in_nodes:
                link_index = linkIndexGraph[in_node, node]
                constraint[link_offset + link_index - 1] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

            ## Flow coming out <= 1
            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

             # Flow coming in == detection edge
            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node] = 1
            for in_node in in_nodes:
                link_index = linkIndexGraph[in_node, node]
                constraint[link_offset + link_index - 1] = 1
            constraint[node] = -1
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1

             # Flow coming out == detection edge
            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node], constraint[node] = 1, -1
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1
        elif out_nodes.shape[0] == 0 and in_nodes.shape[0] == 0:
            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node] = 1
            A_ub[leq_index, :], b_ub[leq_index] = constraint, 1
            leq_index += 1

            constraint = np.zeros(num_constraints)
            constraint[entry_offset + node], constraint[node] = 1, -1
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1

            constraint = np.zeros(num_constraints)
            constraint[exit_offset + node], constraint[node] = 1, -1
            A_eq[eq_index, :], b_eq[eq_index] = constraint, 0
            eq_index += 1    
    return A_eq, b_eq, A_ub, b_ub

def buildConstraintTracklet(tracklets):
    """
    build constraint for tracklet stitching.
    tracklets: frame, id, x, y, w, h maybe, needs to be confirmed.
    """
    num_tracklets = np.unique(tracklets[:, 1]).shape[0]
    linkIndexGraphTracklet = np.zeros((num_tracklets, num_tracklets), dtype=np.int32)
    edge_index = 0
    for src_id in range(num_tracklets):
        for dst_id in range(num_tracklets):
            #print('tracklet %d and %d'%(src_id,dst_id))
            src_tracklet = tracklets[tracklets[:, 1] == src_id, :]
            dst_tracklet = tracklets[tracklets[:, 1] == dst_id, :]
            if src_tracklet[:, 0].max() < dst_tracklet[:, 0].min():
                #print('tracklet %d ends at %d and %d starts at %d'%(src_id,src_tracklet[:, 0].max(),dst_id,dst_tracklet[:, 0].min()))
                edge_index += 1
                linkIndexGraphTracklet[src_id, dst_id] = edge_index

    A_eq, b_eq, A_ub, b_ub = buildConstraint(linkIndexGraphTracklet)  
    return linkIndexGraphTracklet, A_eq, b_eq, A_ub, b_ub

def clusterTracklets(tracklets, curr_app_feat_norm, dist_thres, app_thres):
    
    mean_app_feats = []
    for i in np.unique(tracklets[:, 1]):
        tracklet = tracklets[tracklets[:, 1] == i]
        inds = tracklet[:, -1] # retrieving relavent detections.
        mean_app_feat = curr_app_feat_norm[inds].mean(axis=0)
        mean_app_feats.append(mean_app_feat)    
    mean_app_feats = np.array(mean_app_feats)
    mean_app_feats /= np.linalg.norm(mean_app_feats, axis=1, keepdims=True)
    linkIndexGraphTracklet, A_eq, b_eq, A_ub, b_ub = buildConstraintTracklet(tracklets)
    num_tracklets = linkIndexGraphTracklet.shape[0]
    edge_cost = []
    
    for i in range(num_tracklets):
        for j in range(num_tracklets):
            if linkIndexGraphTracklet[i,j] != 0:
                src_tracklet = tracklets[tracklets[:, 1] == i, :]
                dst_tracklet = tracklets[tracklets[:, 1] == j, :]
                src_centers = ((src_tracklet[:, 2:4] + src_tracklet[:, 4:6])/2).astype(int)
                dst_centers = ((dst_tracklet[:, 2:4] + dst_tracklet[:, 4:6])/2).astype(int)
                src_vel = (src_centers[1:src_centers.shape[0]] - src_centers[0:-1]).mean(axis=0)
                dst_vel = (dst_centers[1:dst_centers.shape[0]] - dst_centers[0:-1]).mean(axis=0)
                vel_diff = np.linalg.norm(src_vel - dst_vel)
                
                frame_gap = dst_tracklet[0, 0] - src_tracklet[-1, 0] #always assume dst tracklet happens after src
                estimated_pos = src_centers[-1] + frame_gap * src_vel
                
                dist = np.linalg.norm(estimated_pos - dst_centers[0])
                if dist > dist_thres:
                    edge_cost.append(50)
                else:
                    if np.dot(mean_app_feats[i], mean_app_feats[j]) > app_thres:
                        edge_cost.append(1 - np.dot(mean_app_feats[i], mean_app_feats[j]))
                    else:
                        edge_cost.append(50)
            
    trackletCost = np.concatenate([-5 * np.ones(num_tracklets),np.ones(num_tracklets), 
                                   np.ones(num_tracklets),np.array(edge_cost)])

    tracklet_sol = forwardLP(trackletCost, A_eq, b_eq, A_ub, b_ub)
    assignment_list = [] #cluster tracklets in current batch
    feature_list = []
    num_nodes = int(A_eq.shape[0]/2)
    startPoints = np.where(tracklet_sol[num_tracklets:num_tracklets * 2] == 1)[0]
    for d in startPoints:
        curr_tracklet = [d]
        curr_node = d
        out_nodes = np.where(linkIndexGraphTracklet[curr_node, :] != 0)[0]
        out_edge_inds = linkIndexGraphTracklet[curr_node,:][out_nodes]
        while len(out_edge_inds) != 0:
            madeLink = False
            for edge_ind in out_edge_inds:
                edge_ind = int(edge_ind)
                if tracklet_sol[num_nodes*3:][edge_ind-1]:
                    madeLink = True
                    next_node = np.where(linkIndexGraphTracklet[curr_node, :] == edge_ind)[0].item()
                    curr_tracklet.append(next_node)
                    break
            if madeLink:
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
