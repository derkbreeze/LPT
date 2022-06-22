import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gurobipy as gp

def make_gurobi_model_tracking(G, h, A, b, Q,test=False):
    '''
    Convert to Gurobi model. Copied from 
    https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/gurobi_.py
    '''
    vtype = gp.GRB.CONTINUOUS# gp.GRB.BINARY if test else gp.GRB.CONTINUOUS
    n = A.shape[1] if A is not None else G.shape[1]
    model = gp.Model()
    model.params.OutputFlag = 0
    x = [model.addVar(vtype= vtype, name='x_%d'%i,lb=0, ub=1) for i in range(n)]
    model.update()   # integrate new variables

    # subject to
    #     G * x <= h
    inequality_constraints = []
    if G is not None:
        for i in range(G.shape[0]):
            row = np.where(G[i] != 0)[0]
            inequality_constraints.append(model.addConstr(gp.quicksum(G[i,j]*x[j] for j in row) <= h[i]))

    # subject to
    #     A * x == b
    equality_constraints = []
    if A is not None:
        for i in range(A.shape[0]):
            row = np.where(A[i] != 0)[0]
            equality_constraints.append(model.addConstr(gp.quicksum(A[i,j]*x[j] for j in row) == b[i]))

    obj = gp.QuadExpr()
    if Q is not None:
        rows, cols = Q.nonzero()
        for i, j in zip(rows, cols):
            obj += x[i]*Q[i, j]*x[j]

    return model, x, inequality_constraints, equality_constraints, obj

def _remove_redundant_rows(A_eq):
    # remove redundant (linearly dependent) rows from equality constraints
    n_rows_A = A_eq.shape[0]
    redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                          "improve performance, check the problem formulation "
                          "for redundant equality constraints.")
    small_nullspace = 5
    if  A_eq.size > 0:
        try:  # TODO: instead use results of first SVD in _remove_redundancy
            rank = np.linalg.matrix_rank(A_eq)
        except Exception:  # oh well, we'll have to go with _remove_redundancy_dense
            rank = 0
    if A_eq.size > 0 and rank < A_eq.shape[0]:
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        dim_row_nullspace = A_eq.shape[0]-rank
        if dim_row_nullspace <= small_nullspace:
            d_removed,  status, message = _remove_redundancy(A_eq)
        if dim_row_nullspace > small_nullspace :
            d_removed,  status, message = _remove_redundancy_dense(A_eq)
        if A_eq.shape[0] < rank:
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            status = 4
        if status != 0:
            complete = True
    if np.linalg.matrix_rank(A_eq) == A_eq.shape[0]:
        return None
    else:
        return d_removed

def build_constraint_torch(A_eq, b_eq, A_ub, b_ub):
    rows_to_be_removed = _remove_redundant_rows(A_eq)
    if rows_to_be_removed is not None:
        A_eq = np.delete(A_eq, rows_to_be_removed, axis=0)
        A = torch.from_numpy(A_eq).float()
        b_eq = np.delete(b_eq, rows_to_be_removed, axis=0)
        b = torch.from_numpy(b_eq).float()
    else:
        A = torch.from_numpy(A_eq).float()
        b = torch.from_numpy(b_eq).float()
    b = b.squeeze() #convert size nx1 to n
    G = torch.from_numpy(A_ub).float()
    h = torch.from_numpy(b_ub).float().squeeze()
    return A, b, G, h

def build_constraint(data, maxFrameGap):
    """
    Args:
    data: torch_geometric.data.Data
    maxFrameGap: maximal frame gap used to connect two detections across frames. Set to 1 during training
    
    returns: A_eq, b_eq, A_ub, b_ub: equality and in-equality constraints in linear program
             x_gt: ground truth annotation of person tracks
             tran_indicator: indicating which parts of the edges(among temporal fully-connected edges) are selected
    """
    num_nodes = data.x.shape[0]
    edges = data.edge_index.t().numpy()
    timestamps = data.ground_truth[:, 0].astype(int)
    entry_offset, exit_offset, link_offset = num_nodes, num_nodes * 2, num_nodes * 3
    entry_indicator = np.zeros((num_nodes, 1), dtype=np.int32)      #Indicating which detections are selected as gt start 
    exit_indicator = np.zeros((num_nodes, 1), dtype=np.int32)       #Indicating which detections are selected as gt terminal
    tran_indicator = np.zeros((edges.shape[0], 1), dtype=np.int32)

    linkIndexGraph = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    gtlinkIndexGraph = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    linkVector = data.y.numpy()
    edge_ind = 0
    for i in range(linkIndexGraph.shape[0]):
        for j in range(linkIndexGraph.shape[0]):
            frameGap = timestamps[j] - timestamps[i]
            if frameGap >= 1 and frameGap <= maxFrameGap:        
                inds = np.logical_and(edges[:, 0] == i, edges[:, 1] == j) #bool np.array, array([False, True, False...])
                if inds.sum() != 0:
                    edge_ind += 1
                    ind = np.where(inds == True)[0][0] #array([1]) or so, so add one [0]
                    tran_indicator[ind] = 1 #indicating this transition edge has been selected
                    linkIndexGraph[i, j] = edge_ind
                    gtlinkIndexGraph[i, j] = linkVector[ind]
                
    assert (linkIndexGraph.flatten() != 0).sum() == edge_ind, 'Shape Mismatch'
    assert tran_indicator.sum() == edge_ind, 'Shape Mismatch'
    num_constraints = link_offset + edge_ind
    for i in range(linkIndexGraph.shape[0]):
        incoming_flows = edges[:, 1] == i
        if not np.any(incoming_flows):
            start_node = i
            entry_indicator[i] = 1
            #print('starting from node {}'.format(i))
            while np.any(gtlinkIndexGraph[i, :]):
                i = np.where(gtlinkIndexGraph[i, :] == 1)[0][0]
                #print('proceeding at node {}'.format(i))
            terminate_node = i
            if terminate_node != start_node:
                exit_indicator[terminate_node] = 1
                #print('terminates at node {}'.format(terminate_node))
            else:
                entry_indicator[i] = 0
    assert entry_indicator.shape == exit_indicator.shape, "shape mismatch"
    assert entry_indicator.sum() ==  exit_indicator.sum(), "GT flow in != flow out" 

    #Initialize the constraint matrices
    x_gt = np.zeros((edge_ind, 1), dtype=np.float32)
    A_ub = np.zeros((num_nodes * 2, num_constraints), dtype=np.float32)
    b_ub = np.zeros((num_nodes * 2, 1), dtype=np.float32)
    leq_index = 0
    A_eq = np.zeros((num_nodes * 2, num_constraints), dtype=np.float32)
    b_eq = np.zeros((num_nodes * 2, 1), dtype=np.float32)
    eq_index = 0

    for node in range(linkIndexGraph.shape[0]):
        out_nodes = np.where(linkIndexGraph[node, :] != 0)[0]
        in_nodes = np.where(linkIndexGraph[:, node] != 0)[0]
        # Note that linkIndex starts at 1, so minus 1 at certain indexes
        if out_nodes.shape[0] != 0:
            for out_node in out_nodes:
                link_index = linkIndexGraph[node, out_node]
                x_gt[link_index - 1] = gtlinkIndexGraph[node, out_node]
        if in_nodes.shape[0] != 0:
            for in_node in in_nodes:
                link_index = linkIndexGraph[in_node, node]
                x_gt[link_index - 1] = gtlinkIndexGraph[in_node, node]

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

    det_indicator = np.ones((entry_offset, 1))
    x_gt = np.concatenate([det_indicator, entry_indicator, exit_indicator, x_gt], axis=0)

    return A_eq, b_eq, A_ub, b_ub, x_gt, tran_indicator