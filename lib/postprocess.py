import numpy as np

def recoverTracklets(curr_dets, sol, linkIndexGraph, prune_len=3):
    """
    recover tracklets based on first stage LP's output.
    sol: solution produced by Linear Program solver(Gurobi), nx1, where top 3*num_dets are unary costs.
    linkIndexGraph: a graph structure indicating which of the two nodes are connected.
    """
    num_nodes = linkIndexGraph.shape[0]
    startPoints = np.where(sol[num_nodes:num_nodes*2] == 1)[0] # detection, entry/exit link nodes.
    tracklets = []
    tracklet_id = 0
    for d in startPoints:
        curr_tracklet = [d]
        curr_node = d
        out_nodes = np.where(linkIndexGraph[curr_node, :] != 0)[0]
        out_edge_inds = linkIndexGraph[curr_node, :][out_nodes]
        while len(out_edge_inds) != 0:
            madeLink = False
            for edge_ind in out_edge_inds:
                edge_ind = int(edge_ind)
                
                #If this linke is active, then proceed to next node.
                if sol[num_nodes*3:][edge_ind-1]:
                    madeLink = True
                    next_node = np.where(linkIndexGraph[curr_node, :] == edge_ind)[0].item()
                    curr_tracklet.append(next_node)
                    break
                    
            if madeLink:
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
            tracklet = np.concatenate([tracklet_id*np.ones((tracklet.shape[0], 1)),tracklet],axis=1)
            tracklets.append(tracklet) #tracklet:local tracklet id,frame,x1,y1,x2,y2,score,local det_index
            tracklet_id += 1

    tracklets = np.concatenate(tracklets)
    tracklets = tracklets.astype(np.int)
    tracklets[:, [0, 1]] = tracklets[:, [1, 0]]
    tracklets = np.delete(tracklets, -2, axis=1) #frame,local tracklet id,x1,y1,x2,y2,local det index
    return tracklets

def recoverClusteredTracklets(tracklets, assignment_list):
    
    """
    assignment_list: second stage tracklet clustering result. e.g. [[1,2,5],[3,4],[6]]
    tracklets: tracklets of the first stage tracking result.
    """
    tracks = []
    track_id = 0
    for i in assignment_list:
        if len(i) == 1:
            tracks.append(np.concatenate([tracklets[tracklets[:, 1] == i, 0][:, None], 
                                          track_id * np.ones([(tracklets[:, 1] == i).sum(), 1]), 
                                          tracklets[tracklets[:, 1] == i, 2:6]], axis=1)) 
                                          #frame,id,x1,y1,x2,y2
        else:
            uninterp_tracklets = []
            for j in i:
                uninterp_tracklets.append(np.concatenate([tracklets[tracklets[:, 1] == j, 0][:, None], 
                                                          tracklets[tracklets[:, 1] == j, 2:6]], axis=1))
                                                          #append clustered tracklets together
            uninterp_tracklets = np.concatenate(uninterp_tracklets) #tracklets without interpolation yet
            #如果list里存的是array， 只能np.concatenate
            interp_list = []
            for ind in range(uninterp_tracklets.shape[0]-1): 
                #uninterp_tracklets format: frame,x1,y1,x2,y2
                if uninterp_tracklets[ind+1][0] - uninterp_tracklets[ind][0] > 1:
                    start_node = uninterp_tracklets[ind]
                    end_node = uninterp_tracklets[ind+1]
                    start_frame = int(start_node[0])
                    end_frame = int(end_node[0])
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
    tracks = np.concatenate(tracks).astype(int) #tracks in the current batch
    
    return tracks

def mergeTracklets(tracks_list, features_list):

    finalTracksAssignment = dict() #This saves the node(tracklet) labeling in each segment.
    thresh = 6
    for segmentInd in range(len(tracks_list)-1):
        oldTracklets = tracks_list[segmentInd]
        newTracklets = tracks_list[segmentInd+1]
        numOldTracklets = np.unique(oldTracklets[:, 1]).shape[0]
        numNewTracklets = np.unique(newTracklets[:, 1]).shape[0]
        if segmentInd == 0:
            maxID = numOldTracklets
            IDSOld = np.unique(oldTracklets[:, 1])
            finalTracksAssignment[segmentInd] = IDSOld
        avg_spatial_dist = np.zeros([numOldTracklets, numNewTracklets])
        avg_app_dist = np.zeros([numOldTracklets, numNewTracklets])
        vel_spatial_dist = np.zeros([numOldTracklets, numNewTracklets])
        IDSNew = -1 * np.ones(numNewTracklets)
        for indOldTracklet in range(numOldTracklets):
            for indNewTracklet in range(numNewTracklets):
                oldTracklet = oldTracklets[oldTracklets[:, 1] == indOldTracklet, :] #frame,id,x1,y1,x2,y2
                newTracklet = newTracklets[newTracklets[:, 1] == indNewTracklet, :]
                oldFrames = oldTracklet[:, 0]
                newFrames = newTracklet[:, 0]

                src_centers = ((oldTracklet[:, 2:4] + oldTracklet[:, 4:6])/2).astype(np.int)
                dst_centers = ((newTracklet[:, 2:4] + newTracklet[:, 4:6])/2).astype(np.int)
                src_vel = (src_centers[1:src_centers.shape[0]] - src_centers[0:-1]).mean(axis=0)
                frame_gap = newTracklet[0, 0] - oldTracklet[-1, 0]
                estimated_pos = src_centers[-1] + frame_gap * src_vel
                dist = np.linalg.norm(estimated_pos - dst_centers[0])
                vel_spatial_dist[indOldTracklet,indNewTracklet] = dist

                intersectFrames = np.intersect1d(oldFrames, newFrames) #Retrive tracklet coordinates in overlapping frames 

                if max(oldFrames) < min(newFrames) or intersectFrames.shape[0] == 0:
                    avg_spatial_dist[indOldTracklet, indNewTracklet] = 300
                else:
                    intersectFrames = np.intersect1d(oldFrames, newFrames) #Retrive tracklet coordinates in overlapping frames 
                    indOld = np.logical_and(oldTracklet[:, 0] >= intersectFrames.min(), 
                                            oldTracklet[:, 0] <= intersectFrames.max())
                    indNew = np.logical_and(newTracklet[:, 0] >= intersectFrames.min(), 
                                            newTracklet[:, 0] <= intersectFrames.max())

                    coorOld = oldTracklet[indOld, 2:4] + oldTracklet[indOld, 4:6]/2
                    coorNew = newTracklet[indNew, 2:4] + newTracklet[indNew, 4:6]/2
                    overlap = np.linalg.norm(coorOld - coorNew) / intersectFrames.shape[0]
                    avg_spatial_dist[indOldTracklet, indNewTracklet] = overlap
                avg_app_dist[indOldTracklet,indNewTracklet]=1-np.dot(features_list[segmentInd][indOldTracklet],
                                                                     features_list[segmentInd+1][indNewTracklet])

        inds = np.argmin(avg_spatial_dist, axis=0)
        matched_old = []
        for col_ind in range(inds.shape[0]):
            row_ind = inds[col_ind]
            if avg_spatial_dist[row_ind, col_ind] < thresh:
                IDSNew[col_ind] = finalTracksAssignment[segmentInd][row_ind] #Stitch two tracklets
                #print('track {} and {} combined with dist {}'.format(row_ind, col_ind, dist))
                matched_old.append(row_ind)
            else:
                index = np.argmin(avg_app_dist[:, col_ind])
                app_dist = avg_app_dist[index][col_ind]
                #print('track {} and {} combined with appear cost {:.3f}'.format(tmp, col_ind, app_dist))
                if vel_spatial_dist[index][col_ind] < 100 and app_dist < 0.1 and index not in matched_old:
                    #print('seg {} track {} and {} track {}, app {}'.format(segmentInd, index, segmentInd+1, col_ind, app_dist))
                    IDSNew[col_ind] = finalTracksAssignment[segmentInd][index]
                    matched_old.append(index)
                else:
                    IDSNew[col_ind] = maxID
                    maxID += 1
        finalTracksAssignment[segmentInd+1] = IDSNew

    tracksDict = dict()
    for segmentInd in range(len(tracks_list)): 
        currTracks = tracks_list[segmentInd]
        for localID in range(finalTracksAssignment[segmentInd].shape[0]):
            trackID = finalTracksAssignment[segmentInd][localID]
            trackID = int(trackID)
            if not trackID in tracksDict.keys():
                tracksDict[trackID] = [];
            tracksDict[trackID].append(currTracks[currTracks[:, 1] == localID, :])

    finalTracks = []
    for k in tracksDict.keys():
        currTrack = np.concatenate(tracksDict[k])
        currTrack = np.concatenate([currTrack[:, 0][:,None], currTrack[:, 2:6]], axis=1)
        currTrack = currTrack[np.argsort(currTrack[:, 0])]
        interpTrack = []
        Track = []
        for ind in range(currTrack.shape[0]-1):
            curr_frame = currTrack[ind][0]
            next_frame = currTrack[ind+1][0]
            bbox = currTrack[currTrack[:, 0] == curr_frame, 0:5]
            if bbox.shape[0] == 1 and next_frame - curr_frame > 1:
                Track.append(bbox.squeeze())
                start_node = currTrack[ind] 
                end_node = currTrack[ind+1]
                start_frame = start_node[0]
                end_frame = end_node[0]
                for frame in range(start_frame+1, end_frame):
                    tmp = start_node + ((end_node-start_node)/(end_frame-start_frame))*(frame-start_frame) 
                    interpTrack.append(tmp)
            elif bbox.shape[0] == 1 and next_frame - curr_frame == 1:
                Track.append(bbox.squeeze())
                continue
            else:
                if curr_frame == currTrack[ind-1][0]:
                    continue
                else:
                    Track.append(np.mean(bbox, axis=0).squeeze())
        if not np.all(currTrack[ind] == currTrack[ind+1]):
            Track.append(currTrack[-1, 0:5])

        if len(interpTrack) != 0:
            Track = np.concatenate([np.array(Track), np.array(interpTrack)], axis=0).astype(np.int)
        else:
            Track = np.array(Track).astype(np.int)
        Track = Track[np.argsort(Track[:, 0])]
        Track = np.concatenate([k*np.ones(Track.shape[0])[:,None], Track], axis=1)
        finalTracks.append(Track)

    finalTracks = np.concatenate(finalTracks)
    finalTracks[:, 4:6] = finalTracks[:, 4:6] - finalTracks[:, 2:4] # x1,y1,x2,y2 to x1,y1,w,h
    finalTracks[:, [0, 1]] = finalTracks[:, [1, 0]] 
    finalTracks[:, 1] += 1
    finalTracks = finalTracks[np.argsort(finalTracks[:, 0]), :] #Convert to matlab evaluation format
    finalTracks = np.concatenate([finalTracks, -1*np.ones((finalTracks.shape[0], 4))], axis=1).astype(np.int)
    return finalTracks
