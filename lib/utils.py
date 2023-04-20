import os
import cv2
import numpy as np

def computeBoxFeatures(bbox1, bbox2):
    """
    Assuming bbox1, bbox2 are in the xmin, ymin, xmax, ymax format
    """
    xmin_1, ymax_1 = bbox1[0], bbox1[3]
    xmin_2, ymax_2 = bbox2[0], bbox2[3]
    width_1, height_1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    width_2, height_2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

    y_rel_dist = 2 * (ymax_1 - ymax_2) / (height_1 + height_2)
    x_rel_dist = 2 * (xmin_1 - xmin_2) / (height_1 + height_2)
    y_rel_size = np.log(height_1 / height_2)
    x_rel_size = np.log(width_1 / width_2)
    return [y_rel_dist, x_rel_dist, y_rel_size, x_rel_size]

def getIoU(bbox1, bbox2):
    """
    Assuming bbox1, bbox2 are in the xmin, ymin, xmax, ymax format
    """
    ixmin = max(bbox1[0], bbox2[0])
    ixmax = min(bbox1[2], bbox2[2])
    iymin = max(bbox1[1], bbox2[1])
    iymax = min(bbox1[3], bbox2[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)
    intersection = iw*ih
    
    union = ((bbox1[2]-bbox1[0]+1.) * (bbox1[3]-bbox1[1]+1.)+(bbox2[2]-bbox2[0]+1.) * (bbox2[3]-bbox2[1]+1.)-intersection)
    iou = intersection / union
    return iou

def visGroundTruthData(data):
    colors = np.random.rand(500, 3)
    resize_scale = 0.5
    for frame in range(int(data.ground_truth[:, 0].min()), int(data.ground_truth[:, 0].max())+1):
        img_file = os.path.join('/home/lishuai/Experiment/MOT/MOT16/train/{}/img1/{:06d}.jpg'.format(data.sequence, frame))
        img = cv2.imread(img_file)
        img = cv2.resize(img, (int(resize_scale*img.shape[1]), int(resize_scale*img.shape[0])))
        cv2.putText(img, '{:04}'.format(frame), (0,50) ,cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,255), thickness=2)

        bboxes = data.ground_truth[data.ground_truth[:, 0] == frame, 1:]
        for i in range(bboxes.shape[0]):
            ID = int(bboxes[i][0])
            x  = int(resize_scale*(bboxes[i][1]))
            y  = int(resize_scale*(bboxes[i][2]))
            w  = int(resize_scale*(bboxes[i][3])) - x
            h  = int(resize_scale*(bboxes[i][4])) - y
            cv2.rectangle(img, (x,y),(x+w,y+h), 255*colors[ID], thickness=2)
            cv2.putText(img, str(ID), (x,y) ,cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255*colors[ID], thickness=2)
        cv2.imwrite('/home/lishuai/Experiment/MOT/MOT16/train/{}/img2/{:06d}.jpg'.format(
            data.sequence, frame), img)

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r = i / dist
        x, y = int((pt1[0]*(1-r)+pt2[0]*r)+.5), int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img, p, thickness,color, -1)
    else:
        s, e = pts[0], pts[0]
        i=0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)
    
    
#####################Code used for postprocess the Network Flow output###########################
def trackletNMS(src_tracklet, dst_tracklet):
    src_start_frame, src_end_frame = src_tracklet[:, 0].min(), src_tracklet[:, 0].max()
    dst_start_frame, dst_end_frame = dst_tracklet[:, 0].min(), dst_tracklet[:, 0].max()
    
    if not src_tracklet[:, 0].min() < dst_tracklet[:, 0].min():
        tmp = dst_tracklet
        dst_tracklet = src_tracklet
        src_tracklet = tmp 

    assert src_tracklet[:, 0].min() <= dst_tracklet[:, 0].min(), 'Need to swap'
    if src_tracklet[:, 0].min() < dst_tracklet[:, 0].min() and src_tracklet[:, 0].max() > dst_tracklet[:, 0].max():
        intersect_frames = np.intersect1d(src_tracklet[:, 0], dst_tracklet[:, 0])
        
        src_ind = np.logical_and(src_tracklet[:, 0] >= intersect_frames[0],
                                 src_tracklet[:, 0] <= intersect_frames[-1])
        src_tracklet = src_tracklet[src_ind][:, 2:6]

        dst_ind = np.logical_and(dst_tracklet[:, 0] >= intersect_frames[0],
                                 dst_tracklet[:, 0] <= intersect_frames[-1])
        dst_tracklet = dst_tracklet[dst_ind][:, 2:6]

        iou_list = []
        for ind in range(src_tracklet.shape[0]):
            bbox1 = src_tracklet[ind]
            bbox2 = dst_tracklet[ind]
            iou = getIoU(bbox1, bbox2)
            iou_list.append(iou)
        return np.mean(iou_list)
        
    else:
        return 0
    
def pruneTracks(tracks, nms_thresh):
    nms_array = np.zeros(np.unique(tracks[:, 1]).shape[0], dtype=np.int)
    for i in range(nms_array.shape[0]-1):
        for j in range(i+1, nms_array.shape[0]):
            iou = trackletNMS(tracks[tracks[:, 1] == i, :], tracks[tracks[:, 1] == j, :])
            if iou > nms_thresh:
                nms_array[j] = 1
    nms_tracks = []
    track_id = 0

    for ind in range(nms_array.shape[0]):
        if nms_array[ind] != 1:
            track = tracks[tracks[:, 1] == ind]
            track = np.concatenate([track, track_id * np.ones((track.shape[0], 1))], axis=1)
            nms_tracks.append(track)
            track_id += 1

    tracks = np.concatenate(nms_tracks)
    tracks[:, [1, 6]] = tracks[:, [6, 1]]
    tracks = tracks[:, :-1]
    return tracks, nms_array

def interpolateTrack(track):
    """
    A track is assumed to be the format of frame, track_id, xmin, ymin, xmax, ymax format
    """
    interpolate_list = []
    for i in range(0, track.shape[0]-1):
        curr_frame, next_frame = track[i, 0], track[i+1, 0]
        frame_gap = next_frame - curr_frame
        curr_box, next_box = track[i], track[i+1]
        if frame_gap > 1:
            count = 1
            for inter_frame in range(curr_frame+1, next_frame):
                inter_id = curr_box[1]
                inter_x1 = curr_box[2] + (count / frame_gap) * (next_box[2] - curr_box[2])
                inter_y1 = curr_box[3] + (count / frame_gap) * (next_box[3] - curr_box[3])
                inter_x2 = curr_box[4] + (count / frame_gap) * (next_box[4] - curr_box[4])
                inter_y2 = curr_box[5] + (count / frame_gap) * (next_box[5] - curr_box[5])
                interpolate_list.append((inter_frame, inter_id, inter_x1, inter_y1, inter_x2, inter_y2))
                count += 1
                
    if len(interpolate_list) != 0:
        track = np.concatenate((track, np.stack(interpolate_list)))
    inds = np.argsort(track[:, 0])
    track = track[inds].astype(np.int)
    return track
    
def interpolateTracks(tracks):    
    interpolated_tracks = []
    for track_id in np.unique(tracks[:, 1]):
        track = tracks[tracks[:, 1] == track_id]
        track_ = interpolateTrack(track)
        interpolated_tracks.append(track_)

    interpolated_tracks = np.concatenate(interpolated_tracks)
    return interpolated_tracks
