import os
import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
import networkx as nx

import lanelet2
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split

from lanelet2.core import AttributeMap, TrafficLight, Lanelet, LineString3d, Point2d, Point3d, getId, \
    LaneletMap, BoundingBox2d, BasicPoint2d
from lanelet2.projection import UtmProjector

projector = UtmProjector(lanelet2.io.Origin(0, 0))

traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                              lanelet2.traffic_rules.Participants.Vehicle)

def reconstruct_one_map(file, segment, shred=50):
    laneletmap = lanelet2.io.load(file, projector)
    graph = lanelet2.routing.RoutingGraph(laneletmap, traffic_rules)
    llts = [llt for llt in laneletmap.laneletLayer]

    nb = len(llts)
    A = np.zeros((nb, nb))
    Al = np.zeros((nb,nb))
    start_type = np.zeros(nb)
    end_type = np.zeros(nb)
    left_turn_type = np.zeros(nb)
    right_turn_type = np.zeros(nb)
    intersection_type = np.zeros(nb)
    lanelet_width = np.zeros(nb)
    centerlines = []

    # construct low-level lanelet information
    for llt in llts:
        Left = graph.left(llt)
        Right = graph.right(llt)
        Previous = graph.previous(llt)
        Following = graph.following(llt)

        if Left:
            left_turn_type[llts.index(llt)] = 1
        if Right:
            right_turn_type[llts.index(llt)] = 1

        if llt.leftBound.attributes["type"] == "virtual" and llt.rightBound.attributes["type"] == "virtual":
            intersection_type[llts.index(llt)] = 1

        if Following:
            for llt_f in Following:
                A[llts.index(llt), llts.index(llt_f)] = 1
        if Left:
            for llt_l in [Left]:
                Al[llts.index(llt), llts.index(llt_l)] = 1

        if (not Following) or len(Following)>1 or (len(Following)==1 and len(graph.previous(Following[0]))>1 ):
            end_type[llts.index(llt)] = -1
        if (not Previous) or len(Previous)>1 or (len(Previous)==1 and len(graph.following(Previous[0]))>1 ):
            start_type[llts.index(llt)] = 1
        arcline = [(pt.x, pt.y) for pt in llt.centerline]
        centerlines.append(arcline)
        lanelet_width[llts.index(llt)] = get_lane_width(llt.leftBound, llt.rightBound)

    # chain decomposition of directed road networks
    if ("Merging" in file) or ('LaneChange' in file):
        start_llt = [llt for llt, c in zip(llts, list(start_type)) if c==1]
        end_llt = [llt for llt, c in zip(llts, list(end_type)) if c==-1]
        assert(len(start_llt)==len(end_llt))
        assert(1 in A)

        # construct new high-level lanelets
        new_llt_list = []
        new_centerlines = []
        new_length = []
        used_llts = []
        # we start from every possible start low-level lanlet
        for llt in start_llt:
            llt_order = [llts.index(llt)]
            current_llt = llt
            while current_llt not in end_llt:
                next_llt = graph.following(current_llt)[0]
                llt_order.append(llts.index(next_llt))
                current_llt = next_llt
            used_llts += llt_order
            new_llt_list.append(llt_order)
            linestring = []
            for i in range(len(llt_order)):
                if i == 0:
                    linestring += centerlines[llt_order[i]]
                else:
                    linestring += centerlines[llt_order[i]][1:]
            cstring, length = uniform_centerlines(linestring, segment)
            new_centerlines.append(cstring)
            new_length.append(length)

        if len(used_llts)!=len(llts):
            remaining_llt = list( set( list(range(nb))-set(used_llt) ) )
            print("following lanelets not constructed:", remaining_llt)
            for r_llt in remaining_llt:
                print(len(graph.previous(llts[r_llt])), len(graph.following(llts[r_llt])))
                print(centerlines[r_llt])

        # split new lanelets that are too long
        for i in range(len(new_length)):
            parts = int(new_length[i]/shred)+1
            split = min(parts, len(new_llt_list[i]))
            if split > 1 and split == len(new_llt_list[i]):
                for ind in new_llt_list[i]:
                    start_type[ind]=1
                    end_type[ind]=-1
            if split > 1 and split < len(new_llt_list[i]):
                step = int(len(new_llt_list[i])/split)+1
                for j in range(len(new_llt_list[i])):
                    if new_llt_list[i][j]%step == 1:
                        end_type[new_llt_list[i][j]]=-1
                        if j<len(new_llt_list[i])-1:
                            start_type[new_llt_list[i][j+1]]=1


    start_llt = [llt for llt, c in zip(llts, list(start_type)) if c==1]
    end_llt = [llt for llt, c in zip(llts, list(end_type)) if c==-1]
    assert(len(start_llt)==len(end_llt))
    assert(1 in A)
    # construct new high-level lanelets
    new_llt_list = []
    new_centerlines = []
    new_length = []
    new_left = []
    new_right = []
    new_intersection = []
    new_width = []
    used_llts = []
    # we start from every possible start low-level lanlet
    for llt in start_llt:
        llt_order = [llts.index(llt)]
        current_llt = llt
        while current_llt not in end_llt:
            next_llt = graph.following(current_llt)[0]
            llt_order.append(llts.index(next_llt))
            current_llt = next_llt
        used_llts += llt_order
        new_llt_list.append(llt_order)
        linestring = []
        for i in range(len(llt_order)):
            if i == 0:
                linestring += centerlines[llt_order[i]]
            else:
                linestring += centerlines[llt_order[i]][1:]
        #new_centerlines.append(linestring)
        cstring, length = uniform_centerlines(linestring, segment)
        new_centerlines.append(cstring)
        new_length.append(length)
        new_left.append(left_turn_type[llt_order[0]])
        new_right.append(right_turn_type[llt_order[0]])
        new_intersection.append(intersection_type[llt_order[0]])
        new_width.append(np.amax(lanelet_width[llt_order]))

    if len(used_llts)!=len(llts):
        remaining_llt = list( set( list(range(nb))-set(used_llt) ) )
        print("following lanelets not constructed:", remaining_llt)
        for r_llt in remaining_llt:
            print(len(graph.previous(llts[r_llt])), len(graph.following(llts[r_llt])))
            print(centerlines[r_llt])
            
    # construct new adjacency matrix
    dim = len(new_llt_list)
    Ar = np.zeros((dim, dim))
    for i in range(dim):
        all_left_type = np.any(Al[new_llt_list[i]],axis=0)
        left_indices = np.nonzero(all_left_type)[0].tolist()
        for j in range(dim):
            last = new_llt_list[i][-1]
            first = new_llt_list[j][0]
            if A[last, first]==1:
                Ar[i,j] = 1
            if left_indices:
                if any(ind in new_llt_list[j] for ind in left_indices):
                    Ar[i,j] = -1

    new_left = np.array(new_left)
    new_right = np.array(new_right)
    new_intersection = np.array(new_intersection)
    new_width = np.array(new_width)
    new_length = np.array(new_length)
    return (new_centerlines, new_llt_list, new_left, new_right, new_intersection, new_width, new_length, Ar)

def get_lane_width(leftbound, rightbound):
    """
    get width/2 of each lanelet
    """
    left = LineString([(pt.x, pt.y) for pt in leftbound])
    right = LineString([(pt.x, pt.y) for pt in rightbound])
    return left.distance(right)/2

def uniform_centerlines(arc, segment):
    line = LineString(arc)
    res = (line.length)/segment
    distances = np.arange(0, line.length, res)
    points = [(line.interpolate(distance).x, line.interpolate(distance).y) for distance in distances]
    end = [arc[-1]]
    return points+end, line.length

def get_all_mapinfo(root, segment=8, shred=80.):
    files = os.listdir(root)
    laneinfo = {}
    for file in files:
        if '.osm' in file:
            mapinfo = reconstruct_one_map(root+file, segment, shred)
            laneinfo[file] = mapinfo
    return laneinfo

def get_egolist_train(root):
    files = os.listdir(root)
    files.sort()
    candidates = []
    for i in range(len(files)):
        df = pd.read_csv(root+files[i])
        #print(df)
        cindex = df['case_id'].values*100 + df['track_id'].values + (i+1)*1e8
        df['cindex'] = list(cindex.astype(int))
        df1 = df.groupby(['cindex'], sort=True).head(1)
        df2 = df.groupby(['cindex'], sort=True).tail(1)

        ind1 = list((df1['case_id'].values*100 + df1['track_id'].values + (i+1)*1e8).astype(int))
        ind2 = list((df2['case_id'].values*100 + df2['track_id'].values + (i+1)*1e8).astype(int))
        df1['new_id'] = ind1
        df2['new_id'] = ind2
        
        df1 = df1.sort_values('case_id')
        df2 = df2.sort_values('case_id')

        ids1 = list(df1[(df1['frame_id']==1) & (df1['agent_type']=='car')]['new_id'].values.astype(int).astype(str))
        ids2 = list(df2[(df2['frame_id']==40) & (df2['agent_type']=='car')]['new_id'].values.astype(int).astype(str))

        ids = list(set(ids1) & set(ids2))
        print(len(ids))
        candidates += ids
        
    case_id = np.array([c[:-2] for c in candidates])
    track_id = np.array([c[-2:] for c in candidates])
    case_list = list(set(case_id))
    
    car_id = [track_id[np.where(case_id==cand)].astype(int) for cand in tqdm(case_list)]
    return case_list, car_id
