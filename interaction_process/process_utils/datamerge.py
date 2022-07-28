import lanelet2
import tempfile
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import collections
import random
from shapely.geometry import LineString, MultiPoint, Point, Polygon
import shapely
from shapely.ops import split
import math
from scipy.sparse import csr_matrix
from matplotlib.path import Path

from lanelet2.core import AttributeMap, TrafficLight, Lanelet, LineString3d, Point2d, Point3d, getId, \
    LaneletMap, BoundingBox2d, BasicPoint2d
from lanelet2.projection import UtmProjector

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(42)

ref_list = [21315, 20108, 15068, 10752, 10470, 8457, 6099, 5304, 771, 20059, 15772, 6765, 10267]

def rotation_matrix(psi_rad):
    rad = psi_rad-math.pi/2
    rotate_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    return rotate_matrix


class InteractionDataset(Dataset):
    def __init__(self, paradict):
        super().__init__()
        self.rootdir = paradict['datadir']
        self.datafiles = os.listdir(paradict['datadir'])
        self.datafiles.sort()
        self.rootdirmap = paradict['mapdir']
        self.mapfiles = os.listdir(self.rootdirmap)
        self.train = paradict['train']
        self.vision_range = paradict['vision']
        self.max_segment = paradict['max_segment']

        self.device = paradict['device']
        self.max_distance = paradict['max_distance']
        self.alane = paradict['alane']
        self.dataset = []

        for file in self.datafiles:
            print(file)
            df = pd.read_csv(self.rootdir+file)
            self.dataset.append(df)

        self.projector = UtmProjector(lanelet2.io.Origin(0, 0))
        self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                           lanelet2.traffic_rules.Participants.Vehicle)

        self.maps = []
        self.graphs = []
        self.all_lanelets = []
        self.polygons = []
        for mapfile in self.mapfiles:
            if '.osm' in mapfile:
                laneletmap = lanelet2.io.load(self.rootdirmap + mapfile, self.projector)
                self.maps.append(laneletmap)
                graph_ = lanelet2.routing.RoutingGraph(laneletmap, self.traffic_rules)
                self.graphs.append(graph_)
                self.all_lanelets.append([lanelet for lanelet in laneletmap.laneletLayer])

                drivable = []
                for ll in laneletmap.laneletLayer:
                    points = [(pt.x, pt.y) for pt in ll.polygon2d()]
                    drivable.append(Path(points))
                self.polygons.append(drivable)

        self.case_list = paradict['case_list']
        self.car_id = paradict['car_id']

        x_, y_ = np.meshgrid(np.arange(-22.5, 23, 1), np.arange(-11.5, 75, 1))
        mesh_ = np.stack([x_.T, y_.T], 2)
        self.mesh = mesh_.reshape(-1, 2)

        assert len(self.case_list) == len(self.car_id)
        self.laneinfo = paradict['laneinfo']

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        return 1 # add before releasing code

    def get_traj(self, df, agents_selected, x0, y0, rotate_matrix):
        df = df.set_index('track_id').loc[agents_selected].reset_index(inplace=False)
        agent_type = list(df.groupby(['track_id']).tail(1)['agent_type'].values)
        agent_type = [1 if t=='car' else -1 for t in agent_type]

        traj_init = [ np.stack([df[(df['track_id']==agent) & (df['frame_id']<=10)]['x'].values, df[(df['track_id']==agent) & (df['frame_id']<=10)]['y'].values], axis=1) for agent in agents_selected]
        traj_norm = [(point-np.array([x0,y0])).dot(rotate_matrix) for point in traj_init if (len(point)>1)]
        v_init = [ np.stack([df[(df['track_id']==agent) & (df['frame_id']<=10)]['vx'].values, df[(df['track_id']==agent) & (df['frame_id']<=10)]['vy'].values], axis=1) for agent in agents_selected]
        v_norm = [point.dot(rotate_matrix) for point in v_init if (len(point)>1)]

        traj_features = np.zeros((len(traj_norm), 9, 8))
        for i in range(len(traj_norm)):
            l = traj_norm[i]
            vl = v_norm[i]
            start, end = l[:-1], l[1:]
            vend = vl[1:]
            a_type = np.ones((len(l)-1,1))*agent_type[i]
            timestamps = np.expand_dims(10-np.arange(11-len(l), 10), axis=1)*0.1
            traj_features[i][1-len(l):] = np.concatenate([start, end, vend, timestamps, a_type], axis=-1)

        return traj_features

    def get_mapinfo(self, x0, y0, v, rad, mapfile, file_id,  rotate_matrix):
        centerlines, _, left, right, intersection, width, length, adj  = self.laneinfo[mapfile]
        available_lanes = self.alane[mapfile]
        # construct rectangles
        b = self.vision_range[file_id]
        d = 3*v+b
        side = min(35, d)
        back = min(35, d)
        polygon1 = Polygon([(x0+d,y0+side),(x0+d,y0-side),(x0-back, y0-side),(x0-back, y0+side)])
        bbox1 = shapely.affinity.rotate(polygon1, rad, origin=Point((x0,y0)), use_radians=True)
        polygon = Polygon([(x0+80,y0+35),(x0+80,y0-35),(x0-35, y0-35),(x0-35, y0+35)])
        bbox = shapely.affinity.rotate(polygon, rad, origin=Point((x0,y0)), use_radians=True)

        if ("Merging" not in mapfile) and ("LaneChange" not in mapfile):
            ccl_inside = [ccl for ccl in centerlines if bbox.intersects(LineString(ccl))]
        else:
            ccl_inside = [ccl for ccl in centerlines if ( ( bbox.intersects(LineString(ccl)) ) and (centerlines.index(ccl) in available_lanes) )]
        indices = [centerlines.index(ccl) for ccl in ccl_inside]
        l = length[indices]
        w = width[indices]+1
        le = left[indices]
        ri = right[indices]
        inter = intersection[indices]
        lanelet_feature = np.stack([w, l,le,ri, inter],axis=1)
        A = adj[indices][:,indices]

        # normalize
        ccl = np.array(ccl_inside).reshape(-1,2)
        ccl = (ccl-np.array([x0,y0])).dot(rotate_matrix)
        ccl = ccl.reshape(-1,self.max_segment+1, 2)
        order_ = np.tile(np.expand_dims(np.arange(1,self.max_segment+1),1), (ccl.shape[0],1,1))
        #print(order.shape, ccl.shape)
        arclines = np.concatenate([ccl[:,:-1], ccl[:,1:], order_], -1)

        return arclines, lanelet_feature, A, indices, bbox1, ccl_inside
    
    def get_mapinfo_semantics(self, x0, y0, rad, mapfile):
        centerlines = self.laneinfo[mapfile][0]
        available_lanes = self.alane[mapfile]
        polygon = Polygon([(x0+80,y0+35),(x0+80,y0-35),(x0-35, y0-35),(x0-35, y0+35)])
        bbox = shapely.affinity.rotate(polygon, rad, origin=Point((x0,y0)), use_radians=True)

        if ("Merging" not in mapfile) and ("LaneChange" not in mapfile):
            ccl_inside = [ccl for ccl in centerlines if bbox.intersects(LineString(ccl))]
        else:
            ccl_inside = [ccl for ccl in centerlines if ( ( bbox.intersects(LineString(ccl)) ) and (centerlines.index(ccl) in available_lanes) )]
        indices = [centerlines.index(ccl) for ccl in ccl_inside]

        return indices

    def get_lanescore_manual(self, indices, mapfile, mapid, llmap, xf, yf, ccl_inside):
        dist = [LineString(ccl).distance(Point(xf,yf)) for ccl in ccl_inside]
        return dist.index(min(dist))

    def preprocess(self, case, car):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]
        df_f = df[(df['track_id']==track_id) & (df['frame_id']==40)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        xf, yf = df_f['x'].values[0], df_f['y'].values[0] # final position
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        v = (vx**2+vy**2)**0.5 # current speed
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi
        rotate_matrix = rotation_matrix(rad) # rotation matrix
        label = np.array([xf,yf])
        label = (label-np.array([x0,y0])).dot(rotate_matrix)

        if self.train:
            mapfile = self.datafiles[file_id][:-10]+'.osm'
        else:
            mapfile = self.datafiles[file_id][:-8]+'.osm'
        map_id = self.mapfiles.index(mapfile)
        centerlines, lanefeatures, adj, indices, bbox, ccl_inside = self.get_mapinfo(x0, y0, v, rad, mapfile, file_id, rotate_matrix)

        # find agents in scope
        df_agents = df[(df['frame_id']==10)]
        dfx, dfy = df_agents['x'].values.tolist(), df_agents['y'].values.tolist()
        x_selected = [x for x,y in zip(dfx, dfy) if bbox.contains(Point(x,y))]
        agents_selected = df_agents[df_agents['x'].isin(x_selected)]['track_id'].values.tolist()
        agents_selected.insert(0, agents_selected.pop(agents_selected.index(track_id)))

        # get trajectory inputs and corresponding labels:
        trajs = self.get_traj(df, agents_selected, x0, y0, rotate_matrix)
        lanescore = self.get_lanescore_manual(indices, mapfile, map_id, self.maps[map_id], xf, yf, ccl_inside)

        TRAJ = np.zeros((26,9,8))
        SPLINES = np.zeros((55,self.max_segment,5))
        LANELETS = np.zeros((55,5))
        ADJ = np.zeros((55,55))
        TRAJ[:trajs.shape[0]] = trajs
        SPLINES[:centerlines.shape[0]] = centerlines
        LANELETS[:lanefeatures.shape[0]] = lanefeatures
        ADJ[:centerlines.shape[0]][:,:centerlines.shape[0]] = adj

        return TRAJ, SPLINES, label, trajs.shape[0], centerlines.shape[0], csr_matrix(LANELETS), csr_matrix(ADJ,dtype=np.int8), lanescore
    
    def preprocess_val(self, case, car):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        v = (vx**2+vy**2)**0.5 # current speed
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi

        return np.array([x0,y0]), rad

    def get_drivable(self, case, car):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        v = (vx**2+vy**2)**0.5
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi
        rotate_matrix = rotation_matrix(math.pi-rad) # rotation matrix

        if self.train:
            mapfile = self.datafiles[file_id][:-10]+'.osm'
        else:
            mapfile = self.datafiles[file_id][:-8]+'.osm'
        map_id = self.mapfiles.index(mapfile)
        
        if ("Merging" not in mapfile) and ("LaneChange" not in mapfile):
            polygon_list = self.polygons[map_id]
        else:
            ind = self.alane[mapfile]
            llt_i = self.laneinfo[mapfile][1]
            ind_valid = sum([llt_i[i] for i in ind], [])
            polygon_list = [self.polygons[map_id][i] for i in ind_valid]

        coordinates = (self.mesh.dot(rotate_matrix)+np.array([x0, y0]))
        mask_list = [p.contains_points(coordinates) for p in polygon_list]
        mask = np.any(np.array(mask_list), 0)
        mask = mask.reshape(46, 87)
        return csr_matrix(mask)

    def get_semantic(self, case, car):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        v = (vx**2+vy**2)**0.5
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi
        rotate_matrix = rotation_matrix(math.pi-rad) # rotation matrix

        if self.train:
            mapfile = self.datafiles[file_id][:-10]+'.osm'
        else:
            mapfile = self.datafiles[file_id][:-8]+'.osm'
        map_id = self.mapfiles.index(mapfile)
        
        llt_i = self.laneinfo[mapfile][1]
        ind = self.get_mapinfo_semantics(x0, y0, rad, mapfile)
        #print(len(ind))
        llt_valid = [llt_i[i] for i in ind]
        polygon_list = self.polygons[map_id]

        coordinates = (self.mesh.dot(rotate_matrix)+np.array([x0, y0]))
        semantics = np.zeros((46*87, 3))
        mask_list = [p.contains_points(coordinates) for p in polygon_list]
        mask = np.array(mask_list).transpose()

        for i in range(len(mask)):
            ms = mask[i]
            if np.any(ms):
                llt_indices = np.nonzero(ms)[0].tolist()
                lane_indices = [llt_valid.index(llt_v)+1 for llt in llt_indices for llt_v in llt_valid if llt in llt_v]
                if lane_indices:
                    lane_indices = np.array(list(set(lane_indices))[:3])
                    semantics[i][:len(lane_indices)] = lane_indices
        #semantics = semantics.reshape(46, 87, 3)
        return csr_matrix(semantics)

    def get_semantic_test(self, case, car, lind):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        v = (vx**2+vy**2)**0.5
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi
        if lind in ref_list:
            rad = rad+math.pi
        rotate_matrix = rotation_matrix(math.pi-rad) # rotation matrix

        if self.train:
            mapfile = self.datafiles[file_id][:-10]+'.osm'
        else:
            mapfile = self.datafiles[file_id][:-8]+'.osm'
        map_id = self.mapfiles.index(mapfile)
        
        llt_i = self.laneinfo[mapfile][1]
        ind = self.get_mapinfo_semantics(x0, y0, rad, mapfile)
        #print(len(ind))
        llt_valid = [llt_i[i] for i in ind]
        polygon_list = self.polygons[map_id]

        coordinates = (self.mesh.dot(rotate_matrix)+np.array([x0, y0]))
        semantics = np.zeros((46*87, 3))
        mask_list = [p.contains_points(coordinates) for p in polygon_list]
        mask = np.array(mask_list).transpose()

        for i in range(len(mask)):
            ms = mask[i]
            if np.any(ms):
                llt_indices = np.nonzero(ms)[0].tolist()
                lane_indices = [llt_valid.index(llt_v)+1 for llt in llt_indices for llt_v in llt_valid if llt in llt_v]
                if lane_indices:
                    lane_indices = np.array(list(set(lane_indices))[:3])
                    semantics[i][:len(lane_indices)] = lane_indices
        #semantics = semantics.reshape(46, 87, 3)
        return csr_matrix(semantics)

    def get_drivable_test(self, case, car, lind):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        v = (vx**2+vy**2)**0.5
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi
        if lind in ref_list:
            rad = rad+math.pi
        rotate_matrix = rotation_matrix(math.pi-rad) # rotation matrix

        if self.train:
            mapfile = self.datafiles[file_id][:-10]+'.osm'
        else:
            mapfile = self.datafiles[file_id][:-8]+'.osm'
        map_id = self.mapfiles.index(mapfile)
        
        if ("Merging" not in mapfile) and ("LaneChange" not in mapfile):
            polygon_list = self.polygons[map_id]
        else:
            ind = self.alane[mapfile]
            llt_i = self.laneinfo[mapfile][1]
            ind_valid = sum([llt_i[i] for i in ind], [])
            polygon_list = [self.polygons[map_id][i] for i in ind_valid]

        coordinates = (self.mesh.dot(rotate_matrix)+np.array([x0, y0]))
        mask_list = [p.contains_points(coordinates) for p in polygon_list]
        mask = np.any(np.array(mask_list), 0)
        mask = mask.reshape(46, 87)
        return csr_matrix(mask)

    def filterout_(self, shred=500):
        available_lanes = {}
        for file_id in range(len(self.dataset)):
            print(file_id)
            if self.train:
                mapfile = self.datafiles[file_id][:-10]+'.osm'
            else:
                mapfile = self.datafiles[file_id][:-8]+'.osm'
            centerlines, _, _, _, _, _, _, _  = self.laneinfo[mapfile]
            ccl_str = [LineString(ccl) for ccl in centerlines]
            lanes_candidate = len(ccl_str)

            df = self.dataset[file_id]
            df_current = df[(df['frame_id']==10)]
            x, y = df_current['x'].values, df_current['y'].values

            l_indices = []
            count = 0
            for r in range(len(x)):
                if len(l_indices) == lanes_candidate or count >shred:
                    break
                else:
                    dist = [ccl.distance(Point(x[r],y[r])) for ccl in ccl_str]
                    ind = dist.index(min(dist))
                    if ind not in l_indices:
                        l_indices.append(ind)
                        count = 0
                    else:
                        count += 1
            available_lanes[mapfile] = l_indices
        return available_lanes

    def preprocess_traj(self, case, car):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_full = df[(df['track_id']==track_id) & (df['frame_id']>10)]
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        v = (vx**2+vy**2)**0.5 # current speed
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi
        rotate_matrix = rotation_matrix(rad) # rotation matrix
        traj = np.stack([df_full['x'].values, df_full['y'].values],1)
        traj = (traj-np.array([x0,y0])).dot(rotate_matrix)

        return traj

    def preprocess_traj_test(self, case, car):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_full = df[(df['track_id']==track_id) & (df['frame_id']>10)]
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0]
        rotate_matrix = rotation_matrix(rad) # rotation matrix
        traj = np.stack([df_full['x'].values, df_full['y'].values],1)
        traj = (traj-np.array([x0,y0])).dot(rotate_matrix)

        return traj

    def preprocess_lanescore(self, case, car):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]
        df_f = df[(df['track_id']==track_id) & (df['frame_id']==40)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        v = (vx**2+vy**2)**0.5 # current speed
        xf, yf = df_f['x'].values[0], df_f['y'].values[0] # final position
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        rotate_matrix = rotation_matrix(rad) # rotation matrix

        # find surronding reconstructed lanelets, and get map elements
        if self.train:
            mapfile = self.datafiles[file_id][:-10]+'.osm'
        else:
            mapfile = self.datafiles[file_id][:-8]+'.osm'
        map_id = self.mapfiles.index(mapfile)

        lanescore = self.get_lanescore_new(x0, y0, v, rad, file_id, rotate_matrix, mapfile, map_id, self.maps[map_id], xf, yf)
        return lanescore

    def preprocess_test(self, case, car, lind):
        file_id, case_id = int(case[:-6])-1, int(case[-6:])
        track_id = int(car)

        # load data and get origin anchors and rotate vector
        df = self.dataset[file_id].query('case_id=='+str(case_id)) # current case
        df_ego = df[(df['track_id']==track_id) & (df['frame_id']==10)]

        x0, y0 = df_ego['x'].values[0], df_ego['y'].values[0] # origin point
        vx, vy = df_ego['vx'].values[0], df_ego['vy'].values[0]
        v = (vx**2+vy**2)**0.5 # current speed
        rad = df_ego['psi_rad'].values[0] # angle between yaw direction and +x axis
        if v>0 and (math.cos(rad)*vx+math.sin(rad)*vy<0):
            rad = rad+math.pi
        if lind in ref_list:
            rad = rad+math.pi
        rotate_matrix = rotation_matrix(rad) # rotation matrix
        
        if self.train:
            mapfile = self.datafiles[file_id][:-10]+'.osm'
        else:
            mapfile = self.datafiles[file_id][:-8]+'.osm'
        map_id = self.mapfiles.index(mapfile)
        centerlines, lanefeatures, adj, indices, bbox, ccl_inside = self.get_mapinfo(x0, y0, v, rad, mapfile, file_id, rotate_matrix)
        
        df_agents = df[(df['frame_id']==10)]
        dfx, dfy = df_agents['x'].values.tolist(), df_agents['y'].values.tolist()
        #dfx.remove(x0)
        #dfy.remove(y0)
        x_selected = [x for x,y in zip(dfx, dfy) if bbox.contains(Point(x,y))]
        agents_selected = df_agents[df_agents['x'].isin(x_selected)]['track_id'].values.tolist()
        agents_selected.insert(0, agents_selected.pop(agents_selected.index(track_id)))
        trajs = self.get_traj(df, agents_selected, x0, y0, rotate_matrix)

        TRAJ = np.zeros((26,9,8))
        SPLINES = np.zeros((55,self.max_segment,5))
        LANELETS = np.zeros((55,5))
        ADJ = np.zeros((55,55))
        TRAJ[:trajs.shape[0]] = trajs
        SPLINES[:centerlines.shape[0]] = centerlines
        LANELETS[:lanefeatures.shape[0]] = lanefeatures
        ADJ[:centerlines.shape[0]][:,:centerlines.shape[0]] = adj

        return TRAJ, SPLINES, trajs.shape[0], centerlines.shape[0], csr_matrix(LANELETS), csr_matrix(ADJ,dtype=np.int8), np.array([x0,y0]), rad
