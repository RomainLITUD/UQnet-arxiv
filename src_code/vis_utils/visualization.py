import numpy as np
import pickle
import os
import math
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.transforms

import xml.etree.ElementTree as xml
import pyproj
import sys
import pandas as pd
from . import map_vis_without_lanelet

def get_polygon_cars(center, width, length, radian):
    x0, y0 = center[0], center[1]
#     lowleft = (x0 - length / 2., y0 - width / 2.)
#     lowright = (x0 + length / 2., y0 - width / 2.)
#     upright = (x0 + length / 2., y0 + width / 2.)
#     upleft = (x0 - length / 2., y0 + width / 2.)
    lowleft = (- length / 2., - width / 2.)
    lowright = ( + length / 2., - width / 2.)
    upright = ( + length / 2., + width / 2.)
    upleft = ( - length / 2., + width / 2.)
    rotate_ = rotation_matrix(radian)
    
    return (np.array([lowleft, lowright, upright, upleft])).dot(rotate_)+center

def get_polygon_peds(center, width, length):
    x0, y0 = center[0], center[1]
    lowleft = (x0 - length / 2., y0 - width / 2.)
    lowright = (x0 + length / 2., y0 - width / 2.)
    upright = (x0 + length / 2., y0 + width / 2.)
    upleft = (x0 - length / 2., y0 + width / 2.)
    
    return np.array([lowleft, lowright, upright, upleft])
    

def rotation_matrix(rad):
    psi = rad - math.pi/2
    return np.array([[math.cos(psi), -math.sin(psi)],[math.sin(psi), math.cos(psi)]])

def Visualize(index, title, xrange, yrange, resolution=1., stage='test', save=False, figname=None):
    if stage=='test':
        data = np.load('./results/test_st.npz', allow_pickle=True)
        H = data['heatmap'][index]
        datap = np.load('./results/test_st.npz', allow_pickle=True)
        Yp = datap['points'][index]
#         trajs = data['trajectories'][index]
#         Na = data['nb_agents'][index]
        
        visdata = np.load('./interaction_merge/test.npz', allow_pickle=True)
        origin = visdata['origin'][index]
        radian = visdata['radian'][index]
        
        filenames = os.listdir('./interaction_data/data/test/')
        filenames.sort()
        
        with open('./interaction_data/data/reference/test_index.pickle', 'rb') as handle:
            s = pickle.load(handle)
        caselist, carid = s[0], s[1]
        
    if stage=='val':
        data = np.load('./results/val_results_f.npz', allow_pickle=True)
        Yp = data['points'][index]
        #Y = data['labels'][index]
        H = data['heatmap'][index]
#         trajs = data['trajectories'][index]
#         Na = data['nb_agents'][index]
        
        visdata = np.load('./interaction_merge/vis_val.npz', allow_pickle=True)
        origin = visdata['origin'][index]
        radian = visdata['radian'][index]
        
        filenames = os.listdir('./interaction_data/data/val/')
        filenames.sort()
        
        with open('./interaction_data/data/reference/val_index.pickle', 'rb') as handle:
            s = pickle.load(handle)
        caselist, carid = s[0], s[1]
    
    rotate = rotation_matrix(radian)
    
    file_id, case_id = int(caselist[index][:-6])-1, int(caselist[index][-6:])
    track_id = int(carid[index])
    
    file_to_read = filenames[file_id]
    
    mapfile = './interaction_data/data/maps/'+filenames[file_id][:-8]+'.osm'
    
    # Visualization module
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(8.5, 8.5)
    fig.canvas.set_window_title("Interaction Dataset Visualization")
    map_vis_without_lanelet.draw_map_without_lanelet(mapfile, axes, origin, rotate, xrange, yrange)
    
    if stage=='test':
        df = pd.read_csv('./interaction_data/data/test/'+file_to_read)
    if stage=='val':
        df = pd.read_csv('./interaction_data/data/val/'+file_to_read)
        
    df = df.query('case_id=='+str(case_id))
    df_e = df[df['frame_id']<=10]
    
    all_agents = set(df_e['track_id'].values)
    
    for ind in all_agents:
        dfc = df_e[df_e['track_id']==ind]
        agent_type = dfc['agent_type'].values[0]
        traj_obs = np.stack([dfc['x'].values, dfc['y'].values], -1)
        traj_obs = (traj_obs-origin).dot(rotate)
        width = dfc['width'].values[-1]
        length = dfc['length'].values[-1]
        center = traj_obs[-1]
        
        if agent_type=='car':
            type_dict = dict(color="green", linewidth=3, zorder=11)
            yaw = radian-dfc['psi_rad'].values[-1]
            bbox = get_polygon_cars(center, width, length, yaw)
            if center[0]==0:
                rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='red', zorder=20, edgecolor='red', linewidth=1, alpha=0.5)
            else:
                rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='blue', zorder=20, edgecolor='blue', linewidth=1, alpha=0.5)
            axes.add_patch(rect)
        else:
            type_dict = dict(color="pink", linewidth=3, zorder=11)
            bbox = get_polygon_peds(center, width, length)
            rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='none', zorder=20, edgecolor='pink')
            axes.add_patch(rect)
        
        plt.plot(traj_obs[:,0], traj_obs[:,1], **type_dict)
    x = np.arange(-22.75, 23.25, 0.5)
    y = np.arange(-11.75, 75.25, 0.5)
    s = H/np.amax(H)
    s[s<0.006]=np.nan
    axes.pcolormesh(x, y, s.transpose(), cmap='Reds',zorder=0)
    for i in range(6):
        axes.scatter(Yp[i,0], Yp[i,1], s=200, alpha=1, zorder=199, marker='*', 
                   facecolors='none', edgecolors='yellow', linewidth=2)
    axes.set_title(title, fontsize=24)
        
#     plt.tick_params(
#                         axis='both',          # changes apply to the x-axis
#                         which='both',      # both major and minor ticks are affected
#                         bottom=False,      # ticks along the bottom edge are off
#                         top=False,         # ticks along the top edge are off
#                         left = False,
#                         labelbottom=False,
#                         labelleft=False) 
    
    if save:
        plt.savefig('./figs/'+figname+'.png', dpi=800)
    
    plt.show()
        

#     for r in range(Na):
#         t = trajs[r]
#         for vector in list(t):
#             axes.arrow(vector[0], vector[1], (vector[2]-vector[0]), (vector[3]-vector[1]), color='blue',head_width=0.8, width=0.2)
    
    
    
    
    
    
        