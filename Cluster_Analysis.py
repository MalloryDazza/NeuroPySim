#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Network
import nngt

#Maths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import itertools as it
import shapely.geometry as sg
from scipy import stats
from scipy.integrate import simps

from scipy.ndimage.filters import gaussian_filter

from NB_Detection import open_data

def CM(net, num_burst, cumu = True, largest = True ):
    '''
    Compute the positions of the Center of Masse of the detected clusters
    as a function of time 
    
    Parameters :
    ------------
    - net : '~nngt.SpatialNetwork',
            Spatial Network with cluster attribute : 'Cluster_step_b' and 
            'Cluster_num_b'.
    - num_burst : int,
            Number of the burst to analyse
    - cumu : bool,
            Wether to use cumulative number of neurons or not 
    - largest : bool,
            wether to return the largest trajectory or all of them
            
    Return : list of list
    --------
            first one with size number of clusters, second one with size
            number of time step. 
    '''
    lbl_cl_step = 'Cluster_step_b' + str(num_burst)
    lbl_cl_num  = 'Cluster_num_b' + str(num_burst)
    
    clust_step  = np.array(net.get_node_attributes(name=lbl_cl_step))
    N_nodes     = net.node_nb()
    all_nodes   = np.arange(0,N_nodes,1)
    nb_clusts   = len(set(net.get_node_attributes(name=lbl_cl_num)))
    
    s_start     = np.min(clust_step)
    s_end       = np.max(clust_step)
    
    CM   = []
    dist = []
    
    for clt in range(1,nb_clusts+1):
        cl_CM = []
        #mask = True if ids in clt
        mask_clust      = ( net.get_node_attributes(name = lbl_cl_num ) == clt )
        for s in np.arange(s_start,s_end+1,1):
            if cumu:
                mask_cumul  = ( net.get_node_attributes(name = lbl_cl_step) <= s )
            else:
                mask_cumul  = ( net.get_node_attributes(name = lbl_cl_step) == s )
            cl_pos          = net.get_positions(all_nodes[mask_cumul & mask_clust])
            cl_CM.append(list(np.mean(cl_pos, axis=0)))
        CM.append(np.array(cl_CM))
        
    if largest == True:
        CM = sorted(CM, key = lambda x:len(x), reverse=True)[0]
    return CM

def CLT_Mass(net, num_burst, cumu = True, normed = 'False', all_clt = False):
    '''
    Determine the cluster mass of the detected clusters
    
    Parameters :
    ------------
    - net : '~nngt.SpatialNetwork',
            Spatial Network with cluster attribute : 'Cluster_step_b' and 
            'Cluster_num_b'.
    - num_burst : int,
            Burst number to analyse 
    - cumu : bool,
            Wether to use all neurons(True) or only those at each time step
            (False)
    - normed : string : 'False' , 'total' , 'total_step' wether to normalised 
            or not and by what : total = all nodes ; total_step : all neurons 
            detected in a cluster at each time step
    
    return : 
    -------- 
             list (size : nb of clusters) of list (size : number of time step)  
    '''
    lbl_cl_step = 'Cluster_step_b' + str(num_burst)
    lbl_cl_num  = 'Cluster_num_b' + str(num_burst)
    
    clust_step  = np.array(net.get_node_attributes(name=lbl_cl_step))
    N_nodes     = net.node_nb()
    all_nodes   = np.arange(0,N_nodes,1)
    if all_clt:
        nb_clusts   = len(set(net.get_node_attributes(name=lbl_cl_num)))
    else:
        nb_clusts = 1
        
    s_start     = np.min(clust_step[np.nonzero(clust_step+1)])
    s_end       = np.max(clust_step)
    n_steps     = int(s_end - s_start)
    mass = []
    cl_mass = []
 
    for clt in range(1,nb_clusts+1):
        cl_mass    = []
        #mask = True if ids in clt
        mask_clust = ( net.get_node_attributes(name = lbl_cl_num) == clt )
        for s in np.arange(s_start, s_end+1,1):
            
            # mask = True if step is < s
            if cumu:
                mask_step_cumul = ( net.get_node_attributes(name = lbl_cl_step) <= s )
            else:
                mask_step_cumul = ( net.get_node_attributes(name = lbl_cl_step) == s )
            if all_clt:
                cl_nodes = all_nodes[mask_step_cumul]
            else:
                cl_nodes = all_nodes[mask_clust & mask_step_cumul]
            # mass of the cluster
            if normed == 'False':
                cl_mass.append(len(cl_nodes))     
            elif normed == 'total':
                cl_mass.append(len(cl_nodes)/float(N_nodes))
            elif normed == 'total_step':
                mask_step = ( net.get_node_attributes(name = lbl_cl_step) == s )
                if len(all_nodes[mask_step & mask_clust]) != 0:
                    cl_mass.append(len(cl_nodes)/float(len(all_nodes[mask_step & mask_clust])))
                else:
                    cl_mass.append(1)
    
    mass.append(cl_mass)
    
    if all_clt:
		return mass[0]
    else:
		return mass



def ring_mass(center, pos, r_0, r_cst, s_start, s_end, norm):
    '''
    Return the mass (total number of neurons inside it) of a growing 
    ring from 'center' of radius r_0 + r_cst*s for s between s_start and s_end
    
    Parameters :
    ------------
    - center : 2D coordinates
            Coordinates of the center
    - pos : array, 2 x number of neurons
            Positions of all neurons
    - r_0 : float,
            Initial radius
    - r_cst : float
            Increment of radius at each time step
    - s_start : int,
            Starting time point
    - s_end : int,
            End time point
    - norm : float,
            Used to normalize the mass
            
    Return : 1D array
    --------
            Mass of the growing circle
    
    '''
    mp_all = sg.MultiPoint(pos)
    #initialize
    circle1 = sg.Point(center).buffer(r_0)
    mass = []
    for i,s in enumerate(np.arange(s_start, s_end+1,1)):
        nodes = circle1.intersection(mp_all)
        try:
            mass.append(len(nodes)/float(norm))
        except:
            mass.append(0)
        circle1 = sg.Point(center).buffer(r_0+(i+1)*r_cst)
    return np.array(mass)
    
def CLT_perimeter(net, num_burst, normed = 'False'):
    '''
    Determine the perimeter (as the number of out neighbors) of the 
    detected clusters
    
    Parameters :
    ------------
    - net : '~nngt.SpatialNetwork',
            Spatial Network with cluster attribute : 'Cluster_step_b' and 
            'Cluster_num_b'.
    - num_burst : int,
            number of burst 
    
    - normed : string,
            Wether to normalize or not and by what : 
            'cl_size' = cluster size ; 'no_clust_neighbors' : neighbors 
            that as not yet been detected in a cluster ; 'total' = total 
            number of neurons
    
    return : 
    -------- 
             list (size : nb of clusters) of list (size : number of time step)  
    '''
    lbl_cl_step = 'Cluster_step_b' + str(num_burst)
    lbl_cl_num  = 'Cluster_num_b' + str(num_burst)
    Adj         = nngt.analysis.adjacency_matrix(net)
    
    clust_step  = np.array(net.get_node_attributes(name=lbl_cl_step))
    N_nodes     = net.node_nb()
    all_nodes   = np.arange(0,N_nodes,1)
    nb_clusts   = len(set(net.get_node_attributes(name=lbl_cl_num)))
    
    s_start     = np.min(clust_step)
    s_end       = np.max(clust_step)
    
    per = []
    for clt in range(1,nb_clusts+1,1):
        #mask = True if ids in clt
        mask_clust = ( net.get_node_attributes(name = lbl_cl_num) == clt )
        cl_per     = []
        for s in np.arange(s_start,s_end+1,1):
            #mask = True if ids in time step
            mask_step = ( net.get_node_attributes(name = lbl_cl_step) <= s )
            
            per_idx = set()
            #neighbors of the cluster 
            for idx in all_nodes[mask_clust & mask_step]:
                per_idx  = per_idx.union(set(Adj.getrow(idx).nonzero()[1]))
            #must not be in any cluster
            #~ per_idx = per_idx - set(all_nodes[mask_clust & mask_step])
            per_idx = per_idx - set(all_nodes[mask_step])
            if normed == 'False':
                cl_per.append(len(per_idx))
            elif normed == 'no_clust_neighbors':
                if len(all_nodes[~mask_step]) != 0:
                    cl_per.append(len(per_idx)/float(len(all_nodes[~mask_step])))
                else:
					cl_per.append(1)
            elif normed == 'cl_size':
                if len(all_nodes[mask_clust & mask_step]) != 0:
                    cl_per.append(len(per_idx)/float(len(all_nodes[mask_clust & mask_step])))
                else:
					cl_per.append(1)
            elif normed == 'total':
                    cl_per.append(len(per_idx)/N_nodes)
        per.append(cl_per)
    return per

def plot_burst(net, num_burst, show = True, fig = None, CM_color = 'b', 
cult_colors = None, markers = None, num_clt = 'all'):
    '''
    Plot the culture according to the activty recorded in the network 'net'
    for the burst 'num_burst'
    
    Parameters :
    ------------
    - net : '~nngt.SpatialNetwork',
            Spatial Network with cluster attribute : 'Cluster_step_b' and 
            'Cluster_num_b'.
    - num_burst : int, 
            Burst number to be plotted
    - show : bool,
            Wether to show the plot or not
    - fig : '~matplotlib.pyplot.figure'
            Figure to add the plot in
    - cult_colors : list of matplotlib.pyplot friendly colors (size, number of time steps)
            colors use to plot the time evolution of the culture
    - markers : list matplotlib.pyplot friendly, size number of lcuster
            markers use to differentiate the clusters
    - num_clt : string 'all' or int the number of the cluster to plot the Center of Masss
    '''
    
    lbl_cl_step    = 'Cluster_step_b' + str(num_burst) 
    lbl_cl_num     = 'Cluster_num_b' + str(num_burst)
    
    clust_step     = np.array(net.get_node_attributes(name=lbl_cl_step))
    N_nodes        = net.node_nb()
    all_nodes      = np.arange(0,N_nodes,1)
    nb_clusts      = len(set(net.get_node_attributes(name=lbl_cl_num)))
                   
    s_start        = np.min(clust_step[np.nonzero(clust_step+1)])
    s_end          = np.max(clust_step)
    n_steps        = int(s_end - s_start)
    
    if cult_colors is None:
        cult_colors  = [plt.cm.viridis(each) for each in np.linspace(0, 1, n_steps+1)]
    if markers is None:
        markers = it.cycle([ "x" , "P" , "o" , "d" , "s"]) 
    if fig == None:
        plt.figure()
        culture = plt.subplot()
        culture.set_aspect('equal')
    else:
        plt.figure(fig)
        culture = plt.subplot()
        culture.set_aspect('equal')
    mask_all_node = np.zeros(N_nodes)
    
    for clt, mkr in zip(range(1,nb_clusts+1), markers):
        mask_clust        = ( net.get_node_attributes(name = lbl_cl_num) == clt )
        CM = []
        for s in np.arange(s_start,s_end+1,1):
            # mask = True if step is s
            mask_step       = ( net.get_node_attributes(name = lbl_cl_step) == s )
            # mask = True if step is < s
            mask_step_cumul = ( net.get_node_attributes(name = lbl_cl_step) <= s )
            pos_step        = net.get_positions(neurons=all_nodes[mask_clust & mask_step])
            pos_clust       = net.get_positions(neurons=all_nodes[mask_clust & mask_step_cumul])
            CM.append(list(np.mean(pos_clust , axis = 0)))
            culture.plot( pos_step[:,0] , pos_step[:,1] , marker = mkr, ls="" , markersize = 4 , markeredgecolor = cult_colors[int(s - s_start)], markerfacecolor = cult_colors[int(s - s_start)] )
        CM = np.array(CM)
        if num_clt == 'all':
            culture.plot( CM[:,0] , CM[:,1], ls = '-', color = CM_color,)
        elif clt == num_clt:
            culture.plot( CM[:,0] , CM[:,1], ls = '-', color = CM_color,)
        culture.legend()
    #~ positions = net.get_positions()
    #~ culture.plot(positions[:,0], positions[:,1], '.', ls = '' , color = 'k', markersize = 1)
    if show == True:
		plt.show()

def dist_max(single_point, list_points):
    '''
    Return the maximum distance from single_point to all of list_points
    
    Parameters :
    - single_point : coordinate of the single point (x,y)
    
    - list_points : list of coordinates of the different points
    [(x1,y1),(x2,y2)...] 
    
    Return : float,
    --------
            Maximal distance
    '''
    dist = 0
    for i in range(len(list_points)):
        d = np.sqrt( (single_point[0] - list_points[i][0])**2 
                    +(single_point[1] - list_points[i][1])**2 ) 
        if d > dist:
            dist = d
    return dist

def dist_min(single_point, list_points, init_dist = 2000.):
    '''
    Return the maximum distance from single_point to all of list_points
    
    Parameters :
    ------------
    - single_point : coordinate of the single point (x,y)
    
    - list_points : list of coordinates of the different points
                [(x1,y1),(x2,y2)...] 
    Return : float,
    --------
            minimal distance
    '''
    dist = init_dist
    for i in range(len(list_points)):
        d = np.sqrt( (single_point[0] - list_points[i][0])**2 
                    +(single_point[1] - list_points[i][1])**2 ) 
        if d < dist:
            dist = d
    return dist

def center_mass(points):
    '''
    Return the coordinates of the center of mass os the 'points'
    
    Parameters : 
                - list of coordinates of the points floats
    '''
    return np.mean(points, axis=0)
    
def nodes_in_ring(list1, list2):
    '''
    Return the coordinates of the nodes in the ring 
    formed by two circles center in the center of mas of the first list 
    and of radius the max distance to all other nodes of list1 for 
    circle 1 and of list 2 for circle 2
    
    Parameters : 
                - list1 : list of coordinates [(x1,y1), (x2,y2), ...]
                
                - list2 : same
    '''
    CM = center_mass(list1)
    
    r1 = dist_max(CM, list1)+0.5
    r2 = dist_max(CM, list2)+0.5
    mp2 = sg.MultiPoint(list2)
    circle1 = sg.Point(CM).buffer(r1)
    circle2 = sg.Point(CM).buffer(r2)
    
    ring = circle2.difference(circle1)
    nodes = ring.intersection(mp2)
    
    ret = []
    for i in range(len(nodes)):
        ret.append([nodes.geoms[i].x , nodes.geoms[i].y])
    
    return ret

def nearest_culture_point(cm):
    '''
    return the coordinates of nearest point in the boundary of the culture
    '''    
    cult = sg.LinearRing(sg.Point([0.,0.]).buffer(1000.).exterior.coords)
    min_d = cult.project(sg.Point(cm))
    ret = cult.interpolate(min_d)
    
    return list(ret.coords)[0]
    
def logical_sum(logical_list):
    '''
    Return the logical sum of the list along the axis 0 
    
    Parameters : 
                - logical_list : list of numpy arrays of same size
    '''
    size = len(logical_list[0])
    r = np.array(logical_list[0])
    for i in range(1,len(logical_list)):
        r += logical_list[i]
        
    return r

def clt_propagation(net, num_burst, normed = True):
    '''
    Return the normalized displacement of cluster in time as the sum of all 
    vector from the center of mass of the cluster at time s to the nodes in 
    the exterior ring at time s+ds
    
    Parameters :
    ------------
    - net : '~nngt.SpatialNetwork',
            Spatial Network with cluster attribute : 'Cluster_step_b' and 
            'Cluster_num_b'.
            
    - num_burst : int, number of the burst to analyse
    
    - normed : bool, wether to normalize the result or not
    
    Return : dict with keys, time step, of dict with keys cluster num of
    --------   each step
            
    '''
    position  = net.get_positions()
    node_nb   = net.node_nb()
    all_nodes = np.arange(0,node_nb,1)
    step_name = 'Cluster_step_b' + str(num_burst)
    steps     = net.get_node_attributes(name = step_name).astype(int)
    min_step = np.min(steps[np.nonzero(steps+1)])
    max_step  = np.max(steps)
    time      = np.arange(min_step+1, max_step+1, 1)
    
    #initialze first CM 
    #cluster detected at time step s
    at_step_clt_num_name = 'Burst'+str(num_burst)+'_in_clt_at'+str(min_step)
    #number of cluster for each nodes
    at_step_clt_num      = net.get_node_attributes(name = at_step_clt_num_name)
    #all cluster detected (-1 means not belonging to any cluster) 
    at_step_clt_set      = set(at_step_clt_num) - set([-1])
    past_CM    = []
    present_CM = []
    #initialize CM
    past_nodes_mask    = []
    present_nodes_mask = []

    for clt_num in list(at_step_clt_set):
        #mask cluster at min_step
        mask_clt_at_step = ( at_step_clt_num == clt_num )
        #initialize the past CM
        past_CM.append(center_mass( position[mask_clt_at_step]))
        present_CM.append(center_mass( position[mask_clt_at_step]))
        #initialise the past nodes
        past_nodes_mask.append(mask_clt_at_step)
    past_nodes_mask = logical_sum(past_nodes_mask)
    vector_v = dict({ i : { } for i in time })

    for s in time:
        #cluster detected at time step s
        at_step_clt_num_name = 'Burst'+str(num_burst)+'_in_clt_at'+ str(s)
        #number of cluster for each nodes
        at_step_clt_num      = net.get_node_attributes(name = at_step_clt_num_name)
        #all cluster detected
        at_step_clt_set      = set(at_step_clt_num) - set([-1])
            
        for present_clt in list(at_step_clt_set):
            #keep track if the cluster is new or not
            is_it_new = 0 
            mask_clt_at_step = ( at_step_clt_num == present_clt )
            mask_new_nodes = mask_clt_at_step & ~past_nodes_mask
            for i,cm in enumerate(past_CM):
                #if there is no new nodes
                if ~mask_new_nodes.any():
                    #if it is the same cluster
                    if is_same_clt(cm, position[mask_clt_at_step]):
                        vector_v[s][i] = 0.
                        break
                        # no need to update present_CM : didn't move
                #else, there is new nodes
                else:
                    #if it is the same cluster, ad the value to this cluster
                    if is_same_clt(cm, position[mask_clt_at_step]):
                        cult_p = np.array(nearest_culture_point(cm)) - cm
                        vv = np.sum(position[mask_new_nodes]-cm, axis = 0) / float(len(all_nodes[mask_new_nodes]))
                        vector_v[s][i] = vv
                        #update the present CM 
                        present_CM[i] = center_mass( position[mask_clt_at_step] )
                        break
                    else: # new cluster update the present CM
                        is_it_new += 1 
                #you find a new cluster, be proud of you !
                if is_it_new > len(past_CM)-1:
                    present_CM.append(center_mass(position[mask_clt_at_step]))
            present_nodes_mask.append(mask_clt_at_step)
        #update past mask and center of mass
        #and null present ones
        past_nodes_mask    = logical_sum(present_nodes_mask)
        past_CM            = list(present_CM)
        present_nodes_mask = []
        
        if normed == True:
            for j in range(len(vector_v[s])):
                if np.linalg.norm(vector_v[s][j]) != 0:
                    vector_v[s][j] /= float(np.linalg.norm(vector_v[s][j]))
    return vector_v

def CLT_velocity(net, num_burst, time_step = None, plot_error = False):
    '''
    Compute the velocity of a growing cluster
    
    Parameters :
    ------------
    - net : '~nngt.SpatialNetwork',
            Spatial Network with cluster attribute : 'Cluster_step_b' and 
            'Cluster_num_b'.   
    - num_burst : int,
            Burst number to study
    - r_0 : float,
            Initial radius
    - r_cst : float
            Constant increment of radius for each time step
    - time_step : flaot, default mean delay of the connections
            Time step to ompute the velocity
    - plot_error : bool,
            Wether to plot the error or not
            
    Return : lists
    -------- 
             Velocity of first detected cluster , quadratic error made 
             by estimating the mass of the cluster with a circle
    '''
    if plot_error:
        fig, ax = plt.subplots()
    # Set time step
    if time_step is None:
        time_step = np.around(np.mean(net.get_delays()),2)
    # number of nodes for normalization
    n_nodes = net.node_nb()
    # get all nodes positions
    pos = net.get_positions()
    maxpos = np.max(pos[:,0]) - np.min(pos[:,0])
    # get the steps
    lbl_cl_step = 'Cluster_step_b' + str(num_burst)
    step = net.get_node_attributes(name = lbl_cl_step)
    max_step = np.max(step)
    min_step = np.min(step[np.nonzero(step+1)])
    mask_min_step = ( step == min_step )
    # Initial center of mass
    cm = np.mean(pos[mask_min_step], axis = 0)
    # cluster real mass
    clt_mass = np.array(CLT_Mass(net, num_burst, cumu = True, normed = 'total', all_clt = True)[0])
    area = np.inf
    best_r = 0
    for r in np.linspace(0.005*maxpos, 0.1*maxpos, 7):    
        # Circle mass
        circle_mass = np.array(ring_mass(cm, pos, r, r, min_step, max_step, norm = n_nodes))
        # Error
        Er = (np.array(circle_mass) - np.array(clt_mass))**2
        area_r = simps(Er, dx = time_step)
        if plot_error:
            ax.plot(r,area_r,'d')
        if area_r < area:
            area = area_r
            best_r = r

    ret = best_r
    for r in np.linspace(best_r - 0.2*best_r, best_r + 0.2*best_r, 20):    
        # Circle mass
        circle_mass = np.array(ring_mass(cm, pos, r, r, min_step, max_step, norm = n_nodes))
        # Error
        Er = (np.array(circle_mass) - np.array(clt_mass))**2
        area_r = simps(Er, dx = time_step)
        if plot_error:
            ax.plot(r,area_r,'.')
        if area_r < area:
            area = area_r
            ret = r
    if plot_error:
        ax.set_ylabel('Total Error made on a single burst')
        ax.set_xlabel('radius of growing circle')
        plt.show()
    return ret/time_step

def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def norm(arr):
    '''
    Return a normalized array
    '''
    return (arr - np.min(arr))/float(np.max(arr)-np.min(arr))

def Activity_wave(net, spikes, senders, num_burst):
	'''
	Compute and plot the activity wave 
	'''
	
	#Determine the minimal size of the x and y grids.
	#Create the 2D grid.
	positions = net.get_positions()
	ux = np.unique(positions[:, 0])
	uy = np.unique(positions[:, 1])
	xx        = np.sort(ux)
	yy        = np.sort(uy)
	xmin      = np.min(xx)
	xmax      = np.max(xx)
	ymin      = np.min(yy)
	ymax      = np.max(yy)
	
	space_step = 2.
	
	# For the heatmaps
	xstep     = max(0.5*np.min(np.diff(xx)), space_step)
	ystep     = max(0.5*np.min(np.diff(yy)), space_step)
	height    = ymax - ymin + 2*ystep
	width     = xmax - xmin + 2*xstep
	centroid  = (0.5*(xmin + xmax), 0.5*(ymin + ymax))
	xsize     = int(np.ceil(width / xstep))
	ysize     = int(np.ceil(height / ystep))
	if xsize % 2:
		xsize += 1
	if ysize % 2:
		ysize += 1
	shape = (ysize,xsize)
	
	halfsize_x = 0.5*xsize*xstep
	halfsize_y = 0.5*ysize*ystep
	xlim       = (centroid[0] - halfsize_x, centroid[0] + halfsize_x)
	ylim       = (centroid[1] - halfsize_y, centroid[1] + halfsize_y)
	xbins      = np.linspace(xlim[0] - 0.5*xstep, xlim[1] + 0.5*xstep, xsize + 1)
	ybins      = np.linspace(ylim[0] - 0.5*ystep, ylim[1] + 0.5*ystep, ysize + 1)
	dx = np.diff(xbins)[0]
	dy = np.diff(ybins)[0]
	digit_x = np.digitize(positions[:, 0], xbins)
	digit_y = np.digitize(positions[:, 1], ybins)
	xx = xbins[:-1] - 0.5*dx
	yy = ybins[:-1] - 0.5*dy
	circ_mask = sector_mask((ysize, xsize), (0.5*ysize, 0.5*xsize), 0.51*xsize, (0, 360))
	
	time_step = np.mean(net.get_delays())
	fig, ax = plt.subplots()
	print(time_step)
	for k in range(20):
		#start/stop of the burst in time
		#~ tb = net.get_node_attributes(name = 'burst' +str(num_burst)+'_start')[0]
		tb = 5300
		#start/stop of the burst in index 
		start = np.where(spikes > tb + k*time_step)[0][0]
		#stop of the burst in index 
		stop = np.where(spikes > tb + (k+1)*time_step)[0][0]
		
		spk = spikes[start:stop]
		sdr = senders[start:stop].astype(int) - 1
		
		nb_nodes = net.node_nb()
		print(start, stop, sdr)
		treated_neurons = []
		to_plot = np.zeros(nb_nodes)
		
		for time in range(len(spk)):
			if sdr[time] in treated_neurons:
				continue
			else:
				to_plot[sdr[time]] = 1
				treated_neurons.append(sdr[time])	
		print(to_plot)
		grid = np.zeros((ysize, xsize))
		np.add.at(grid, [digit_y, digit_x], to_plot)
		convolved = np.zeros((ysize, xsize))
		
		smooth_hm = 100.
		convolved  = gaussian_filter(grid, smooth_hm)
		convolved  = norm(np.ma.masked_array(convolved, mask=~circ_mask))
		
		heatmap = ax.imshow(convolved, aspect='equal', extent=(xmin-0.5*dx, xmax+0.5*dx, ymin-0.5*dx, ymax+0.5*dx), vmin=np.min(convolved), vmax=np.max(convolved), origin="lower", cmap = 'Reds')
		#~ cb = plt.colorbar(heatmap, ax=ax)
		#~ cb.set_label('Activity')
		plt.pause(0.01)
	plt.show()
