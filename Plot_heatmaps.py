#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Network
import nngt

#Maths
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy import ndimage as nd
from scipy.ndimage.filters import gaussian_filter

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

def Plot2D_Network_Attribute(net = None, step_cp = 2, step_ht = 1., smooth_cp = 10., 
n_contour = 6, heatmap_map = 'Reds', contour_map = 'Greens', 
smooth_hm = None, Contour_Plot = None, Heatmap_Plot = None,
show = True, save_fig = None, to_plot = None, where_to_plot = None, 
Param_name = None):
    '''
    Plot on the same figure two attributes as a contour plot and a heatmap 
    Voronoi
    
    Parameters :
    ------------
    - net : :class: '~nngt.SpatialNetwork'
            Network to ge the attributes
    - step_cp : float,
            Min step of the grid for the contour plot
    - step_hm : float, default 1. 
            Min step of the grid for the heatmap
    - smooth_cp : float,
            Smooth value for the contour plot
    - smooth_hm : float, default : no smoothing
            Smooth value for the contour plot
    - n_contour : int, dfault 10
            number of lines in the contour plot
    - Contour_Plot : string,
            Name of the attribute to contour plot.
    - Heatmap_Plot : string,
            Name of the attribute to plot as heatmap.
    - heatmap_map : string, default 'Reds'
            Name of the colormap to use for heatmap
    - contour_map : string, default 'Greens'
            Name of the colormap to use for contour plot
    - show : bool,
            Wether to show the figure of not
    - save_fig : string,
            Path to save the figure with extension
    - to_plot : 1D array of size number of neurons
            Parameter to plot
    - where_to_plot : string
            How do you want to plot 'to_plot', wether 'heatmap' or 'contour'
    - Param_name : string,
            Name of the parameter 'to_plot'
            
    Return : Notting
    --------
    '''
    fig, ax = plt.subplots(figsize=(12, 9))
    
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
    
    # For the heatmaps
    xstep     = max(0.5*np.min(np.diff(xx)), step_ht)
    ystep     = max(0.5*np.min(np.diff(yy)), step_ht)
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
    
    # If you know what you want to plot, plot it where you want it
    if to_plot is not None:
        
        grid = np.zeros((ysize, xsize))
        np.add.at(grid, [digit_y, digit_x], to_plot)
        convolved = np.zeros((ysize, xsize))
        #initialise flag
        #~ flag_din = np.ones(shape, dtype = bool)
        #~ flag_din[grid != 0] = 0
        
        #~ ind = nd.distance_transform_edt(flag_din, return_distances = False, return_indices = True) 
        #~ grid = grid[ind[0], ind[1]]
        
        if smooth_hm is not None:
            convolved = gaussian_filter(grid, smooth_hm)
            #~ convolved /= np.sum(convolved)
            
        convolved  = norm(np.ma.masked_array(convolved, mask=~circ_mask))
        
        if where_to_plot == 'heatmap':
            heatmap = ax.imshow(convolved, aspect='equal', extent=(xmin-0.5*dx, xmax+0.5*dx, ymin-0.5*dx, ymax+0.5*dx), vmin=np.min(convolved), vmax=np.max(convolved), origin="lower", cmap = heatmap_map)
            cb = plt.colorbar(heatmap, ax=ax)
            cb.set_label('Normalized' + Param_name)
            
        if where_to_plot == 'contour':
            contour = ax.contour(xx, yy, convolved, n_contour, cmap = contour_map)
            cb = plt.colorbar(contour)
            cb.set_label('Normalized '+Param_name)
            
    # Compute what you have to plot
    if Heatmap_Plot is not None:
        if Heatmap_Plot == 'in-degree':
            to_plot_hm = net.get_degrees('in')
            
        elif Heatmap_Plot == 'out-degree':
            to_plot_hm = net.get_degrees('out')
            
        elif Heatmap_Plot == 'total-degree':
            to_plot_hm = net.get_degrees('total')
            
        elif Heatmap_Plot == 'clustering':
            to_plot_hm = nngt.analysis.local_clustering(net)
            
        elif Heatmap_Plot == 'betweenness':
            to_plot_hm = nngt.analysis.node_attributes(net, attributes = Heatmap_Plot )
        
        elif Heatmap_Plot == 'closeness':
            to_plot_hm = nngt.analysis.closeness(net)
        
        elif Heatmap_Plot == 'subgraph_centrality':
            to_plot_hm = nngt.analysis.node_attributes(net, attributes = Heatmap_Plot)
            
        elif Heatmap_Plot == 'mean_delay_in':
            edges     = net.edges_array
            delay     = net.get_delays()
            indegree  = net.get_degrees(deg_type = 'in')
            n_nodes   = net.node_nb()
            in_mean_delay  = [0.]*n_nodes

            for e,(i,j) in enumerate(edges):
                in_mean_delay[j]  += delay[e]
            
            for i in range(n_nodes):
                in_mean_delay[i]  /= float(indegree[i])
            to_plot_hm = in_mean_delay

        elif Heatmap_Plot == 'mean_delay_out':
            edges     = net.edges_array
            delay     = net.get_delays()
            outdegree = net.get_degrees(deg_type = 'out')
            n_nodes   = net.node_nb()
            out_mean_delay  = [0.]*n_nodes

            for e,(i,j) in enumerate(edges):
                out_mean_delay[i]  += delay[e]
            
            for i in range(n_nodes):
                out_mean_delay[i]  /= float(outdegree[i])
            to_plot_hm = out_mean_delay
            
        elif Heatmap_Plot == 'mean_delay_tot':
            edges     = net.edges_array
            delay     = net.get_delays()
            indegree  = net.get_degrees(deg_type = 'in')
            outdegree = net.get_degrees(deg_type = 'out')
            n_nodes   = net.node_nb()
            out_mean_delay = [0.]*n_nodes
            in_mean_delay  = [0.]*n_nodes
            
            for e,(i,j) in enumerate(edges):
                out_mean_delay[i] += delay[e]
                in_mean_delay[j]  += delay[e]
            
            for i in range(n_nodes):
                in_mean_delay[i]  /= float(indegree[i])
                out_mean_delay[i] /= float(outdegree[i])
            to_plot_hm = np.mean([in_mean_delay,out_mean_delay], axis = 0)
            
        elif Heatmap_Plot == 'first_to_cluster':
            N_bursts   = net.get_node_attributes(name = 'N_bursts')[0]
            n_nodes    = net.node_nb()
            to_plot_hm = [0]*n_nodes
            for b in range(1,N_bursts):
                step          = net.get_node_attributes(name = 'Cluster_step_b' + str(b))
                #step is -1 if never detected in a cluster
                min_step    = np.min(step[np.nonzero(step+1)])
                mask        = ( step == min_step )
                to_plot_hm += mask / float(N_bursts-1)
            #~ to_plot_hm += 0.00001*np.min(to_plot[np.nonzero(to_plot_hm)])

        grid = np.zeros((ysize, xsize))
        np.add.at(grid, [digit_y, digit_x], to_plot_hm)
        convolved = np.zeros((ysize, xsize))
        #initialise flag
        #~ flag_din = np.ones(shape, dtype = bool)
        #~ flag_din[grid != 0] = 0
        
        #~ ind = nd.distance_transform_edt(flag_din, return_distances = False, return_indices = True) 
        #~ grid = grid[ind[0], ind[1]]
        
        if smooth_hm is not None:
            convolved  = gaussian_filter(grid, smooth_hm)   
            convolved  = convolved
        else:
			convolved  = grid
        convolved  = norm(np.ma.masked_array(convolved, mask=~circ_mask))
        
        heatmap = ax.imshow(convolved, aspect='equal', extent=(xmin-0.5*dx, xmax+0.5*dx, ymin-0.5*dx, ymax+0.5*dx), vmin=np.min(convolved), vmax=np.max(convolved), origin="lower", cmap = heatmap_map)
        cb = plt.colorbar(heatmap, ax=ax)
        cb.set_label('Normalized ' + Heatmap_Plot)

    # For the contour plot
    xstep     = max(0.5*np.min(np.diff(xx)), step_cp)
    ystep     = max(0.5*np.min(np.diff(yy)), step_cp)
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
    dim   = len(shape)
    slcs  = [slice(None)]*dim
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
    circ_mask = sector_mask((ysize, xsize), (0.5*ysize, 0.5*xsize), 0.5*xsize, (0, 360))
    
    # Compute what you have to plot
    if Contour_Plot is not None:
        if Contour_Plot == 'density':
            to_plot_cp = np.ones(net.node_nb())
            
        elif Contour_Plot == 'in-degree':
            to_plot_cp = net.get_degrees('in')
            
        elif Contour_Plot == 'out-degree':
            to_plot_cp = net.get_degrees('out')
            
        elif Contour_Plot == 'total-degree':
            to_plot_cp = net.get_degrees('total')
            
        elif Contour_Plot == 'clustering':
            to_plot_cp = nngt.analysis.local_clustering(net)
            
        elif Contour_Plot == 'betweenneess':
            to_plot_cp = nngt.analysis.node_attributes(net, attributes = Contour_Plot )
        
        elif Contour_Plot == 'closeness':
            to_plot_cp = nngt.analysis.closeness(net)
        
        elif Contour_Plot == 'subgraph_centrality':
            to_plot_cp = nngt.analysis.node_attributes(net, attributes = Contour_Plot)
            
        elif Contour_Plot == 'mean_delay_in':
            edges     = net.edges_array
            delay     = net.get_delays()
            indegree  = net.get_degrees(deg_type = 'in')
            n_nodes   = net.node_nb()
            in_mean_delay  = [0.]*n_nodes

            for e,(i,j) in enumerate(edges):
                in_mean_delay[j]  += delay[e]
            
            for i in range(n_nodes):
                in_mean_delay[i]  /= float(indegree[i])
            to_plot_cp = in_mean_delay

        elif Contour_Plot == 'mean_delay_out':
            edges     = net.edges_array
            delay     = net.get_delays()
            outdegree = net.get_degrees(deg_type = 'out')
            n_nodes   = net.node_nb()
            out_mean_delay  = [0.]*n_nodes

            for e,(i,j) in enumerate(edges):
                out_mean_delay[i]  += delay[e]
            
            for i in range(n_nodes):
                out_mean_delay[i]  /= float(outdegree[i])
            to_plot_cp = out_mean_delay
            
        elif Contour_Plot == 'mean_delay_tot':
            edges     = net.edges_array
            delay     = net.get_delays()
            indegree  = net.get_degrees(deg_type = 'in')
            outdegree = net.get_degrees(deg_type = 'out')
            n_nodes   = net.node_nb()
            out_mean_delay = [0.]*n_nodes
            in_mean_delay  = [0.]*n_nodes
            
            for e,(i,j) in enumerate(edges):
                out_mean_delay[i] += delay[e]
                in_mean_delay[j]  += delay[e]
            
            for i in range(n_nodes):
                in_mean_delay[i]  /= float(indegree[i])
                out_mean_delay[i] /= float(outdegree[i])
            to_plot_cp = np.mean([in_mean_delay,out_mean_delay], axis = 0)
            
        elif Contour_Plot == 'first_to_cluster':
            N_bursts   = net.get_node_attributes(name = 'N_bursts')[0]
            n_nodes    = net.node_nb()
            to_plot_cp = [0]*n_nodes
            for b in range(1,N_bursts):
                step          = net.get_node_attributes(name = 'Cluster_step_b' + str(b))
                #step is -1 if never detected in a cluster
                min_step    = np.min(step[np.nonzero(step+1)])
                mask        = ( step == min_step )
                to_plot_cp += mask / float(N_bursts-1)
        
        grid = np.zeros((ysize, xsize))
        np.add.at(grid, [digit_y, digit_x], to_plot_cp)
        convolved = np.zeros((ysize, xsize))
        #initialise flag
        #~ flag_din = np.ones(shape, dtype = bool)
        #~ flag_din[grid != 0] = 0
        
        #~ ind = nd.distance_transform_edt(flag_din, return_distances = False, return_indices = True) 
        #~ grid = grid[ind[0], ind[1]]
        
        if smooth_cp is not None:
            convolved = gaussian_filter(grid, smooth_cp)
        else:
			convolved = grid
            
        convolved  = norm(np.ma.masked_array(convolved, mask=~circ_mask)) 
        
        contour = ax.contour(xx, yy, convolved, n_contour, cmap = contour_map)
        cb = plt.colorbar(contour)
        cb.set_label('Normalized '+Contour_Plot)
    
	ax.set_xlabel('x $\mu m$') 
	ax.set_ylabel('y $\mu m$') 
        
    if save_fig is not None:
        fig.save_fig(save_fig)
    if show == True:
        plt.show()

if __name__ == '__main__':
    net_file = '/home/mallory/Documents/NBs nucleation  - Internhip Mallory DAZZA/DATA/net1AS_minis0.55_mr15.0.el'
    net = nngt.load_from_file(net_file)
    
    edges     = net.edges_array
    delay     = net.get_delays()
    indegree  = net.get_degrees(deg_type = 'in')
    outdegree  = net.get_degrees(deg_type = 'out')
    n_nodes   = net.node_nb()
    in_mean_delay  = [0.]*n_nodes
    out_mean_delay = [0.]*n_nodes
    
    for e,(i,j) in enumerate(edges):
        in_mean_delay[j]  += delay[e]
        out_mean_delay[i] += delay[e]
        
    for i in range(n_nodes):
        in_mean_delay[i]  /= float(indegree[i])
        out_mean_delay[i] /= float(outdegree[i])
    
    tot_mean_delay = np.mean([in_mean_delay,out_mean_delay], axis = 0)
    
    arr  = norm(-nngt.analysis.local_clustering(net))
    arr *= norm(-np.array(tot_mean_delay))
    
    Plot2D_Network_Attribute(net, step_cp = 5., step_ht = 1., smooth_cp = 5., 
    n_contour = 5, heatmap_map = 'Reds', contour_map = 'Greens', 
    smooth_hm = 25., Contour_Plot = 'first_to_cluster', Heatmap_Plot = 'in-degree',
    show = True, save_fig = None, to_plot = None, where_to_plot = None, 
    Param_name = None)
    
