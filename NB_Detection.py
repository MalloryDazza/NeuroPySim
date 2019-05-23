#!/usr/bin/env python3
#-*- coding:utf-8 -*-

# Maths
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as localext
from scipy.signal import convolve
from scipy.integrate import simps
'''
Temporal Network Burst Detection
'''
def open_data(name):
    '''
    Collect the data
    
    Parameter : 
    -----------
    - name : string
            Path to the activity file, must be in the form of column :
            Nest id ~ spike time ~ X position ~ Y position 
            
    Return :
    --------
            numpy.array of size 4 x number of spike 
    '''
    return_list = []
    with open(name, "r") as fileobject:
        for i, line in enumerate(fileobject):
            if not line.startswith('#'):
                lst = line.rstrip('\n').split(' ')
                return_list.append([float(lst[0]),float(lst[1]),float(lst[2]),float(lst[3])])
    return_list = sorted(return_list, key = lambda x:x[1])
    return np.array(return_list)

def firing_rate_exp_convolve(spikes, tau, dt=None, crop=5.):
    '''
    Convolve a spiketrain by an exponential kernel with characteristic
    timescale `tau`.

    Parameters
    ----------
    spikes : 1D array
        Spike times.
    tau : double
        Timescale for the exponential decay.
    dt : double, optional (default: 0.1*tau)
        Timestep for the "continuous" time arrays that will be returned.
    crop : double, optional (default: 5.)
        Crop the exponential decay after `crop*tau`.

    Returns
    -------
    ts, fr : 1D arrays
    '''
    tmin, tmax = np.min(spikes), np.max(spikes)
    dt = 0.1*tau if dt is None else dt
    # generate the time bins
    bins = np.arange(tmin - 0.5*dt, tmax + crop*tau + 1.5*dt, dt)
    # get the histogram of the spikes
    fr, bins = np.histogram(spikes, bins)
	#set the firing rate nomalisation with dt
    fr = np.array(fr)/dt
    # set the times associated to counts
    ts = bins[:-1] + 0.5*dt
    # create the exponential kernel
    tkernel  = np.arange(0, crop*tau + dt, dt)
    ekernel  = np.exp(-tkernel/tau)
    ekernel /= np.sum(ekernel)
    # convolve
    fr = convolve(fr, ekernel, "same")
    return ts, fr

def convolve_gauss(fr, sigma, dt, crop=5.):
    '''
    Convolve a spiketrain by an gaussian kernel with width `sigma`.

    Parameters
    ---------
    - fr : 1D array
        Firing rate.
    - sigma : double
        Timescale for the gaussian decay.
    - dt : double
        Timestep for the "continuous" time arrays that will be returned.
    - crop : double, optional (default: 5.)
        Crop the gaussian after `crop*sigma`.

    Returns
    -------
    ts, fr : 1D arrays
    '''
    # create the gaussian kernel
    tkernel  = np.arange(-crop*sigma, crop*sigma + dt, dt)
    ekernel  = np.exp(-(tkernel/(2*sigma))**2)
    ekernel /= np.sum(ekernel)
    # convolve
    fr = np.array(convolve(fr, ekernel, "same"))
    return fr

def max_annihilation(fr, bin_size, ind_max, N_max, beta):
    '''
    maxima elimination task
    
    Params :
    --------
    - fr : numpy array
        firing rate all neurons
    - N_max : int
           Number of local maximum
    - bin_size : float
              size of time step in the computation of the firing rate
    - ind_max : int
            index in the firing rate of the maximum
            
    Return : bool,
    --------
            True if the area associated to maximum ind_max is large 
            enough and False 
            otherwise.
    '''
    fr_inf = np.where(fr<=fr[ind_max],fr,0)
    area_inf = simps(fr_inf,dx=bin_size)
    area = simps(fr,dx=bin_size)
    if area_inf < beta*area:
        return False
    else:
        return True

def Burst_detection(spike, bin_size = 1., alpha = 0.5, beta = 0.36, withplot = False, lbl = None, col = None, it=2):
    '''
    Burst detection algorithm of a train spike
    
    - spike    : list of float
            list of spike times 
    - bin_size : float
            time step in the firing rate computation in milliseconds
            N.B. expo time and gaussian width are 3*bin_size
    - alpha : float
            parameters for high value of the firing rate maxima 
            discrimination. The maxima such that their firing rate is 
            less than alpha*max(firing rate) will be discarded.
    - beta : float
            local maxima such that the area below them is less than 
            beta*total_area will be discarded.
            For gaussian-like curve beta~0.36 and for a lorentzienne 
            beta~0.57 correspond to discarded local max below half the max. 
    - withplot : boolean 
            show a plot or not (default)
    - col : string
            color of the plot
             
    return the smoothed firing rate, time conrresponding 
    '''
    # order the data
    spike = sorted(spike)
    # compute the smoothed firing rate
    ts , fr = firing_rate_exp_convolve(spike, tau=3*bin_size, dt = bin_size, crop = 3.)
    fr = convolve_gauss(fr, sigma=2*bin_size, dt = bin_size, crop = 3.)
    # sort the local maxima with small value of fr 
    list_max = list(localext(fr,np.greater)[0])
    sorted_max = sorted(list_max,key = lambda x:fr[x])
    for Max in sorted_max:
        N_max = len(list_max)
        if max_annihilation(fr,bin_size,Max,N_max,beta) == False:
            list_max.remove(Max)
    # eliminate the maxima too close to each other (same peak)
    list_max.sort()
    list_burst = list(list_max)
    for q in range(it):
        for i,idm in enumerate(list_max):
            fr_max = fr[idm]
            if i != len(list_max)-1:
                if len(np.where(fr[idm:list_max[i+1]] < fr_max*alpha)[0]) == 0:
                    rmv = min(list_max[i],list_max[i+1], key = lambda x:fr[x])
                    if rmv in list_burst:
				        list_burst.remove(rmv)
        list_max = list(list_burst)
    if withplot == True:
		plt.plot(ts,fr, color = col)
		plt.plot(ts[list_burst],fr[list_burst],'*', color = 'r')
		plt.xlabel('Time (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.legend()
		plt.show()
        
    return fr , ts , list_burst

def pk_width(firing_rate, ts, bursts):
    '''
    Width (90% of the maximum) determination of NBs
    
    - fr : numpy array
            firing rate
        
    - bursts : list
            index of the bursts
    return 
    width and troncated firing rate 
    '''
    wth = []
    wth_idx = []
    fr = np.array(firing_rate)
    for i in range(len(bursts)-1):
        idmax = int(( bursts[i+1] + bursts[i] )/2.)
        fr_max = fr[bursts[i]]
        stt = np.where(fr[0:bursts[i]] > fr_max*0.1)[0][0]
        fr[0:bursts[i]]=fr_max
        stp = np.where(fr[0:idmax] < fr_max*0.1)[0][0]
        fr[0:idmax]=0
        wth.append((ts[stp] - ts[stt]))
        wth_idx.append(stp-stt)
    return wth , wth_idx

def Burst_Firing_Rate_Separation(fr, ts, bursts, withplot = False, orgswap = False):
    '''
    seperate the firing rate into different bursts occurences.
    return np array of size number of bursts, filled with lists which 
    contain the fr and corresponding times
    '''
    wth_idx = pk_width(fr,ts,bursts)[1]
    burst_fr = []
    burst_ts = []
    burst_fr.append(fr[0:bursts[0] + wth_idx[0]])
    burst_ts.append(ts[0:bursts[0] + wth_idx[0]])
    for i in range(len(bursts)):
        burst_fr.append(fr[bursts[i-1] + int(np.mean(wth_idx)) : bursts[i] + int(np.mean(wth_idx))])
        burst_ts.append(ts[bursts[i-1] + int(np.mean(wth_idx)) : bursts[i] + int(np.mean(wth_idx))])
    if orgswap == True:
        for i in range(len(burst_ts)):
            burst_ts[i] -= ts[bursts[i]]
    if withplot == True:
        for i in range(len(burst_fr)):
            plt.plot(burst_ts[i],burst_fr[i],label=str(i))
            plt.legend()
        plt.show()
    
    return np.array(burst_fr) , np.array(burst_ts)

def Burst_Spike_SeparationNTXY(listeNTXY, ts_bursts):
    '''
    return list of lists which contain the spike events separated by bursts
    '''
    IIB = [(ts_bursts[i] + ts_bursts[i+1])/2 for i in range(len(ts_bursts)-1)]
    t = []
    x = []
    y = []
    n = []
    fret = []
    spike = listeNTXY[:,1]
    tmin , tman = np.min(spike), np.max(spike)
    for i in range(len(ts_bursts)):
        if i == 0:
            stt = 0
            stp = np.where(spike > ts_bursts[1])[0][0]
        elif i == len(ts_bursts)-1:
            stt = np.where(spike > IIB[i-1])[0][0]
            stp = len(spike)
        else:
            stt = np.where(spike > IIB[i-1])[0][0]
            stp = np.where(spike > ts_bursts[i+1])[0][0]
        t = spike[stt:stp]
        x = listeNTXY[:,2][stt:stp]
        y = listeNTXY[:,3][stt:stp]
        n = listeNTXY[:,0][stt:stp]
        fret.append(np.array([[int(n[i]),t[i],x[i],y[i]] for i in range(len(t))]))
    return fret
    

def Best_dt(spike, burst_ref = None , dt_int = None, withplot = False):
    '''
    Find the optimal time step as the smallest one that still give the 
    same number of bursts
    
    Params :
    --------
    - spike : 1D array
            Spike times
    - burst_ref : int, default None,
            Reference number of bursts
    - dt_int : 1D array
            time step to try 
    - withplot : bool,
            Wether to show a plot or not
    
    Return : float
    --------
                The optimal time step
    '''
    # reference number of burst
    if burst_ref == None:
        N_burst = len(Burst_detection(spike, bin_size = 1.)[2])
    else:
        N_burst = burst_ref
    # find the optimal dt
    dt_opti = dt_int[0]
    for dt in dt_int:
        fr , ts , bursts = Burst_detection(spike, bin_size = dt)
        if len(bursts) > N_burst:
            if withplot == True:
                fr , ts , bursts = Burst_detection(spike, bin_size = dt_opti)
                plt.plot(ts,fr)
                plt.plot(ts[bursts],fr[bursts],'*')
                plt.show()
            break
        dt_opti = dt
    return dt_opti

def max_annihilation(fr, bin_size, ind_max, N_max, beta):
    '''
    maxima elimination task
    
    fr : numpy array
        firing rate all neurons
    N_max : int
           Number of local maximum
    bin_size : float
              size of time step in the computation of the firing rate
    ind_max : int
            index in the firing rate of the maximum
            
    Return True if the area associated to maximum ind_max is large 
    enough (more than the total area / number of bursts) and False 
    otherwise.
    '''
    fr_inf = np.where(fr<=fr[ind_max],fr,0)
    area_inf = simps(fr_inf,dx=bin_size)
    area = simps(fr,dx=bin_size)
    if area_inf < beta*area:
        return False
    else:
        return True

def Burst_detection_fb(spike, senders, bin_size_n = 10, bin_size_fr = 0.5, norm=1., alpha = 0.5, beta = 0.36, withplot = False, lbl = None, col = None, it=2):
	'''
	Burst detection algorithm of a train spike
	
	- spike    : list of float
			list of spike times 
	- bin_size : float
			time step in the firing rate computation in milliseconds
			N.B. expo time and gaussian width are 3*bin_size
	- alpha : float
			parameters for high value of the firing rate maxima 
			discrimination. The maxima such that their firing rate is 
			less than alpha*max(firing rate) will be discarded.
	- beta : float
			local maxima such that the area below them is less than 
			beta*total_area will be discarded.
			For gaussian-like curve beta~0.36 and for a lorentzienne 
			beta~0.57 correspond to discarded local max below half the max. 
	- withplot : boolean 
			show a plot or not (default)
	- col : string
			color of the plot
			
	return the smoothed firing rate, time conrresponding 
	'''
	# order the data
	spike = sorted(spike)
	crop = 3
	tau = 3*bin_size_fr
	# compute the smoothed firing rate
	tmin, tmax = np.min(spike), np.max(spike)
	bin_size_fr = 0.1*tau if bin_size_fr is None else bin_size_fr
	# generate the time bins
	bins = np.arange(tmin - 0.5*bin_size_fr, tmax + crop*tau + 1.5*bin_size_fr, bin_size_fr)
	# get the histogram of the spike
	fr, bins = np.histogram(spike, bins)
	#set the firing rate nomalisation with bin_size_fr
	fr = np.array(fr)/bin_size_fr
	# set the times associated to counts
	ts = bins[:-1] + 0.5*bin_size_fr
	# create the exponential kernel
	tkernel  = np.arange(0, crop*tau + bin_size_fr, bin_size_fr)
	ekernel  = np.exp(-tkernel/tau)
	ekernel /= np.sum(ekernel)
	# convolve
	fr = convolve(fr, ekernel, "same")
	fr_alone = np.array(fr)
	dt_n = bin_size_n*bin_size_fr
	
	bins_n = np.arange(tmin - 0.5*dt_n, tmax + crop*tau + 1.5*dt_n, dt_n)
	digit = np.digitize(spike, bins_n)
	prop_n = []
	for d in range(len(bins_n)):
		mask = (digit == d)
		prop_n.append(len(set(senders[mask]))/float(norm))
	digit = np.digitize(bins[:-1], bins_n)
	for i,idx in enumerate(set(digit)):
		mask = (digit == idx)
		fr[mask] *= prop_n[i]
	
	fr = convolve_gauss(fr, sigma=3*bin_size_fr, dt = bin_size_fr, crop = 3.)
	fr_alone = convolve_gauss(fr_alone, sigma=2*bin_size_fr, dt = bin_size_fr, crop = 3.)
	# sort the local maxima with small value of fr 
	list_max = list(localext(fr,np.greater)[0])
	sorted_max = sorted(list_max,key = lambda x:fr[x])
	for Max in sorted_max:
		N_max = len(list_max)
		if max_annihilation(fr,bin_size_fr,Max,N_max,beta) == False:
			list_max.remove(Max)
	# eliminate the maxima too close to each other (same peak)
	list_max.sort()
	list_burst = list(list_max)
	for q in range(it):
		for i,idm in enumerate(list_max):
			fr_max = fr[idm]
			if i != len(list_max)-1:
				if len(np.where(fr[idm:list_max[i+1]] < fr_max*alpha)[0]) == 0:
					rmv = min(list_max[i],list_max[i+1], key = lambda x:fr[x])
					if rmv in list_burst:
						list_burst.remove(rmv)
		list_max = list(list_burst)
	if withplot == True:
		plt.plot(ts,fr, color = col)
		#~ plt.plot(ts, fr_alone, color = 'r')
		plt.plot(ts[list_burst],fr[list_burst],'*', color = 'r')
		plt.xlabel('Time (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.legend()
		plt.show()
		
	return fr , ts , list_burst

'''
Spatial Network Burst Detection
'''
# Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import itertools as it

def Ids_in_cluster(burst_NXY, epsilon, neighbors):
    '''
    DBSCAN Cluster detection function
    
    Parameters : 
    ------------
    - burst_NXY : list (neuron id , X position , Y position)
            sorted in increasing time of spiking 
    - epsilon : float 
            epsilon parameter in DBSCAN : range of clustering for a 
            node
    - neighbors : int
            minimum number of neighbors near a node to be in a cluster. 
    Return     : dict of dict
    ------------
            cluster ids as first keys, and neurons ids in the detected 
            cluster as second keys
    '''
    n_ids = burst_NXY[:,0]
    X = np.delete(burst_NXY,0,1)
    
    #DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=neighbors , metric='euclidean', algorithm='ball_tree', leaf_size=5).fit(X)

    # Mask = True if in a detected core
    labels = db.labels_
    unique_labels = set(labels)

    in_core_mask = np.zeros_like(labels, dtype=bool)
    in_core_mask[db.core_sample_indices_] = True
    
    r_ids = dict()
    for k in unique_labels:
        if k != -1:  # -1 = non-labeled neuron (not in a cluster)
            k_core_mask  = (labels == k)
            cluster_size = len(n_ids[k_core_mask & in_core_mask])
            r_ids[k] = dict(zip(n_ids[k_core_mask & in_core_mask], [None]*cluster_size))
                  
    return r_ids
    
def Cluster_Attribute_from_file(net, fr_bin_size, it, 
epsilon, activity_file, return_network = False, neighbors = None , save_net = None, 
t_bins = None, alpha = 0.5, beta = 0.36, step_attribute = False):
    '''
    Give clustering attribute to the network given the activity file 
    and save it to save_net 
    
    Params :
    --------
    - net : :class: '~nngt.SpatialNetwork',
            network to study
    - NTXY : Array of size number of spikes, needed if activity_file is None
            Must be arrange : ids Senders, time, Xpos and Ypos in column
    - activity_file : string, default None needed if NTXY is None
            Path to the activity file, must be in the form of column :
            Nest id ~ spike time ~ X position ~ Y position 
    - save_net : string,
            Path to the file where the future network will be saved
    - fr_bin_size : float,
            Bin size for calculating the firing rate in the burst detection algorithm
    - it : int,
            Number of iteration in the burst detection algorithm
    - epsilon : float
            epsilon parameter in DBSCAN : range of clustering for a 
            node
    - neighbors : int,
             Number of neighbors for a nodes to be considered in a cluster
             If None, it will be estimated with the density of the culture
    - t_bins : float, default None, will take the mean delay of the network
            Time bins to take for the cluster detection algorithm.
    - alpha : float,
            Parameter of the burst detection 
    - beta  : float,
            Parameter of the burst detection
    - return_network : bool, default False
            Wether to return the new network or not
    - step_attribute : bool, default False
            Wether to add attributes to all nodes at each time step for 
            each bursts being the number of cluster they are detected in
    Return : :class: '~nngt.SpatialNetwork',
    -------
            Spatial network with cluster attributes : step and cluster num
            at time step of detection. 
    '''
    listeNTXY = open_data(activity_file)
    fr, ts, bursts = Burst_detection(listeNTXY[:,1], alpha = alpha, beta = beta ,
        bin_size = fr_bin_size , it = it, withplot = True)
    tb = ts[bursts]
    N_bursts = len(bursts)
    if N_bursts > 1:
        burstNTXY = Burst_Spike_SeparationNTXY(listeNTXY, tb)
        # Network
        total_neurons = net.node_nb()
        net.new_node_attribute('N_bursts', 'int', val = N_bursts)
        # Set t_bins
        if t_bins is None:
            t_bins = np.mean(net.get_delays())
        # Set DBSCAN params
        if neighbors is None:
            area      = net.shape.area
            density   = net.node_nb() / float(area)
            neighbors = density * np.pi * epsilon**2
            neighbors = int(neighbors - np.sqrt(neighbors))
        # Loop on all bursts
        for b in range(N_bursts):
            #Create new attribute : cluster_step and cluster_num at which nodes appears in a cluster
            clt_step = 'Cluster_step_b' + str(b)
            clt_num  = 'Cluster_num_b' + str(b)
            net.new_node_attribute(clt_step, 'double', val = -1.)
            net.new_node_attribute(clt_num , 'double', val = -1.)
            burst_i = burstNTXY[b]
            #time
            tstart, tstop = burst_i[0, 1], burst_i[-1, 1]
            #max number of time step 
            n_steps = int((tstop - tstart) / t_bins)
            # Set IC for loop in time
            step = 0
            recruited_neurons = 0
            # store detected clusters
            sv_clust = []
            
            t_burst = - np.inf
            while step < n_steps and recruited_neurons < total_neurons:
                step += 1
                if step_attribute:
                    clt_step_name = 'Burst' + str(b) + '_in_clt_at'+ str(step)
                    net.new_node_attribute(clt_step_name, 'double', val = -1.)
                # create the data inside a time bin of size step*t_bins
                # stp is stop
                stp = np.where(burst_i[:,1] > tstart + t_bins*step )[0][0]
                NXY = np.delete(burst_i,1,1)
                NXY = NXY[0:stp]
                ids = Ids_in_cluster(NXY, epsilon, neighbors)
                if ids:
                    if t_burst < 0:
                        t_burst = tstart + t_bins*(step-1)
                        net.new_node_attribute(str(b)+'_burst_start', 'double', val = t_burst)
                    for c in range(len(ids)):
                        # ids in nest start at 1
                        nodes = np.array(list(ids[c].keys()), dtype=int) - 1
                        if step_attribute:
                            net.set_node_attribute(clt_step_name, val = c, nodes = nodes)
                        # mask = True if ids in 'nodes' have not been detected in a cluster sooner
                        # those that have not been recruited yet have their attribute to -1
                        mask_step = ( net.get_node_attributes(name=clt_step, nodes = nodes) < 0 )
                        # set Cluster_step attribute for mask_step
                        net.set_node_attribute(clt_step, val = step, nodes = nodes[mask_step])
                        #set Cluster_num attribute
                        if len(sv_clust) == 0 :
                            # add neurons that have been detected in the cluster (c = 0 here)
                            sv_clust.append(set(ids[c].keys()))
                            # set node attibute the first detected cluster is number 1
                            net.set_node_attribute(clt_num, val = 1, nodes = nodes)
                            recruited_neurons += len(ids[c].keys())
                        else:
                            # we already add detected a cluster
                            new_clt = 0
                            for idx in range(len(sv_clust)):
                                # if every neurons in the detected cluster have been stored in sv_clust
                                if set(ids[c].keys()).issuperset(sv_clust[idx]):
                                    #mask = True if ids detected just now have no Cluster_num 
                                    mask_num = ( net.get_node_attributes(name=clt_num, nodes = nodes) < 0 )
                                    recruited_neurons += np.sum(mask_num)
                                    #set node attibute
                                    net.set_node_attribute(clt_num, val = idx+1, nodes = nodes[mask_num])
                                    #update saved cluster 'sv_clust'
                                    sv_clust[idx] = set(ids[c].keys())
                                    break
                                else:
                                    new_clt += 1
                            #new cluster detecteds
                            if new_clt == len(sv_clust):
                                #mask = True if ids detected just now has no Cluster_num 
                                mask_num = ( net.get_node_attributes(name=clt_num, nodes = nodes) < 0 )
                                recruited_neurons += np.sum(mask_num)
                                #set node attibute
                                net.set_node_attribute(clt_num, val = len(sv_clust) + 1, nodes = nodes[mask_num])
                                #update saved cluster 'sv_clust'
                                sv_clust.append(set(ids[c].keys()))
        if save_net is not None:
            net.to_file(save_net)
    if return_network:
        return net

def Cluster_Attribute_from_array(net, epsilon, listeNTXY, N_bursts, tmax_burst,
return_network = False, neighbors = None , save_net = None, t_bins = None, 
step_attribute = False):
    '''
    Give clustering attribute to the network given the activity file 
    and save it to save_net 
    
    Params :
    --------
    - net : :class: '~nngt.SpatialNetwork',
            network to study
    - listeNTXY : Array of size number of spikes, needed if activity_file is None
            Must be arrange : ids Senders, time, Xpos and Ypos in column
    - N_bursts : int,
            Number of bursts
    - tmax_burst : 1D array
            times of the bursts (as maximum)
    - save_net : string,
            Path to the file where the future network will be saved
    - epsilon : float
            epsilon parameter in DBSCAN : range of clustering for a 
            node
    - neighbors : int,
             Number of neighbors for a nodes to be considered in a cluster
             If None, it will be estimated with the density of the culture
    - t_bins : float, default None, will take the mean delay of the network
            Time bins to take for the cluster detection algorithm.
    - return_network : bool, default False
            Wether to return the new network or not
    - step_attribute : bool, default False
            Wether to add attributes to all nodes at each time step for 
            each bursts being the number of cluster thy are detected in
    Return : :class: '~nngt.SpatialNetwork',
    -------
            Spatial network with cluster attributes : step and cluster num
            at time step of detection.
    '''
    burstNTXY = Burst_Spike_SeparationNTXY(listeNTXY, tmax_burst)
    # Network
    total_neurons = net.node_nb()
    net.new_node_attribute('N_bursts', 'int', val = N_bursts)
    
    positions = net.get_positions()
    print(positions)
    # Set t_bins
    if t_bins is None:
        t_bins = np.mean(net.get_delays())
    # Set DBSCAN params
    print(t_bins)
    if neighbors is None:
        area      = net.shape.area
        density   = net.node_nb() / float(area)
        neighbors = density * np.pi * epsilon**2
        neighbors = int(neighbors - np.sqrt(neighbors))
    print(neighbors)
    # Loop on all bursts
    for b in range(N_bursts):
    #~ for b in [3]:
        #~ fig = plt.figure()
        nump = 0
        #Create new attribute : cluster_step and cluster_num at which nodes appears in a cluster
        clt_step = 'Cluster_step_b' + str(b)
        clt_num  = 'Cluster_num_b' + str(b)
        net.new_node_attribute(clt_step, 'double', val = -1.)
        net.new_node_attribute(clt_num , 'double', val = -1.)
        burst_i = burstNTXY[b]
        print(len(burst_i))
        #time
        tstart, tstop = burst_i[0, 1], burst_i[-1, 1]
        print(tstart, tstop)
        #max number of time step 
        n_steps = int((tstop - tstart) / t_bins)
        print(n_steps)
        # Set IC for loop in time
        step = 0
        recruited_neurons = 0
        # store detected clusters
        sv_clust = []
        
        t_burst = -1.
        while step < n_steps and recruited_neurons < total_neurons:
            
            step += 1
            if step_attribute:
                clt_step_name = 'Burst' + str(b) + '_in_clt_at'+ str(step)
                net.new_node_attribute(clt_step_name, 'double', val = -1.)
            # create the data inside a time bin of size step*t_bins
            # stp is stop
            stp = np.where(burst_i[:,1] > tstart + t_bins*step )[0][0]
            NXY = np.delete(burst_i,1,1)
            NXY = NXY[0:stp]
            nump += 1
            #~ if nump > 9:
                #~ fig.tight_layout()
                #~ plt.show()
            #~ ax = fig.add_subplot(3,3,nump)
            #~ ax.set_aspect('equal')
            #~ ax.plot(NXY[:,1],NXY[:,2], c = 'k', marker = '.', ls = '')
            #~ plt.plot(NXY[:,1],NXY[:,2], c = 'k', marker = '.', ls = '')
            ids = Ids_in_cluster(NXY, epsilon, neighbors)
            if ids:
                if t_burst < 0:
                    t_burst = tstart + t_bins*(step-1)
                    net.new_node_attribute('burst' +str(b)+'_start', 'double', val = t_burst)
                for c in range(len(ids)):
                    # ids in nest start at 1
                    nodes = np.array(list(ids[c].keys()), dtype=int) - 1
                    if step_attribute:
                        net.set_node_attribute(clt_step_name, val = c, nodes = nodes)
                    # mask = True if ids in 'nodes' have not been detected in a cluster sooner
                    # those that have not been recruited yet have their attribute to -1
                    mask_step = ( net.get_node_attributes(name=clt_step, nodes = nodes) < 0 )
                    # set Cluster_step attribute for mask_step
                    net.set_node_attribute(clt_step, val = step, nodes = nodes[mask_step])
                    to_plot = positions[nodes[mask_step]]
                    #~ ax.plot(to_plot[:,0],to_plot[:,1], c = 'blue', marker = str(c), ls = '')
                    #~ plt.plot(to_plot[:,0],to_plot[:,1], c = 'blue', marker = str(c+1), ls = '')
                    #set Cluster_num attribute
                    if len(sv_clust) == 0 :
                        # add neurons that have been detected in the cluster (c = 0 here)
                        sv_clust.append(set(ids[c].keys()))
                        # set node attibute the first detected cluster is number 1
                        net.set_node_attribute(clt_num, val = 1, nodes = nodes)
                        #~ ax.plot(positions[nodes][:,0],positions[nodes][:,1], c = 'blue', marker = 'd', ls = '')
                        #~ plt.plot(positions[nodes][:,0],positions[nodes][:,1], c = 'blue', marker = str(c+1), ls = '')
                        recruited_neurons += len(ids[c].keys())
                    else:
                        # we already add detected a cluster
                        new_clt = 0
                        for idx in range(len(sv_clust)):
                            # if every neurons in the detected cluster have been stored in sv_clust
                            if set(ids[c].keys()).issuperset(sv_clust[idx]):
                                #mask = True if ids detected just now have no Cluster_num 
                                mask_num = ( net.get_node_attributes(name=clt_num, nodes = nodes) < 0 )
                                recruited_neurons += np.sum(mask_num)
                                #set node attibute
                                net.set_node_attribute(clt_num, val = idx+1, nodes = nodes[mask_num])
                                #~ ax.plot(positions[nodes[mask_num]][:,0],positions[nodes[mask_num]][:,1], c =  'blue', marker = 'd', ls = '')
                                #~ plt.plot(positions[nodes[mask_num]][:,0],positions[nodes[mask_num]][:,1], c =  'blue', marker = str(c+1), ls = '')
                                #update saved cluster 'sv_clust'
                                sv_clust[idx] = set(ids[c].keys())
                                break
                            else:
                                new_clt += 1
                        #new cluster detecteds
                        if new_clt == len(sv_clust):
                            #mask = True if ids detected just now has no Cluster_num 
                            mask_num = ( net.get_node_attributes(name=clt_num, nodes = nodes) < 0 )
                            recruited_neurons += np.sum(mask_num)
                            #set node attibute
                            net.set_node_attribute(clt_num, val = len(sv_clust) + 1, nodes = nodes[mask_num])
                            #~ ax.plot(positions[nodes[mask_num]][:,0],positions[nodes[mask_num]][:,1], c = 'blue', marker = 'd', ls = '')
                            #~ plt.plot(positions[nodes[mask_num]][:,0],positions[nodes[mask_num]][:,1], c = 'blue', marker = str(c+1), ls = '')
                            #update saved cluster 'sv_clust'
                            sv_clust.append(set(ids[c].keys()))
                #~ plt.show()
    if save_net is not None:
        net.to_file(save_net)
    if return_network:
        return net


