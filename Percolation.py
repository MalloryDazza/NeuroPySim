#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Network
import nngt
import networkx as nx

#Maths
import numpy as np
import matplotlib.pyplot as plt

#Multiprocessing
import multiprocessing


def Dynamical_Quorum_Percolation(graph, threshold, simulation_time, spk_file = None):
    '''
    Dynamical Quorum Percolation process. 
    
    Nodes are active for only one time step. 
    
    Parameter :
    -----------
    
    graph : DiGraph, networkx directed graph
    
    threshold : int, value of the quorum often noted 'm'
    
    simulation_time : int, number of time step
    
    return :
    --------
    
        1D numpy array of spike times from 1 for the initially activated
        nodes to numpy.inf for non spiking neurons.
        
    '''
    if spk_file is not None:
        savefile = open(spk_file, 'w')
        savefile.write('# N T X Y')
        # ~ savefile.write('# N T')
        savefile.write('\n')
        savefile.close()
    
    #all nodes
    nodes = np.array(graph.nodes())
    pos = graph.get_positions()
    
    DT = 1.
    dt = DT/ float(len(nodes))
    
    #Initialize nodes
    potential = np.random.randint(0,threshold-1, size = len(nodes))
    
    #Store avalanches
    AVALANCHES = []
    for _ in range(simulation_time):
        
        #activate one random node
        rdm_node = np.random.randint(0, len(nodes))
        #start of thr avalanche
        avalanche = [rdm_node]
        #Change the potenetial accordingly
        potential[rdm_node] = 0
        potential[list(graph.neighbours(rdm_node, mode = 'out'))] += 1
        count = 0
        while (potential > threshold).any():
            msk = (potential > threshold)
            
            avalanche.append(nodes[msk])
        
            for n in nodes[msk]:
                potential[list(graph.neighbours(n, mode = 'out'))] += 1
                
            potential[msk] = 0
            
            count += 1
            if count > len(nodes):
                print('infinity is my limit')
                break
                
        if spk_file is not None:
            savefile = open(spk_file, 'a')
            t = -DT
            for spk_n in avalanche:
                t += dt
                if type(spk_n) == int:
                    savefile.write(str(spk_n) + ' ')
                    savefile.write(str(t) + ' ')
                    savefile.write(str(pos[spk_n][0]) + ' ')
                    savefile.write(str(pos[spk_n][1]))
                    savefile.write('\n')
                    
                else:
                    for n in spk_n:
                        savefile.write(str(n) + ' ')
                        savefile.write(str(t) + ' ')
                        savefile.write(str(pos[n][0]) + ' ')
                        savefile.write(str(pos[n][1]))
                        savefile.write('\n')
                        
            savefile.close()
            
        if spk_file is None:
            AVALANCHES.append(avalanche)
        
        if count > len(nodes):
            potential = np.random.randint(0,threshold-1, size = len(nodes))
            if spk_file is not None:
                savefile = open(spk_file, 'w')
                savefile.write('# N T X Y')
                # ~ savefile.write('# N T')
                savefile.write('\n')
                savefile.close()
    return AVALANCHES

if __name__ == "__main__":
    
    nngt.seed(10)
    
    # ~ root = '/media/mallory/My Passport/Mallory2/These/Percolation/DynamicalQP/'
    
    # ~ N = 10000
    # ~ deg = 100
    # ~ m = 102
    # ~ R = 2500
    # ~ scale = 100
    
    # ~ sim_time = 1000
    
    # ~ culture = nngt.geometry.Shape.disk(R)
    
    # ~ filename = 'DQP_m_' + str(m) + '_ER_activity.txt' 
        
    # ~ graph = nngt.generation.erdos_renyi(nodes = N, avg_deg = deg)
    
    # ~ graph = nngt.generation.distance_rule(scale, rule='exp', 
                    # ~ shape=culture, nodes=N, avg_deg=deg, unit='um')
    
    # ~ avalanches = Dynamical_Quorum_Percolation(graph, m, sim_time)
    
    # ~ DT = 1.
    # ~ dt = DT/ float(N)
    # ~ neuron = []
    # ~ time = []
    # ~ for i,avl in enumerate(avalanches):
        # ~ t = DT*i
        # ~ for spk_n in avl:
            # ~ t += dt
            # ~ if type(spk_n) == int:
                # ~ neuron.append(spk_n)
                # ~ time.append(t)
                
            # ~ else:
                # ~ neuron.extend(spk_n)
                # ~ time.extend([t]*len(spk_n))
    
    # ~ plt.figure()
    # ~ plt.plot(time, neuron, ls = '', marker = '.')
    # ~ plt.show()
    
    # ~ exit()
    
    root = '/media/mallory/My Passport/Mallory2/These/Percolation/DynamicalQP/'
    
    N = 10000
    R = 2500.
    deg = 100 # lambda critic = R/10 = 250 microns
    m = 100
    culture = nngt.geometry.Shape.disk(R)
    
    sim_time = 10000
    
    DT = 1.
    dt = DT/ float(N)
    
    scale = [100,175,250,325,400]
    
    quorum = [102,103,104,105, 106]
    
    for lmbd in scale:
        for m in quorum:
            
            graph = nngt.generation.distance_rule(lmbd, rule='exp', 
                    shape=culture, nodes=N, avg_deg=deg, unit='um')
            # ~ graph = nngt.generation.erdos_renyi(nodes = N, avg_deg = deg)
            
            filename = 'DQP_m_' + str(m) + '_EDR_lmbd_' + str(lmbd) + 'activity.txt' 
            # ~ filename = 'DQP_m_' + str(m) + '_ER_activity.txt' 
            
            print('done network', m, lmbd)
            # ~ print('done network', m)
            
            avalanches = Dynamical_Quorum_Percolation(graph, m, sim_time, root + filename)
            print('simulation done')
            
            nngt.save_to_file(graph, root + 'EDRnetwork_N104_deg100_lmbd' + str(lmbd) + '_.el' )
            '''
            savefile = open(root + filename, 'w')
            savefile.write('# N T X Y')
            savefile.write('\n')
            
            pos = graph.get_positions()
            
            t = -DT
            # ~ neuron = []
            # ~ time = []
            # ~ sizes = []
            for avl in avalanches:
                t += DT
                # ~ s = 0
                for spk_n in avl:
                    t += dt
                    if type(spk_n) == int:
                        # ~ neuron.append(spk_n)
                        # ~ time.append(t)
                        # ~ s += 1
                        
                        savefile.write(str(spk_n) + ' ')
                        savefile.write(str(t) + ' ')
                        savefile.write(str(pos[spk_n][0]) + ' ')
                        savefile.write(str(pos[spk_n][1]))
                        savefile.write('\n')
                        
                    else:
                        neuron.extend(spk_n)
                        # ~ time.extend([t]*len(spk_n))
                        # ~ s += len(spk_n)
                        
                        for n in spk_n:
                            savefile.write(str(n) + ' ')
                            savefile.write(str(t) + ' ')
                            savefile.write(str(pos[n][0]) + ' ')
                            savefile.write(str(pos[n][1]))
                            savefile.write('\n')
                            
            savefile.close()
            '''
                    # ~ sizes.append(s)
            
    # ~ plt.figure()
    # ~ plt.plot(time, neuron, ls = '', marker = '.')
    # ~ plt.show()
    
    # ~ bins_number = 10
    # ~ c = np.nanmax(sizes)**(1./(bins_number))

    # ~ bins = c**np.arange(1,bins_number+1,1)
    
    # ~ h, b = np.histogram(sizes, bins)
    
    # ~ b = b[:-1] + np.diff(b)/2.
    
    # ~ plt.figure()
    # ~ plt.loglog(b,h)
    # ~ plt.show()
    exit()
def Quorum_Percolation(graph, threshold, f_init):
    '''
    Quorum Percolation process 
    
    Parameter :
    -----------
    
    graph : DiGraph, networkx directed graph
    
    threshold : int, value of the quorum often noted 'm'
    
    f_init : float, proportion of initiatily activated nodes
    
    return :
    --------
    
        1D numpy array of spike times from 1 for the initially activated
        nodes to numpy.inf for non spiking neurons.
        
    '''
    #all nodes
    nodes = np.array(graph.nodes())
    
    #activate some nodes
    init = np.random.choice(nodes, size=int(f_init * len(nodes)), 
                            replace=False,)
    
    spike_time = np.ones(len(nodes)) + np.inf
    spike_time[init] = 1
    
    keepongoing = True
    step = 1
    while keepongoing:
        #----------------------------------------#
        # spike every time step after activation #
        #----------------------------------------#
        
        #Stop if no one spike in this time step 
        keepongoing = False
        step += 1
        spiking = (spike_time < np.inf)
        #Turn neurons on if they reached the quorum
        #here an active neurons if always spiking
        for n in nodes[~spiking]:
            if np.sum(spiking[[p 
                                for p in graph.predecessors(n)]]) >= threshold:
                spike_time[n] = step
                #do not stop now
                keepongoing = True
        
        '''
        !!! A TESTER !!!
        #-----------------------------------------------#
        # neuron is active for only one time step after activation #
        #-----------------------------------------------#
        
        #Stop if no one spike in this time step 
        keepongoing = False
        step += 1
        spiking = (spike_time == step - 1)
        
        #Turn neurons on if they reached the quorum
        for n in nodes[(spike_time == np.inf)]:
            if np.sum(spiking[[p 
                                for p in graph.predecessors(n)]]) >= m:
                spike_time[n] = step
                #do not stop now
                keepongoing = True
        '''
        
    return spike_time

class Worker_Quorum_Percolation(multiprocessing.Process):
    '''
    Class to operate the Quorum Percolation process with multiple
    processes for different networks
    '''
    
    def __init__(self, graph, QP_params, iteration, name, root):
        '''
        Parameters:
        -----------
        graph : DiGraph, networkx directed graph
        
        QP_params : dict, values of the proportion of initially activated 
                  nodes (dict values) for each value of the quorum 
                  thresholds (dict keys)
                  
        iteration : int, number of iteration of the QP
        
        name : string, name of the process use to save the data
        '''
        multiprocessing.Process.__init__(self)
        self.graph = nx.DiGraph(graph)
        self.QP_params = QP_params
        self.iteration = iteration
        self.name = name
        self.root = root
        
    def run(self):
        '''
        run the QP and save the interesting quantities
        '''
        
        nodes = np.array(self.graph.nodes())
        
        print('working on ' + self.name)
        
        for m in self.QP_params.keys():
            
            for f in self.QP_params[m]:
                
                #avalanche properties
                avch_sizes = []
                avch_duration = []
                
                #mean activity (number of active nodes) over time steps
                #from 1 to 300
                activity = np.zeros(300)
                count = 0
                
                for it in range(iteration):
                
                    spk_time = Quorum_Percolation(self.graph, m, f)
                    
                    savefile = open(root + self.name + 'SpikeTime_threshold' + 
                        str(m) + '_finit' + str(f) +'.txt', 'a')
                    
                    for v in spk_time:
                        
                        savefile.write(str(v))
                        savefile.write(' ')
                        
                    savefile.write('\n')
                    
                    savefile.close()
                    
                    ## Analyse the spike timing ## 
                    
                    #find avalanches
                    msk = (spk_time < np.inf)
                    subg = nx.DiGraph(self.graph.subgraph(nodes[msk]))
                    edges = np.array([[a,b] for a,b in subg.edges])
                    
                    if len(edges) != 0:
                        count += 1
                        #causal edges
                        mask = (np.diff(spk_time[edges], axis = 1) > 0) & (
                                np.diff(spk_time[edges], axis = 1) < np.inf)
                        #remove edges that cannot be in the avalanche because of causality
                        subg.remove_edges_from(edges[~mask.reshape(
                                                            mask.shape[0])])
                        Gcc = sorted(nx.weakly_connected_components(subg),
                                                 key = len, reverse = True)
                    
                        avch_sizes.extend([len(g) for g in Gcc])
                        
                        avch_duration.extend([np.max(spk_time[[n for n in list(g)
                                             ]]) - 1 for g in Gcc])
                        
                        time_steps, act = np.unique(spk_time[msk], 
                                                        return_counts = True)
                        # !! time step starts at 1 !!
                        if np.max(time_steps) - 1 > 299:
                            continue
                        else:
                            activity[(time_steps - 1).astype(int)] += act
                
                msk = activity != 0
                
                activity /= float(count)
                
                branching = activity[msk][1:].astype(float) / activity[msk][:-1].astype(float)
                
                np.save(root + self.name + 'MeanActivity_threshold' + str(m) + 
                           '_finit' + str(f) +'.npy', activity[msk])
                
                np.save(root + self.name + 'AvchSizes_threshold' + 
                        str(m) + '_finit' + str(f) +'.npy', avch_sizes)
                
                np.save(root + self.name + 'AvchDuration_threshold' + 
                        str(m) + '_finit' + str(f) +'.npy', avch_duration)
                        
                np.save(root + self.name + 'Branching_threshold' + 
                        str(m) + '_finit' + str(f) +'.npy', branching)
                        
            print(self.name + ' with threshold  ' + str(m) + ' is DONE')
''' 
if __name__ == "__main__":
    
    iteration = 10000
    
    N = 10000
    
    
    ### DONE ###
    root = '/media/mallory/Hive/Percolation/Space/'
    
    
    ###-----------------###
    ### Gausian Network ###
    ###-----------------###
    
    # STD 4 
    std = 4
    
    graph = nngt.load_from_file(root + 'GaussNet_std' + str(std) + '.el')
    graph = nx.DiGraph(graph)
    
    m = 4 
    QP_param1 = {m : np.linspace(0.07,0.08, 5, endpoint = False)}
    p1 = Worker_Quorum_Percolation(graph, QP_param1, iteration,
                                      'GaussNet_std' + str(std), 
                                      root)
    
    m = 5 
    QP_param2 = {m : np.linspace(0.12,0.15, 5, endpoint = False)}
    p2 = Worker_Quorum_Percolation(graph, QP_param2, iteration,
                                      'GaussNet_std' + str(std), 
                                      root)
    m = 6
    QP_param3 = {m : np.linspace(0.19,0.22, 5, endpoint = False)}
    p3 = Worker_Quorum_Percolation(graph, QP_param3, iteration,
                                      'GaussNet_std' + str(std), 
                                      root)
    m = 7 
    QP_param4 = {m : np.linspace(0.29,0.33, 5, endpoint = False)}
    p4 = Worker_Quorum_Percolation(graph, QP_param4, iteration,
                                      'GaussNet_std' + str(std), 
                                      root)
    
    #STD 3.5
    std = 3.5
    
    graph = nngt.load_from_file(root + 'GaussNet_std' + str(std) + '.el')
    graph = nx.DiGraph(graph)
    
    m = 5 
    QP_param5 = {m : np.linspace(0.135,0.15, 5, endpoint = False)}
    p5 = Worker_Quorum_Percolation(graph, QP_param5, iteration,
                                      'GaussNet_std' + str(std), 
                                      root)
    
    m = 7 
    QP_param6 = {m : np.linspace(0.29,0.33, 5, endpoint = False)}
    p6 = Worker_Quorum_Percolation(graph, QP_param6, iteration,
                                      'GaussNet_std' + str(std), 
                                      root)
    
    #STD 3
    std = 3
    
    graph = nngt.load_from_file(root + 'GaussNet_std' + str(std) + '.el')
    graph = nx.DiGraph(graph)
    
    m = 5 
    QP_param7 = {m : np.linspace(0.1725,0.18, 5)}
    p7 = Worker_Quorum_Percolation(graph, QP_param7, iteration,
                                      'GaussNet_std' + str(std), 
                                      root)
    
    ###-------------###
    ### Erdos Renyi ###
    ###-------------###
    
    
    root = '/media/mallory/Hive/Percolation/Space/Erdos_Renyi/'
    
    graph = nngt.generation.erdos_renyi(nodes = N, avg_deg = 10)
    nngt.save_to_file(graph, root + 'Erdos_Renyi_deg10.el')
    graph = nx.DiGraph(graph)
    
    QP_param8 = { 2 : np.linspace(0.001,0.012,15),
                  3 : np.linspace(0.03,0.05,15),
                  4 : np.linspace(0.07,0.11,15) }
    p8 = Worker_Quorum_Percolation(graph, QP_param8, iteration,
                                      'Erdos_Renyi_deg10', 
                                      root)
                                      
    QP_param9 = { 5 : np.linspace(0.12,0.17,15),
                  6 : np.linspace(0.19,0.27,15) }
    p9 = Worker_Quorum_Percolation(graph, QP_param9, iteration,
                                      'Erdos_Renyi_deg10', 
                                      root)
                                      
    QP_param10 = {7 : np.linspace(0.29,0.37,15),
                  8 : np.linspace(0.42,0.57,15) }
    p10 = Worker_Quorum_Percolation(graph, QP_param10, iteration,
                                      'Erdos_Renyi_deg10', 
                                      root)
    
    for p in [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]:
        p.start()
        
    for p in [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]:
        p.join()
    
    
    ###-------------###
    ### EDR Network ###
    ###-------------###
    
    root = '/media/mallory/Hive/Percolation/Space/EDRNetwork/'
    
    # ~ QP_param1 = { # 2 : np.linspace(0.002,0.007,15)[12:],  # done
                  #3 : np.linspace(0.009,0.023,15) }    # done
                                                       
    # ~ QP_param2 = { 5 : np.linspace(0.1,0.2,15)[14:],     # done
                  # ~ 6 : np.linspace(0.17,0.31,15) }      # done
                  
    jobs = []
    
    lmbd = 35
    graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
    jobs.append( Worker_Quorum_Percolation(graph, {6 : np.linspace(0.17,0.31,15)},
                                           iteration, 'EDRNet_lmbd' + str(lmbd), 
                                           root))
    
    for lmbd in [15,25,35,50]:
        graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
        graph = nx.DiGraph(graph)
    
        p1 = Worker_Quorum_Percolation(graph, QP_param1, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      root)
        #p2 = Worker_Quorum_Percolation(graph, QP_param2, iteration, # done
        #                             'EDRNet_lmbd' + str(lmbd),     # done 
        #                             root)                          # done
        
        jobs.append(p1)
        # ~ jobs.append(p2) # done
    
    # ~ jobs = []
    
    # ~ QP_param3 = { 2 : np.linspace(0.0015,0.01,15)[10:], # done
                  # ~ 3 : np.linspace(0.012,0.045,15)}      # done
                  
    # ~ QP_param4 = { 6 : np.linspace(0.18,0.23,15)[2:],    # done 
                  # 8 : np.linspace(0.38,0.47,15)[13:]  # done
                  }   
    
    for lmbd in [75,100,150,200]:
        graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
        graph = nx.DiGraph(graph)
    
        p3 = Worker_Quorum_Percolation(graph, QP_param3, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      root)
        p4 = Worker_Quorum_Percolation(graph, QP_param4, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      root)
        
        jobs.append(p3)
        jobs.append(p4)
    
    
    print('PREMIER RUN')
    
    for j in jobs:
        j.start()
    
    for j in jobs:
        j.join()
    
    print('DONE')
    # ~ exit()
    
    
    print('DEUXIEME RUN')
     
    for j in jobs:
        j.start()
    
    for j in jobs:
        j.join()
    
    print('DONE')
    
    print('PREMIER RUN')
    root = '/media/mallory/My Passport/Mallory2/These/Percolation/Space/EDRNetwork/'
    
    jobs = []
    
    QP_param1 = { 6 : np.linspace(0.18,0.23,15)[6:], # done
                # 3 : np.linspace(0.03,0.045,15)  # done
                }   
    
    # ~ QP_param2 = { 6 : np.linspace(0.19,0.25,15)[8:],  # done
                # ~ }
    
    # ~ for lmbd in [400,800,1600,3000,6000]:
    for lmbd in [75]:
        graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
        graph = nx.DiGraph(graph)
    
        p1 = Worker_Quorum_Percolation(graph, QP_param1, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      root)
        # ~ p2 = Worker_Quorum_Percolation(graph, QP_param2, iteration,
                                      # ~ 'EDRNet_lmbd' + str(lmbd), 
                                      # ~ root)
        
        jobs.append(p1)
        # ~ jobs.append(p2)

    # ~ print('DEUXIEME RUN')
    
    for j in jobs:
        j.start()
    
    for j in jobs:
        j.join()
    
    exit()
    jobs = []
    
    print('DONE')
    
    # ~ QP_param3 = { 7 : np.linspace(0.24,0.4,15)[6:]} # done 
                  
    # ~ for lmbd in [15,25,35,50,75,100]:
        # ~ graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    # ~ str(lmbd)+ '_directed.el' )
        # ~ graph = nx.DiGraph(graph)
    
        # ~ p3 = Worker_Quorum_Percolation(graph, QP_param3, iteration,
                                      # ~ 'EDRNet_lmbd' + str(lmbd), 
                                      # ~ root)
        # ~ jobs.append(p3)
    
    # ~ print('TROISIEME RUN')
    
    # ~ for j in jobs:
        # ~ j.start()
    
    # ~ for j in jobs:
        # ~ j.join()
    
    # ~ print('DONE')
    
    # ~ jobs = []
    
    QP_param1 = { # 4 : np.linspace(0.065,0.105,15), # done
                  7 : np.linspace(0.27,0.37,15) }  # corrigé
    
    for lmbd in [150,200,400,800,1600,3000,6000]:
        graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
        graph = nx.DiGraph(graph)
        p1 = Worker_Quorum_Percolation(graph, QP_param1, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      '/media/mallory/')
        
        jobs.append(p1)
    
    print('TROISIEME RUN')
    
    for j in jobs:
        j.start()
    
    for j in jobs:
        j.join()
    
    print('DONE')
    exit()
    jobs = []
    
    QP_param2 = { 4 : np.linspace(0.04,0.11,15)} # corrigé
                  
    for lmbd in [15,25,35]:
        graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
        graph = nx.DiGraph(graph)
    
        p2 = Worker_Quorum_Percolation(graph, QP_param2, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      root)
        jobs.append(p2)
    
    QP_param3 = {5 : np.linspace(0.145,0.17,15), # corrigé
                 8 : np.linspace(0.42,0.52,15)}  # corrigé
    
    for lmbd in [1600,3000,6000]:
        graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
        graph = nx.DiGraph(graph)
    
        p3 = Worker_Quorum_Percolation(graph, QP_param3, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      root)
        jobs.append(p3)
    
    print('SIXIEME RUN')
    
    for j in jobs:
        j.start()
    
    for j in jobs:
        j.join()
    
    print('DONE')
    
    jobs = []
    
    
    #######    DONE     #######
    
    # ~ QP_param1 = { 8 : np.linspace(0.31,0.45,15) } # DONE
    
    # ~ for lmbd in [15,25,35,50]:
        # ~ graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    # ~ str(lmbd)+ '_directed.el' )
        # ~ graph = nx.DiGraph(graph)
    
        # ~ p1 = Worker_Quorum_Percolation(graph, QP_param1, iteration,
                                      # ~ 'EDRNet_lmbd' + str(lmbd), 
                                      # ~ root)
        
        # ~ jobs.append(p1)
    
    graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(50)+ '_directed.el' )
    graph = nx.DiGraph(graph)
    
    p3 = Worker_Quorum_Percolation(graph, {4 : np.linspace(0.035,0.08,15)[1:]}, 
                                   iteration, 'EDRNet_lmbd' + str(50), 
                                   root)
        
    jobs.append(p3)
    
    QP_param4 = { # 4 : np.linspace(0.035,0.08,15), done
                 5 : np.linspace(0.1,0.14,15)}      # corrigé 
    
    for lmbd in [75,100]:
        graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    str(lmbd)+ '_directed.el' )
        graph = nx.DiGraph(graph)
    
        p4= Worker_Quorum_Percolation(graph, QP_param4, iteration,
                                      'EDRNet_lmbd' + str(lmbd), 
                                      root)
        jobs.append(p4)
        
    
    #######    DONE     ####### 
    
    # ~ for lmbd in [400,800]:
        # ~ graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    # ~ str(lmbd)+ '_directed.el' )
        # ~ graph = nx.DiGraph(graph)
    
        # ~ p5= Worker_Quorum_Percolation(graph, {8 : np.linspace(0.38,0.47,15)}, 
                                      # ~ iteration, 'EDRNet_lmbd' + str(lmbd), 
                                      # ~ root)
        # ~ jobs.append(p5)
    
    # ~ QP_param5 = {5 : np.linspace(0.125,0.15,15)}
    
    # ~ for lmbd in [150,200,400,800]:
        # ~ graph = nngt.load_from_file(root + 'EDRGraph_N104_deg10_lmbd' + 
                                    # ~ str(lmbd)+ '_directed.el' )
        # ~ graph = nx.DiGraph(graph)
    
        # ~ p5= Worker_Quorum_Percolation(graph, QP_param5, 
                                      # ~ iteration, 'EDRNet_lmbd' + str(lmbd), 
                                      # ~ root)
        # ~ jobs.append(p5)
    
    
    print('LAST RUN')
    
    for j in jobs:
        j.start()
    
    for j in jobs:
        j.join()
    
    print('DONE')
'''
