#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import Analysis as anl


""""
get data from NNGT saved files_double_event
"""

rootpath = "/home/mallory/Documents/These/javier-avril-CR/Simulations2/GaussianNetworks/IBneurons/"
filename = "2000Gaussian_30_6.0_weight166.67withminis.txt"

tmin = 0.
tmax = 200000.

raster = anl.load_raster(rootpath + filename, tmin=tmin, tmax=tmax, 
                         with_space = False)

"""
Find the bursts : ce passage prend du temps, tu peux le passer si tu 
veux juste la phase et ses stat. 
"""
plot_phase = True

#Choose time step in ms
step = 5.

# Ici je change les tmin et tmax parce que la phase n'est definit que 
# entre deux spike, donc je regarde uniquement à ces temps la.
tmin = np.nanmin(raster[:,0])
tmax = np.nanmax(raster[:,-1])

times = np.arange(tmin, tmax, step)

time_bursts, network_pahse = anl.Burst_times(raster, times, 
                                         th_high = 0.6, th_low = 0.4, 
                                         ibi_th = 0.9, plot = plot_phase)
if plot_phase:
    plt.show()

"""
Compute statistical values
"""
tmin, tmax = np.nanmin(raster[:,0]), np.nanmax(raster[:,-1])
step = 5. 
times = np.arange(tmin, tmax, step)
network_phase = anl.all_time_network_phase(raster, times,
                                           after_spike_value=0.5)

# Le parameter 'after_spike_value' donne la valeur de la phase dans le 
# cas avant le premier spike ou après le dernier spike. (ici 0.5 pour 
# réduire l'effet sur la moyenne)

mean_value = np.mean(network_phase)
std_value  = np.std(network_phase)

print(mean_value, std_value)
