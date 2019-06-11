#!/usr/bin/env python
#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as an
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from shapely.geometry import Point, MultiPoint, Polygon, LineString, MultiLineString, MultiPolygon
from shapely.prepared import prep
import numpy as np 

from Analysis import all_time_neuron_phase, load_raster, load_activity

from Space_Analysis import PolygonPatch

def nrn_in_clt(positions, surface):
    '''
    return nngt ids of neurons in surface
    '''
    points   = [Point(p) for p in positions]
    data     = zip(points, range(0,len(positions)))
    #in which ring neurons belong
    prepared_polygon = prep(surface)
    N_filter = filter(lambda x: prepared_polygon.intersects(x[0]), data)
    #neuron NNGT ids in the ring one is looking at
    N_filter = [N_filter[i][1] for i in range(len(N_filter))]
    return N_filter

def update_anim(num, scat, phases, cum_spk, plot_phase):
    '''
    num: int, iteration index
    scat: matplotlib Collection, where neurons are plot
    colors: array of size time_step x num_neurons x 3 
            rgb colors of all neurons 
    '''
    colors  = [colormap(each) for each in phases[:,num]]
    scat.set_color(colors)

    time_text.set_text('   time\n' + str(time_array[num] - 
                        time_array[0]) + ' ms' )
    
    cum_spk.set_data(time_array[0:num] - time_array[0], fspk[0:num])
    
    plot_phase.set_data(time_array[0:num] - time_array[0], mph[0:num])
    
    if time_array[num] > time_bursts[bid,0]:
        cx.vlines([time_bursts[bid,0]] - time_array[0], [-0.1],
         [1.1], lw=.6)
        bx.vlines([time_bursts[bid,0] - time_array[0]], [-0.1], 
        [1.1], lw=.6)
        bst.set_visible(True)
    if time_array[num] > time_bursts[bid,1]:
        bx.vlines([time_bursts[bid,1] - time_array[0]], [-0.1], 
        [1.1], lw=.6)
        cx.vlines([time_bursts[bid,1] - time_array[0]], [-0.1], 
        [1.1], lw=.6)
        bet.set_visible(True)
    if time_array[num] > t_max_dens:
        bx.vlines([t_max_dens - time_array[0]], [-0.1], [1.1], 
        lw = 0.7, color = 'darkgreen')
        cx.vlines([t_max_dens - time_array[0]], [-0.1], [1.1], 
        lw = 0.7, color = 'darkgreen')
        cltt.set_visible(True)

#set movie parameters
colormap = plt.cm.viridis

culture_radius = 800.

#files
num_neurons, degree, std, weight = 2000, 100, 50., 125.
root       = '/home/mallory/Documents/These/javier-avril-CR/Simulations2/EDRNetworks/NDneurons/'

spike_name = str( str(num_neurons) + 'EDR_' + str(degree) + 
                '_lambda' + str(std) + '_weight' + str(weight) + '.txt' )

filename   = str( str(num_neurons) + 'EDR_' + str(degree) + 
                '_lambda' + str(std) + '_weight' + str(weight) )

#load raster plot
activity, positions = load_raster(root + spike_name)

#activity = load_activity(root + spike_name)

#r = np.random.uniform(0,culture_radius-50,num_neurons)
#t = np.random.uniform(0,2*np.pi,num_neurons)
#x = r * np.cos(t)
#y = r * np.sin(t)

#positions = np.dstack((x,y))[0]
#print(positions.shape)

#load cluster surface
INIT_CLT = np.load(root + filename + '_init_clt.npy', allow_pickle = True).item()

#time on which to compute the phase
bid = INIT_CLT.keys()[3]
time_bursts = np.load(root + 'time_bursts_' + filename + '.npy')

tmin, tmax = time_bursts[bid][0] - 15, time_bursts[bid][1] + 15
step = 1.
time_array = np.arange(tmin, tmax, step)
surface = INIT_CLT[bid]

if type(surface) == list:
    surface = MultiPolygon(surface)

nrn_surface = nrn_in_clt(positions, surface)

fspk = []
for nrn in activity:
    i = np.where(nrn > time_array[0])[0][0]
    fspk.append(nrn[i])
fspk = np.array(fspk)

h,b = np.histogram(fspk[nrn_surface], np.append(np.array([0.]),time_array))
h = h.cumsum() / float(len(nrn_surface))
t_max_dens = b[np.where(h > 0.8)[0][0]]
#t_max_dens = np.inf

phases = all_time_neuron_phase(activity, time_array)

mph = np.mean(phases, axis = 0)

#figure initialisation
fig = plt.figure(figsize = (4,1.5))
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[:, 0])
bx = fig.add_subplot(gs[0, 1])
cx = fig.add_subplot(gs[1, 1])
    
#phases of all neurons
colors  = [colormap(each) for each in phases[:,0]]
scat    = ax.scatter(positions[:,0], positions[:,1], s = 3, c = colors,
                  marker = '.')
                  
time_text = ax.text(.4, 1.02,'   time\n 0. ms' ,
                    transform=ax.transAxes, fontsize = 4, color = 'black')

patch = PolygonPatch(surface, alpha = 0.3, fc = 'darkgreen', ec = 'w')
ax.add_patch(patch)

ax.set_xlim([-culture_radius, culture_radius])
ax.set_ylim([-culture_radius, culture_radius])
ax.set_xlabel('x ($\mu$m)', fontdict = {'fontsize':4})
ax.set_ylabel('y ($\mu$m)', fontdict = {'fontsize':4})
ax.tick_params(labelsize=3, width=.5)

cax = fig.add_axes([0.46, 0.2, 0.011, 0.55])
cb = mpl.colorbar.ColorbarBase(cax, cmap=colormap)
cb.set_label('Neuron Phase', fontdict = {'fontsize':4})
cb.set_ticks([0.,0.5,1.])
cb.set_ticklabels(['0.', '0.5', '1.'])
cb.ax.tick_params(labelsize=3, width=.5)
ax.set_aspect('equal')

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)
    bx.spines[axis].set_linewidth(0.5)
    cx.spines[axis].set_linewidth(0.5)
    cb.ax.spines[axis].set_linewidth(0.5)

h,b = np.histogram(fspk, np.append(np.array([0.]),time_array))

fspk = h.cumsum() / float(num_neurons)

cum_spk, = bx.plot(time_array[0], fspk[0], lw = '.5')
cx.set_xlim([0., time_array[-1] - time_array[0]])
bx.set_xlim([0., time_array[-1] - time_array[0]])
bx.set_xticklabels([])
#bx.set_xticks([])
bx.set_ylim([-0.1,1.1])
bx.set_ylabel('cumulative activity', fontdict = {'fontsize':4})
cx.set_xlabel('time (ms)', fontdict = {'fontsize':4})
bx.yaxis.set_label_position("right")
bx.yaxis.tick_right()
bx.tick_params(labelsize=3, width=.5)

plot_phase, = cx.plot([0.], mph[0], lw = .5)
cx.set_ylim([-0.1,1.1])
cx.set_ylabel('Network Phase', fontdict = {'fontsize':4})
cx.yaxis.set_label_position("right")
cx.yaxis.tick_right()
cx.tick_params(labelsize=3, width=.5)

bst = bx.text(time_bursts[bid,0] - time_array[0], 1.2, 'Burst Start', 
              horizontalalignment='center', verticalalignment='center',
              fontsize = 4, color = 'black')
bst.set_visible(False)

cltt = bx.text(t_max_dens - time_array[0], -0.23, '80% in \n cluster', 
              horizontalalignment='center', verticalalignment='center',
              fontsize = 4, color = 'darkgreen')
cltt.set_visible(False)

bet = bx.text(time_bursts[bid,1] - time_array[0], 1.2, 'Burst End', 
              horizontalalignment='center', verticalalignment='center',
              fontsize = 4, color = 'black')
bet.set_visible(False)

anim = an.FuncAnimation(fig, update_anim, frames = len(time_array), 
                        fargs=(scat, phases, cum_spk, plot_phase))
# ~ plt.show()

anim.save('ND-lambda50_weight125.mp4', fps = 7, dpi = 700, bitrate = 1100)
