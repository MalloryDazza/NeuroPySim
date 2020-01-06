import nngt 
import dense as ds
import numpy as np 
import matplotlib.pyplot as plt

def spatial_activity_fromfile(activity_file, dt, tmin, tmax, ncol, nrow,
                                show = True, save_file = None):
    '''
    '''
    return_list = []
    with open(activity_file, "r") as fileobject:
        for i, line in enumerate(fileobject):
            if not line.startswith('#'):
                lst = line.rstrip('\n').split(' ')
                return_list.append([int(lst[0]),float(lst[1]),float(lst[2]),float(lst[3])])
    NTXY = np.array(sorted(return_list, key = lambda x:x[1]))
    neurons = [i for i in set(NTXY[:,0])]
    
    spk = NTXY[:,1]
    pos = NTXY[...,2:]
    
    num_neurons = len(set(NTXY[:,0]))
    #Get the activity of each neurons in the same matrix : active 
    activity = []
    positions = []
    for nn in neurons:
        #Get positions
        idx = list(NTXY[:,0]).index(nn)
        positions.append(pos[idx])
        #single neuron spike train
        act = []
        for i,sdr in enumerate(NTXY[:,0]):
            #nest ids start at 1
            if sdr == nn:
                act.append(spk[i])
        activity.append(np.array(act))
    
    if tmin is None and tmax is None:
        tmin , tmax = np.min(spk) , np.max(spk)
    
    time_array = np.arange(tmin,tmax+dt,dt)
    
    fig, axa = plt.subplots(nrow,ncol)
    for ax in axa.flatten():
        ax.scatter(np.array(positions)[:,0],np.array(positions)[:,1], marker = '.', c = 'k', s = 2)
        
    for tt,time in enumerate(time_array):
        tksup = np.where(spk > time + dt/2)[0][0]
        tkinf = np.where(spk > time - dt/2)[0][0]
    
        for i in np.arange(1,nrow+1,1):
            if tt in np.arange((i-1)*ncol,i*ncol,1).astype(int):
                line = i-1
                column = tt-ncol*(i-1)
                
        axa[line][column].scatter(np.array(pos[tkinf:tksup])[:,0],np.array(pos[tkinf:tksup])[:,1], c = 'r', s = 4)
        axa[line][column].set_title (str(int(time))+'ms')
        axa[line][column].set_aspect('equal')
        
        if line < nrow - 1:
            axa[line][column].set_xticklabels([])
        elif line == nrow - 1:
            axa[line][column].set_xlabel('x ($\mu$m)')
            
        if column != 0:
            axa[line][column].set_yticklabels([])
        elif column == 0:
            axa[line][column].set_ylabel('y ($\mu$m)')
        
    if show == True:
        plt.show()
    
    if save_file is not None:
        fig.savefig(save_file)
    
def spatial_activity_fromspikes(spikes, pos, positions, time_array, dt, ncol, nrow, show = True, save_file = None):
    '''
    '''
    
    fig, axa = plt.subplots(nrow,ncol)
    
    for ax in axa.flatten():
        ax.scatter(np.array(positions)[:,0],np.array(positions)[:,1], marker = '.', c = 'k', s = 2)
    
    for tt,time in enumerate(time_array):
        tksup = np.where(spikes > time + dt/2)[0][0]
        tkinf = np.where(spikes > time - dt/2)[0][0]
        
        for i in np.arange(1,nrow+1,1):
            if tt in np.arange((i-1)*ncol,i*ncol,1).astype(int):
                line = i-1
                column = tt-ncol*(i-1)
        
        axa[line][column].scatter(np.array(pos[tkinf:tksup])[:,0],np.array(pos[tkinf:tksup])[:,1], c = 'r', s = 4)
        axa[line][column].set_title (str(int(time))+'ms')
        axa[line][column].set_aspect('equal')
        
        

        if line == nrow - 1:

            axa[line][column].set_xlabel('x ($\mu$m)')
        elif line != nrow-1:

            axa[line][column].set_xticklabels([])
            
        if column != 0:
            axa[line][column].set_yticklabels([])
        elif column == 0:
            axa[line][column].set_ylabel('y ($\mu$m)')
            
    if show == True:
        plt.show()
    
    if save_file is not None:
        fig.savefig(save_file)


if __name__ == '__main__':
    spatial_activity_fromfile(activity_file = '/home/mallory/Documents/These/september2018/RP report/Simu1/N1000_spikes2.txt', dt = 25, tmin = 4800, tmax = 5075, ncol = 4, nrow = 3, show = True, save_file = None)
