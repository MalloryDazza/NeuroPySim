#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nest



def susceptibility(V, w, model, params):
    '''
    Return the neuronal susceptibility of
    an aEIF neuron.
    
    Parameters
    ----------
    V : double or numpy array
        Membrane potential of the neuron.
    w : double or numpy array
        Adaptation current potential of the neuron.
    model : str
        NEST model (e.g. "aeif_psc_alpha").
    params : dict
        Non-default parameters.
    '''
    import numpy as np
    from scipy.special import lambertw
    # complete params
    p_def = nest.GetDefaults(model)
    p_def.update(params)
    # get params
    EL  = p_def["E_L"]
    Vth = p_def["V_th"]
    gL  = p_def["g_L"]
    DT  = p_def["Delta_T"]
    Ie  = p_def["I_e"]
    a   = p_def["a"]
    tw  = p_def["tau_w"]
    tm  = p_def["C_m"] / gL
    # define w_min and V-nullcline function
    w_min = Ie + gL*(EL + DT - Vth)

    # test the conditions
    if isinstance(V, (float, int, np.integer, np.floating)):
        Vnv = V_nv(w)
        if V <= Vnv and w >= w_min:
            return (Vnv - V)/DT
        else:
            h = 1 if V > Vth else 0
            return (EL - V + 2*(1-h)*(V - Vth))/DT + np.exp(h*(V-Vth)/DT) + (Ie-w)/(gL*DT)
    else:
        res  = []
        #res2 = []
        VVnv = V_nv(w)
        hh   = V > Vth 
        for v, W, Vnv, h in zip(V, w, VVnv, hh):
            if v <= Vnv and W >= w_min:
                res.append((v - Vnv)/DT) # Sc = -q
                #res2.append(5)
            else:
                res.append((EL - v + 2*(1-h)*(v - Vth))/DT + np.exp(h*(v-Vth)/DT) + (Ie-W)/(gL*DT))
                #res2.append(0)
        return np.array(res)

def V_nv(w):
    arg_lw = -np.exp((gL*(EL-Vth) + Ie - w)/(gL*DT))
    lw     = None
    if isinstance(w, (float, int, np.integer, np.floating)):
        if arg_lw < -1/np.e:
            return np.NaN
        else:
            lw = np.real(lambertw(arg_lw, -1))
        return EL + (Ie - w)/gL - DT*lw
    else:
        lw      = np.zeros(len(w))
        valid = np.greater(arg_lw, -1/np.e)
        lw[~valid] = np.NaN
        lw[valid]  = np.real(lambertw(arg_lw[valid], -1))
        return EL + (Ie - w)/gL - DT*lw

def w_Vnull(Vs, I):
    gL = p_RS_osc["g_L"]
    EL = p_RS_osc["E_L"]
    VT = p_RS_osc["V_th"]
    DT = p_RS_osc["Delta_T"]
    Ie = p_RS_osc["I_e"]
    return -gL * ((Vs-EL) - DT*np.exp((Vs-VT) / DT)) + Ie + I
