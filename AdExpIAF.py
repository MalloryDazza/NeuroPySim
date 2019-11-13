#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nest

def normalize_params(params):
    '''
    return all parameters in a dictionnary, but normalized
    '''
    p_osc = nest.GetDefaults('aeif_psc_alpha')
    ret = {}
  
    if params is not None:
        p_osc.update(params)
    
    gL   = p_osc["g_L"]
    EL   = p_osc["E_L"]
    Vth  = p_osc["V_th"]
    DT   = p_osc["Delta_T"]
    Ie   = p_osc["I_e"]
    a    = p_osc['a']
    b    = p_osc['b']
    Vr   = p_osc["V_reset"]
    Cm   = p_osc['C_m']
    tauw = p_osc['tau_w']
    taum = Cm / gL
    
    ret["E_L"]  = p_osc["E_L"] - Vth
    ret["E_L"] /= DT
    
    ret["I_e"]   = p_osc["I_e"] / (gL * DT)
    
    ret['tau_w'] = p_osc['tau_w'] / taum
    
    ret['a'] = p_osc['a'] / gL
    
    ret['V_reset'] = p_osc["V_reset"] - Vth
    ret['V_reset'] /= DT

    ret['b']  = b / ( gL * DT)
    
    ret['Delta_T'] = DT
    ret['g_L'] = gL
    ret['V_th'] = Vth
    
    return ret  

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

def w_Vnull(params, Vs, I=0):
    '''
    return the V nullcline in the 
    AdExp model where nest defaults 
    values and used if not present 
    in params for all values of the 1D array Vs
    '''
    p_def = nest.GetDefaults('aeif_psc_alpha')
    p_def.update(params)
    gL = p_def["g_L"]
    EL = p_def["E_L"]
    VT = p_def["V_th"]
    DT = p_def["Delta_T"]
    Ie = p_def["I_e"]
    return -gL * ((Vs-EL) - DT*np.exp((Vs-VT) / DT)) + Ie + I
