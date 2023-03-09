
import numpy as np
import lmfit
import emcee
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
#solves python>=3.8 issues, see https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
from multiprocessing import Pool
from contextlib import closing
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import sys
import copy
#import astropy.units as u
#from numpy.polynomial import Polynomial as polyn
import logging
import warnings
warnings.filterwarnings('ignore')
import radvel

#--- Change LOG level ----------------------------------------------------------
LOG_LEVEL = "warning"
#LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
################################################################################

def saltirise(params,velo,K_p,obs,fixpar,data,err=None,method='leastsq',func='Gauss'):
    '''
    Saltirise your data!

    This fuction will prefom an lmfit fitting to your CCF map to obtain first parameters (default: least squares).
    
    params: lmfit parameter file, specifying: K,v_sys,height,sum_amp,dif_amp,sigmax1,sigmax2
    x,y: np.meshgrid(v_sys,K_p) - explored systemic velocity and semi amplitude.
    obs: [obstimes,weights,planet]- observation parameters
    fixpar: np.array([period,T0,ecc,omega]) - orbital parameters (each parameters has to be an array od len(K_p))
    data: 2d CCF map
    err: (optional) error of map data

    '''
    x,y = np.meshgrid(velo,K_p)
    #build dictionary of parameters
    if err is None:
        kws={'obs': obs,'fixpar':fixpar.T,'data':data,'func':func}#,'err':err_data}
    else:
        kws={'obs': obs,'fixpar':fixpar.T,'data':data,'err':err,'func':func}

    return lmfit.minimize(fit_residual, params=params, args=(x,y),kws=kws, method=method)    

def fit_residual(pars,x, y,obs,fixpar,data=None,err=None,func='Gauss'):

    #fixpar = [period,T_0,ecc,omega]
    # unpack parameters: extract .value attribute for each parameter
    
    parvals = pars.valuesdict()
    v_sys = parvals['v_sys']
    K = parvals['K']
    height = parvals['height']
    sum_amp = parvals['sum_amp']
    dif_amp = parvals['dif_amp']
    sigmax1 = parvals['sigmax1']
    sigmax2 = parvals['sigmax2']
    
    #unpack observation parameters:
    obstimes,weigths,planet = obs
    theta, offset = [0.,0.]
    
    
    model  = np.zeros(x.shape)#(len(K_p),len(x)))
    for i in range(len(y.T[0])):

        
        if planet:
            #1.) comparison model with exact orbital parameters at start position
            #parm = [K, K_oth[i], theta[i], v_sys]
            parm = [K, 0., theta, v_sys]
            rv0,_ = radvel.RV_Model(parm,obstimes,fixpar[i])
            #2.) Model with exact orbital parameters at explored position
            parm = [y.T[0][i],0., theta, offset]
            rv,_ = radvel.RV_Model(parm,obstimes,fixpar[i]) 
        else:
            #1.) comparison model with exact orbital parameters at start position
            #parm = [K_oth[i],K, theta[i], v_sys]
            parm = [0.,K, theta, v_sys]
            _,rv0 = radvel.RV_Model(parm,obstimes,fixpar[i])
            #2.) Model with exact orbital parameters at explored position
            parm = [0.,y.T[0][i], theta, offset]
            _,rv = radvel.RV_Model(parm,obstimes,fixpar[i])

                    
        #apply expected semi amplitude K
        rv=-rv+rv0
        #Derive gausian with mean at each of these points and return average
        xs,rv = np.meshgrid(x[0],rv)
        if func=='Lorentz':
            model[i] = height + np.average(DbLor1d(xs,mean=rv,sum_amp=sum_amp,dif_amp=dif_amp,
                                        sigmax1=sigmax1,sigmax2=sigmax2),weights=weigths,axis=0)
        else:
            model[i] = height + np.average(DbGaus1d(xs,mean=rv,sum_amp=sum_amp,dif_amp=dif_amp,
                                        sigmax1=sigmax1,sigmax2=sigmax2),weights=weigths,axis=0)
    
    if data is None:
        return model.flatten()
    if err is None:
        return (data - model).flatten()
    return ((data-model)/err).flatten()

def lnlike(pos):#x,y,params,varnames,obs,fixpar,data=None,err=None,func='Gauss'):
    
    '''
    residual model
    Works with lmfit params set (includes limit for the posteriors
    pos are the posteriors for 'K', 'v_sys', 'height', 'sum_amp', 'dif_amp', 'sigmax1', 'sigmax2', 'jitter' 
    '''
    #all parameters have been defined as 'global' in the 'run_mcmc' function call and can be used here.
    #This is important to use multiprocessing for emcee.
    x,y,params,varnames,obs,fixpar,data,err,func = args
    
    #copy and update param object for each parameter
    #assume uniform priors
    parcopy = params.copy()
    for i, p in enumerate(varnames):
        v = pos[i]
        if (v < parcopy[p].min) or (v > parcopy[p].max):
            return -np.inf
        parcopy[p].value = v
        
    #unpack relevant parameters
    parvals = parcopy.valuesdict()
    K = parvals['K']
    v_sys = parvals['v_sys']
    height = parvals['height']
    sum_amp = parvals['sum_amp']
    dif_amp = parvals['dif_amp']
    sigmax1 = parvals['sigmax1']
    sigmax2 = parvals['sigmax2']
    jitter  = parvals['jitter']

    #unpack observation parameters
    obstimes,weigths,planet = obs
    theta, offset = [0.,0.]
    
    #derive residuals
    model  = np.zeros(x.shape)#(len(K_p),len(x)))
    for i in range(len(y.T[0])):

        
        if planet:
            #1.) comparison model with exact orbital parameters at start position
            parm = [K, 0., theta, v_sys]
            rv0,_ = radvel.RV_Model(parm,obstimes,fixpar[i])
            #2.) Model with exact orbital parameters at explored position
            parm = [y.T[0][i],0., theta, offset]
            rv,_ = radvel.RV_Model(parm,obstimes,fixpar[i]) 
        else:
            #1.) comparison model with exact orbital parameters at start position
            parm = [0.,K, theta, v_sys]
            _,rv0 = radvel.RV_Model(parm,obstimes,fixpar[i])
            #2.) Model with exact orbital parameters at explored position
            parm = [0.,y.T[0][i], theta, offset]
            _,rv = radvel.RV_Model(parm,obstimes,fixpar[i])

                    
        #apply expected semi amplitude K
        rv=-rv+rv0
        #Derive gausian with mean at each of these points and return average
        xs,rv = np.meshgrid(x[0],rv)
        if func=='Lorentz':
            model[i] = height + np.average(DbLor1d(xs,mean=rv,sum_amp=sum_amp,dif_amp=dif_amp,
                                        sigmax1=sigmax1,sigmax2=sigmax2),weights=weigths,axis=0)
        else:
            model[i] = height + np.average(DbGaus1d(xs,mean=rv,sum_amp=sum_amp,dif_amp=dif_amp,
                                        sigmax1=sigmax1,sigmax2=sigmax2),weights=weigths,axis=0)
    
    if data is None:
        return model#.flatten()
    if err is None:
        #add errors of 'fixed' parameters
        #K_oth does not influences the result
        #fixpar includes orbital parameters: period,T_0,ecc,omega
        #those are not varied, but if then they should be included like:
        #lnprior = -0.5*(55137.7628 - T_0)**2/(0.0015)**2
        #lnprior += -0.5*(7.45635 - period)**2/(0.000018)**2
        #add for actual 
        lnprior = 0
        
        wt = 1/(jitter**2)
        return -0.5*(np.sum((data-model)**2*wt - np.log(wt))) + lnprior#.flatten()
    
    lnprior = 0
    wt = 1/(jitter**2 + err**2)
    return -0.5*(np.sum((data-model)**2*wt - np.log(wt))) + lnprior

def run_MCMC(params,x,y,e_p,obs,fixpar,data,err=None,filename = "save.h5",overwrite=True,nwalkers=50,nburnin=1000,nsteps=100,thin=40,n_threads=4,func='Gauss'):
    
    #Read variables and bounds from parameter file
    varvals = []
    varerrs = []
    varnames = []
    for p in params:
        if params[p].vary:
            varnames.append(p)
            varvals.append(params[p].value)
            if params[p].stderr is None:
                varerrs.append(0.1*(params[p].max-params[p].min))
            else:
                varerrs.append(params[p].stderr)
                
    varvals = np.array(varvals)
    varerrs = np.array(varerrs)
    n_var = len(varvals)
    #define 'args' as global parameter to be read by lnlike
    global args
    args = (x,y,params,varnames,obs,fixpar.T,data,err,func)

    #Create valid starting parameters for each walker
    pos = []
    for i in range(nwalkers):
        params_tmp = params.copy()
        lnlike_i = -np.inf
        while lnlike_i == -np.inf:
            pos_i = varvals + varerrs*np.random.randn(n_var)*e_p
            lnlike_i = lnlike(pos_i)#,x, y,varnames=varnames,params=params_tmp,obs=obs,fixpar=fixpar.T,data=data)#,err=fit_err)
        pos.append(pos_i)

    def run_sampler(pos,nwalkers, n_var,nsteps,backend,burn=True):
        #run sampler parallel on several CPUs
        with closing(Pool(n_threads)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, n_var,lnlike,pool=pool,backend=backend)
            if burn:
                print('Running burn-in ...')
                pos, _, _ = sampler.run_mcmc(pos, nburnin, store=False, 
                                skip_initial_state_check=True, progress=True)
                sampler.reset()
            print('Running sampler ...')
            state = sampler.run_mcmc(pos, nsteps, skip_initial_state_check=True, 
                                thin_by=thin, progress=True, store=True)
        return sampler
    
    #Check filename exists
    burn = True
    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, n_var)
        else: #continue from last mcmc run
            print('Continue last MCMC run, skipping burn-in ...')
            backend = emcee.backends.HDFBackend(filename)
            pos = backend.get_chain()[-1,:,:]
            burn = False

    else:
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, n_var)

    sampler = run_sampler(pos,nwalkers, n_var,nsteps,backend,burn=burn)    
    
    return sampler,varnames
    

def DbGaus1d(xs,mean,sum_amp,dif_amp,sigmax1,sigmax2):
    '''
        Double Gausian at x-position
    '''
    if dif_amp==0:
        amplitude2=0
    else:
        amplitude2 = sum_amp/(1/dif_amp + 1)
    amplitude1 = sum_amp/(dif_amp + 1)
    #print(amplitude1,amplitude2)
    
    xp1 = (xs - mean) #- (y - centery1)
    gausx1 = amplitude1 * np.exp(-((xp1/sigmax1)**2)/2)
    #add a second gaus only in x axis
    gausx2 = amplitude2 * np.exp(-((xp1/sigmax2)**2)/2)
    
    model = gausx1+gausx2
    return model

def DbLor1d(xs,mean,sum_amp,dif_amp,sigmax1,sigmax2):
    '''
        Double Lorentzian funtion at x-position
    '''

    if dif_amp==0:
        amplitude2=0
    else:
        amplitude2 = sum_amp/(1/dif_amp + 1)
    amplitude1 = sum_amp/(dif_amp + 1)
    #print(amplitude1,amplitude2)
    
    xp1 = (xs - mean) #- (y - centery1)
    lorx1 = amplitude1 * sigmax1**2 / ( sigmax1**2 + ( xp1 )**2)
    #add a second gaus only in x axis
    lorx2 = amplitude2 * sigmax2**2 / ( sigmax2**2 + ( xp1 )**2)
    
    model = lorx1+lorx2
    return model

def plot_axis2D(x,y,data,xlabel='v_{sys}\,[km\,s^{-1}]',ylabel='K_{P}\,[km\,s^{-1}]', outim='_.png',savefig=True,xlim=None,vmin=None,vmax=None):

    '''
    Plots 2D data with right axis labels.
    Data need to be equstitant sampled (Gaps are not properly displayed).
    xlim: tuple with starting and end value in units of x.
    '''

    y_values_full = y

    if xlim!=None:
        x_part = [(x>=xlim[0]) & (x<=xlim[1])]
        x_values_full = x[x_part]
        data_plot=(data.T[x_part]).T
    else:
        x_values_full = x
        data_plot = copy.deepcopy(data)
    
    fig, ax = plt.subplots(figsize=(8,6))

    #derive optimal labels (values are assumed to continuus)
    if len(y_values_full)>=10:
            y_values      = np.linspace(y_values_full[0],y_values_full[-1],10)#len(x_values_full)-1)
    else:
            y_values      = np.linspace(y_values_full[0],y_values_full[-1],len(y_values_full)-1)
    y_values = ["%.1f" % y for y in y_values]
    y_positions = np.linspace(0,len(y_values_full)-1,len(y_values))

    

   

    if len(x_values_full)>=11:
            x_values      = np.linspace(x_values_full[0],x_values_full[-1],11)#len(x_values_full)-1)
    else:
            x_values      = np.linspace(x_values_full[0],x_values_full[-1],len(x_values_full)-1)
    x_values = ["%.1f" % x for x in x_values]
    x_positions = np.linspace(0,len(x_values_full)-1,len(x_values))

    #y_smooth =polyn.fit(y_values_full,np.arange(len(y_values_full)),2)
    #y_smooth2=polyn.fit(np.arange(len(y_values_full)),y_values_full,2)

    #ax.axvline(y_smooth(K2),ls='--',color='black',alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_values, rotation=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values) #empty between each subplot
    ax.set_xlabel( r'$\rm '+xlabel+'$',fontsize=14)
    ax.set_ylabel( r'$\rm '+ylabel+'$',fontsize=14)

    if vmin==None or vmax==None:
        plt.imshow(data_plot,cmap='Blues',aspect='auto',origin='lower',alpha=1)
    else:
        plt.imshow(data_plot,cmap='Blues',aspect='auto',origin='lower',alpha=1,vmin=vmin,vmax=vmax)

    plt.colorbar()
    plt.tight_layout()
    if savefig:
        plt.savefig(outim,dpi=300)
    plt.show()
    del data_plot

def cutplot(data,model,velo,K_p,v_sys,K_opt,save=False,file='.png',plotting=True,xlabel1 ='v_{sys}\,[km\,s^{-1}]',xlabel2 ='K_{p}\,[km\,s^{-1}]',margin=0.1):
    '''
    Generates a cut plot through the CCF map at given coordinates (v_sys,K_opt).


    '''

    #find the data (and model) values, closest to the resulting parameters

    velo_min = np.where(velo<=v_sys)[0][-1]
    K_p_min = np.where(K_p<=K_opt)[0][-1]
    ycut = [K_p_min,K_p_min+1]#22,23,24]
    xcut=[velo_min,velo_min+1]#np.linspace(85,95,11).astype(int)#[89,90,91]
    
    fig, axs = plt.subplots(1, 2,figsize=(10,5),sharey=True)
    
    #define y scale of plot
    min = np.min(data)
    max = np.max(data)
    if np.sign(min) == np.sign(max):
        axs[0].set_ylim((1-margin)*min,(1+margin)*max)
        axs[1].set_ylim((1-margin)*min,(1+margin)*max)
    else:
        axs[0].set_ylim((1+margin)*min,(1+margin)*max)
        axs[1].set_ylim((1+margin)*min,(1+margin)*max)
    
    for i in ycut:
        axs[0].set_ylabel(r'$\rm '+'CCF'+'$',fontsize=14)
        axs[0].set_xlabel(r'$\rm '+xlabel1+'$',fontsize=14)
        axs[0].plot(velo,data[i],marker='o',color='black')
        axs[0].plot(velo,model[i],color='red',alpha=0.5)
    
    for i in xcut:
        axs[1].set_xlabel(r'$\rm '+xlabel2+'$',fontsize=14)
        axs[1].plot(K_p,data.T[i],color='black',marker='o',ls='')
        axs[1].plot(K_p,model.T[i],color='red',alpha=0.5)
    plt.tight_layout()
    
    if save:
        plt.savefig(file,dpi=300)
    if plotting:
        plt.show()
    plt.close()

def rem_walkers(samples,parameter=0,sigma=1):
    '''
    removes walkers that are on average > sigma away from the others (
    !! Check it's not removing a valid second solution !!

    samples - emcee samples to be cleaned
    parameter - parameter to be evaluated (My affect all params, but only one has to be selected).
    '''
    data = np.average(samples[:,:,:],axis=0)
    #remove walkers that are on average > sigma away
    val = np.median(data.T[parameter])
    err = np.std(data.T[parameter])
    
    part = np.abs(data.T[parameter]-val)<=sigma*err
    #plt.plot(np.abs(data.T[parameter]-val))
    #plt.axhline(err)
    #plt.show()
    return samples[:,part,:]
    