# -*- coding: utf-8 -*-

import numpy as np
from numpy import random
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib import pyplot
from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
#import kf_book.book_plots as bp


def rts_kalman(xaxis, data, Q_process, R_state, x_state, dx_state, show_velocity=False):
    
    fk = KalmanFilter(dim_x=2, dim_z=1)
    
    fk.x = np.array([x_state, dx_state])      # state (x and dx)

    fk.F = np.array([[1., 1],
                     [0., 1]])    # state transition matrix

    fk.H = np.array([[1, 0.]])    # Measurement function
    
    fk.P = 100000.                     # covariance matrix
    
    fk.R = R_state                   # state uncertainty
    
    fk.Q = Q_process                      # process uncertainty
    #fk.Q = Q_discrete_white_noise(dim = 2, dt = .5, var = .1)
    #fk.Q = Q_continuous_white_noise( dim = 2, dt = 1, spectral_density = .1)
    # create noisy data
    #zs = np.asarray([t + randn()*noise for t in range (40)])
    zs = np.nan_to_num(data)

    # filter data with Kalman filter, than run smoother on it
    mu, cov, _, _ = fk.batch_filter(zs)
    M, P, C, _ = fk.rts_smoother(mu, cov)

    # plot data
    if show_velocity:
        index = 1
        print('gu')
    else:
        index = 0
    if not show_velocity:
        fig = plt.figure()
        #plot_measurements(zs, lw=1, lines=True)
        plt.plot(xaxis, zs, c='silver',)
    #fig = plt.figure(figsize=(20,40))
    #plt.plot(xaxis, M[:, index], c='b', label='RTS', linewidth = 3)
    plt.plot(xaxis, mu[:, index], c='g', label='KF output',linewidth =3)
    if not show_velocity:
        N = len(zs)
        #plt.plot([0, N], [0, N], 'k', lw=2, label='track') 
    plt.legend(loc=4)
    plt.show()
    return  mu
#mu = rts_kalman(np.sum(norm_result_unpumped[0][:,1:],axis=0))

def fx(x, dt):
    xout = np.empty_like(x)
    xout[0] = x[1] * dt + x[0]
    xout[1] = x[1]
    return xout

def hx(x):
    return x[:1] # return position [x] 

def ukf_kalman(xaxis, data, Q_process, dt, R_state, x_state, dx_state, show_velocity=False):
    
    points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=0.)
    
    
    #dt = 12
    
    fk = UKF(dim_x=2, dim_z=1,dt=dt, fx=fx, hx=hx, points=points)
    
    #fk.x = np.array([x_state, dx_state])      # state (x and dx)

    #fk.F = np.array([[1., 1],
    #                 [0., 1]])    # state transition matrix

    
    
    fk.P = 10000000.                     # covariance matrix
    
    fk.R = R_state                   # state uncertainty
    
    fk.Q = Q_process                      # process uncertainty
    #fk.Q = Q_discrete_white_noise(dim = 2, dt = .5, var = .1)
    #fk.Q = Q_continuous_white_noise( dim = 2, dt = 1, spectral_density = .1)
    # create noisy data
    #zs = np.asarray([t + randn()*noise for t in range (40)])
    zs = np.nan_to_num(data)

    # filter data with Kalman filter, than run smoother on it
    mu, cov, = fk.batch_filter(zs)
    M, P, C = fk.rts_smoother(mu, cov)

    # plot data
    if show_velocity:
        index = 1
        print('gu')
    else:
        index = 0
    if not show_velocity:
        fig = plt.figure()
        #plot_measurements(zs, lw=1, lines=True)
        plt.plot(xaxis, zs, c='silver')
    #fig = plt.figure(figsize=(20,40))
    plt.plot(xaxis, M[:, index], c='b', label='RTS', linewidth = 3)
    plt.plot(xaxis, mu[:, index], c='g', label='KF output',linewidth =3)
    if not show_velocity:
        N = len(zs)
        #plt.plot([0, N], [0, N], 'k', lw=2, label='track') 
    plt.legend(loc=4)
    plt.show()
    return  mu

def floatrange(x, y, jump):
  while x < y:
    yield x
    x += jump
    
def pp_cut(tof_low,tof_high,result1_new,pptime,dldtime,color='terrain',log=False,nout=False):
    
    from scipy import interpolate
    
    
    k_low=find_nearest(dldtime,tof_low)[0][0]
    #index_low = np.where(dldtime==(k_low))
    #index_2=index_low[0][0]
    index_2=k_low
    
    k_high=find_nearest(dldtime,tof_high)[0][0]
    #index_high = np.where(dldtime==(k_high))
    #index_1=index_high[0][0]
    index_1=k_high
    
    result_cut =result1_new[index_2:index_1,:]
    
    
    #plt.imshow(result_cut,cmap="nipy_spectral")
    if nout==False:
        plt.figure()
        if log==False:
            plt.imshow(result_cut, extent = (pptime[0],pptime[-1],dldtime[index_1],dldtime[index_2]),cmap=color)
        else:
            plt.imshow(result_cut, extent = (pptime[0],pptime[-1],dldtime[index_1],dldtime[index_2]),norm=LogNorm(),cmap=color)
             
    cut_array=np.sum(result_cut,axis=0)
    #mean = cut_array.mean(axis=0)
    extent = (pptime[0],pptime[-1])
    extent2 = (dldtime[index_1],dldtime[index_2])

    if nout==False:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

    pp_div=abs((extent[0]-extent[-1])/result_cut.shape[1])
    grad=np.gradient(result_cut)
    if nout==False:
        ax1.set_xlim(xmin=pptime[0], xmax=pptime[-1])
    pp_range=list(floatrange(extent[0],extent[1],pp_div))
    pp_range_smo=result_cut.shape[1]/100
    #ax1.plot(pp_range[:-1],grad[0])#-mean)def pp_cut(tof_low,tof_high,result_cut,pptime,dldtime):
    if nout==False:
        try:
            if log==False:
                ax1.imshow(result_cut, extent = (pptime[0],pptime[-1],dldtime[index_1],dldtime[index_2]),cmap=color)
            else:
                ax1.imshow(result_cut, extent = (pptime[0],pptime[-1],dldtime[index_1],dldtime[index_2]),norm=LogNorm(),cmap=color)
            ax1.imshow(result_cut, extent = (pptime[0],pptime[-1],dldtime[index_1],dldtime[index_2]),norm=LogNorm(),cmap=color)
            ax2.plot(pp_range[:],cut_array,color='r')#-mean)
            ax1.set_xlabel('delay time [ps]')
            ax1.set_ylabel('yield')
            print(cut_array.min())
        except:
            ax2.plot(pp_range[:-1],cut_array,color='r')
            if log==False:
                ax1.imshow(result_cut, extent = (pptime[0],pptime[-1],dldtime[index_1],dldtime[index_2]),cmap=color)
            else:
                ax1.imshow(result_cut, extent = (pptime[0],pptime[-1],dldtime[index_1],dldtime[index_2]),norm=LogNorm(),cmap=color)
            ax1.set_xlabel('delay time [ps]')
            ax1.set_ylabel('yield')
    return (result_cut, pp_range, cut_array, pp_range_smo, grad)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return np.where(array == array[idx])  

def find_nearest_range(array, value1, value2):
    array = np.asarray(array)
    idx = (np.abs(array - value1)).argmin()
    idx2 = (np.abs(array - value2)).argmin()
    return np.where(array == array[idx]), np.where(array == array[idx2]) 

def pumpProbePlot(runNumber,begin,end,timeFrom, timeTo, timeStep, delayFrom, delayTo, delayStep, N, processor=None, returnProc=False, useBAM=False):
    if processor==None:
        processor = DldFlashProcessor()
        processor.runNumber=runNumber
        
        try:
            processor.appendDataframeParquet()
            print('schnell')
        except:
            processor.readData(path='/asap3/flash/gpfs/pg2/2019/data/11006509/raw/hdf/online-1/')
            print('langsam')
    if useBAM:
        processor.postProcess(bamCorrectionSign=1)
        axisname='pumpProbeTime'
        
    else:
        processor.postProcess()
        axisname='delayStage'
        
    processor.resetBins()  
    #processor.addFilter('opticalDiode', 1000,4000)
    #processor.addFilter('macroBunchPulseId',0,243691)
    #processor.addFilter('gmdBda',2.,10)
    ToF = processor.addBinning('dldTime', timeFrom, timeTo, timeStep)
    PPT = processor.addBinning(axisname, delayFrom, delayTo, delayStep)
    processor.addBinning('dldMicrobunchId',begin,end,end-begin)
    

    result = processor.computeBinnedData()
    result = np.sum(result,axis=2)
    result = np.nan_to_num(result)
    result1 = np.zeros_like(result)
    
    #for i in range(result.shape[0]):
        #if useBAM:
            #hist=processor.pumpProbeHistogram
        #    hist=processor.histograms
        #    hist=hist['pumpProbeTime'].compute()
        #else:
        #    #hist=processor.delaystageHistogram
        #    hist=processor.histograms
        #    hist=hist['pumpProbeTime'].compute()
            
        #result1[i,:] = result[i,:] / hist
    result1=processor.normalizeDelay(result)
    
    mean = np.nan_to_num(result1).mean(axis=1) 

    dldtime = np.arange(timeFrom,timeTo-timeStep,timeStep)
    pptime = np.arange(delayFrom,delayTo-delayStep,delayStep)


    plt.figure()
    try:
        plt.plot(ToF,mean[:])
    except:
        plt.plot(ToF,mean[:-1])

    plt.grid()
    plt.xlabel('dldTime (ns)')
    plt.ylabel('Counts')

    rolling_result1 = np.convolve(np.sum(result1[:,:],axis=0), np.ones((N,))/N, mode='valid')

    plt.figure(figsize = (12,6))

    try:
        plt.plot(PPT, result1[:,:].sum(axis=0))
        #plt.axvline(x=80.5, linewidth=2, alpha=0.4 , color='cyan')
    except:
        plt.plot(PPT, result1[:,:-1].sum(axis=0))
        #plt.axvline(x=80.5, linewidth=2, alpha=0.4 , color='cyan')

    plt.xlabel('pumpProbedelay (ps)')
    plt.ylabel('energy histogram')
    plt.grid()

    # compute difference
    diff = np.zeros_like(result)
    for i in range(diff.shape[1]):
        diff[:,i]=result1[:,i]-mean #result1


    # give plots of result1 and diff
    plt.figure(figsize = (12,6))
    plt.imshow(result1,aspect = 'auto',extent = (pptime[0],pptime[-1],dldtime[-1],dldtime[0]),cmap='terrain')
    #plt.axvline(x=80.5, linewidth=2, alpha=0.4 , color='cyan')
    plt.colorbar()
    plt.ylabel('dldTime (ns)')
    plt.xlabel('pumpProbedelay (ps)')


    if returnProc:
        return (dldtime,mean,result, result1, diff,pptime,PPT, processor)

    return (dldtime,mean,result, result1, diff,pptime,PPT)

def orbitals(runNumber,begin,end,*args, processor=None, returnProc=False):
    if processor==None:
        processor = DldFlashProcessor()
        processor.runNumber=runNumber
        processor.readData(path='/asap3/flash/gpfs/pg2/2019/data/11006509/raw/hdf/online-1/')
    n=len(args)
    print(n)
    r=int(np.floor(np.sqrt(n)))
    c=int(np.ceil(n/r))
    #fig = plt.figure(figsize=(5,5))
        
    #ax1 = plt.subplot2grid((r+1,c), (0,0), colspan=2, rowspan=n)
    
    #fig, axes = plt.subplots(nrows=r, ncols=c,figsize=(5*c,5*r))
    fig, axes = plt.subplots(nrows=r, ncols=c)
    
    #spec3 = gridspec.GridSpec(ncols=c, nrows=r+1,wspace=2.0, hspace=2.0)
    
    #axes.flat[1] = plt.subplot2grid((r,c), (3,,figsize=(5*c,5*r))
    
    processor.postProcess()
    #processor.readDataframes(runNumber)
    orbitals_all =[]
    
    for i,item in enumerate(args):
        
        processor.resetBins()
        processor.addBinning('dldMicrobunchId',begin,end,end-begin)
        processor.addBinning('dldPosX',405,905,4)
        processor.addBinning('dldPosY',402,902,4)
        processor.addBinning('dldTime',item[0],item[1],processor.TOF_STEP_TO_NS*8)
        
        result=processor.computeBinnedData(skip_metadata=True)
        
        #processor.postProcess()
        #result_norm = processor.normalizePumpProbeTime(result)
        
        result1 = np.sum(result[:,:,:,:], axis=(0,3))
        #fig.add_subplot(spec3(r,c))
        ##ax= plt.subplot(r+1,c,i+1)
        #fig.add_subplot(spec3[i//2,i//3])
        axes.flat[i].imshow(np.sum(result[:,:,:,:], axis=(0,3)),cmap='nipy_spectral')
        axes.flat[i].set_title(str(item))
        orbitals_all.append(result1)        
    

        #fig, axes=plt.subplots(nrows=1, ncols=2,figsize=(14,5))
        #ax1,ax2=axes
        #im=ax1.imshow(result1)
        #ax1.set_title('LUMO dldTime 193.5 ~ 194.5ns')
        #fig.colorbar(im, ax=ax1)
    if returnProc:
        return(orbitals_all, processor)
    
    return(orbitals_all)


def delayorbitals(runNumber,*args, processor=None, returnProc=False):
    if processor==None:
        processor = DldFlashProcessor()
        processor.runNumber=runNumber
        processor.readData(path='/asap3/flash/gpfs/pg2/2019/data/11006509/raw/hdf/online-1/')
    n=len(args)
    print(n)
    r=int(np.floor(np.sqrt(n)))
    c=int(np.ceil(n/r))
    #fig = plt.figure(figsize=(5,5))
        
    #ax1 = plt.subplot2grid((r+1,c), (0,0), colspan=2, rowspan=n)
    fig, axes = plt.subplots(nrows=r, ncols=c,figsize=(5*c,5*r))
    #spec3 = gridspec.GridSpec(ncols=c, nrows=r+1,wspace=2.0, hspace=2.0)
    
    #axes.flat[1] = plt.subplot2grid((r,c), (3,,figsize=(5*c,5*r))
    
    processor.postProcess(bamCorrectionSign=1)    #processor.readDataframes(runNumber)
    orbitals_all =[]
    
    for i,item in enumerate(args):
        
        processor.resetBins()

        #processor.addBinning('dldMicrobunchId',begin,end,end-begin)
        processor.addBinning('dldPosX',405,905,4)
        processor.addBinning('dldPosY',402,902,4)
        processor.addBinning('dldTime',item[0],item[1],processor.TOF_STEP_TO_NS*8)
        processor.addBinning('pumpProbeTime',item[2],item[3],item[3]-item[2])
        result=processor.computeBinnedData()
        #processor.postProcess()
        #result_norm = processor.normalizePumpProbeTime(result)
        
        #fig.add_subplot(spec3(r,c))
        ##ax= plt.subplot(r+1,c,i+1)
        #fig.add_subplot(spec3[i//2,i//3])
        axes.flat[i].imshow(np.sum(result[:,:,:,:], axis=(2,3)),cmap='nipy_spectral')
        axes.flat[i].set_title(str(item))
        orbitals_all.append(result)        
    
    if returnProc:
        return(orbitals_all, processor)
    
    return(orbitals_all)

def movieOrbital(runNumber,orbital, processor=None, returnProc=False):
    if processor==None:
        processor = DldFlashProcessor()
        processor.runNumber=runNumber
        try:
            processor.readDataframes()
            print('schnell')
        except:    
            processor.readData(path='/asap3/flash/gpfs/pg2/2019/data/11006509/raw/hdf/online-1/')
            processor.storeDataframes()
            print('langsam')    
        
   
    times = np.arange(orbital[2],orbital[3],orbital[4])
    n=len(times)
    print(n)
    r=int(np.floor(np.sqrt(n)))
    c=int(np.ceil(n/r))
    #fig = plt.figure(figsize=(5,5))
        
    #ax1 = plt.subplot2grid((r+1,c), (0,0), colspan=2, rowspan=n)
    fig, axes = plt.subplots(nrows=r, ncols=c,figsize=(1*c,1*r))
    #spec3 = gridspec.GridSpec(ncols=c, nrows=r+1,wspace=2.0, hspace=2.0)
    
    #axes.flat[1] = plt.subplot2grid((r,c), (3,,figsize=(5*c,5*r))
    
    processor.postProcess(bamCorrectionSign=1)    #processor.readDataframes(runNumber)
    orbitals_all =[]
    

     
    processor.resetBins()
    #processor.addBinning('dldMicrobunchId',begin,end,end-begin)
    processor.addBinning('dldPosX',405,905,12)
    processor.addBinning('dldPosY',402,902,12)
    processor.addBinning('dldTime',orbital[0],orbital[1],processor.TOF_STEP_TO_NS*8)
    processor.addBinning('pumpProbeTime',orbital[2],orbital[3],orbital[4])
    result=processor.computeBinnedData()
    
    steps = len(times)
        
    for k in range(steps):
        
        #processor.postProcess()
        #result_norm = processor.normalizePumpProbeTime(result)
        
        #fig.add_subplot(spec3(r,c))
        ##ax= plt.subplot(r+1,c,i+1)
        #fig.add_subplot(spec3[i//2,i//3])
        axes.flat[k].imshow(np.sum(result[:,:,:,k], axis=(2)),cmap='nipy_spectral')
        axes.flat[k].set_title('delay of '+str('{:06.2f}'.format(times[k]))+' ps')
    #orbitals_all.append(result)        
    
    if returnProc:
        return(result, processor)
    
    return(result)