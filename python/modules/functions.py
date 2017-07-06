#!/usr/bin/python

"""
Functions for fractal analysis
"""

__author__ = 'Luiz Gustavo de Andrade Alves'
__email__ = 'gustavoandradealves@gmail.com'

import sys
import os
import pathlib
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.stats import norm
from scipy import stats
import matplotlib.mlab as mlab
import statsmodels.stats.multitest as smm
import matplotlib.ticker as plticker
import random
from scipy import optimize
from matplotlib import ticker
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['xtick.major.pad'] = 10
plt.rcParams['xtick.minor.pad'] = 10
plt.rcParams['ytick.major.pad'] = 10
plt.rcParams['ytick.minor.pad'] = 10
plt.rcParams['xtick.direction']="in"
plt.rcParams['ytick.direction']="in"
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

def increase_font(ax, fontsize=18):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

def datainfo():
    print("Aging data set summary")
    print("\n")
    summary=[]
    for age in [1,2,3,4,5,6]:
        sample_dir=pathlib.Path('../data/age/{}'.format(age))
        tracks=[i for i in sample_dir.glob('*')]
        summary.append([age,len(tracks)])
        
    print(pd.DataFrame(summary,columns=["Age","# tracks"]))
     
    print("\n")
    summary=[]
    for t in [15,20,25]:
        sample_dir=pathlib.Path('../data/temperature/{}'.format(t))
        tracks=[i for i in sample_dir.glob('*')]
        summary.append([t,len(tracks)])
        
    print("Temperature data set summary\n")
        
    print(pd.DataFrame(summary,columns=["Temperature","# tracks"]))
    

def read_tracks(tracks):
    df_tracks=[]
    for track in tracks:
        df_track=pd.read_csv(str(track), index_col=0)
        df=df_track[df_track.time>=1800] #Select points with time t>30 minutes
        df.sort_values(by='time',inplace=True)
        if len(df)>100:
            df_tracks.append(df)

    len_max=max([len(x) for x in df_tracks])

    df_x=pd.DataFrame()
    for i,df in enumerate(df_tracks):
        x=np.array(df["x"])
        x=np.array(x-x[0])
        x0=np.zeros(len_max-len(x))
        x=np.concatenate((x,x0), axis=0)
        df_x[i]=x


    df_y=pd.DataFrame()
    for i,df in enumerate(df_tracks):
        y=np.array(df["y"])
        y=np.array(y-y[0])
        y0=np.zeros(len_max-len(y))
        y=np.concatenate((y,y0), axis=0)
        df_y[i]=y
    return df_x,df_y

def msd(df_x,df_y):
    variance=[np.nanvar(df_x.T[t])+np.nanvar(df_y.T[t]) for t in range(0,len(df_y))]
    return variance

def msd_by_age(age):
    sample_dir=pathlib.Path('../data/age/{}'.format(age))
    tracks=[i for i in sample_dir.glob('*')]
    
    df_x,df_y=read_tracks(tracks)
        
    variance=msd(df_x,df_y)
    
    var_df=pd.DataFrame(np.transpose([np.arange(0,len(df_y))/5,variance]),columns=["time","variance"])
    var_df.to_csv("../results/msd_age-{}.csv".format(age))

def increase_font(ax, fontsize=18):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

def datainfo():
    print("Aging data set summary")
    print("\n")
    summary=[]
    for age in [1,2,3,4,5,6]:
        sample_dir=pathlib.Path('../data/age/{}'.format(age))
        tracks=[i for i in sample_dir.glob('*')]
        summary.append([age,len(tracks)])
        
    print(pd.DataFrame(summary,columns=["Age","# tracks"]))
     
    print("\n")
    summary=[]
    for t in [15,20,25]:
        sample_dir=pathlib.Path('../data/temperature/{}'.format(t))
        tracks=[i for i in sample_dir.glob('*')]
        summary.append([t,len(tracks)])
        
    print("Temperature data set summary\n")
        
    print(pd.DataFrame(summary,columns=["Temperature","# tracks"]))
    

def read_tracks(tracks):
    df_tracks=[]
    for track in tracks:
        df_track=pd.read_csv(str(track), index_col=0)
        df=df_track[df_track.time>=1800] #Select points with time t>30 minutes
        df.sort_values(by='time',inplace=True)
        if len(df)>100:
            df_tracks.append(df)

    len_max=max([len(x) for x in df_tracks])

    df_x=pd.DataFrame()
    for i,df in enumerate(df_tracks):
        x=np.array(df["x"])
        x=np.array(x-x[0])
        x0=np.zeros(len_max-len(x))
        x=np.concatenate((x,x0), axis=0)
        df_x[i]=x


    df_y=pd.DataFrame()
    for i,df in enumerate(df_tracks):
        y=np.array(df["y"])
        y=np.array(y-y[0])
        y0=np.zeros(len_max-len(y))
        y=np.concatenate((y,y0), axis=0)
        df_y[i]=y
    return df_x,df_y

def msd(df_x,df_y):
    variance=[np.nanvar(df_x.T[t])+np.nanvar(df_y.T[t]) for t in range(0,len(df_y))]
    return variance


def piecewise_linear(x, x0, y0, k1):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:y0])

def line(x, a, b):
    return a * x + b

def winm(x, y, nw):
    xw = []
    yw = []
    step = (max(x) - min(x)) / nw
    lw = [min(x) + step * i for i in range(0, nw)]
    for i in range(0, len(lw) - 1):
        if len(y[x > lw[i]][x[x > lw[i]] < lw[i + 1]]) > 0:
            xw.append(np.mean(x[x > lw[i]][x[x > lw[i]] < lw[i + 1]]))
            yw.append(np.mean(y[x > lw[i]][x[x > lw[i]] < lw[i + 1]]))
    return (xw, yw)
    
def msd_by_age(age):
    sample_dir=pathlib.Path('../data/age/{}'.format(age))
    tracks=[i for i in sample_dir.glob('*')]
    
    df_x,df_y=read_tracks(tracks)
        
    variance=msd(df_x,df_y)
    
    var_df=pd.DataFrame(np.transpose([np.arange(0,len(df_y))/5,variance]),columns=["time","variance"])
    var_df.to_csv("../results/msd_age-{}.csv".format(age))

def read_dataset(dataset,label):
    """label can be age or temperature, this is, 1,2,3,4,5,6 or 15,20,25"""
    if dataset=="age":
        sample_dir=pathlib.Path('../data/age/{}'.format(label))
    else:
        sample_dir=pathlib.Path('../data/temperature/{}'.format(label))
    tracks=[i for i in sample_dir.glob('*')]
    df_x,df_y=read_tracks(tracks)
    return df_x,df_y

def slope_random_sample(df_x,df_y):
    randomsample=list(np.random.choice(list(df_x.columns),df_x.shape[1]))
    ndf_x=df_x[randomsample]
    ndf_y=df_x[randomsample]

    variance=msd(ndf_x,ndf_y)
    var_df=pd.DataFrame(np.transpose([np.arange(0,len(ndf_y))/5,variance]),
                        columns=["time","variance"])
    y=np.log10(var_df['variance'].tolist()[1:])
    x=np.log10(var_df["time"].tolist()[1:])

    xw, yw=winm(x,y,200)
    xw, yw=np.array(xw), np.array(yw)

    param_bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,2])
    p , pcov = optimize.curve_fit(piecewise_linear, xw[1:], yw[1:],bounds=param_bounds) 
    return(p[-1])
