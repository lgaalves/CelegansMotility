#!/usr/bin/python

"""
Functions for fractal analysis
"""

__author__ = 'Luiz Gustavo de Andrade Alves'
__email__ = 'gustavoandradealves@gmail.com'

import sys
import subprocess
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
import statsmodels.api as sm
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
    

##### MSD ######

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

def stdm(x,y,nw):
    xw=[]
    yw=[]
    step=(max(x)-min(x))/nw
    lw=[min(x)+step*i for i in range(0,nw)]
    for i in range(0,len(lw)-1):
        if len(y[x>lw[i]][x[x>lw[i]]<lw[i+1]])>0:
            xw.append(np.std(x[x>lw[i]][x[x>lw[i]]<lw[i+1]]))
            yw.append(np.std(y[x>lw[i]][x[x>lw[i]]<lw[i+1]]))
    return(xw,yw)

def plaw(fitt,x):

    y=fitt[1]+fitt[0]*x

    return y

def msd_by_age(age):
    sample_dir=pathlib.Path('../data/age/{}'.format(age))
    tracks=[i for i in sample_dir.glob('*')]
    
    df_x,df_y=read_tracks(tracks)
        
    variance=msd(df_x,df_y)
    
    var_df=pd.DataFrame(np.transpose([np.arange(0,len(df_y))/5,variance]),columns=["time","variance"])
    var_df.to_csv("../results/msd_age-{}.csv".format(age))

def msd_by_temp(temperature):
    sample_dir=pathlib.Path('../data/temperature/{}'.format(temperature))
    tracks=[i for i in sample_dir.glob('*')]
    
    df_x,df_y=read_tracks(tracks)
        
    variance=msd(df_x,df_y)
    
    var_df=pd.DataFrame(np.transpose([np.arange(0,len(df_y))/5,variance]),columns=["time","variance"])
    var_df.to_csv("../results/msd_temperature-{}.csv".format(temperature))

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

####### Fractal dimension #######
def timeseriestoeventtimes(dataframe, timeseries):
    """
    input:
        dataframe: dataframe with all timeseries
        timeseries: string, boolean timeseries name
    output:
        eventtimes: timeseries of the time which an event occurs
    """
    timeseries = str(timeseries)
    eventtimes = dataframe.time[dataframe[timeseries] == 1].tolist()
    return eventtimes


def count_boxes(data, box_size):
    data = pd.Series(data)
    result = set()
    for value in data:
        result.add(np.int(np.floor(value / box_size)))
    M = data.max()
    N = np.int(np.floor(M / box_size))
    result.discard(N)
    # print(sorted(list(result)))
    return len(result)


def nbox_boxsize(data, number_of_sizes=12):
    """
    input:
        data: timeseries, 1-d list, array or pandas series of events
        number_of_sizes: number of different sizes of box
    output:
        r: Box size list
        N: Number of boxes that contain an event given a box size
    """
    L = max(data)
    r = np.array([L / (2.0**i) for i in range(0, number_of_sizes, 1)])
    N = [count_boxes(data, ri) for ri in r]
    return r, N


def fit_data(r, N):
    """
    input:
        r: Box size list
        N: Number of boxes that contain an event given a box size
    output:
        A: Interception
        Df: Fractal dimension
        r_value**2: R-square from fit
        p_value: two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
        std_err: Standard error of the estimate

    """

    Df, A, r_value, p_value, std_err = stats.linregress(np.log(1 / r), np.log(N))

    return Df, A, r_value**2, p_value, std_err


def find_stable_slope(r, N, size_window):
    list_df = []
    error_df = []
    step = size_window
    for i in range(0, len(N) - step + 1):
        Df, A, r_value, p_value, std_err = fit_data(r[i:i + step],
                                                    N[i:i + step])
        list_df.append(Df)
        error_df.append(std_err)
    return list_df, error_df


def fractal_plot(r, N, A, Df, drop_first_points, drop_last_points,
                 column_name):
    """
    input:
        r: Box size list
        N: Number of boxes that contain an event given a box size
        A: Interception
        Df: Fractal dimension
    output:
        log-log figure
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(1. / r, N, 'ob', markersize=8)
    ax.plot(
        1. / r[drop_first_points:drop_last_points],
        N[drop_first_points:drop_last_points],
        'or',
        markersize=8)
    ax.plot(1. / r, np.exp(A) * 1. / r**Df, '--k', linewidth=2.5, alpha=1.0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Box size, 1/s')
    plt.ylabel('Number of boxes, N(s)')
    increase_font(ax, 20)
    plt.savefig(
        "../figures/fractal_dimension.pdf",
        bbox_inches='tight')
    plt.show()


##### DFA ####
def integrate(y):
    result = []
    previous = 0
    for value in y:
        if not np.isnan(value):
            previous += value
        result.append(previous)
    return np.array(result)


def dfa(thelist,order=1):
    """
        Needs dfa.c installed: https://www.physionet.org/physiotools/dfa/

    INPUT
        order:      Detrend using a polynomial of degree K (default: K=1 -- linear detrending)

        thelist: The standard input should contain one column of data in text format.
    OUTPUT
        out_df:     The standard output is two columns: log(n) and log(F) [base 10 logarithms],
                    where n is the box size and F is the root mean square fluctuation.

    """

    temp_in_file = "temp_in.dat"
    temp_out_file = "temp_out.dat"
    dfa_code_path = "../c/dfa_macos"

    thelist=integrate(thelist)
    with open(temp_in_file, 'w') as file:
        for item in thelist:
            file.write("{}\n".format(item))

    if order==1:
        command= dfa_code_path + " -i <" + temp_in_file + " >" + temp_out_file
    else:
        command= dfa_code_path + " -d " +str(order) + " -i <" + temp_in_file + " >" + temp_out_file

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    out_df=pd.read_csv(temp_out_file, sep=" ", names=["n", "F"])
    os.system("rm {0} && rm {1}".format(temp_in_file, temp_out_file))
    return out_df

def linmodel_fit(x,y):
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    return results
