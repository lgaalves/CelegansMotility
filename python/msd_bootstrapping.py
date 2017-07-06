#!/usr/bin/python

"""
Run bootstrapping MSD
"""


__author__ = 'Luiz Gustavo de Andrade Alves'
__email__ = 'gustavoandradealves@gmail.com'

import numpy as np
import pandas as pd
import pathlib
from scipy import optimize
import random
import warnings
warnings.filterwarnings('ignore')

# Locals
from modules.functions import *

def main():
    data_plot=[]
    for age in [1,2,3,4,5,6]:
        df_x,df_y=read_dataset("age",label=age)
        slope_bootstrapping=[]
        for i in range(0,100):
            print(age,i)
            slope=slope_random_sample(df_x,df_y)
            slope_bootstrapping.append(slope)
        data_plot.append(slope_bootstrapping)
        data_plot = pd.DataFrame(data_plot)
        data_plot.to_csv('../results/msd_bootstrapping_{}.csv'.format(age), index=False, header=False)


if __name__ == "__main__":
    main()