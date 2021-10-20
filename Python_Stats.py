# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:29:28 2021

@author: Mahmoud Suliman
"""

# =============================================================================
# importing libraries

import os
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import PIL
from PIL import Image, ImageDraw
from pathlib import Path
from functools import partial

import pymannkendall as mk
from sklearn import linear_model
from statsmodels.graphics.regressionplots import abline_plot
import statsmodels.api as sm

# =============================================================================
# =============================================================================

# defining head directory
headdir=r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P'

# getting a list of all subdirectories within the head directory
subdir = [f.path for f in os.scandir(headdir) if f.is_dir()]

# getting all climatic csv file directories
clist=[]
for i in range(0, len(subdir)):
    clist1= glob.glob(subdir[i]+'\\*.csv')
    # print(clist1)
    clist.append(clist1[0])
    clist.append(clist1[1])
  
# =============================================================================
# =============================================================================

emp=np.empty((720,7)); emp[:]=np.nan
stperiod=np.arange(1960, 2020).astype(str).tolist()
stdates = pd.date_range('1960-01-01','2020-01-01' , freq='1M')-pd.offsets.MonthBegin(1)
emp=np.empty((720,7)); emp[:]=np.nan

# Precipitation data wrangling

# Precipitation Paths

plist=[]
for i in range(0, len(clist)):
    if clist[i].find('smhi-opendata_22') == -1:
        if clist[i].find('T.csv') == -1:
            if clist[i].find('t.csv') == -1:
                plist.append(clist[i])
# praw
dpraw={}
for i in range (0,len(plist)):
    zapath=Path(plist[i])
    parentpath=zapath.parent.absolute() # gets parent path of directory
    splitparpath=os.path.split(parentpath) # splits parent path
    stname = splitparpath[1] # gets station name from split path
    dpraw[stname]=pd.read_csv(plist[i],  sep=';', names=list('abcdefg'))

# pdata
dpdata={}
for i in range (0,len(plist)):
    zapath=Path(plist[i])
    parentpath=zapath.parent.absolute() # gets parent path of directory
    splitparpath=os.path.split(parentpath) # splits parent path
    stname = splitparpath[1] # gets station name from split path
    x23=pd.read_csv(plist[i], sep=';', names=list('abcdefg'))
    startingindex = x23.loc[x23['a']=='Från Datum Tid (UTC)'].index.tolist()
    dpdata[stname]=pd.read_csv(plist[i], sep=';', header=startingindex)

# tfin
# study period converted to string array and then to a list
stperiod=np.arange(1960, 2020).astype(str).tolist()

dpfin={}
for i in range (0,len(plist)):
    zapath=Path(plist[i])
    parentpath=zapath.parent.absolute() # gets parent path of directory
    splitparpath=os.path.split(parentpath) # splits parent path
    stname = splitparpath[1] # gets station name from split path
    x23=pd.read_csv(plist[i], sep=';', names=list('abcdefg'))
    startingindex = x23.loc[x23['a']=='Från Datum Tid (UTC)'].index.tolist()
    x232=pd.read_csv(plist[i], sep=';', header=startingindex)
    # converting column to datetime
    x232['Representativ månad']=pd.to_datetime(x232['Representativ månad'])
    x232=x232.set_index(x232['Representativ månad']) # changing index
    xcv=pd.DataFrame(data=emp ,columns=x232.columns) # creating a df with nan
    xcv['Representativ månad']= stdates
    xcv=xcv.set_index(stdates) # setting the index to the full period
    xcv.update(x232) # Updating the df with the full period
    dpfin[stname] = xcv
    # dtfin[stname]=x232[x232['Från Datum Tid (UTC)'].str.contains('|'.join(stperiod))] # filtering study years using the regex character '|'

# =============================================================================
# =============================================================================
# Temperature data wrangling
# Temperature Paths
tlist=[]
for i in range(0, len(clist)):
    if clist[i].find('smhi-opendata_23') == -1:
        if clist[i].find('P.csv') == -1:
            if clist[i].find('p.csv') == -1:
                tlist.append(clist[i])

# traw
dtraw={}
for i in range (0,len(tlist)):
    zapath=Path(tlist[i])
    parentpath=zapath.parent.absolute() # gets parent path of directory
    splitparpath=os.path.split(parentpath) # splits parent path
    stname = splitparpath[1] # gets station name from split path
    dtraw[stname]=pd.read_csv(tlist[i],  sep=';', names=list('abcdefg'))


x=pd.read_csv(tlist[0], sep=';', names=list('abcdefg'))

# tdata
dtdata={}
for i in range (0,len(tlist)):
    zapath=Path(tlist[i])
    parentpath=zapath.parent.absolute() # gets parent path of directory
    splitparpath=os.path.split(parentpath) # splits parent path
    stname = splitparpath[1] # gets station name from split path
    x23=pd.read_csv(tlist[i], sep=';', names=list('abcdefg'))
    startingindex = x23.loc[x23['a']=='Från Datum Tid (UTC)'].index.tolist()
    dtdata[stname]=pd.read_csv(tlist[i], sep=';', header=startingindex)

# tfin
# study period converted to string array and then to a list
stperiod=np.arange(1960, 2020).astype(str).tolist()
stdates = pd.date_range('1960-01-01','2020-01-01' , freq='1M')-pd.offsets.MonthBegin(1)
emp=np.empty((720,7)); emp[:]=np.nan
# cols = list(x232.columns)

dtfin={}
for i in range (0,len(tlist)):
    zapath=Path(tlist[i])
    parentpath=zapath.parent.absolute() # gets parent path of directory
    splitparpath=os.path.split(parentpath) # splits parent path
    stname = splitparpath[1] # gets station name from split path
    x23=pd.read_csv(tlist[i], sep=';', names=list('abcdefg'))
    startingindex = x23.loc[x23['a']=='Från Datum Tid (UTC)'].index.tolist()
    x232=pd.read_csv(tlist[i], sep=';', header=startingindex)
    # converting column to datetime
    x232['Representativ månad']=pd.to_datetime(x232['Representativ månad'])
    x232=x232.set_index(x232['Representativ månad']) # changing index
    xcv=pd.DataFrame(data=emp ,columns=x232.columns) # creating a df with nan
    xcv['Representativ månad']= stdates
    xcv=xcv.set_index(stdates) # setting the index to the full period
    xcv.update(x232) # Updating the df with the full period
    dtfin[stname] = xcv
    # dtfin[stname]=x232[x232['Från Datum Tid (UTC)'].str.contains('|'.join(stperiod))] # filtering study years using the regex character '|'

# converting column to datetime
x222=dtfin['10.Skövde']['Representativ månad']
y222=dtfin['10.Skövde']['Lufttemperatur']

plt.plot(x222, y222,)

# =============================================================================
# =============================================================================
# linear regression

# using statmodels


data=dtfin['10.Skövde']
data['Representativ månad'] = data.index.to_julian_date() # convert to pddatetime

linrx = data['Representativ månad'] #dtfin['10.Skövde']['Representativ månad']
linry = data['Lufttemperatur']

# Note that the xs are not datetimes
model = sm.OLS(linry, sm.add_constant(linrx)).fit()
# predictions = model.predict(linrx) # make the predictions by the model
linrp = model.params

# Print out the statistics
model.summary()

# plotting

# scatter-plot data
ax = data.plot(x='Representativ månad', y='Lufttemperatur', kind='scatter')
abline_plot(model_results=model, ax=ax, color='g')

slope=linrp['Representativ månad']

xtest1=1; xtest2=2
ytest2=2; ytest1=ytest2-slope

plt.plot([xtest1,xtest2],[ytest1,ytest2])
plt.xlim(0,3)
plt.ylim(1.99,2.01)

#------------------------------------------------------------------------------

# using sklearn

lm = linear_model.LinearRegression()

data=dtfin['10.Skövde']
lmx = data['Representativ månad'].values.reshape(-1, 1)
lmy = data['Lufttemperatur'].values.reshape(-1, 1)

# values to convert to numpy then reshape to fix error
model = lm.fit(lmx,lmy)

lm.score(lmx,lmy) #r2score
lm.coef_ # slope
lm.intercept_ # intercept

ax = data.plot(x='Representativ månad', y='Lufttemperatur', kind='scatter')
predictions = lm.predict(lmx)
plt.plot(data['Representativ månad'],predictions, 'g')

# =============================================================================
# =============================================================================
# Seasonal Mann-Kendall and sen slope

# autocorrelation plot (high ac = we should use mk)
fig, ax = plt.subplots(figsize=(12, 8))
sm.graphics.tsa.plot_acf(data['Lufttemperatur'], lags=20, ax=ax)

# seasonal Mann-Kendall
smk=mk.seasonal_test(data['Lufttemperatur'],period=12)
print(smk)

smk.trend #trend
smk.h         # hypothesis (false= no trend, true= there is trend)
smk.p         # if p-value < 0.1/0.05/0.01 => theres statistically significant 
              # evidence that a trend is present in the time series data
smk.z         # normalized test statistics
smk.Tau       # Kendall Tau
smk.s         # Mann-Kendal's score
smk.var_s     # Variance S
smk.slope     # Theil-Sen estimator/slope
smk.intercept # intercept of Kendall-Theil Robust Line, for seasonal test, 
              # full period cycle consider as unit time step

# =============================================================================
