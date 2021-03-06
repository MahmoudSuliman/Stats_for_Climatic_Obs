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
    startingindex = x23.loc[x23['a']=='Fr??n Datum Tid (UTC)'].index.tolist()
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
    startingindex = x23.loc[x23['a']=='Fr??n Datum Tid (UTC)'].index.tolist()
    x232=pd.read_csv(plist[i], sep=';', header=startingindex)
    # converting column to datetime
    x232['Representativ m??nad']=pd.to_datetime(x232['Representativ m??nad'])
    x232=x232.set_index(x232['Representativ m??nad']) # changing index
    xcv=pd.DataFrame(data=emp ,columns=x232.columns) # creating a df with nan
    xcv['Representativ m??nad']= stdates
    xcv=xcv.set_index(stdates) # setting the index to the full period
    xcv.update(x232) # Updating the df with the full period
    dpfin[stname] = xcv
    # dtfin[stname]=x232[x232['Fr??n Datum Tid (UTC)'].str.contains('|'.join(stperiod))] # filtering study years using the regex character '|'

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
    startingindex = x23.loc[x23['a']=='Fr??n Datum Tid (UTC)'].index.tolist()
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
    startingindex = x23.loc[x23['a']=='Fr??n Datum Tid (UTC)'].index.tolist()
    x232=pd.read_csv(tlist[i], sep=';', header=startingindex)
    # converting column to datetime
    x232['Representativ m??nad']=pd.to_datetime(x232['Representativ m??nad'])
    x232=x232.set_index(x232['Representativ m??nad']) # changing index
    xcv=pd.DataFrame(data=emp ,columns=x232.columns) # creating a df with nan
    xcv['Representativ m??nad']= stdates
    xcv=xcv.set_index(stdates) # setting the index to the full period
    xcv.update(x232) # Updating the df with the full period
    dtfin[stname] = xcv
    # dtfin[stname]=x232[x232['Fr??n Datum Tid (UTC)'].str.contains('|'.join(stperiod))] # filtering study years using the regex character '|'

# converting column to datetime
x222=dtfin['1.Tullinge A']['Representativ m??nad']
y222=dtfin['1.Tullinge A']['Lufttemperatur']

plt.plot(x222, y222,)

# =============================================================================
