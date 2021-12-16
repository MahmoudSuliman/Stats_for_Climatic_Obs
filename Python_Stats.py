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

# getting a list of all immediate subdirectories within the head directory
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

# pvalue
pval=model.pvalues['Representativ månad']

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
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# classification part

# getting all subdirectories within head directories
subdirall= [x[0] for x in os.walk(headdir)]

# getting all classification csv files
clalist=[]
for i in range(0, len(subdirall)):
    clalist1= glob.glob(subdirall[i]+'\\*pixres.csv')
    # print(clalist1)
    if clalist1==[]:
        print('Nothing here!')
    else:
        clalist.append(clalist1[0])
    
clalist[0]
x=pd.read_csv(clalist[0], sep=',')

# =============================================================================
# =============================================================================
# 
# stations list
stlist=[]
for i in range (0,len(subdir)):
    zapath=Path(subdir[i])
    splitparpath=os.path.split(zapath)
    stname = splitparpath[1]
    stlist.append(stname)


claraw={}
for i in range (0, len(clalist)):
    for j in range(0, len(stlist)):
        if clalist[i].find(stlist[j]) != -1:
            if clalist[i].find('1960') != -1:
                his = pd.read_csv(clalist[i], sep=',')
                his.columns = his.iloc[0]
                his = his.drop (his.index[0])
                his = his.rename(index={1:'hist'})
                his = his.rename(columns={0:'null', 2:'hveg2', 3:'iveg3', 4:'lveg4', 5:'urb5'})
                his= his.drop('null', 1)
                his['pixelnr']= his.sum(1)
                his['precurb']= (his['urb5'][0]/his['pixelnr'][0])*100
            else:
                latest= pd.read_csv(clalist[i], sep=',')
                latest.columns = latest.iloc[0]
                latest = latest.drop (latest.index[0])
                latest = latest.rename(index={1:'latest'})
                latest = latest.rename(columns={0:'null', 2:'hveg2', 3:'iveg3',
                                                4:'lveg4', 5:'urb5'})
                latest= latest.drop('null', 1)
                latest['pixelnr']= latest.sum(1)
                latest['precurb']= (latest['urb5'][0]/latest['pixelnr'][0])*100
            histlat= pd.DataFrame(data=[], columns=['hveg2','iveg3','lveg4','urb5'])
            histlat=pd.concat([histlat,his,latest])
            claraw[stlist[j]]=histlat        

clafin={}
for j in range(0, len(stlist)):
    clafin[stlist[j]]=claraw[stlist[j]]['precurb'][1]-claraw[stlist[j]]['precurb'][0]

his['pixelnr']=his['hveg2'][0]+his['iveg3'][0]+his['lveg4'][0]+his['urb5'][0]
his['precurb']= (his['urb5'][0]/his['pixelnr'][0])*100
# =============================================================================
fldec = []
flnocha = []
# flmininc = []
flinc = []
flarinc= []

for keys, vals in clafin.items():
    if vals <= -3:
        fldec.append(keys)
    if -3 < vals <= 3:
        flnocha.append(keys)
    if 3 < vals <= 20:
        flinc.append(keys)
    if vals > 20:
        flarinc.append(keys)

flbins={}
flbins['decrease']= fldec
flbins['no change']= flnocha
flbins['increase']= flinc
flbins['large increase']= flarinc


# =============================================================================
# 
# =============================================================================
# =============================================================================
# 

# using statmodels

dtslopes=pd.DataFrame(data=[], columns=['stname','linregslope','smkslope'])
for keys, vals in dtfin.items():    
    data=dtfin[keys]
    data['Representativ månad'] = data.index.to_julian_date() # convert to pddatetime
    # seasonal MK
    linrx = data['Representativ månad'] 
    linry = data['Lufttemperatur']
    smk=mk.seasonal_test(data['Lufttemperatur'],period=12)
    smkslope= smk.slope
    smkpval=smk.p
    # Linear regression (drops nan values)
    linrx = data[data['Lufttemperatur'].notna()]['Representativ månad']
    linry = data[data['Lufttemperatur'].notna()]['Lufttemperatur']
    model = sm.OLS(linry, sm.add_constant(linrx)).fit()
    linrp = model.params
    slope=linrp['Representativ månad']
    pval = model.pvalues['Representativ månad']
    # saving
    x=pd.DataFrame(data=[], columns=['stname','linregslope', 'linregpval','smkslope','smkpval'])
    x.loc[0]= [keys , slope, pval, smkslope, smkpval]
    dtslopes=pd.concat([dtslopes,x])


dpslopes=pd.DataFrame(data=[], columns=['stname','linregslope','smkslope'])
for keys, vals in dtfin.items():    
    data=dpfin[keys]
    data['Representativ månad'] = data.index.to_julian_date() # convert to pddatetime

    linrx = data['Representativ månad'] 
    linry = data['Nederbördsmängd']
    smk=mk.seasonal_test(data['Nederbördsmängd'],period=12)
    smkslope= smk.slope
    smkpval=smk.p

    linrx = data[data['Nederbördsmängd'].notna()]['Representativ månad']
    linry = data[data['Nederbördsmängd'].notna()]['Nederbördsmängd']
    model = sm.OLS(linry, sm.add_constant(linrx)).fit()
    linrp = model.params    
    slope=linrp['Representativ månad']
    pval = model.pvalues['Representativ månad']

    x=pd.DataFrame(data=[], columns=['stname','linregslope', 'linregpval','smkslope','smkpval'])
    x.loc[0]= [keys , slope, pval, smkslope, smkpval]
    dpslopes=pd.concat([dpslopes,x])

stprec=pd.DataFrame(data=[])
for keys, vals in clafin.items():
    x=pd.DataFrame(data=[[keys,vals]])
    stprec=pd.concat([stprec,x])

finres=pd.concat([dtslopes,dpslopes, stprec], axis=1)

resdec=pd.DataFrame(data=[])
for i in range (0,len(finres)):
    for j in range(0, len(fldec)):
        if finres.iloc[i,0]== fldec[j]:
            x=finres.iloc[[i]]
            resdec=pd.concat([resdec,x])

resnocha=pd.DataFrame(data=[])
for i in range (0,len(finres)):
    for j in range(0, len(flnocha)):
        if finres.iloc[i,0]== flnocha[j]:
            x=finres.iloc[[i]]
            resnocha=pd.concat([resnocha,x])

resinc=pd.DataFrame(data=[])
for i in range (0,len(finres)):
    for j in range(0, len(flinc)):
        if finres.iloc[i,0]== flinc[j]:
            x=finres.iloc[[i]]
            resinc=pd.concat([resinc,x])

reslarinc=pd.DataFrame(data=[])
for i in range (0,len(finres)):
    for j in range(0, len(flarinc)):
        if finres.iloc[i,0]== flarinc[j]:
            x=finres.iloc[[i]]
            reslarinc=pd.concat([reslarinc,x])

# t vs precentage
plt.scatter(finres.iloc[:,7], finres.iloc[:,2])

# p vs precentage
plt.scatter(finres.iloc[:,7], finres.iloc[:,5])

# categorized t
plt.figure(figsize=(8,5))
plt.scatter(resdec.iloc[:,7], resdec.iloc[:,2], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,7], resnocha.iloc[:,2], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,7], resinc.iloc[:,2], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,7], reslarinc.iloc[:,2], c='crimson', label='Large Increase')
plt.hlines(np.average(resdec.iloc[:,2]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnocha.iloc[:,2]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinc.iloc[:,2]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinc.iloc[:,2]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Sen Slope (Temperature)')
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\SenT.jpg', dpi=300, bbox_inches='tight')

np.average(resdec.iloc[:,2])
np.average(resnocha.iloc[:,2])
np.average(resinc.iloc[:,2])
np.average(reslarinc.iloc[:,2])

# categorized p
plt.figure(figsize=(8,5))
plt.scatter(resdec.iloc[:,7], resdec.iloc[:,5], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,7], resnocha.iloc[:,5], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,7], resinc.iloc[:,5], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,7], reslarinc.iloc[:,5], c='crimson', label='Large Increase')
plt.hlines(np.average(resdec.iloc[:,5]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnocha.iloc[:,5]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinc.iloc[:,5]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinc.iloc[:,5]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Sen Slope (Precipitation)')
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\SenP.jpg', dpi=300, bbox_inches='tight')

# linreg categorized t
plt.figure(figsize=(8,5))
plt.scatter(resdec.iloc[:,7], resdec.iloc[:,1], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,7], resnocha.iloc[:,1], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,7], resinc.iloc[:,1], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,7], reslarinc.iloc[:,1], c='crimson', label='Large Increase')
plt.hlines(np.average(resdec.iloc[:,1]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnocha.iloc[:,1]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinc.iloc[:,1]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinc.iloc[:,1]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Linear regression slope (Temperature)')
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\LinregT.jpg', dpi=300, bbox_inches='tight')

# linreg categorized p
plt.figure(figsize=(8,5))
plt.scatter(resdec.iloc[:,7], resdec.iloc[:,4], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,7], resnocha.iloc[:,4], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,7], resinc.iloc[:,4], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,7], reslarinc.iloc[:,4], c='crimson', label='Large Increase')
plt.hlines(np.average(resdec.iloc[:,4]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnocha.iloc[:,4]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinc.iloc[:,4]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinc.iloc[:,4]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Linear regression slope (Precipitation)')
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\LinregP.jpg', dpi=300, bbox_inches='tight')

# =============================================================================
# =============================================================================
