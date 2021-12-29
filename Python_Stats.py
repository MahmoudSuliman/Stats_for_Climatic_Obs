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
from glob import glob
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
    clist1= glob(subdir[i]+'\\*.csv')
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
    clalist1= glob(subdirall[i]+'\\*pixres.csv')
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
            elif clalist[i].find('_IRF_') != -1:
                irf = pd.read_csv(clalist[i], sep=',')
                irf.columns = irf.iloc[0]
                irf = irf.drop (irf.index[0])
                irf = irf.rename(index={1:'irf'})
                irf = irf.rename(columns={5:'null', 4:'hveg2', 3:'iveg3', 2:'lveg4', 1:'urb5'})
                irf= irf.drop('null', 1)# number of pixels with 4(h.veg), 3(i.veg), 2(l.veg), 1(urban), 5(nan)
                irf['pixelnr']= irf.sum(1)
                irf['precurb']= (irf['urb5'][0]/irf['pixelnr'][0])*100
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
            histlat=pd.concat([histlat,his,latest,irf])
            claraw[stlist[j]]=histlat        

# for key, val in claraw.items():
#     val.to_csv(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\class results\rs_'+ key +'.csv')

clafin={}
for j in range(0, len(stlist)):
    clafin[stlist[j]]=claraw[stlist[j]]['precurb'][1]-claraw[stlist[j]]['precurb'][0]

# his['pixelnr']=his['hveg2'][0]+his['iveg3'][0]+his['lveg4'][0]+his['urb5'][0]
# his['precurb']= (his['urb5'][0]/his['pixelnr'][0])*100
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
# =============================================================================
# IR images

irfdiff=pd.DataFrame([])
for key, val in claraw.items():
    x = val['precurb']['latest']
    y = val['precurb']['irf']
    absdiff = np.abs(x-y)
    avgxy=(x+y)/2
    # abspdiff = (absdiff/avgxy)*100
    z=pd.DataFrame([[key,absdiff]])
    irfdiff=pd.concat([irfdiff, z])    

np.mean(irfdiff[1])
plt.scatter(np.arange(0, len(irfdiff)), irfdiff[1])


# good, v.good
irfg = ['Åsele', 'Åtorp', 'Uppsala Flygplats', 'Södertälje', 'Ställdalen',
        'Skövde', 'Oxelösund', 'Mariestad', 'Lund', 'Höljes', 'Forse', 
        'Fredriksberg']

# ok
irfok = ['Berga Mo', 'Falun-Lugnet', 'Gustavsfors', 'Hallands Väderö A', 
         'Karlsborg Mo', 'Norderön']

# low ok
irflok = ['Falsterbo', 'Sunne A', 'Tullinge A']

# bad
irfbad = ['Bollerup', 'Borås','Brämön A', 'Hagshult Mo', 'Hudiksvall', 'Järvsö',
          'Karlshamn', 'Kristinehamn', 'Ljusnedal', 'Lycksele A', 'Malmslätt', 
          'Mora A', 'Mörbylånga', 'Osby', 'Oskarshamn', 'Prästkulla', 'Ramsjöholm',
          'Ronneby-Bredåkra', 'Stockholm', 'Säffle', 'Såtenäs', 'Ulricehamn', 
          'Varberg', 'Vänersborg', 'Västerås', 'Ölands norra udde', 'Örskär A']

irfzag=['Åsele', 'Åtorp', 'Uppsala Flygplats', 'Södertälje', 'Ställdalen',
        'Skövde', 'Höljes', 'Forse', 
        'Fredriksberg', 'Falun-Lugnet', 'Gustavsfors', 'Hallands Väderö A', 
         'Norderön', 'Bollerup', 'Järvsö', 'Kristinehamn',
         'Ljusnedal', 'Såtenäs', 'Varberg']


rirfg=pd.DataFrame([])
rirfzag=pd.DataFrame([])
rirfok=pd.DataFrame([])
rirflok=pd.DataFrame([])
rirfbad=pd.DataFrame([])

for i in range(0, len(irfg)):
    for j in range(0, len(irfdiff)):
        if irfdiff.iloc[j,0].find(irfg[i]) != -1:
            print(irfdiff.iloc[j,0])
            x=irfdiff.iloc[[j]]
            rirfg=pd.concat([rirfg,x])

for i in range(0, len(irfzag)):
    for j in range(0, len(irfdiff)):
        if irfdiff.iloc[j,0].find(irfzag[i]) != -1:
            print(irfdiff.iloc[j,0])
            x=irfdiff.iloc[[j]]
            rirfzag=pd.concat([rirfzag,x])

for i in range(0, len(irfok)):
    for j in range(0, len(irfdiff)):
        if irfdiff.iloc[j,0].find(irfok[i]) != -1:
            print(irfdiff.iloc[j,0])
            x=irfdiff.iloc[[j]]
            rirfok=pd.concat([rirfok,x])

for i in range(0, len(irflok)):
    for j in range(0, len(irfdiff)):
        if irfdiff.iloc[j,0].find(irflok[i]) != -1:
            print(irfdiff.iloc[j,0])
            x=irfdiff.iloc[[j]]
            rirflok=pd.concat([rirflok,x])

for i in range(0, len(irfbad)):
    for j in range(0, len(irfdiff)):
        if irfdiff.iloc[j,0].find(irfbad[i]) != -1:
            print(irfdiff.iloc[j,0])
            x=irfdiff.iloc[[j]]
            rirfbad=pd.concat([rirfbad,x])

mgood=np.mean(rirfg[1])
mok=np.mean(rirfok[1])
mlok=np.mean(rirflok[1])
mbad=np.mean(rirfbad[1])
mzagood=np.mean(rirfzag[1])

m1=np.mean([mgood,mok,mlok])
np.mean([m1,mbad])

zz=irfdiff[1][irfdiff[1]<10]
    

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

# percentages for the stations (classification)
stprec=pd.DataFrame(data=[])
for keys, vals in clafin.items():
    x=pd.DataFrame(data=[[keys,vals]])
    stprec=pd.concat([stprec,x])

# concatenating all the slopes and the stations percentages
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
#------------------------------------------------------------------------------
# p value filtration

ssthreshold = 0.05 
hsthreshold = 0.001

# decreased
# statistically significant
resdeclrssT = resdec.copy()[resdec.copy().iloc[:,3]<= ssthreshold] # temperature linear regression
resdeclrssP = resdec.copy()[resdec.copy().iloc[:,6]<= ssthreshold] # precipitation linear regression
resdecsmkssT = resdec.copy()[resdec.copy().iloc[:,4]<= ssthreshold] # t smk
resdecsmkssP = resdec.copy()[resdec.copy().iloc[:,7]<= ssthreshold] # p smk

# statistically highly significant
resdeclrhsT = resdec.copy()[resdec.copy().iloc[:,3]<= hsthreshold] # temperature linear regression
resdeclrhsP = resdec.copy()[resdec.copy().iloc[:,6]<= hsthreshold] # precipitation linear regression
resdecsmkhsT = resdec.copy()[resdec.copy().iloc[:,4]<= hsthreshold] # t smk
resdecsmkhsP = resdec.copy()[resdec.copy().iloc[:,7]<= hsthreshold] # p smk

# no change
# statistically significant
resnochalrssT = resnocha.copy()[resnocha.copy().iloc[:,3]<= ssthreshold] # temperature linear regression
resnochalrssP = resnocha.copy()[resnocha.copy().iloc[:,6]<= ssthreshold] # precipitation linear regression
resnochasmkssT = resnocha.copy()[resnocha.copy().iloc[:,4]<= ssthreshold] # t smk
resnochasmkssP = resnocha.copy()[resnocha.copy().iloc[:,7]<= ssthreshold] # p smk

# statistically highly significant
resnochalrhsT = resnocha.copy()[resnocha.copy().iloc[:,3]<= hsthreshold] # temperature linear regression
resnochalrhsP = resnocha.copy()[resnocha.copy().iloc[:,6]<= hsthreshold] # precipitation linear regression
resnochasmkhsT = resnocha.copy()[resnocha.copy().iloc[:,4]<= hsthreshold] # t smk
resnochasmkhsP = resnocha.copy()[resnocha.copy().iloc[:,7]<= hsthreshold] # p smk

# increase
# statistically significant
resinclrssT = resinc.copy()[resinc.copy().iloc[:,3]<= ssthreshold] # temperature linear regression
resinclrssP = resinc.copy()[resinc.copy().iloc[:,6]<= ssthreshold] # precipitation linear regression
resincsmkssT = resinc.copy()[resinc.copy().iloc[:,4]<= ssthreshold] # t smk
resincsmkssP = resinc.copy()[resinc.copy().iloc[:,7]<= ssthreshold] # p smk

# statistically highly significant
resinclrhsT = resinc.copy()[resinc.copy().iloc[:,3]<= hsthreshold] # temperature linear regression
resinclrhsP = resinc.copy()[resinc.copy().iloc[:,6]<= hsthreshold] # precipitation linear regression
resincsmkhsT = resinc.copy()[resinc.copy().iloc[:,4]<= hsthreshold] # t smk
resincsmkhsP = resinc.copy()[resinc.copy().iloc[:,7]<= hsthreshold] # p smk

# large increase
# statistically significant
reslarinclrssT = reslarinc.copy()[reslarinc.copy().iloc[:,3]<= ssthreshold] # temperature linear regression
reslarinclrssP = reslarinc.copy()[reslarinc.copy().iloc[:,6]<= ssthreshold] # precipitation linear regression
reslarincsmkssT = reslarinc.copy()[reslarinc.copy().iloc[:,4]<= ssthreshold] # t smk
reslarincsmkssP = reslarinc.copy()[reslarinc.copy().iloc[:,7]<= ssthreshold] # p smk

# statistically highly significant
reslarinclrhsT = reslarinc.copy()[reslarinc.copy().iloc[:,3]<= hsthreshold] # temperature linear regression
reslarinclrhsP = reslarinc.copy()[reslarinc.copy().iloc[:,6]<= hsthreshold] # precipitation linear regression
reslarincsmkhsT = reslarinc.copy()[reslarinc.copy().iloc[:,4]<= hsthreshold] # t smk
reslarincsmkhsP = reslarinc.copy()[reslarinc.copy().iloc[:,7]<= hsthreshold] # p smk

# -----------------------------------------------------------------------------
# normal plot

# categorized t
plt.figure(figsize=(8,5))
plt.scatter(resdec.iloc[:,11], resdec.iloc[:,2], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,11], resnocha.iloc[:,2], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,11], resinc.iloc[:,2], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,11], reslarinc.iloc[:,2], c='crimson', label='Large Increase')
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
plt.scatter(resdec.iloc[:,11], resdec.iloc[:,7], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,11], resnocha.iloc[:,7], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,11], resinc.iloc[:,7], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,11], reslarinc.iloc[:,7], c='crimson', label='Large Increase')
plt.hlines(np.average(resdec.iloc[:,7]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnocha.iloc[:,7]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinc.iloc[:,7]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinc.iloc[:,7]), -20, 70, 'crimson', '--')
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
plt.scatter(resdec.iloc[:,11], resdec.iloc[:,1], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,11], resnocha.iloc[:,1], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,11], resinc.iloc[:,1], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,11], reslarinc.iloc[:,1], c='crimson', label='Large Increase')
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
plt.scatter(resdec.iloc[:,11], resdec.iloc[:,6], c='Skyblue', label='Decrease')
plt.scatter(resnocha.iloc[:,11], resnocha.iloc[:,6], c='Lawngreen', label='No change')
plt.scatter(resinc.iloc[:,11], resinc.iloc[:,6], c='Orange', label='Increase')
plt.scatter(reslarinc.iloc[:,11], reslarinc.iloc[:,6], c='crimson', label='Large Increase')
plt.hlines(np.average(resdec.iloc[:,6]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnocha.iloc[:,6]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinc.iloc[:,6]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinc.iloc[:,6]), -20, 70, 'crimson', '--')
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
# statistically significant plot

# categorized t
plt.figure(figsize=(8,5))
plt.scatter(resdecsmkssT.iloc[:,11], resdecsmkssT.iloc[:,2], c='Skyblue', label='Decrease')
plt.scatter(resnochasmkssT.iloc[:,11], resnochasmkssT.iloc[:,2], c='Lawngreen', label='No change')
plt.scatter(resincsmkssT.iloc[:,11], resincsmkssT.iloc[:,2], c='Orange', label='Increase')
plt.scatter(reslarincsmkssT.iloc[:,11], reslarincsmkssT.iloc[:,2], c='crimson', label='Large Increase')
plt.hlines(np.average(resdecsmkssT.iloc[:,2]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochasmkssT.iloc[:,2]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resincsmkssT.iloc[:,2]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarincsmkssT.iloc[:,2]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Sen Slope (Temperature), p='+str(ssthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\SenT'+str(ssthreshold)+'.jpg', dpi=300, bbox_inches='tight')

np.average(resdec.iloc[:,2])
np.average(resnocha.iloc[:,2])
np.average(resinc.iloc[:,2])
np.average(reslarinc.iloc[:,2])

# categorized p
plt.figure(figsize=(8,5))
plt.scatter(resdecsmkssP.iloc[:,11], resdecsmkssP.iloc[:,7], c='Skyblue', label='Decrease')
plt.scatter(resnochasmkssP.iloc[:,11], resnochasmkssP.iloc[:,7], c='Lawngreen', label='No change')
plt.scatter(resincsmkssP.iloc[:,11], resincsmkssP.iloc[:,7], c='Orange', label='Increase')
plt.scatter(reslarincsmkssP.iloc[:,11], reslarincsmkssP.iloc[:,7], c='crimson', label='Large Increase')
plt.hlines(np.average(resdecsmkssP.iloc[:,7]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochasmkssP.iloc[:,7]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resincsmkssP.iloc[:,7]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarincsmkssP.iloc[:,7]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Sen Slope (Precipitation), p='+str(ssthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\SenP'+str(ssthreshold)+'.jpg', dpi=300, bbox_inches='tight')

# linreg categorized t
plt.figure(figsize=(8,5))
plt.scatter(resdeclrssT.iloc[:,11], resdeclrssT.iloc[:,1], c='Skyblue', label='Decrease')
plt.scatter(resnochalrssT.iloc[:,11], resnochalrssT.iloc[:,1], c='Lawngreen', label='No change')
plt.scatter(resinclrssT.iloc[:,11], resinclrssT.iloc[:,1], c='Orange', label='Increase')
plt.scatter(reslarinclrssT.iloc[:,11], reslarinclrssT.iloc[:,1], c='crimson', label='Large Increase')
plt.hlines(np.average(resdeclrssT.iloc[:,1]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochalrssT.iloc[:,1]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinclrssT.iloc[:,1]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinclrssT.iloc[:,1]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Linear regression slope (Temperature), p='+str(ssthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\LinregT'+str(ssthreshold)+'.jpg', dpi=300, bbox_inches='tight')

# linreg categorized p
plt.figure(figsize=(8,5))
plt.scatter(resdeclrssP.iloc[:,11], resdeclrssP.iloc[:,6], c='Skyblue', label='Decrease')
plt.scatter(resnochalrssP.iloc[:,11], resnochalrssP.iloc[:,6], c='Lawngreen', label='No change')
plt.scatter(resinclrssP.iloc[:,11], resinclrssP.iloc[:,6], c='Orange', label='Increase')
plt.scatter(reslarinclrssP.iloc[:,11], reslarinclrssP.iloc[:,6], c='crimson', label='Large Increase')
plt.hlines(np.average(resdeclrssP.iloc[:,6]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochalrssP.iloc[:,6]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinclrssP.iloc[:,6]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinclrssP.iloc[:,6]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Linear regression slope (Precipitation), p='+str(ssthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\LinregP'+str(ssthreshold)+'.jpg', dpi=300, bbox_inches='tight')

# =============================================================================
# =============================================================================
# highly statistically significant plot

# categorized t
plt.figure(figsize=(8,5))
plt.scatter(resdecsmkhsT.iloc[:,11], resdecsmkhsT.iloc[:,2], c='Skyblue', label='Decrease')
plt.scatter(resnochasmkhsT.iloc[:,11], resnochasmkhsT.iloc[:,2], c='Lawngreen', label='No change')
plt.scatter(resincsmkhsT.iloc[:,11], resincsmkhsT.iloc[:,2], c='Orange', label='Increase')
plt.scatter(reslarincsmkhsT.iloc[:,11], reslarincsmkhsT.iloc[:,2], c='crimson', label='Large Increase')
plt.hlines(np.average(resdecsmkhsT.iloc[:,2]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochasmkhsT.iloc[:,2]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resincsmkhsT.iloc[:,2]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarincsmkhsT.iloc[:,2]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Sen Slope (Temperature), p='+str(hsthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\SenT'+str(hsthreshold)+'.jpg', dpi=300, bbox_inches='tight')

np.average(resdec.iloc[:,2])
np.average(resnocha.iloc[:,2])
np.average(resinc.iloc[:,2])
np.average(reslarinc.iloc[:,2])

# categorized p
plt.figure(figsize=(8,5))
plt.scatter(resdecsmkhsP.iloc[:,11], resdecsmkhsP.iloc[:,7], c='Skyblue', label='Decrease')
plt.scatter(resnochasmkhsP.iloc[:,11], resnochasmkhsP.iloc[:,7], c='Lawngreen', label='No change')
plt.scatter(resincsmkhsP.iloc[:,11], resincsmkhsP.iloc[:,7], c='Orange', label='Increase')
plt.scatter(reslarincsmkhsP.iloc[:,11], reslarincsmkhsP.iloc[:,7], c='crimson', label='Large Increase')
plt.hlines(np.average(resdecsmkhsP.iloc[:,7]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochasmkhsP.iloc[:,7]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resincsmkhsP.iloc[:,7]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarincsmkhsP.iloc[:,7]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Sen Slope (Precipitation), p='+str(hsthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\SenP'+str(hsthreshold)+'.jpg', dpi=300, bbox_inches='tight')

# linreg categorized t
plt.figure(figsize=(8,5))
plt.scatter(resdeclrhsT.iloc[:,11], resdeclrhsT.iloc[:,1], c='Skyblue', label='Decrease')
plt.scatter(resnochalrhsT.iloc[:,11], resnochalrhsT.iloc[:,1], c='Lawngreen', label='No change')
plt.scatter(resinclrhsT.iloc[:,11], resinclrhsT.iloc[:,1], c='Orange', label='Increase')
plt.scatter(reslarinclrhsT.iloc[:,11], reslarinclrhsT.iloc[:,1], c='crimson', label='Large Increase')
plt.hlines(np.average(resdeclrhsT.iloc[:,1]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochalrhsT.iloc[:,1]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinclrhsT.iloc[:,1]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinclrhsT.iloc[:,1]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Linear regression slope (Temperature), p='+str(hsthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\LinregT'+str(hsthreshold)+'.jpg', dpi=300, bbox_inches='tight')

# linreg categorized p
plt.figure(figsize=(8,5))
plt.scatter(resdeclrhsP.iloc[:,11], resdeclrhsP.iloc[:,6], c='Skyblue', label='Decrease')
plt.scatter(resnochalrhsP.iloc[:,11], resnochalrhsP.iloc[:,6], c='Lawngreen', label='No change')
plt.scatter(resinclrhsP.iloc[:,11], resinclrhsP.iloc[:,6], c='Orange', label='Increase')
plt.scatter(reslarinclrhsP.iloc[:,11], reslarinclrhsP.iloc[:,6], c='crimson', label='Large Increase')
plt.hlines(np.average(resdeclrhsP.iloc[:,6]), -20, 70, 'Skyblue','--')
plt.hlines(np.average(resnochalrhsP.iloc[:,6]), -20, 70, 'Lawngreen', '--')
plt.hlines(np.average(resinclrhsP.iloc[:,6]), -20, 70, 'Orange','--')
plt.hlines(np.average(reslarinclrhsP.iloc[:,6]), -20, 70, 'crimson', '--')
plt.xlim(-20,60)
plt.grid(True, linestyle='--',  color='black', alpha=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
                   fancybox=True, shadow=True, ncol=5, fontsize=12)
plt.ylabel('Slope', fontsize=16)
plt.xlabel ('Increase in Urban LC', fontsize=16)
plt.title('Linear regression slope (Precipitation), p='+str(hsthreshold))
plt.savefig(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\figures\LinregP'+str(hsthreshold)+'.jpg', dpi=300, bbox_inches='tight')
