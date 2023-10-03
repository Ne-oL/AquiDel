# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:37:22 2022
 
@author: khalid
"""

import warnings
import pandas as pd
import numpy as np
import pathlib
import copy
import time
import datetime
import logging
import os
# import holoviews as hv
# from holoviews import opts
# import hvplot.pandas
from joblib import Parallel, delayed
import matplotlib.dates
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler 
from scipy import stats
import scipy 
import nevergrad as ng
# from multiprocessing import  Pool
import ray
from ray.util.multiprocessing.pool import Pool
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from hebo.optimizers.hebo import HEBO
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
import nevergrad as ng
import multiprocessing as mp
import pickle
import math
import itertools
import hdbscan
from prophet import Prophet
from minisom import MiniSom
# from tslearn.barycenters import dtw_barycenter_averaging
# from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import rand_score, adjusted_mutual_info_score, adjusted_rand_score
from dtaidistance import ed, dtw, clustering, preprocessing
from dtaidistance.subsequence.dtw import subsequence_alignment
from dtaidistance import dtw_visualisation as dtwvis
from scipy.cluster.hierarchy import linkage, single, complete, average, ward, dendrogram, fcluster
from scipy.spatial.distance import squareform
#from tqdm.notebook import tqdm_notebook
# from chart_studio.plotly import plot, iplot as py
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.offline import iplot, init_notebook_mode
# # Using plotly + cufflinks in offline mode
# import cufflinks
# import seaborn as sns
mydir=r'/home/200734761/data/TX/GWDBDownload' #r'K:\Code' #r'C:\Users\KhALiD\OneDrive - UAE University\PhD\Jupyter'
mydir2=r'/home/200734761/notebooks/vars'
Freq='MS' # TimeStep Frequency, Supported & Tested Frequencies are Annual (AS) or Monthly (MS)

path= pathlib.Path(mydir, 'WaterLevelsMajor.txt') #gets the full path appropriate for the system
data= pd.read_csv(path, sep='|',encoding='cp1252', low_memory=False)

data=data.rename(columns={"MeasurementYear": "year", "MeasurementMonth": "month","MeasurementDay": "day"})
data.loc[data['day'] < 1.0, 'day'] = 1.0
data.loc[data['month'] < 1.0, 'month'] = 1.0
data[['MeasurementDate']]=pd.DataFrame(pd.to_datetime(data[['day','year','month']]))

# Groundwater Elevation
data=data[['StateWellNumber','MeasurementDate','WaterElevation']]
data=data.rename(columns={"MeasurementDate": "Date", "StateWellNumber": "WellCode", "WaterElevation": "GWL"}) #we rename the columns to a more identifiable names

# #Groundwater Table
# data=data[['StateWellNumber','MeasurementDate','DepthFromLSD']]
# data=data.rename(columns={"MeasurementDate": "Date", "StateWellNumber": "WellCode", "DepthFromLSD": "GWL"}) #we rename the columns to a more identifiable names

data[['WellCode']] = data[['WellCode']].astype('str') 
dataClean=data.groupby('WellCode') #groups the dataset into batchs each containing time series of individual well

#dataClean=dataClean.filter(lambda x: x.count().GWL >= 10) #Filter the minimum acceptable number of readings per well, in this case it was "12" for variable reading periods
#dataCleanMonthly=dataClean.groupby('WellCode').resample('AS', on='Date').mean()
# datagroup=dataCleanMonthly.groupby('WellCode')
tsperiod=pd.date_range(data.Date.min(),data.Date.max(),freq=Freq)

ray.init(log_to_driver=False)


         
            
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)        
            
            
# for z in tqdm_notebook(parameters):
# def calculating_matrix(StudyPeriod,BasinSize,Comperiod,Outliers,Interpolation,Similarity,Unsynced):
def calculating_matrix(config):
    # Clean and impute the missing data
    def dataclean(dataCleanMonthlyRR):
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        logging.getLogger('prophet').setLevel(logging.ERROR)
        name=dataCleanMonthlyRR[0]
        dataCleanMonthlyRR=copy.copy(dataCleanMonthlyRR[1])
    #         Avg=dataCleanMonthlyRR.mean()
    #         std=dataCleanMonthlyRR.std()*SD
    #         outlier=abs(dataCleanMonthlyRR-Avg)>std
        outlier=abs(dataCleanMonthlyRR-Avg[name])>std[name]
        dataCleanMonthlyRR[outlier]=None #Outliers detection
        dataCleanMonthlyRR=dataCleanMonthlyRR[dataCleanMonthlyRR.dropna().index[0]:dataCleanMonthlyRR.dropna().index[-1]]
        if Freq=='MS':
            MinPeriod=np.round((dataCleanMonthlyRR.index[-1][1] - dataCleanMonthlyRR.index[0][1])/np.timedelta64(1, 'M')*covPeriod)
        elif Freq=='AS':
            MinPeriod=np.round((dataCleanMonthlyRR.index[-1][1] - dataCleanMonthlyRR.index[0][1])/np.timedelta64(1, 'Y')*covPeriod)
        else:
            print('The chosen Time Step Frequency is not correctly defined/supported, please check the Freq variable, it should be AS or MS')
        if dataCleanMonthlyRR.count()[0] < MinPeriod or dataCleanMonthlyRR.count()[0] < studPeriod: #Filter the minimum acceptable number of readings per well, in this case it was "12" months
            dataCleanMonthlyRR = None
            return None
        if Interploation == 'linear':
            dataCleanMonthlyRR=dataCleanMonthlyRR.interpolate(limit_direction='both', method='linear')
        elif Interploation == 'mean':
            dfb = dataCleanMonthlyRR.fillna(method='bfill')
            dff = dataCleanMonthlyRR.fillna(method='ffill')
            dataCleanMonthlyRR = (dfb+dff)/2
        elif Interploation == 'median':        
            df=dataCleanMonthlyRR.reset_index().set_index('Date')
            df['GWL']=df['GWL'].interpolate(method='pad')
            dataCleanMonthlyRR['GWL']=df['GWL'].values
        elif Interploation == 'prophet':
            df=dataCleanMonthlyRR.reset_index().set_index('Date')
            fit_data=pd.DataFrame({'ds':df.index,'y':df['GWL'].values})
            model = Prophet()
            with suppress_stdout_stderr():
                model.fit(fit_data)
            future = model.make_future_dataframe(periods=0, freq=Freq)
            forecast = model.predict(future)
            forecast=forecast[['ds','yhat']].rename(columns={'ds':'Date', 'yhat':'GWL'}).set_index('Date')
            mask = dataCleanMonthlyRR.isnull()
            dataCleanMonthlyRR[mask] = forecast[mask]
        if scaling== 'standard':
            scaler = StandardScaler()
            dataCleanMonthlyRR['GWL']= scaler.fit_transform(dataCleanMonthlyRR).flatten()
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
            dataCleanMonthlyRR['GWL']= scaler.fit_transform(dataCleanMonthlyRR).flatten()
        elif scaling == 'zscore':
            temp = np.array(stats.zscore(dataCleanMonthlyRR['GWL']))
            temp = preprocessing.differencing(temp).reshape(-1,1).flatten()
            # Since the differencing decrease the inputs by one, we add the mean of the data at the end so we car return it back to the DataFrame
            dataCleanMonthlyRR['GWL']= np.append(temp, temp.mean())
        return dataCleanMonthlyRR    

    ##################################### Prediction ########################################################        

    # Clean and impute the missing data
    def prediction(dataCleanMonthlyRR):
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        logging.getLogger('prophet').setLevel(logging.ERROR)
        name=dataCleanMonthlyRR[0]
        dataCleanMonthlyRR=copy.copy(dataCleanMonthlyRR[1])
        dateinit=dataCleanMonthlyRR['GWL'].index[-1][1] ############
        initdate=tsperiod.get_loc(dateinit)
        enddate=-1
        studates=pd.date_range(tsperiod[initdate],tsperiod[enddate],freq=Freq)
        periods=len(studates)
        df=dataCleanMonthlyRR.reset_index().set_index('Date')
        fit_data=pd.DataFrame({'ds':df.index,'y':df['GWL'].values})
        model = Prophet()
        with suppress_stdout_stderr():
            model.fit(fit_data)
        future = model.make_future_dataframe(periods=periods, freq=Freq)
        forecast = model.predict(future)
        dataCleanMonthlyRR=forecast[['ds','yhat']].set_index('ds')
        dataCleanMonthlyRR['WellCode']=name
        dataCleanMonthlyRR=dataCleanMonthlyRR.reset_index()
        dataCleanMonthlyRR=dataCleanMonthlyRR.iloc[:, [2,0,1]]
        dataCleanMonthlyRR=dataCleanMonthlyRR.rename(columns = {'ds':'Date', 'yhat':'GWL'})
        dataCleanMonthlyRR.set_index(['WellCode','Date'], inplace=True)
        return dataCleanMonthlyRR


    ##################################### DataMat ######################################################## 



    def DataMat(i):    
        dataCleanMonthlyR=ray.get(a)
        datagroup=dataCleanMonthlyR.groupby('WellCode')
        n_series = len(datagroup.groups)
        distance_matrix = np.zeros(shape=(n_series, n_series))
        for j in range(i,n_series):
            if i == j:
                distance_matrix[i,j]=0       
                continue
            dateinit=dataCleanMonthlyR['GWL'][indx[i]].index[0]
            datend=dataCleanMonthlyR['GWL'][indx[i]].index[-1]
            dateinit2=dataCleanMonthlyR['GWL'][indx[j]].index[0]
            datend2=dataCleanMonthlyR['GWL'][indx[j]].index[-1]
            # in this confusing section we are trying to find the earlier occuring Well, if the earlier one is (i) we choose her first Date as the initial date
            # otherwise we choose (j) first date as the initial Date, Depending on which we choose, we will set the Value for (var) variable
            # so that we can use this information later on
            if tsperiod.get_loc(dateinit)>tsperiod.get_loc(dateinit2): 
                initdate=tsperiod.get_loc(dateinit)      # if (i) is earlier, we choose its first date as the initial date
                var=2                                    # We Also set initdate2 as the end date of the earliers date, this will be used to calculate Prophet prediction, We Flag this choice with var=2
            elif tsperiod.get_loc(dateinit)<tsperiod.get_loc(dateinit2):
                initdate=tsperiod.get_loc(dateinit2)
                var=1
            else:
                initdate=tsperiod.get_loc(dateinit)
                var=3
            if tsperiod.get_loc(datend)<tsperiod.get_loc(datend2): 
                enddate=tsperiod.get_loc(datend)
            else:
                enddate=tsperiod.get_loc(datend2)
            studates=pd.date_range(tsperiod[initdate],tsperiod[enddate],freq=Freq)
            if dateinit == dateinit2 and datend == datend2:
                equals=True
            else:
                equals=False
            if len(studates) > Comperiod or equals==True:
                s1=np.array(dataCleanMonthlyR['GWL'][indx[i]][studates],dtype=np.double)
                s2=np.array(dataCleanMonthlyR['GWL'][indx[j]][studates],dtype=np.double)
                if Similarity== 'dtw':
                    dist = dtw.distance_fast(s1, s2)
                    distance_matrix[i,j]=(dist/len(studates))    # Critical Bug Fix: deviding by the number of compared periods
                elif Similarity== 'euclidean':
                    dist = ed.distance_fast(s1, s2)
                    distance_matrix[i,j]=(dist/len(studates))
            # Add more methods
            #             elif Similarity== 'euclidean':
            #                 dist = ed.distance(s1, s2)
            #                 distance_matrix[i,j]=dist
            else:    
                if unsynced == 'dtw':
                    s1=np.array(dataCleanMonthlyR['GWL'][indx[i]],dtype=np.double)
                    s2=np.array(dataCleanMonthlyR['GWL'][indx[j]],dtype=np.double)
                    pstudates=np.mean([len(s1),len(s2)]).astype(int)
                    dist = dtw.distance_fast(s1, s2)
                    distance_matrix[i,j]=(dist/pstudates) 
                elif unsynced == 'mean':
                    s1=Avg[indx[i]]
                    s2=Avg[indx[j]]
                    dist = np.abs(s1-s2)
                    distance_matrix[i,j]=dist
                elif unsynced == 'null':
                    distance_matrix[i,j]=np.inf
                elif unsynced == 'mode':
                    s1=dataCleanMonthlyR['GWL'][indx[i]].mode()[0]
                    s2=dataCleanMonthlyR['GWL'][indx[j]].mode()[0]
                    dist = np.abs(s1-s2)
                    distance_matrix[i,j]=dist
                elif unsynced == 'median':
                    s1=dataCleanMonthlyR['GWL'][indx[i]].median()
                    s2=dataCleanMonthlyR['GWL'][indx[j]].median()
                    dist = np.abs(s1-s2)
                    distance_matrix[i,j]=dist
                elif unsynced=='prophet':
                    dataCleanMonthlyY=ray.get(b)
                    if var==3:
                        if tsperiod.get_loc(datend)<tsperiod.get_loc(datend2): 
                            var=2
                        else:
                            var=1
                    if var==1:
                        studates=pd.date_range(tsperiod[tsperiod.get_loc(dateinit2)],tsperiod[tsperiod.get_loc(datend2)],freq=Freq)
                        s1=np.array(dataCleanMonthlyY['GWL'][indx[i]][studates].values,dtype=np.double)
                        s2=np.array(dataCleanMonthlyR['GWL'][indx[j]].values,dtype=np.double)
                    else:
                        studates=pd.date_range(tsperiod[tsperiod.get_loc(dateinit)],tsperiod[tsperiod.get_loc(datend)],freq=Freq)
                        s1=np.array(dataCleanMonthlyR['GWL'][indx[i]].values,dtype=np.double)
                        s2=np.array(dataCleanMonthlyY['GWL'][indx[j]][studates].values,dtype=np.double)
                    if Similarity== 'dtw':
                        dist = dtw.distance_fast(s1, s2)
                        distance_matrix[i,j]=(dist/len(studates))
                    elif Similarity== 'euclidean':
                        dist = ed.distance_fast(s1, s2)
                        distance_matrix[i,j]=(dist/len(studates))
        return distance_matrix[i]    


    ######################################## PWCCM ###################################################

    # Function to check the Accuracy of the Model Prediction (Purity Based)
    def PWCCM (WellsDic, CWells):
        WellsResults=pd.DataFrame()
        report=pd.DataFrame()
        for basin in WellsDic:
            for well in WellsDic[basin]:
                cluster=CWells['Cluster'][well]
    #             WellsResults=WellsResults.append([[basin, well, cluster]])
                temp=pd.DataFrame([[basin, well, cluster]])
                WellsResults= pd.concat((WellsResults, temp), axis=0)
        WellsResults.columns=['Basin','Wells', 'Cluster']
        WellsResults=WellsResults.set_index('Basin')
        WellsResultsG=WellsResults.groupby('Basin')
        for basins in WellsResultsG.groups:
            testG=WellsResultsG.get_group(basins).reset_index()
            BasinClust=testG['Cluster'].mode()    #Cluster Name
            BasinClust=BasinClust[0]
            BasinWells=testG.shape[0]
            ClustWells=testG['Cluster'].value_counts()[0]
            TotClustWells=CWells['Cluster'].value_counts()[BasinClust]
            Accuracy= (ClustWells)/((BasinWells-ClustWells)+(TotClustWells-ClustWells)+ClustWells) #Our Method for Cluster Purity
    #         Accuracy= (ClustWells/TotClustWells) #Cluster Purity based on Literature 
            temp2=pd.DataFrame([[basins, BasinClust, BasinWells, ClustWells, TotClustWells, Accuracy]])
            report= pd.concat((report, temp2), axis=0)
    #         report=report.append(temp2,ignore_index=True)
        report.columns=['Basin', 'MostProbableCluster', 'NumberOfWellsinBasin', 'CorrectlyPredictedWells', 'NumberOfWellsinCluster' ,'PredictionAccuracy']
    #     print('Overall Accuracy: ', round((report['PredictionAccuracy'].mean()*100),2), '%')    
        return report    

                   ############################################################################ 
    
    
    
# def calculating_matrix(param, data):
    # load The Pickle File
    with open(pathlib.Path(mydir2, 'dataCleanMonthlyTX.pkl'), 'rb') as file:

        # Call load method to deserialze
        dataCleanMonthly = pickle.load(file)

#     z=['original', 0.6, StudyPeriod, BasinSize, Comperiod, Outliers, Interpolation, Similarity,Unsynced] 
    z=[config['Scaling'], 0.5, 400, 0, config['Comperiod'], config['Outliers'] , config['Interpolation'], config['Similarity'], config['Unsynced'], config['Linkmeth']]
#     z=['original', 0.7, param['StudyPeriod'], param['BasinSize'], param['Comperiod'], param['Outliers'], param['Interpolation'], param['Similarity'], param['Unsynced']]


    finalReport=pd.DataFrame(columns=['Iteration','Scaling', 'CoveredPeriod', 'StudyPeriod', 'BasinSize', 'Comperiod','Outliers',
                                  'Interpolation', 'Similarity', 'Unsynced', 'Linkage', 'NoWells', 'Rand Index',
                                  'AdjustedRandIndex', 'AdjustedMutualInfoScore', 'AveragePurity', 'CalculationTime', 'FileName'])
    methods=['single','average','complete','weighted','centroid','median','ward', 'hdbscan']
    
    
#     print(f'Iteration Number: (Tuning) \n')
    curr_time = datetime.datetime.now()
    print('\nIteration Starting time: ',curr_time.strftime("%Y-%m-%d %H:%M:%S"), '\n')
    iterstart = time.time()
    print(f'The Testing Parameters are: \nScaling: {z[0]}, \nMinimum Data Percentage: {z[1]*100}%, \nMinimum Study Period: {z[2]} Months, \nMinimum Number of Wells per Basin: {z[3]} Well, \nMinimum Comparison period: {z[4]} Months, \nOutliers: {z[5]}, Interpolation: {z[6]}, Similarity: {z[7]}, Unsynced: {z[8]} \n')
    print('step One: Loading and Setting the hyperparameters')
    # This Cell determine the hyperparameters that can tune the model, namely:
    start = time.time()
    scaling=z[0]         #'original'
    covPeriod=z[1]    # The minimum acceptable data coverage for every Well record, its set to 70% by default 
    studPeriod=z[2]    # the minimum length of every Well record, its set to 12 months by default
    MinBasinSize=z[3]  # The Minimum acceptable number of wells per basin
    Comperiod=z[4]     # The Minimum acceptable number of overlapping months between the comapred wells (less than this threshold will be treated as non-overlapping) defaults to 1
    Outliers=z[5] 
    if  Outliers  == 'SD6':
        SD=6
    elif Outliers == 'SD5':
        SD=5
    elif Outliers == 'SD4':
        SD=4
    elif Outliers == 'SD3':
        SD=3
    elif Outliers == 'SD2':
        SD=2
    elif Outliers == 'SD1':
        SD=1
    else:
        print('No Standard Deviation input was Detected, SD defaults to 6 for Outliers Detection')
        SD=6
    Interploation=z[6] 
    Similarity=z[7] 
    unsynced =z[8]
    meth=z[9]
    dataCleanMonthlyR=copy.copy(dataCleanMonthly)
    datagroup=dataCleanMonthlyR.groupby('WellCode')
    
    latlong= pd.read_csv(pathlib.Path(mydir, 'WellMain.txt'), sep='|',encoding='cp1252', low_memory=False)
    latlong=latlong[['StateWellNumber', 'Classification', 'LatitudeDD', 'LongitudeDD', 'Aquifer']]
    latlong=latlong[latlong['Classification']=='Major'].drop(columns='Classification')
    latlong=latlong.rename(columns={'StateWellNumber':'SITE_CODE', 'LatitudeDD':'LATITUDE', 'LongitudeDD':'LONGITUDE', 'Aquifer':'BASIN_NAME'})
    latlong[['SITE_CODE']] = latlong[['SITE_CODE']].astype('str') 
    
    Wells=pd.pivot_table(dataCleanMonthlyR, index = 'Date', columns = 'WellCode', values = 'GWL') #Original
    Avg=Wells.mean()  #Find the average mean of Each Well
    std=Wells.std()*SD #Outlier Detection: Find the Standard Deviation of each well and multiply it by 6
    end = time.time()
    print(f'Done - Elapsed Time: {round(end-start,2)} S\n')

    print('step Two: Clean and impute missing data')
    # This block will clean the data in Parallel
    start = time.time()
    
    #This section will remove any wells that do not fall in Major Aquifers, it does this by comparing the wells
    #in the observation wells to the wells in the Major aquifers, and then filter out any that does not exist in the list
    
#     datagroup=dataCleanMonthlyR.groupby('WellCode')              
#     majorAQ=pd.DataFrame(datagroup.groups.keys(),columns=['SITE_CODE'])
#     majorAQ=majorAQ[majorAQ['SITE_CODE'].isin(latlong['SITE_CODE'])].SITE_CODE.values
#     dataCleanMonthlyR=datagroup.filter(lambda x: x.name in majorAQ)
    
    datagroup=dataCleanMonthlyR.groupby('WellCode')
    latlong=latlong[latlong['SITE_CODE'].isin(datagroup.groups)]
    WBasins=latlong.set_index(['SITE_CODE'])
    WBasins=WBasins.groupby('BASIN_NAME')                        # After Filtering the wells, we group the remaining wells by their respective basins
    droped=[]
    counts=WBasins.count()                                       # then we count how much wells each basin contains
    counts=counts[counts<=MinBasinSize]                          # and filter any that are below our threshold (set to 10 by default)
    counts.dropna(inplace=True)
    counts=list(counts.index)
    for i in counts:                                             # We create a list of the filtered wells through looping in the dataset
        for j in range(len(WBasins.get_group(i))):
            droped.append(list(WBasins.get_group(i).index)[j])
    dataCleanMonthlyR=datagroup.filter(lambda x: x.name not in droped)
    datagroup=dataCleanMonthlyR.groupby('WellCode')
    
    pool = pools 
    
    results = pool.map(dataclean, [x for x in datagroup]) # PARALLEL HEAVEY COMPUTATION !!!!!!!!!!!!!
#    pool.close()
    results = pd.DataFrame(results).dropna()
    if len(results)==0:
        print('No Wells Remain after Filtering based on the the setted Thresholds')
        score=0.001
        return score


    temp=pd.DataFrame()
    for x in range(len(results)):
        temp=pd.concat((temp,results.iloc[x][0].reset_index()), axis = 0)
    dataCleanMonthlyR=temp.set_index(['WellCode','Date'])
    datagroup= dataCleanMonthlyR.groupby('WellCode')
    
    dic=[]
    for well in datagroup.groups: # loop in the wells to create an index number for each well
        dic.append(well)


    end = time.time()
    print(f'Done - Elapsed Time: {round(end-start,2)} S\n')

    
    print('step Three: Filter Wells based on the determined threshold for the hyperparameter')
    # This code block will also remove wells that have no basins and any wells that falls into basins that have less than the MinBasinSize 
    start = time.time()
    latlong=latlong[latlong['SITE_CODE'].isin(datagroup.groups)] # We save the wells that were filtered in the preprocessing from the dictunary
    noBasin=latlong[latlong.BASIN_NAME.isna()].SITE_CODE.values  # we filter the wellls that have no basins assgined to them
    latlong=latlong[~latlong.BASIN_NAME.isna()]
    latlong=latlong.reset_index().drop('index',axis=1)
    dataCleanMonthlyR=datagroup.filter(lambda x: x.name not in noBasin)
    datagroup=dataCleanMonthlyR.groupby('WellCode')
    latlong=latlong[latlong['SITE_CODE'].isin(datagroup.groups)]
    WBasins=latlong.set_index(['SITE_CODE'])
    WBasins=WBasins.groupby('BASIN_NAME')                        # After Filtering the wells, we group the remaining wells by their respective basins
    droped=[]
    counts=WBasins.count()                                       # then we count how much wells each basin contains
    counts=counts[counts<=MinBasinSize]                          # and filter any that are below our threshold (set to 10 by default)
    counts.dropna(inplace=True)
    counts=list(counts.index)
    for i in counts:                                             # We create a list of the filtered wells through looping in the dataset
        for j in range(len(WBasins.get_group(i))):
            droped.append(list(WBasins.get_group(i).index)[j])
    dataCleanMonthlyR=datagroup.filter(lambda x: x.name not in droped) # we drop the wells that are in the list (of wells below the threshold) from the dataset
    Wells=pd.pivot_table(dataCleanMonthlyR, index = 'Date', columns = 'WellCode', values = 'GWL') #Original
    Avg=Wells.mean()  #Find the average mean of Each Well
    std=Wells.std()*SD #Outlier Detection: Find the Standard Deviation of each well and multiply it by 6
    datagroup=dataCleanMonthlyR.groupby('WellCode')
    latlong=latlong[latlong['SITE_CODE'].isin(datagroup.groups)]
    latlong=latlong[~latlong.duplicated(keep='first')]
    latlong=latlong.reset_index().drop('index',axis=1)
    latlong['BASIN_NAME']=latlong.BASIN_NAME.str.lower() 
    latlong=latlong.set_index('SITE_CODE').loc[list(datagroup.groups)].reset_index()  ####### Reorder the latlong to be identical to dataCleanMonthlyR
    WBasins=latlong.set_index(['SITE_CODE'])
    WBasins=WBasins.groupby('BASIN_NAME')
    WellsDic={}
    for x in WBasins:
        for r in x[1].index:
            if x[0] in WellsDic:
                WellsDic[x[0]].append(r)
            else:
                WellsDic.update({x[0]:[r]})

    timeseries=[]
    indx=[]
    for well in datagroup.groups:
        timeseries.append(np.array(datagroup.get_group(well).values))
        indx.append(well)
    n_series=len(datagroup.groups)   

    #Create basins index and prepare the data for accuracy measurements
    BasinIndx=list(WBasins.groups.keys())
    latlong['basinIndex']=None
    for i in range(len(latlong)):
        latlong.loc[i,'basinIndex']=BasinIndx.index(latlong['BASIN_NAME'][i])
    labels=latlong.basinIndex
    end = time.time()
    print(f'Done - Elapsed Time: {round(end-start,2)} S\n')

    print(f'{len(datagroup.groups)} Well remains out of {len(dataClean.groups)} Wells recived\
    \nAcceptable Data Percentage= {covPeriod*100}% \nMinimum Study Period= {studPeriod/12} Year/s\
    \nMinimum Number of Wells per Basin= {MinBasinSize} \nNumber of Remaining Basins= {(len(WBasins))}\n')


    # Check if the requirements are fullfilled to start the Distance Matrix Calculations
    print('step Four: Checking if the requirements are fullfilled to Initiate the Distance Matrix Calculations\n\n')
    n_series = len(datagroup.groups)
    if n_series<10 or (len(WBasins))<3:
#         print(n_series,(len(WBasins)))    #(parameters.index(z))
        df={'Iteration':['Tuning'] ,'Scaling':[z[0]], 'CoveredPeriod':[z[1]], 'StudyPeriod':[z[2]], 
            'BasinSize':[z[3]], 'Comperiod': [z[4]], 'Outliers':z[5],'Interpolation':z[6], 'Similarity':z[7], 
            'Unsynced':z[8], 'Linkage':None, 'NoWells':[n_series], 'Rand Index':None,
        'AdjustedRandIndex':None, 'AdjustedMutualInfoScore':None, 'AveragePurity':None,'CalculationTime':None, 'FileName':None}
        tempdf=pd.DataFrame(df)
        finalReport = pd.concat((finalReport,tempdf),ignore_index=True, axis=0)
        if n_series<2:
            print('Number of Wells is very low, skipping Distance Matrix Calculations')
        else:
            print('Number of Basins is very low, skipping Distance Matrix Calculations')
        iterend = time.time()
        iterlap=round(iterend-iterstart, 2)
        print(f'Iteration Done - Elapsed Time: {iterlap} S\n')
    #     continue
        score=0.001
        print('The Adjusted Rand Index Score is:', score,'====================================================================================\n')
        return score
#        pool.close()
#         return score
#         from sys import exit
#         exit()
    print('Requirements met, proceeding to the next step\n')

    timeseries=[]
    indx=[]
    for well in datagroup.groups:
        timeseries.append(np.array(datagroup.get_group(well).values))
        indx.append(well)
    n_series=len(datagroup.groups)

    dic=[]
    for well in datagroup.groups: # loop in the wells to create an index number for each well
        dic.append(well)


    if unsynced == 'prophet':
        print('step Five: Predicting values based on Prophet Modelling\n')
#        pool = Pool(mp.cpu_count())
        results = pool.map(prediction, [x for x in datagroup]) # PARALLEL HEAVEY COMPUTATION !!!!!!!!!!!!!
#        pool.close()
        dataCleanMonthlyY=pd.DataFrame() ######################################################################################
        for well in datagroup.groups:  # Loop through the Data to recreate the original dataCleanMonthly but with predicted values instead of original ones
            dataCleanMonthlyY=pd.concat((dataCleanMonthlyY,results[dic.index(well)]), axis = 0)
        datagroup2=dataCleanMonthlyY.groupby('WellCode') #apply and format the results
        b = ray.put(dataCleanMonthlyY)
        end = time.time()    
        print(f'Done - Elapsed Time: {round(end-start,2)} S\n')
        print('step Six: Calculate the Distance Matrix')
    else:
        print('step Five: Calculate the Distance Matrix')
        

#############################################################################################        

    a = ray.put(dataCleanMonthlyR)


    start = time.time()

    # Initialize distance matrix    
    distance_matrix = np.zeros(shape=(n_series, n_series))
    counter=[]
    p=0          

#    pool = Pool(mp.cpu_count())
    results2 = pool.map(DataMat, [i for i in range(n_series)])  # PARALLEL HEAVEY COMPUTATION !!!!!!!!!!!!!
#    pool.close()
            
    # Build distance matrix
    for i in range(n_series):
        for j in range(i,n_series):
            distance_matrix[i,j]=distance_matrix[j,i]=results2[i][j]

                        
##################################### DataMat ######################################################## 


    dist_matrix = squareform(distance_matrix)
    end = time.time()
    print(f'Done - Elapsed Time: {round(end-start,2)} S\n')  




    # Create a Pickle File

    fileName='distance_matrix2(Turning).pkl' #+str(parameters.index(z))+

###########################################################################################

    if meth=='hdbscan':
        clusterer = hdbscan.HDBSCAN(metric='precomputed')
        clusterer.fit(distance_matrix)
        cluster_labels=clusterer.labels_
        y_pred= clusterer.labels_.tolist()
    else:
        linkage_matrix = linkage(dist_matrix, method=meth, metric='euclidean')
        cluster_labels = fcluster(linkage_matrix, len(WBasins), criterion='maxclust')
        y_pred=cluster_labels.tolist()
    
    
    score=adjusted_rand_score(labels, y_pred)
    print('The Adjusted Rand Index Score is:', score,'====================================================================================\n')
    if score==0.0:
        score=0.001
    return score
    
    
    
if __name__ == "__main__":
    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    Freq='MS'  # Supported & Tested Frequencies are Annual (AS) or Monthly (MS)
    config = {
    "Scaling": tune.choice(['minmax', 'zscore', 'standard']),  # Choose one of these options uniformly
#    "CoveredPeriod": tune.quniform(0.7, 0.9, 0.1), #Uniform float between -5 and -1
#    "StudyPeriod": tune.randint(12, 1200),  # Uniform float between -5 and -1
#    "BasinSize": tune.randint(5, 50),  # Uniform float between -5 and -1
    "Comperiod": tune.randint(2, 1200),  # Uniform float between -5 and -1
    "Outliers": tune.choice(['SD6','SD5','SD4','SD3','SD2']),  # Choose one of these options uniformly
    "Interpolation": tune.choice(['linear','prophet', 'mean', 'median']),  # Choose one of these options uniformly
    "Similarity": tune.choice(['dtw','euclidean']),  # Choose one of these options uniformly    
    "Unsynced": tune.choice(['mean','mode','median','prophet','dtw']),  # Choose one of these options uniformly
    "Linkmeth": tune.choice(['single','average','complete','weighted','centroid','median','ward', 'hdbscan'])  # Choose one of these options uniformly
    }

    # load The Pickle File
#    with open(pathlib.Path(mydir2, 'dataCleanAnnual.pkl'), 'rb') as file:

        # Call load method to deserialze
 #       dataCleanMonthly = pickle.load(file)
    
    # previously_run_params = [{'StudyPeriod': 221, 'BasinSize': 23, 'Comperiod': 59, 'Outliers': 'SD2', 'Interpolation': 'linear', 'Similarity': 'euclidean', 'Unsynced': 'mean'}]

    # known_rewards = [0.811941766]

    algo = HEBOSearch(
        random_state_seed=42,  # for reproducibility
        max_concurrent=1
        # points_to_evaluate=previously_run_params,
        # evaluated_rewards=known_rewards,
    )
    
    pools = Pool()
    analysis = tune.run(calculating_matrix, 
                     mode="max", 
                     config=config,
                     name='M50DSOAQ400SP3',
                     search_alg=algo, 
                     num_samples=1000)
    print("Best hyperparameters found were: ", analysis.best_config)
    algo.save('M50DSOAQ400SP3.pkl')
    pools.close()

