
DayHours=24
import datetime  # for month group use
import os  # build directory if it doesn't exist
import time

#%matplotlib inline
#from __future__ import print_function
#import matplotlib as mpl  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta  # for month group use
from sklearn.decomposition import FactorAnalysis , PCA
from factor_analyzer import FactorAnalyzer


def GroupProcess(GpDf,GpMethod,*args,**kwargs): #, InProcName,DiffName): 
    GpGroupResult=GpDf.groupby(GpMethod)
    GpResult=GpGroupResult.agg(kwargs)
    DictKeys=kwargs.keys()
    if ('meanForecastedErrorAbs' and 'AAAErrorAbs') in DictKeys: #等於if ('meanForecastedErrorAbs' in DictKeys and 'AAAErrorAbs' in DictKeys):
        if  len(args)==1:  ##對於分類類型加入贅字HOUR、DAY、WEEK、MONTH
            GpResult['%sAverDiff_MF-AAA'%args]=GpResult['meanForecastedErrorAbs']-GpResult['AAAErrorAbs'] #'%sAverDiff'%(GpMethod  #GpMethod+'AverDiff' 兩種方法皆可用
        else:
            GpResult['%sAverDiff_MF-AAA'%GpMethod]=GpResult['meanForecastedErrorAbs']-GpResult['AAAErrorAbs']
    else:
        pass
    return GpResult

def TimeParseBin(Df, GpType):
    
    def TimeSnEPoinTHandle(startTime,endTime,year):
        startYear=startTime.year
        endYear=endTime.year
        if year==startYear:
            SnEDayTempList=['%d %d %d 0'%(startTime.day,startTime.month,year),'%d %d %d 0'%(1,1,year+1)]
            startDay,EndDay=[datetime.datetime.strptime(i, "%d %m %Y %H") for i in SnEDayTempList]
        elif year==endYear:
            SnEDayTempList=['%d %d %d 0'%(1,1,year),'%d %d %d 0'%(endTime.day,endTime.month,year)]
            startDay,EndDay=[datetime.datetime.strptime(i, "%d %m %Y %H") for i in SnEDayTempList]
        else:
            SnEDayTempList=['%d %d %d 0'%(1,1,year),'%d %d %d 0'%(1,1,year+1)]
            startDay,EndDay=[datetime.datetime.strptime(i, "%d %m %Y %H") for i in SnEDayTempList]

        return startDay,EndDay

    def TimeBinBuild(startYearDay,EndYearDay,GpType):
        bins=[]
        if GpType=='Day':
            EndYearDay=EndYearDay+datetime.timedelta(days=2)
            for i in range(0,366):
                DayParse=startYearDay+datetime.timedelta(days=i)
                DayTimeStamp=DayParse.timestamp()
                if DayTimeStamp < EndYearDay.timestamp():
                    bins.append(DayTimeStamp)
                else:
                    break
        elif GpType=='Week':
            EndYearDay=EndYearDay+datetime.timedelta(days=8)
            startYearDay=startYearDay-datetime.timedelta(days=startYearDay.isoweekday()-1)
            for i in range(0,366,7):
                DayParse=startYearDay+datetime.timedelta(days=i)
                DayTimeStamp=DayParse.timestamp()
                if DayTimeStamp < EndYearDay.timestamp():
                    bins.append(DayTimeStamp)
                else:
                    break
        elif GpType==('Month'):
            startYearDayTemp='%d %d %d 0'%(1,startYearDay.month,startYearDay.year)
            startYearDay=datetime.datetime.strptime(startYearDayTemp, "%d %m %Y %H")
            EndYearDayTemp='%d %d %d 0'%(1,EndYearDay.month,EndYearDay.year)
            EndYearDay=datetime.datetime.strptime(EndYearDayTemp, "%d %m %Y %H")
            EndYearDay=EndYearDay+ relativedelta(months=2)      #need to use datautil to realize the month deviation
            for i in range(0,13):
                DayParse=startYearDay+ relativedelta(months=i)
                DayTimeStamp=DayParse.timestamp()
                if DayTimeStamp < EndYearDay.timestamp():
                    bins.append(DayTimeStamp)
                else:
                    break
        else:
            pass

        return bins

    binsList=[]        
    startTime=datetime.datetime.fromtimestamp(Df.ix[0,'Time1'])
    endTime=datetime.datetime.fromtimestamp(Df.ix[-1,'Time1'])    
    startYear=startTime.year
    endYear=endTime.year
    for i in range(startYear, endYear+1):
        startDay, EndDay=TimeSnEPoinTHandle(startTime,endTime,i)
        binsList.extend(TimeBinBuild(startDay, EndDay,GpType))
    binsList= sorted(set(binsList))  #,key=binsList.index==>保存原來排序用
    bins=np.array(binsList)
    Gpdf=pd.cut(Df['Time1'],bins,labels=bins[:-1],right = False)
    return Gpdf   

def GroupSubplot(GpDf,GpType,PlotType,**kwargs):
    #plt.rcParams['font.family']='SimHei' #⿊體--跑不出來
    #plt.style.use('ggplot') #mpl.style.use('ggplot')  ## matplotlib(mpl).style.use 與 matplotlib.pyplot(plt).style.use皆可用 
    plotnums=len(kwargs.keys())
    xAxis=GpDf.index
    if PlotType is 'subplot':
        plt.figure(figsize=(16, 8))
        # GpDf.to_csv('GpDf.csv')
        for cumu, kwkey in enumerate(kwargs):  
            if cumu is 0:
                ax1=plt.subplot(plotnums*100+10+cumu+1) #plt.subplot(2,1) #plt.subplot(211)
            else:
                plt.subplot(plotnums*100+10+cumu+1,sharex=ax1,sharey=ax1)
            plt.bar(np.arange(len(xAxis)),GpDf[kwkey],color=kwargs[kwkey],align='center')#,figsize=(16,8)
            if len(str(xAxis[1]))>5:
                plt.xticks(np.arange(len(xAxis)),xAxis,rotation=90) #, fontsize =18 ##搭配plt.bar裡的x，自定義排序，否則會照matplotlib裡的規則，Inmatplotlib說明文檔的Custom Tick，In
            else:
                plt.xticks(np.arange(len(xAxis)),xAxis)
            plt.xlabel(GpType)
            plt.ylabel('AveError')
            plt.title('%sAve%s'%(GpType,kwkey))
        plt.show()
    elif PlotType is 'axplot':
        #雙圖併一圖
        xAxis='TimeSeg'
        for cumu, kwkey in enumerate(kwargs):
            if cumu is 0:
                axe=GpDf.plot(x=xAxis,y=kwkey,kind='bar',title=('%sAve%s'%(GpType,kwkey)),align='center',color=kwargs[kwkey],alpha=0.8) #,gridsize=10 
            else:
                GpDf.plot(x=xAxis,y=kwkey,kind='bar',title=('%sAveError'%(GpType)),align='center',color=kwargs[kwkey],alpha=0.6,figsize=(16,8),ax=axe) #在實際出圖上ax傳入的圖標題'AAADayAveError'會消失，不會出來
        plt.xlabel(GpType)
        plt.ylabel('AveError')
        plt.show()
            #day diffence
def DataCome(dfLeft,dfRight,MergeIndex,*args):
    
    def InfoKeep(dfLeft,MergeIndex):
        Info=[]
        colList=dfLeft.columns.values
        for i in colList:
            if ('ID' in i) or (i == 'Time0'): #or (i == 'Time1'):
                Info.append(i)
            else:
                pass
        dfInfo=dfLeft[[i for i in Info]]
        return dfInfo

    MergeMethod='left'
    dfInfo=InfoKeep(dfLeft,MergeIndex)
    Rebuildargs=list(args)
    Rebuildargs.append(MergeIndex)
    #Rebuildargs.append('Time0')
    #Rebuildargs.append('StationID')
    # dfLeft=dfLeft[[i for i in Rebuildargs]]
    # dfRight=dfRight[[i for i in Rebuildargs]]
    # Rename={i:'%sObs'%i for i in args}
    # dfRight = dfRight.rename(columns=Rename)
    DataMerge=pd.merge(dfLeft,dfRight,how=MergeMethod,on=MergeIndex)
    DataMerge=DataMerge.dropna()
    #DataMergeInfo=pd.merge(DataMerge,dfInfo,how=MergeMethod,left_index=True,right_index=True)#,on=MergeIndex
    #DataMergeInfo=DataMergeInfo.dropna()
    return DataMerge#DataMergeInfo

# def missingData(Df,):
# #此段可以抓出我每行經過的資料含index(row.Index裡存row索引)
#     tp=1
#     recordIndex=[]
#     recordTime0=[]
#     for row in Df.itertuples(index=True, name='Pandas'):
#         #print(row.Index)
#         #txxx=getattr(row.SiteID)  #getattr(row,'SiteID')
#         #print(txxx)
#         if row.period%DayHours!=0:
#             if row.period==tp:
#                 tp+=1
#             else:
#                 recordIndex.append(row.Index)
#                 if not row.Time0 in recordTime0:
#                     recordTime0.append(row.Time0)  #有120組天數的資料有缺失
#                 tp=1
#         else:
#             tp=1

def readnPreprocData(SiteID,StationID):
    PowerData = pd.read_csv('nationalgridVsAAA.csv')
    WindData = pd.read_csv('.\WeatherForecastData\WeatherObserved.csv')
    PowerData =PowerData.drop(['AAA_forecast','MeanForecasted','meanForecastedError','AAAError','meanForecastedErrorAbs','AAAErrorAbs','Time0','period'],axis=1)
    WindData =WindData.drop(['TotalCloud','Temperature','Humidity','Radiation'],axis=1)
    PowerData= PowerData[PowerData['SiteID']==SiteID]
    WindData = WindData[WindData['StationID']==StationID]
    return PowerData,WindData

def ortho_rotation(lam, method='varimax',gamma=None,
                   eps=1e-6, itermax=100):
    """
    Return orthogal rotation matrix
    TODO: - other types beyond 
    consider with sklearn's components,or any matrix, suppose that rotation matrix "TM" for components matrix "lam"
    final_result_matrix=lam*TM 
    """
    if gamma == None:
        if (method == 'varimax'):
            gamma = 1.0
        if (method == 'quartimax'):
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):   
            break
        var = var_new

    return R


def main():
    StationID=[3693,3769,3781,3784,3796,3797,99154,99182,99185,353506,355862]
    SiteID=[575]#,576,602]
    for i in SiteID:
        PCAdataSpeed=pd.DataFrame()
        PCAdataAngle=pd.DataFrame()
        SpeedfeaturesName=[]
        for j in StationID:
            PowerData,WindData=readnPreprocData(i,j)
            #
            DataMerge=DataCome(WindData,PowerData,'Time1',*('WindSpeed', 'WindDirection'))
            DataMerge=DataMerge.drop_duplicates('Time1', keep='last')   
            if not os.path.exists('.\WindForecastProceData\MergeWnP%d'%i):
                os.makedirs('.\WindForecastProceData\MergeWnP%d'%i)
            DataMerge.to_csv('.\WindForecastProceData\MergeWnP%d\DataMergeW%dnP%d.csv'%(i,j,i))

            Rename={'WindSpeed':'WindSpeed_%s'%j,'WindDirection':'WindDirection_%s'%j}
            SpeedfeaturesName.append(Rename['WindSpeed'])
            DataMerge = DataMerge.rename(columns=Rename)
            PCAdataSpeed[Rename['WindSpeed']]=DataMerge[Rename['WindSpeed']]
            PCAdataAngle[Rename['WindDirection']]=DataMerge[Rename['WindDirection']]
        PCAdataSpeed['WindPower']=DataMerge['actualAbs']
        PCAdataAngle['WindPower']=DataMerge['actualAbs']
        PCAdataSpeed['Time1']=DataMerge['Time1']
        PCAdataAngle['Time1']=DataMerge['Time1']
        # PCAdataSpeed.to_csv('.\WindForecastProceData\PCAdataSpeed_%d.csv'%i)
        # PCAdataAngle.to_csv('.\WindForecastProceData\PCAdataAngle_%d.csv'%i)

        MeanPower=PCAdataSpeed.mean(axis=0)['WindPower']
        PowerStd=PCAdataSpeed.std(axis=0)['WindPower']
        for speedtuple in PCAdataSpeed.itertuples():
            tp=speedtuple.Index
            if PCAdataSpeed.ix[tp,'WindPower']<(MeanPower-PowerStd):
                PCAdataSpeed.ix[tp,'PowerLevel']=1
            elif MeanPower>PCAdataSpeed.ix[tp,'WindPower']>(MeanPower-PowerStd):
                PCAdataSpeed.ix[tp,'PowerLevel']=2
            elif (MeanPower+PowerStd)>PCAdataSpeed.ix[tp,'WindPower']>MeanPower:
                PCAdataSpeed.ix[tp,'PowerLevel']=3
            else:
                PCAdataSpeed.ix[tp,'PowerLevel']=4

        PCAdataSpeed=PCAdataSpeed.reset_index() #drop=True
        PCAdataSpeed.to_csv('.\WindForecastProceData\PCAdataSpeed_%d.csv'%i)
        PCAdataAngle.to_csv('.\WindForecastProceData\PCAdataAngle_%d.csv'%i)  
        Speedfeatures=np.array(PCAdataSpeed[[i for i in SpeedfeaturesName]].values, dtype='float64')
        PowerLabes=np.array(PCAdataSpeed['PowerLevel'].values, dtype='float64')
        #PCA  
        pca=PCA(n_components=2,svd_solver='full',copy=True) #n_components='mle'n_components=5 ,whiten=True
        TransData=pca.fit_transform(Speedfeatures, y=PowerLabes)
        # #TransData=pca.fit_transform(Speedfeatures)
        # print(TransData)
        # print('-'*50)
        # #print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)
        # OriData=pca.inverse_transform(TransData)
        #print(OriData)
        pos = pd.DataFrame()
        pos['PC1'] = TransData[:,0]
        pos['PC2'] = TransData[:,1]
        pos['class'] = PCAdataSpeed['PowerLevel']
        print(pos['class'])
        plt.figure(figsize=(18,6))
        ax = pos[pos['class']==1].plot(kind='scatter', x='PC1', y='PC2', color='blue', label='smallest P')
        ax = pos[pos['class']==2].plot(kind='scatter', x='PC1', y='PC2', color='green', label='relat small P', ax=ax)
        ax = pos[pos['class']==3].plot(kind='scatter', x='PC1', y='PC2', color='red', label='relat big P', ax=ax)
        ax= pos[pos['class']==4].plot(kind='scatter', x='PC1', y='PC2', color='lime', label='biggest P', ax=ax)
        plt.show()

        # #FA sklearn因素分析裡面的component矩陣(也就是W將舊資料映射之新資料的矩陣)為X資料的covariance矩陣X(X.T)特徵分解所得的特徵向矩陣量再乘以根號的特徵值矩陣，惟尚未轉軸，需另撰寫採varimax的轉軸矩陣以進一步分析內容，詳見https://ccjou.wordpress.com/2017/01/13/%E5%9B%A0%E7%B4%A0%E5%88%86%E6%9E%90/
        # #因素分析中
        # n_fac=4
        # FAestimator=FactorAnalysis(n_components=n_fac)
        # TransData=FAestimator.fit_transform(Speedfeatures, y=PowerLabes)        #(n,n_fac)
        # pos = pd.DataFrame()
        # for i in range(len(TransData[0])):
        #     pos['PC%d'%(i+1)]=TransData[:,i]
        # pos['class'] = PCAdataSpeed['PowerLevel']
        # pos.to_csv('.\WindForecastProceData\pos.csv')
        # # rot_mat=ortho_rotation(FAestimator.components_)#(11,11)       ##這個雖然在4種轉換之乘法的組合上，是唯一可以很清楚分離因素的weather station，但由書籍資料來看，可能是錯誤用法，另這裡的FAestimator.components_所得為WS,S為特徵值對角矩陣且尚未除以[(根號n-1),n為sample數]
        # # rotation_components=np.matrix(FAestimator.components_)*np.matrix(rot_mat)  #(n_fac,11)*(11,11) =(n_fac,11) ，FAestimator.components_在x為(n,m)的Pca分解上為(W.T)，在
        # # rot_component_points=np.array(rotation_components.transpose())        ##
        # rot_mat=ortho_rotation(FAestimator.components_.transpose()) #(n_fac,n_fac)  書籍資料上的用法，詳見https://books.google.com/books?id=Lyxww9PImZUC&pg=PA243&lpg=PA243&dq=%E7%9F%A9%E9%99%A3%E8%BD%89%E8%BB%B8&source=bl&ots=DRHLRm7Xe5&sig=nth5cnidqvpY2mYtzHP1Uuq5j4A&hl=zh-TW&sa=X&ved=0ahUKEwjy4JS64tDbAhUJvlkKHXWWBGEQ6AEIdjAL#v=onepage&q=%E7%9F%A9%E9%99%A3%E8%BD%89%E8%BB%B8&f=false
        # rot_component_points=np.array(np.matrix(FAestimator.components_.transpose())*np.matrix(rot_mat))   #(11,n_fac)*(n_fac,n_fac)詳見https://stats.stackexchange.com/questions/612/is-pca-followed-by-a-rotation-such-as-varimax-still-pca
        # print(rot_mat)
        # correlationFactor_Power=pd.DataFrame()
        # for i in range(len(rot_component_points[0])):
        #     correlationFactor_Power['fac%d'%(i+1)]=rot_component_points[:,i]
        # correlationFactor_Power.to_csv('.\WindForecastProceData\correlationFactor_Power1_maybeWrongWaitToCheck.csv')

        # Speedfeatures=PCAdataSpeed[[i for i in SpeedfeaturesName]]
        # fa = FactorAnalyzer()
        # fa.analyze(Speedfeatures, n_fac, method='ml', rotation='varimax')
        # loading=fa.loadings
        # rot_matrix=fa.rotation_matrix
        # print(rot_matrix)
        # loading.to_csv('.\WindForecastProceData\loading2.csv')

        
        # if n_fac==2:
        #     plt.figure(figsize=(20,8))
        #     ax = pos[pos['class']==1].plot(kind='scatter', x='PC1', y='PC2', color='blue', label='smallest P')
        #     ax = pos[pos['class']==2].plot(kind='scatter', x='PC1', y='PC2', color='green', label='relat small P', ax=ax)
        #     ax = pos[pos['class']==3].plot(kind='scatter', x='PC1', y='PC2', color='red', label='relat big P', ax=ax)
        #     ax= pos[pos['class']==4].plot(kind='scatter', x='PC1', y='PC2', color='lime', label='biggest P', ax=ax)
        #     plt.show()
        # else:
        #     pass

    # # # for i in SiteID:
    # # #     for j in StationID:
    # # #         Time=pd.read_csv('nationalgridVsAAA.csv')
    # # #         #
    # # #         DataMerge=DataCome(WindData,PowerData,'Time1',*('WindSpeed', 'WindDirection'))
    # # #         DataMerge=DataMerge.drop_duplicates('Time1', keep='last')   
    # # #         if not os.path.exists('.\WindForecastProceData\MergeWnP%d'%i):
    # # #             os.makedirs('.\WindForecastProceData\MergeWnP%d'%i)
    # # #         DataMerge.to_csv('.\WindForecastProceData\MergeWnP%d\DataMergeW%dnP%d.csv'%(i,j,i))
        
        
            
    # #     #
    # #     # DataMergeDayGroup=DataMerge.groupby(by='Time0')  #這裡的類型是DFGB，加了agg(為多重運算函式，必加操作，或加其他單項運算函式)則變成DF
    # #     # DataMergeDay=DataMergeDayGroup.agg({'WindSpeed':'mean','WindSpeedObs':'mean'})#.reset_index() #這裡的類型回歸DF，agg先後順序會在資料中排列
    # #     # DataMergeDay['TimeSeg'] = [time.strftime("%a %H %b %d %Y",time.localtime(Time1stamp)) for Time1stamp in DataMergeDay.index]
    # #     # Difeerence='WindSpeed'
    # #     # DataMergeDay['%sDiff'%Difeerence]=DataMergeDay['WindSpeed']-DataMergeDay['WindSpeedObs']
    # #     # DataMergeDay.to_csv('.\WindForecastProceData\Time0Group\DataMerge%dTime0Group.csv'%i)
    # #     # #GroupSubplot(DataMergeDay,'WindSpeed','subplot',**{'WindSpeedDiff':'lightskyblue'})
    # #     # #    
    # #     # ### group for period forcasting by program###
    # #     # MeanDiffDict={'WindSpeed':'mean','WindDirection':'mean'}
    # #     # ### Try Aver group for Day Week Month ###
    # #     # ## now is hour group(basic group) ##
    # #     # WindDataHour=GroupProcess(WindData,'Time1',*('Hour',),**MeanDiffDict)
    # #     # WindDataHour['TimeSeg'] = [time.ctime(Time1stamp) for Time1stamp in WindDataHour.index]  ##轉換時間法1-1:for迴圈只要做一件事的簡化
    # #     # WindDataHour=WindDataHour.reset_index()
    # #     # WindDataHour=WindDataHour.set_index('TimeSeg')
    # #     # WindDataHour.to_csv('.\WindForecastProceData\HourForeAve\WindDataHourAve%d.csv'%i)
    # #     # ## now reorganize into Day group ##
    # #     # WindDataDaySeries=TimeParseBin(WindDataHour,'Day')
    # #     # WindDataDay=GroupProcess(WindDataHour,WindDataDaySeries,*('Day',),**MeanDiffDict)
    # #     # WindDataDay['TimeSeg'] = [time.strftime("%a %b %d %Y",time.localtime(Time1stamp)) for Time1stamp in WindDataDay.index]
    # #     # WindDataDay=WindDataDay.reset_index()
    # #     # WindDataDay=WindDataDay.set_index('TimeSeg')
    # #     # #SD1PreciseDay = SD1PreciseDay.rename(columns={'meanForecastedErrorAbs': 'DayAverOfMFError', 'AAAErrorAbs': 'DayAverOfAAAError'})
    # #     # WindDataDay.to_csv('.\WindForecastProceData\DayForeAve\WindDataDayAve%d.csv'%i)
    # #     # # #plot result
    # #     # # #GroupSubplot(SD1PreciseDay,'Day','subplot',**{'meanForecastedErrorAbs':'lime','AAAErrorAbs':'pink'})
    # #     # # GroupSubplot(WindDataDay,'Day','subplot',**{'WindSpeed':'lightskyblue'})
    # #     # # #GroupSubplot(SD1PreciseDay,'Day','axplot',**{'meanForecastedErrorAbs':'lime','AAAErrorAbs':'pink'})
        

    # # # PowerData = pd.read_csv('.\WeatherForecastData\WeatherObserved.csv')
    # # # WindData = pd.read_csv('.\WeatherForecastData\WeatherForecastFull _3693.csv')
    # # # PowerData =PowerData.drop(['Radiation'],axis=1)
    # # # WindData =WindData.drop(['HighCloud','MediumCloud','LowCloud','Radiation'],axis=1)
    # # # PowerData= PowerData[PowerData['StationID']==3693]
    # # # # ### group for Time0 day and Time1  ###
    # # # # bins=pd.Series(WindData['Time0'])
    # # # # bins.drop_duplicates(keep='first')
    # # # # PowerDataGroup=pd.cut(PowerData['Time1'],bins,labels=bins[:-1],right = False)
    # # # # DataObserveTime0Group=WindData.groupby(PowerDataGroup)
    # # # # DataObserveTime0=DataObserveTime0Group.agg(kwargs)
    # # # # PowerData = PowerData.rename(columns={'WindSpeed': 'WindSpeedObs', 'WindDirection': 'WindDirectionObs'})
    # # # # DataMerge=pd.merge(WindData,PowerData,how='left',on='Time1')
    # # # DataMerge=DataCome(WindData,PowerData,'Time1',*('WindSpeed', 'WindDirection'))
    # # # DataMerge.to_csv('DataMerge.csv')
    

    # # # DataMergeDayGroup=DataMerge.groupby(by='Time0')  #這裡的類型是DFGB，加了agg(為多重運算函式，必加操作，或加其他單項運算函式)則變成DF
    # # # DataMergeDay=DataMergeDayGroup.agg({'WindSpeed':'mean','WindSpeedObs':'mean'})#.reset_index() #這裡的類型回歸DF，agg先後順序會在資料中排列
    # # # #SD1Day['meanForecastedErrorAbs','AAAErrorAbs']=SD1Day['meanForecastedErrorAbs','AAAErrorAbs']/DayHours
    # # # DataMergeDay['TimeSeg'] = [time.strftime("%a %H %b %d %Y",time.localtime(Time1stamp)) for Time1stamp in DataMergeDay.index]
    # # # Difeerence='WindSpeed'
    # # # DataMergeDay['%sDiff'%Difeerence]=DataMergeDay['WindSpeed']-DataMergeDay['WindSpeedObs']
    # # # #SD1Day[['meanForecastedErrorAbs','AAAErrorAbs']]=SD1Day[['meanForecastedErrorAbs','AAAErrorAbs']]/DayHours
    # # # #SD1Day = SD1Day.rename(columns={'meanForecastedErrorAbs': 'AverageOfMFError', 'AAAErrorAbs': 'AverageOfAAAError'})
    # # # DataMergeDay.to_csv('DataMergeTime0DayGroup.csv')
    # # # GroupSubplot(DataMergeDay,'WindSpeed','subplot',**{'WindSpeedDiff':'lightskyblue'})

    # # # # WindDataDayGroup=DataMerge.groupby(by='Time0')  #這裡的類型是DFGB，加了agg(為多重運算函式，必加操作，或加其他單項運算函式)則變成DF
    # # # # WindDataDay=WindDataDayGroup.agg({'WindSpeed':'mean','WindDirection':'mean'}).reset_index() #這裡的類型回歸DF，agg先後順序會在資料中排列
    # # # # #SD1Day['meanForecastedErrorAbs','AAAErrorAbs']=SD1Day['meanForecastedErrorAbs','AAAErrorAbs']/DayHours
    # # # # WindDataDay['TimeSeg'] = [time.strftime("%a %b %d %Y",time.localtime(Time1stamp)) for Time1stamp in WindDataDay.index]
    # # # # #SD1Day[['meanForecastedErrorAbs','AAAErrorAbs']]=SD1Day[['meanForecastedErrorAbs','AAAErrorAbs']]/DayHours
    # # # # #SD1Day = SD1Day.rename(columns={'meanForecastedErrorAbs': 'AverageOfMFError', 'AAAErrorAbs': 'AverageOfAAAError'})
    # # # # WindDataDay.to_csv('WindDataTime0DayGroup.csv')
    # # # ### group for period forcasting by program###
    # # # MeanDiffDict={'WindSpeed':'mean','WindDirection':'mean'}

    # # # ### Try Aver group for Day Week Month ###
    # # # ## now is hour group(basic group) ##
    # # # WindDataHour=GroupProcess(WindData,'Time1',*('Hour',),**MeanDiffDict)
    # # # WindDataHour['TimeSeg'] = [time.ctime(Time1stamp) for Time1stamp in WindDataHour.index]  ##轉換時間法1-1:for迴圈只要做一件事的簡化
    # # # WindDataHour=WindDataHour.reset_index()
    # # # WindDataHour=WindDataHour.set_index('TimeSeg')
    # # # WindDataHour.to_csv('WindDataHourGroup.csv')
    # # # ## now reorganize into Day group ##
    # # # WindDataDaySeries=TimeParseBin(WindDataHour,'Day')
    # # # WindDataDay=GroupProcess(WindDataHour,WindDataDaySeries,*('Day',),**MeanDiffDict)
    # # # WindDataDay['TimeSeg'] = [time.strftime("%a %b %d %Y",time.localtime(Time1stamp)) for Time1stamp in WindDataDay.index]
    # # # WindDataDay=WindDataDay.reset_index()
    # # # WindDataDay=WindDataDay.set_index('TimeSeg')
    # # # #SD1PreciseDay = SD1PreciseDay.rename(columns={'meanForecastedErrorAbs': 'DayAverOfMFError', 'AAAErrorAbs': 'DayAverOfAAAError'})
    # # # WindDataDay.to_csv('WindDataDayGroup.csv')
    # # # #plot result
    # # # #GroupSubplot(SD1PreciseDay,'Day','subplot',**{'meanForecastedErrorAbs':'lime','AAAErrorAbs':'pink'})
    # # # GroupSubplot(WindDataDay,'Day','subplot',**{'WindSpeed':'lightskyblue'})
    # # # #GroupSubplot(SD1PreciseDay,'Day','axplot',**{'meanForecastedErrorAbs':'lime','AAAErrorAbs':'pink'})


if __name__=='__main__':
    main()
