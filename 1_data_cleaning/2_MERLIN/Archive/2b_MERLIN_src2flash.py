import os
import shutil
import pandas as pd
import pickle
import glob 
import sys
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

"""
This method turns the CC source files into flashes, then stores them in netcdf files.  
The CC cource information is stored in nano seconds, so first these data are converted 
into UNIX time, in seconds.  Then the time differences between each source 
are calculated, and if the sources are within 300 milliseconds of each other, 
then the sources are considered to be within the same flash.  
Once all the sources are grouped into flashes, 
the mean lat/lon is used to determine the location of the flash.  
These data are stored in a Pandas dataframe then saved in a netcdf file, for future use.
"""

def turnCCtoflashes():
    data_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Sources/LaunchCast/'
    glob_call = data_dir+'MERLIN_CC_*.pkl'
    files = sorted(glob.glob(glob_call))

    print('getting all of the CC sources from all files')
    for f,file in enumerate(files):
        if f==0:
            cc_df = pickle.load(open(file,'rb'))
        else:
            cc_df = pd.concat([cc_df,pickle.load(open(file,'rb'))],axis=0)
    cc_df = cc_df.sort_index(ascending=True)
    print(len(cc_df),'# of CC sources before flash assignment')
    print('building the times to compute the flashes')
    times = cc_df.index
    ux_times  = times.values.astype('int64') #nano_seconds, convert to a standard reference time in nano seconds
    ux_times = ux_times/1e9 #convert to seconds
    cc_df['ux_times'] = ux_times#add back to the dataframe
    ux_t_diff = abs(ux_times[0:len(cc_df)-1]-ux_times[1:len(cc_df)])#calculate the difference in the times
    ux_t_diff = np.append([0],ux_t_diff)#append 0 to the array
    cc_df['ux_diff'] = ux_t_diff#create new column in array for time differences
    cc_df['t_flash'] = cc_df['ux_diff']<=300e-3#300 milliseconds, close to GLM; create booleans for when time differences are within 300 milliseconds for flash sorting

    flash_id = 0 #initialize the flash id and array to store them
    flash_id_array = np.zeros(len(cc_df))
    cc_t_flash = cc_df['t_flash'].values #convert to numpy array for faster processing 

    print('assigning flash ids to all of the sources')
    #loop through each source to assign it to a flash
    for i in range(len(cc_df)):
        if cc_t_flash[i]:
            #this pulse belongs to the flash we have now
            pass
        else:
            #this is a new flash
            flash_id += 1
        flash_id_array[i] = int(flash_id)#assign the source to a flash

    #store the flash id in the dataframe
    cc_df['flash_id'] = flash_id_array

    #parse out the relevant columns
    cc_test = cc_df[['flash_id','Lat_Decimal','Lon_Decimal','ux_times']]

    #take the mean of each flash id to reduce to one flash per group of sources 
    df = cc_test.groupby('flash_id').mean().reset_index()

    #convert the unix times back into normal times in UTC, then assign them as the data frame object 
    times = df['ux_times']
    times_array = []
    for time in times:
        times_array.append(dt.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S'))
    df.index = pd.to_datetime(times_array)
    print(len(df),'# of CC flashes')
    
    save_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/'
    pickle.dump(df,open(save_dir+'merlin_cc_df.pkl','wb'))

def turnCGtoflashes():
    print('making CG flashes')
    data_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Sources/LaunchCast/'
    pkl_file = 'MERLIN_CG.pkl'
    merlin_cg = pickle.load(open(data_dir+pkl_file,'rb'))
    time_list = []
    for t in merlin_cg.index:
        ts = pd.Timestamp(t)
        time_list.append(ts)
    merlin_cg.index = time_list
    print(merlin_cg)
    merlin_cg = merlin_cg.sort_index(ascending=True)
    times = merlin_cg.index
    ux_times  = times.values.astype('int64') #nano_seconds, convert to a standard reference time in nano seconds
    ux_times = ux_times/1e9 #convert to seconds
    merlin_cg['ux_times'] = ux_times#add back to the dataframe
    ux_t_diff = abs(ux_times[0:len(merlin_cg)-1]-ux_times[1:len(merlin_cg)])#calculate the difference in the times
    ux_t_diff = np.append([0],ux_t_diff)#append 0 to the array
    merlin_cg['ux_diff'] = ux_t_diff#create new column in array for time differences
    merlin_cg['t_flash'] = merlin_cg['ux_diff']<=1000e-3#1 seconds, close to GLM; create booleans for when time differences are within 1 second to account for the ground strokes
    # note the temporal reporting changed in 2023 to include the nano seconds
    
    flash_id = 0 #initialize the flash id and array to store them
    flash_id_array = np.zeros(len(merlin_cg))
    cg_t_flash = merlin_cg['t_flash'].values #convert to numpy array for faster processing

    #loop through each source to assign it to a flash
    for i in range(len(merlin_cg)):
        if cg_t_flash[i]:
            #this pulse belongs to the flash we have now
            pass
        else:
            #this is a new flash
            flash_id += 1
        flash_id_array[i] = int(flash_id)#assign the source to a flash

    #store the flash id in the dataframe
    merlin_cg['flash_id'] = flash_id_array

    #parse out the relevant columns
    cg_test = merlin_cg[['flash_id','Lat','Lon','ux_times']]

    #take the mean of each flash id to reduce to one flash per group of sources 
    df = cg_test.groupby('flash_id').mean().reset_index()
    
    #convert the unix times back into normal times in UTC, then assign them as the data frame object 
    times = df['ux_times']
    times_array = []
    for time in times:
        times_array.append(dt.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S'))
    df.index = pd.to_datetime(times_array)
    print(df)
    print(len(df),'# of CG flashes')
    save_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/'
    pickle.dump(df,open(save_dir+'merlin_cg_df.pkl','wb'))

def plot_diurnal():
    
    print("plotting the diurnal cycle of lightning")
    flash_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/'
    cc_file = 'merlin_cc_df.pkl'
    cg_file = 'merlin_cg_df.pkl'

    cc_df = pickle.load(open(flash_dir+cc_file,'rb'))
    cg_df = pickle.load(open(flash_dir+cg_file,'rb'))

    cc_hrly = cc_df.resample('h').count()
    cc_hrly['count'] = cc_hrly['flash_id']
    cc_hrly = cc_hrly.drop(columns=['flash_id','Lat_Decimal','Lon_Decimal','ux_times'])
    cc_hrly['hr'] = cc_hrly.index.hour

    cg_hrly = cg_df.resample('h').count()
    cg_hrly['count'] = cg_hrly['Lat']
    cg_hrly = cg_hrly.drop(columns=['Lat','Lon'])
    cg_hrly['hr'] = cg_hrly.index.hour

    cc_hr_count = np.zeros(24)
    cc_hr_max = np.zeros(24)
    cc_hr_mean = np.zeros(24)
    cc_hr_min = np.zeros(24)

    cg_hr_count = np.zeros(24)
    cg_hr_max = np.zeros(24)
    cg_hr_mean = np.zeros(24)
    cg_hr_min = np.zeros(24)
    for hr in range(24):
        cc_hr_count[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].sum()
        cc_hr_max[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].max()
        cc_hr_mean[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].mean()
        print('cc_mean',cc_hr_mean[hr])
        cc_hr_min[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].min()

        cg_hr_count[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].sum()
        cg_hr_max[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].max()
        cg_hr_mean[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].mean()
        print('cg_mean',cg_hr_mean[hr])
        cg_hr_min[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].min()

    fig, ax1 = plt.subplots(1,1,figsize=(20,10))
    ax1.plot(range(24),cc_hr_mean,linewidth=3,label='CC Mean',color='green')
    ax1.plot(range(24),cc_hr_max,linewidth=3,label='CC Max',color='green',linestyle='--')
    ax1.plot(range(24),cg_hr_mean,linewidth=3,label='CG Mean',color='red')
    ax1.plot(range(24),cg_hr_max,linewidth=3,label='CG Max',color='red',linestyle='--')
    ax1.legend(fontsize=18,loc='upper left')
    ax1.set_yticks([0,500,1000,1500,2000,2500,3000,3500,4000],
                    ['0','500','1000','1500','2000','2500','3000','3500','4000'],
                    fontsize=18)
    ax1.set_xticks(range(24),
                    ['00','01','02','03','04','05','06',
                    '07','08','09','10','11','12',
                    '13','14','15','16','17','18',
                    '19','20','21','22','23'],
                    fontsize=18)
    ax1.set_ylabel('Mean/Max # of Flashes',fontsize=18)

    ax2 = ax1.twinx()
    ax2.plot(range(24),cc_hr_count,linewidth=6,label='CC Count',color='green')
    ax2.plot(range(24),cg_hr_count,linewidth=6,label='CG Count',color='red')
    ax2.legend(fontsize=18,loc='upper right')
    
    plt.xlabel('Time (UTC)',fontsize=18)
    
    ax2.set_yticks([0,5e4,1e5,1.5e5,2e5,2.5e5,3e5,3.5e5,4e5],
                    ['0.0','0.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0'],
                    fontsize=18)
    ax2.set_ylabel('# of Flashes (1e5)',fontsize=18)
    plt.grid('on')
    plt.suptitle('Diurnal Cycle of CG vs. CC, All Seasons, 2018-2024',fontsize=24)
    plt.savefig('merlin_diurnal.png')
    plt.close()

def plot_seasonal():

    print("plotting the seasonal cycle of lightning")
    flash_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/'
    cc_file = 'merlin_cc_df.pkl'
    cg_file = 'merlin_cg_df.pkl'

    cc_df = pickle.load(open(flash_dir+cc_file,'rb'))
    cg_df = pickle.load(open(flash_dir+cg_file,'rb'))

    cc_moly = cc_df.resample('ME').count()
    cc_moly['count'] = cc_moly['flash_id']
    cc_moly = cc_moly.drop(columns=['flash_id','Lat_Decimal','Lon_Decimal','ux_times'])
    cc_moly['mo'] = cc_moly.index.month

    cg_moly = cg_df.resample('ME').count()
    cg_moly['count'] = cg_moly['Lat']
    cg_moly = cg_moly.drop(columns=['Lat','Lon'])
    cg_moly['mo'] = cg_moly.index.month

    cg_mo_count = np.zeros(12)
    cg_mo_max = np.zeros(12)
    cg_mo_mean = np.zeros(12)
    cg_mo_min = np.zeros(12)

    cc_mo_count = np.zeros(12)
    cc_mo_max = np.zeros(12)
    cc_mo_mean = np.zeros(12)
    cc_mo_min = np.zeros(12)
    print('len_ccdf:',len(cc_df),'len_cgdf:',len(cg_df))

    for mo in range(12):
        cc_mo_count[mo] = cc_moly[cc_moly['mo']==mo]['count'].sum()
        cc_mo_max[mo] = cc_moly[cc_moly['mo']==mo]['count'].max()
        cc_mo_mean[mo] = cc_moly[cc_moly['mo']==mo]['count'].mean()
        cc_mo_min[mo] = cc_moly[cc_moly['mo']==mo]['count'].min()
        
        cg_mo_count[mo] = cg_moly[cg_moly['mo']==mo]['count'].sum()
        cg_mo_max[mo] = cg_moly[cg_moly['mo']==mo]['count'].max()
        cg_mo_mean[mo] = cg_moly[cg_moly['mo']==mo]['count'].mean()
        cg_mo_min[mo] = cg_moly[cg_moly['mo']==mo]['count'].min()

    fig,ax1=plt.subplots(1,1,figsize=(20,10))
    ax1.plot(range(12),cc_mo_mean/1e4,linewidth=3,label='CC Mean',color='green')
    ax1.plot(range(12),cc_mo_max/1e4,linewidth=2,label='CC Max',color='green',linestyle='--')
    ax1.plot(range(12),cg_mo_mean/1e4,linewidth=3,label='CG Mean',color='red')
    ax1.plot(range(12),cg_mo_max/1e4,linewidth=3,label='CG Max',color='red',linestyle='--')
    ax1.set_xticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],fontsize=18)
    ax1.set_yticks([0,5,10,15,20,25,30],['00','05','10','15','20','25','30'],fontsize=18)
    ax1.set_ylabel('Mean/Max # of Flashes (1e4)',fontsize=18)
    ax1.set_xlabel('Time (UTC)',fontsize=18)
    ax1.legend(fontsize=18,loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(range(12),cc_mo_count/1e6,linewidth=8,label='CC Count',color='green')
    ax2.plot(range(12),cg_mo_count/1e6,linewidth=8,label='CG Count',color='red')
    ax2.set_ylabel('# of Flashes (1e6)',fontsize=18)
    ax2.set_yticks([0,.2,.4,.6,.8],['0.0','0.2','0.4','0.6','0.8'],fontsize=18)
    ax2.legend(fontsize=18,loc='upper right')
    plt.grid('on')
    plt.suptitle('Seasonal Cycle of CG vs. CC, All Seasons, 2018-2024',fontsize=24)
    plt.xlabel('Month',fontsize=18)
    plt.savefig('merlin_seasonal.png')
    plt.close()




def plot_annual():
    print("plotting the annual trend of lightning")

def main():
    print('creating flashes from sources')
    turnCCtoflashes()
    turnCGtoflashes()
    plot_diurnal()
    plot_seasonal()

if __name__=='__main__':
    main()