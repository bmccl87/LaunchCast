import os
import shutil
import pandas as pd
import pickle
import glob 
import sys
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import argparse

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")
    return slurm_vars

def parse_args():
    print('parsing args')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',type=int,default=54,help='1-84 int for year month')
    args = parser.parse_args()
    exp = args.exp
    if exp>=1 and exp<=12:
        year='2018'
        month=f"{exp:02}"
    elif exp>=13 and exp<=24:
        year='2019'
        month=f"{exp-12:02}"
    elif exp>=25 and exp<=36:
        year='2020'
        month=f"{exp-24:02}"
    elif exp>=37 and exp<=48:
        year='2021'
        month=f"{exp-36:02}"
    elif exp>=49 and exp<=60:
        year='2022'
        month=f"{exp-48:02}"
    elif exp>=61 and exp<=72:
        year='2023'
        month=f"{exp-60:02}"
    else:
        year='2024'
        month=f"{exp-72:02}"
    return year,month

def turn_to_flashes(year='2022',month='06',ltg_type='CC',ms_thresh=300):
    """
    This method turns the CC source files into flashes, then stores them in netcdf files.  
    The CC cource information is stored in nano seconds, so first these data are converted 
    into UNIX time, in seconds.  Then the time differences between each source 
    are calculated, and if the sources are within 300 milliseconds (CC) and 1000 milliseconds (CG) of each other, 
    then the sources are considered to be within the same flash.  
    Once all the sources are grouped into flashes, 
    the mean lat/lon is used to determine the location of the flash.  
    These data are stored in a Pandas dataframe then saved in a pickle file.
    """

    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2a_MERLIN_%s_Sources/%s%s/'%(ltg_type,year,month)
    files = os.listdir(data_dir)
    print('concatenating all of the CC sources from all files')
    for f,file in enumerate(files):
        ltg_dict = pickle.load(open(data_dir+file,'rb'))
        if f==0:
            cc_df = ltg_dict['merlin_df'][['Lat_Decimal','Lon_Decimal']]
        else:
            cc_df = pd.concat([cc_df,ltg_dict['merlin_df'][['Lat_Decimal','Lon_Decimal']]],axis=0)
        del ltg_dict
    cc_df = cc_df.sort_index(ascending=True)

    print('building the times to compute the flashes')
    times = cc_df.index
    ux_times  = times.values.astype('int64') #nano_seconds, convert to a standard reference time in nano seconds
    ux_times = ux_times/1e9 #convert to seconds
    cc_df['ux_times'] = ux_times#add back to the dataframe
    ux_t_diff = abs(ux_times[0:len(cc_df)-1]-ux_times[1:len(cc_df)])#calculate the difference in the times
    ux_t_diff = np.append([0],ux_t_diff)#append 0 to the array
    cc_df['ux_diff'] = ux_t_diff#create new column in array for time differences
    cc_df['t_flash'] = cc_df['ux_diff']<=(ms_thresh/1000) #300 milliseconds - CC, 1 sec CG; close to GLM; create booleans for when time differences are within 300 milliseconds for flash sorting

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
    unix_times = df['ux_times']
    times_dt64 = (np.array(unix_times) * 1e9).astype("datetime64[ns]")
    df.index = pd.to_datetime(times_dt64)     
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2b_MERLIN_%s_Flashes/'%ltg_type
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    pickle.dump(df,open(save_dir+'%s%s_merlin_%s.pkl'%(year,month,ltg_type),'wb'))

def concatenate_all_flash_files(ltg_type='CC'):
    load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2b_MERLIN_%s_Flashes/'%ltg_type
    files = os.listdir(load_dir)
    df_list = []
    for file in files:
        df_list.append(pickle.load(open(load_dir+file,'rb')))
    df = pd.concat(df_list,axis=0)
    df = df[['Lat_Decimal','Lon_Decimal']]
    df = df.sort_index(ascending=True)
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/'
    fname = 'MERLIN_%s_all_flashes.pkl'%ltg_type
    df.to_pickle(save_dir+fname)
    del df

def check_jobs():
    
    years = ['2018','2019','2020','2021','2022','2023','2024']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    ltg_types = ['CG']
    years_list = []
    months_list = []
    for ltg_type in ltg_types:
        for year in years:
            for month in months:

                check_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2b_MERLIN_%s_Flashes/'%(ltg_type)
                fname = '%s%s_merlin_%s.pkl'%(year,month,ltg_type)

                if os.path.isfile(check_dir+fname)==False:
                    print(fname)
                    years_list.append(year)
                    months_list.append(month)

    return years_list, months_list

def plot_diurnal_seasonal():
    
    print("plotting the diurnal cycle of lightning")
    cc_file = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CC_all_flashes.pkl'
    cg_file = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CG_all_flashes.pkl'

    cc_df = pickle.load(open(cc_file,'rb'))
    cg_df = pickle.load(open(cg_file,'rb'))
    print(len(cc_df),len(cg_df))

    cc_color = '#92c5de'
    cc_marker = 'D'
    cc_ls = 'solid'
    cg_color = '#f4a582'
    cg_marker = 'o'
    cg_ls = 'dashdot'

    cc_hrly = cc_df.resample('h').count()
    cc_hrly['count'] = cc_hrly['Lat_Decimal']
    cc_hrly = cc_hrly.drop(columns=['Lat_Decimal','Lon_Decimal'])
    cc_hrly['hr'] = cc_hrly.index.hour

    cg_hrly = cg_df.resample('h').count()
    cg_hrly['count'] = cg_hrly['Lat_Decimal']
    cg_hrly = cg_hrly.drop(columns=['Lat_Decimal','Lon_Decimal'])
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

    cc_moly = cc_df.resample('ME').count()
    cc_moly['count'] = cc_moly['Lat_Decimal']
    cc_moly = cc_moly.drop(columns=['Lat_Decimal','Lon_Decimal'])
    cc_moly['mo'] = cc_moly.index.month

    cg_moly = cg_df.resample('ME').count()
    cg_moly['count'] = cg_moly['Lat_Decimal']
    cg_moly = cg_moly.drop(columns=['Lat_Decimal','Lon_Decimal'])
    cg_moly['mo'] = cg_moly.index.month

    cg_mo_count = np.zeros(12)
    cg_mo_max = np.zeros(12)
    cg_mo_mean = np.zeros(12)
    cg_mo_min = np.zeros(12)

    cc_mo_count = np.zeros(12)
    cc_mo_max = np.zeros(12)
    cc_mo_mean = np.zeros(12)
    cc_mo_min = np.zeros(12)

    for mo in range(12):
        cc_mo_count[mo] = cc_moly[cc_moly['mo']==mo]['count'].sum()
        cc_mo_max[mo] = cc_moly[cc_moly['mo']==mo]['count'].max()
        cc_mo_mean[mo] = cc_moly[cc_moly['mo']==mo]['count'].mean()
        cc_mo_min[mo] = cc_moly[cc_moly['mo']==mo]['count'].min()
        
        cg_mo_count[mo] = cg_moly[cg_moly['mo']==mo]['count'].sum()
        cg_mo_max[mo] = cg_moly[cg_moly['mo']==mo]['count'].max()
        cg_mo_mean[mo] = cg_moly[cg_moly['mo']==mo]['count'].mean()
        cg_mo_min[mo] = cg_moly[cg_moly['mo']==mo]['count'].min()

    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(26,13))
    ax1.plot(range(24),cc_hr_count,linewidth=6,label='CC Count',color=cc_color,marker=cc_marker,linestyle=cc_ls,markeredgecolor='black',markersize=12.0)
    ax1.plot(range(24),cg_hr_count,linewidth=6,label='CG Count',color=cg_color,marker=cg_marker,linestyle=cg_ls,markeredgecolor='black',markersize=12.0)
    ax1.text(.6,7.3e5,'(a)',fontsize=48,fontweight='heavy')
    ax1.set_xlabel('Time (UTC)',fontsize=18)
    ax1.set_xticks(range(24), 
                    ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23'],
                    fontsize=18)
    ax1.set_yticks(np.squeeze(np.linspace(0,10,21)*1e5),
                    ['0.0','0.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0','4.5','5.0','5.5','6.0','6.5','7.0','7.5','8.0','8.5','9.0','9.5','10.0'],
                    fontsize=18)
    ax1.set_ylabel('# of Flashes (1e5)',fontsize=18)
    ax1.set_ylim([0,8e5])
    ax1.set_title('Diurnal Cycle of CG vs. CC, All Seasons, 2018-2024',fontsize=24)
    ax1.grid('on')

    ax2.plot(range(12),cc_mo_count/1e6,linewidth=6,label='CC Count',color=cc_color,marker=cc_marker,linestyle=cc_ls,markeredgecolor='black',markersize=12)
    ax2.plot(range(12),cg_mo_count/1e6,linewidth=6,label='CG Count',color=cg_color,marker=cg_marker,linestyle=cg_ls,markeredgecolor='black',markersize=12)
    ax2.set_ylabel('# of Flashes (1e6)',fontsize=18)
    ax2.set_xticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],fontsize=18)
    ax2.set_ylim([0,1.4])
    ax2.set_yticks(np.linspace(0,1.4,8),['0.0','0.2','0.4','0.6','0.8','1.0','1.2','1.4'],fontsize=18)
    ax2.legend(fontsize=18,loc='upper right')
    ax2.grid('on')
    ax2.text(.4,1.28,'(b)',fontsize=48,fontweight='heavy')
    ax2.set_title('Seasonal Cycle of CG vs. CC, All Seasons, 2018-2024',fontsize=24)
    ax2.set_xlabel('Month',fontsize=18)
    plt.savefig('merlin_diurnal_seasonal.png')
    plt.savefig('merlin_diurnal_seasonal.pdf')
    plt.close()
    

def main():
    plot_diurnal_seasonal()

if __name__=='__main__':
    main()