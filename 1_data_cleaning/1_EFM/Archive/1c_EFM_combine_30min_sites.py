import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os 
import shutil
import glob
from datetime import datetime, timedelta
import numpy as np
import pickle
import argparse
from helper import *
import pickle

"""
This script takes the single site EFM data generated from 1b_EFM_single_site_30min_50HZ.py and 
stores them in a file for that specific thrity minute interval. It also claculates the correct 
times for each site. 
"""

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
        return slurm_vars
    else:
        print("No SLURM environment variables found.")
        return {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',type=str,default='2024',help='The year')
    parser.add_argument('--exp',type=int,default=180,help='Slurm array index for day of the julian day')
    parser.add_argument('--nogo',action='store_true',default=False)
    args = vars(parser.parse_args())

    year = args['year'] #string
    day = args['day'] #int
    nogo = args['nogo']
    try:
        month, day = get_day_mo(year,day) #returns string, string
        return year, month, day, nogo
    except Exception as e:
        print('Exception:',e)
        return '2018','01','01',True


#build the correct time axis for the 50 Hz data
def fix_efm_time(temp_df):

    #the date and hour are fixed for each data frame so just get the first one
    date_hr = pd.to_datetime(temp_df['date'].iloc[0][1:-1])
    minute = date_hr.minute

    #save it as a list so we can copy it to store in the data frame as column
    date_hours = [date_hr]
    for i in range(len(temp_df)-1):
        date_hours.append(date_hr)
    temp_df['Date_Hr'] = date_hours

    # check if the first minute is zero or 30.  the calculation is different. 
    if minute==0:
        #calculate given the mins in a float, calculate the number of seconds from the base hour in the 'Date_Hr' column
        temp_df['dt_secs'] = temp_df['mins']*60+temp_df['secs']
    else:#30
    #   #calculate given the mins in a float, calculate the number of seconds from the base hour in the 'Date_Hr' column
        temp_df['dt_secs'] = (temp_df['mins']-30)*60+temp_df['secs']

    #now loop through each time step to add the seconds to the base hour
    date_hr_min_secs = []
    temp_date_hour = temp_df['Date_Hr'].values
    temp_seconds = temp_df['dt_secs'].values

    #loop through to calculate the correct time
    for i in range(len(temp_df)):    
        date_hr_min_secs.append(pd.to_datetime(temp_date_hour[i]) + timedelta(seconds=temp_seconds[i]))
    
    #store the new date_time in a column and set it to the dataframe index
    temp_df['Date_Time'] = date_hr_min_secs
    temp_df = temp_df.drop(columns=['Date_Hr','dt_secs','date','mins','secs'])
    temp_df.index = temp_df['Date_Time']
    
    return temp_df

def build_50HZ_30min_dict(year,month,day):
    #declare the data directory
    efm_dir = '/scratch/bmac87/EFM_50HZ_single_site_pkl/'+year+month+'/'
    if os.path.isdir(efm_dir)==False:
        print('wrong directory, try ourdisk')
        return

    #list the station ids for the valid efms
    stnids = ['1','2','4','5','6','7',
            '8','9','10','11','12',
            '14','15','16','17','18',
            '20','21','22','24','25',
            '26','27','28','29','30',
            '31','32','34','35']
    
    #build the hours
    hours = ['01','02','03','04','05','06',
            '07','08','09','10','11','12',
            '13','14','15','16','17','18',
            '19','20','21','22','23']

    #list the half hours
    half_hours = ['00','30']

    for hour in hours:
        for half_hour in half_hours:
            time_str = '%s%s%s%s%s'%(year,month,day,hour,half_hour)

            #loop through the files for each stn id
            #for each station declare a dictionary to hold the data
            #initialize the data with an empty dataframe
            stn_dict = {}
            for s,stnid in enumerate(stnids):
                stn_dict[stnid] = pd.DataFrame()#initialize in case it is empty
                if s>=0:
                    print(stnid)#for tracking

                    #get the files for the specific day
                    efm_file = stnid+'.0_%s.pkl'%time_str
                    
                    #for the files, convert the times to the correct datetime 
                    if os.path.isfile(efm_dir+efm_file)==True:
                        #open the data and convert to a dataframe
                        df = pickle.load(open(efm_file,'rb'))
                        #if the data exists convert the times
                        if len(df)>0:
                            df = fix_efm_time(df)
                    else:
                        print('empty dataframe, no files for '+stnid,year,month,day)
                        df = pd.DataFrame()

                    #sort the dataframe and remove any duplicates
                    df = df.sort_index()
                    df = df.drop_duplicates()
                    stn_dict[stnid] = df
                    del df
            
            #save off the dictionary so it can be merged into a df later. 
            save_dir = '/scratch/bmac87/1c_EFM_30min_pkl/%s%s/'%(year,month)
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            fsave = '%s_efm_dict.pkl'%(time_str)
            print('saving pickle file:',save_dir+fsave)
            pickle.dump(stn_dict,open(save_dir+fsave,'wb'))

def main():
    slurm_vars = extract_slurm_env()
    year, month, day = parse_args()
    build_50HZ_daily_dict(year, month, day)
    
if __name__=='__main__':
    main()