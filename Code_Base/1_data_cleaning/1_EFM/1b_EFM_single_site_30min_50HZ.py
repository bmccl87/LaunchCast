import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

import glob
import wandb
import time
import copy
from datetime import datetime, timedelta
"""
This script pulls the EFM records for 30 minutes. It goes through each of the EFM 
sites that are included in each 30 minute file, then calculates the vertical 
gradient of the electric potential for each .02 seconds, or 50 HZ frequency. 
"""

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
    #create parser to read in the years for parallelization 
    print('parsing the args and generating the files to process')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',type=int,default=181,help='0-997')
    parser.add_argument('--nogo',action='store_true',default=False,help='Whether or not to process the data')
    args = vars(parser.parse_args())
    exp = args['exp']#int
    print('exp: ',exp)

    start_idx=exp*25
    end_idx=start_idx+24
    file_list = pickle.load(open('missing_efm_files.pkl','rb'))
    # file_list = get_file_list()
    if exp==492:
        end_idx=len(file_list)-1
    files = file_list[start_idx:end_idx]
    nogo = args['nogo']#boolean
    return files, nogo, exp

def get_file_list():
    """
    This method returns a list of EFM files.
    Across all years and months. This is NOT experiment number driven.
    """

    #list the station ids for the valid efms
    stnids = ['1','2','4','5','6','7','8','9','10','11','12','14','15','16','17','18','20','21','22','24','25','26','27','28','29','30','31','32','34','35']
    years = ['2018','2019','2020','2021','2022','2023','2024']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    hours = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    half_hours = ['00','30']
    yrs_dict = years_dict()

    file_list = []
    for year in years:
        months_dict = yrs_dict[year]
        for month in months:
            days = months_dict[month]
            efm_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/Field_Mill_50HZ_txt/%s/%s%s/'%(year,year,month)
            efm_files = sorted(os.listdir(efm_dir))
            for efm_file in efm_files:
                file_list.append(efm_dir+efm_file)
    return file_list

def build_dirs():
    """
    This method creates the directories used to save the data so schooner doesn't trip.
    """
    years = ['2018','2019','2020','2021','2022','2023','2024']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for year in years:
        for month in months:
            save_dir = '/scratch/bmac87/EFM_50HZ_single_file_pkl/%s%s/'%(year,month)
            if os.path.isdir(save_dir)==False:
                print('making directory: ', save_dir)
                os.makedirs(save_dir)

def check_files(file_list=[]):
    """
    This function checks for missing files from requeued or failed schooner jobs.
    It takes in the list of the original EFM text files.
    """
    check_dir = '/scratch/bmac87/EFM_50HZ_single_file_pkl/'
    missing_files = []
    for f,file in enumerate(file_list):
        if f>=0:
            fcheck = file[-23:-4]+'.pkl'
            if os.path.isfile(check_dir+fcheck)==False:
                print('missing:',fcheck)
                missing_files.append(file)
    pickle.dump(missing_files,open('missing_efm_files.pkl','wb'))
    print(len(missing_files),'# of missing files for the original: ',len(file_list))

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

def build_pkl_file(efm_files=[],nogo=False):
    if nogo==False:#False=run the code, True=don't run the code

        #build the list of labels for the columns in the dataframe
        labels = []
        labels.append('FM_ID')
        labels.append('mins')
        labels.append('seconds')
        labels_50hz = []#subset of the total number of columns

        for num in range(51):
            labels.append(str(num))
            labels_50hz.append(str(num))

        #declare the time interval for each timestep
        hz = .02 #seconds

        for f,file in enumerate(efm_files):
            if f>=0:
                print('loading:',file)
                file_dict = {}
                fname = file[-16:-4]+'.pkl'
                yearmo = fname[0:6]
                target_dir = '/scratch/bmac87/EFM_50HZ_single_file_pkl/%s/'%(yearmo)
                if os.path.isdir(target_dir)==False:
                    os.makedirs(target_dir)
                if os.path.isfile(target_dir+fname)==True:
                    print('this file exists:',fname)
                    continue

                #read in the first row for the date and time
                date = pd.read_csv(file,nrows=0)#DataFrame

                #load the rest of the file
                efm_50 = pd.read_csv(file,
                                    delimiter=',', 
                                    names=labels,
                                    index_col=False,
                                    skiprows=[0,1],
                                    dtype=float)
            
                #get the id's for each EFM in the file
                site_ids = efm_50['FM_ID'].drop_duplicates()
                
                # #loop through the ids
                start = time.time()
                for site_id in site_ids:

                    #get the data for each EFM
                    single_efm = efm_50.loc[(efm_50['FM_ID']==site_id)]
                    
                    #initialize data structures to append the information to
                    ez_idx=0
                    ez = []
                    secs=[]
                    mins=[]
                    date1=[]

                    #loop through each second of data
                    for e in range(single_efm.shape[0]):
                        sec_efm = single_efm.iloc[e]#one second of efm data
                        secs.append(sec_efm['seconds'])
                        ez_temp = sec_efm[labels_50hz].values #1D numpy array (rows=seconds,columns=Hz sample) (technically this is the potential gradient)

                        for y in range(50):#loop through the 50Hz to retrieve the data
                            #if it is the first ez reading in the row, append to the data structure
                            if y==0:
                                ez.append(ez_temp[y])
                            else:
                                #if it is nan, append the previous ez recording, there is no change in ez
                                if np.isnan(ez_temp[y]):
                                    ez.append(ez[ez_idx-1])
                                
                                #otherwise, per the README, add the value to the previous value
                                else: 
                                    ez.append(ez[ez_idx-1]+ez_temp[y])
                                
                                #append the the seconds to with the additional timestep, rounding down to .02
                                secs.append(round(secs[ez_idx-1]+hz,2))
                            
                            #increment the index for tracking and appending
                            ez_idx=ez_idx+1

                            #save the date and minutes as well
                            date1.append(str(date.columns.values))
                            mins.append(sec_efm['mins'])
                        #end loop through 50HZ
                    #end loop through each second

                    # store
                    print('creating dataframe')
                    single_efm2 = pd.DataFrame({'date':date1,'mins':mins,'secs':secs,'ez':ez})
                    single_efm_fixed_time = copy.deepcopy(fix_efm_time(temp_df = single_efm2))
                    del single_efm2
                    file_dict.update({site_id:single_efm_fixed_time})
                    del single_efm_fixed_time, date1, mins, secs, ez 
                # end loop through site ids
                end = time.time()
                print(f"{(end-start)/60:02f}",'minutes to process,',fname)
                print('saving:',target_dir+fname)
                pickle.dump(file_dict,open(target_dir+fname,'wb'))
                del file_dict
        #     end if control
        # end files for loop

def main():
    print('generating the pickle files for each site from the text files')
    slurm_vars = extract_slurm_env()
    efm_text_files, nogo, exp = parse_args()
    start = time.time()
    build_pkl_file(efm_files=efm_text_files,nogo=nogo)
    end = time.time()
    run_time_hours = ((end - start)/60)/60
    print('Experiment:',exp,'took',f"{run_time_hours:02f}",'to run')
    print('END OF MAIN 1b_EFM_single_site_30min_50HZ.py')

def check_main():
    file_list = get_file_list()
    missing_files = check_files()
    
if __name__ == '__main__':
    # main()
    check_main()
