import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import glob
import argparse
from helper import *
import time
import cartopy.geodesic as gd

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
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--exp', type=int, default=54, help='Integer between 1-84 to drive the experiments by month and year')
    args = vars(parser.parse_args())
    exp = args['exp']#int
    print('exp: ',exp)
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
    print('year:',year,'month:',month)
    return year, month




def check_lightning_data_files():
    years, months, hours, half_hours = time_stuff()
    yrs_dict = years_dict()

    missing_cc_files = []
    missing_cg_files = []

    for year in years:
        cg_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Sources_dat/MERLIN_CG/%s/'%(year)
        #CG file: KSCCG20181231.dat YYYYMMDD.dat

        cc_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Sources_dat/MERLIN_CC/%s/'%(year)
        #CC file: KSCCC20180924.dat YYYYMMDD.dat
        months_dict = yrs_dict[year]
        for month in months:
            days = months_dict[month]
            for day in days:
                CG_file = 'KSCCG%s%s%s.dat'%(year,month,day)
                CC_file = 'KSCCC%s%s%s.dat'%(year,month,day)

                if os.path.isfile(cg_dir+CG_file)==False:
                    missing_cg_files.append(CG_file)
                
                if os.path.isfile(cc_dir+CC_file)==False:
                    missing_cc_files.append(CC_file)

    for file in missing_cg_files:
        print(file)
    print(len(missing_cg_files),' # of missing CG files')

def convert_lat_lon(merlin_df=pd.DataFrame()):
    """
    This function converts the degrees minutes and seconds format of the lat/lon
    into decimal lat/lons. 
    """
    temp_merlin = merlin_df
    temp_merlin[['Lat_Deg','Lat_Min','Lat_Seconds']] =  temp_merlin['Lat'].str.split(':',expand=True)
    temp_merlin['Lat_Deg'] = temp_merlin['Lat_Deg'].astype('float')
    temp_merlin['Lat_Min'] = temp_merlin['Lat_Min'].astype('float')/60. 
    temp_merlin['Lat_Seconds'] = temp_merlin['Lat_Seconds'].astype('float')/3600.
    temp_merlin['Lat_Decimal'] = temp_merlin['Lat_Deg']+temp_merlin['Lat_Min']+temp_merlin['Lat_Seconds']
    
    temp_merlin[['Lon_Deg','Lon_Min','Lon_Seconds']] = temp_merlin['Lon'].str.split(':',expand=True)
    temp_merlin['Lon_Deg'] = temp_merlin['Lon_Deg'].astype('float')
    temp_merlin['Lon_Min'] = temp_merlin['Lon_Min'].astype('float')/60.
    temp_merlin['Lon_Seconds'] = temp_merlin['Lon_Min'].astype('float')/3600. 
    temp_merlin['Lon_Decimal'] = temp_merlin['Lon_Deg']-temp_merlin['Lon_Min']-temp_merlin['Lon_Seconds']
    return temp_merlin

def calc_distances_to_efm_sites(merlin_df=pd.DataFrame(),efm_df=pd.DataFrame()):
    """
    This function calculates the distance between each merlin data point, and the EFM locations.
    """

    print('calculating the distances to the EFM sites')
    site_names = efm_df['SiteName'].values
    site_lats = efm_df['Latitude'].values
    site_lons = efm_df['Longitude'].values

    merlin_lats = merlin_df['Lat_Decimal'].values
    merlin_lons = merlin_df['Lon_Decimal'].values
    sensors = merlin_df['Sensors'].values
    k = gd.Geodesic()

    # Define the two points as NumPy arrays [longitude, latitude]
    # coord1 = np.array([77.343750, 22.593726])
    # coord2 = np.array([86.945801, 23.684774])
    # distance_meters = k.inverse(coord1, coord2)[0, 0]
    num_sensors = []
    for sn in range(len(site_names)):
        ltg_distances = []
        for ml in range(len(merlin_lons)):
            if sn==0:
                num_sensors.append(len(sensors[ml].split(',')))
            if ml%1000==0:
                print(site_names[sn],ml,len(merlin_lons))
            merlin_coord = np.array([merlin_lons[ml],merlin_lats[ml]])
            efm_coord = np.array([site_lons[sn],site_lats[sn]])
            distance_meters = k.inverse(merlin_coord, efm_coord)[0, 0]
            ltg_distances.append(distance_meters)
            del merlin_coord, efm_coord, distance_meters

        merlin_df[site_names[sn]+'_Dist_meters']=ltg_distances
        del ltg_distances
    merlin_df['Num_Sensors'] = num_sensors
    return merlin_df

def dirty_lines(filename):
    """
    This function returns the number of lines with data on them. 
    To be compared with the final line count 
    in the merlin_df dataframe.
    """
    lines=0
    with open(filename) as f:
        for line in f:
            if len(line)!=1:#if the line is not blank
                lines+=1
    return lines

def read_in_CC_dat_files(year='2022',month='06'):
    """
    This function reads in the original MERLIN CC or CG data files
    and returns a pandas dataframe with the duplicated dropped.
    The dataframe for all years is also saved.

    The CC data was cleaned, where empty files were deleted.
    """
    print('reading in the lightning data')
    data_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Sources_dat/MERLIN_CC/%s/'%(year) 
    glob_call = data_dir+'KSCCC%s%s*.dat'%(year,month)
    print('glob_call:',glob_call)
    files = sorted(glob.glob(glob_call))
    print(len(files),' # of files to open')
    efm_locations_df = pd.read_excel('../1_EFM/EFM_Locations.xlsx')

    for f,file in enumerate(files):
        start_time = time.time()
        save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2a_MERLIN_CC_Sources/%s%s/'%(year,month)
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        fsave = file[-17:-4]+'.pkl'
        print(f,fsave)
        if os.path.isfile(save_dir+fsave)==True:
            continue

        try:
            pre_qc_lines = dirty_lines(filename=file)
            print(pre_qc_lines,' lines in text file before qc')

            """
            These lines of code drop the CC QC flags with Ts in them, since it generates a bad line, with an extra marker. 
            """
            merlin_df = pd.read_csv(file,sep='\s+',header=None,low_memory=False,on_bad_lines='skip')
            merlin_df.rename({0:'Date',1:'Time',2:'Lat',3:'Lon',4:'SigStrength_kAmps',5:'Real',6:'SemiMajor_km',7:'SemiMinor_km',8:'Ellipse_Angle_deg',9:'Sensors'},inplace=True,axis=1)
            print('data read in successfully')
            
            merlin_df.dropna(inplace=True,axis=0)
            print('nans dropped')

            merlin_df = convert_lat_lon(merlin_df)
            print('latitude/longitude converted successfully')

            merlin_df['Date_Time'] = pd.to_datetime(merlin_df['Date']+' '+merlin_df['Time'],errors='coerce')
            merlin_df = merlin_df[merlin_df['Date_Time'].isnull()==False]#lose 1000 sources out of 5 million
            merlin_df = merlin_df.set_index('Date_Time')
            merlin_df = merlin_df.sort_index(ascending=True)
            print('date/time converted to pandas index successfully')

            merlin_df = merlin_df[['Lat_Decimal','Lon_Decimal','SemiMajor_km','SemiMinor_km','Ellipse_Angle_deg','Sensors']]
            merlin_df = calc_distances_to_efm_sites(merlin_df=merlin_df,efm_df=efm_locations_df)
            print('efm distances calculated successfully')
            print('post_qc_lines ',len(merlin_df))

            print('saving the df_dict')
            df_dict = {'pre_qc_lines':pre_qc_lines,'post_qc_lines':len(merlin_df),'merlin_df':merlin_df}
            pickle.dump(df_dict,open(save_dir+fsave,'wb'))
            print('df_dict saved successfully')
            del merlin_df
        except Exception as e:
            print(e)
            print('error occurred, probably an empty file:',file)

        end_time = time.time()
        run_time_seconds = (end_time-start_time)
        print('data processing time for CC %s %s: %s seconds'%(year,month,f"{run_time_seconds:.08f}"))
        print()
        print()

def read_in_CG_dat_files(year='2022',month='06'):
    """
    This method processes the CG data. While the files are similar in structure,
    the files are slightly different. The original files are used, and the empty ones
    are included in this script. 

    When writing the manuscript, be sure to state that only files with data in them 
    were used. 

    There is an additional qc flag here, where if the real label is r, then the data 
    kept.
    """

    print('reading in the lightning data')
    data_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Sources_dat/MERLIN_CG/%s/'%(year) 
    glob_call = data_dir+'KSCCG%s%s*.dat'%(year,month)
    print('glob_call:',glob_call)
    files = sorted(glob.glob(glob_call))
    efm_locations_df = pd.read_excel('../1_EFM/EFM_Locations.xlsx')
    for f,file in enumerate(files):
        print(file)
        start_time = time.time()
        save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2a_MERLIN_CG_Sources/%s%s/'%(year,month)
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        fsave = file[-17:-4]+'.pkl'
        print(f,fsave)

        try:
            pre_qc_lines = dirty_lines(filename=file)
            print(pre_qc_lines,' lines in text file before qc')

            """
            These lines of code drop the CC QC flags with Ts in them, since it generates a bad line, with an extra marker. 
            """
            merlin_df = pd.read_csv(file,sep='\s+',header=None,low_memory=False,on_bad_lines='skip')
            merlin_df.rename({0:'Date',1:'Time',2:'Lat',3:'Lon',4:'SigStrength_kAmps',5:'Real_int',6:'Real_label',7:'SemiMajor_km',8:'SemiMinor_km',9:'Ellipse_Angle_deg',10:'Rand_01',11:'Sensors'},inplace=True,axis=1)
            print(merlin_df.head(10))
            print('data read in successfully')
            merlin_df = merlin_df.loc[merlin_df['Real_label']=='r']
            merlin_df.dropna(inplace=True,axis=0)
            print('nans dropped')

            merlin_df = convert_lat_lon(merlin_df)
            print('latitude/longitude converted successfully')

            merlin_df['Date_Time'] = pd.to_datetime(merlin_df['Date']+' '+merlin_df['Time'],errors='coerce')
            merlin_df = merlin_df[merlin_df['Date_Time'].isnull()==False]#lose 1000 sources out of 5 million
            merlin_df = merlin_df.set_index('Date_Time')
            merlin_df = merlin_df.sort_index(ascending=True)
            print('date/time converted to pandas index successfully')
            try:
                merlin_df = merlin_df[['Lat_Decimal','Lon_Decimal','SigStrength_kAmps','SemiMajor_km','SemiMinor_km','Ellipse_Angle_deg','Sensors']]
            except Exception as e:
                print(e)
                print('re-assigning the Rand_01 columns to sensors')
                merlin_df = merlin_df[['Lat_Decimal','Lon_Decimal','SigStrength_kAmps','SemiMajor_km','SemiMinor_km','Ellipse_Angle_deg','Rand_01']]
                merlin_df = merlin_df.rename(columns={'Rand_01':'Sensors'})
                print(merlin_df.columns)
                print('re-assignment successful')
            merlin_df = calc_distances_to_efm_sites(merlin_df=merlin_df,efm_df=efm_locations_df)
            print('efm distances calculated successfully')
            print('post_qc_lines ',len(merlin_df))

            print('saving the df_dict')
            df_dict = {'pre_qc_lines':pre_qc_lines,'post_qc_lines':len(merlin_df),'merlin_df':merlin_df}
            pickle.dump(df_dict,open(save_dir+fsave,'wb'))
            print('df_dict saved successfully')
            del merlin_df
        except Exception as e:
            print(e)
            print('error occurred, probably an empty file:',file)
        end_time = time.time()
        run_time_seconds = (end_time-start_time)
        print('data processing time for CC %s %s: %s seconds'%(year,month,f"{run_time_seconds:.08f}"))

if __name__=='__main__':
    slurm_vars = extract_slurm_env()
    year,month = parse_args()
    # read_in_CC_dat_files(month=month,year=year)
    # read_in_CG_dat_files(month=month,year=year)
    print('END OF 2a_MERLIN_readin2df.py')