import os
import shutil
import pickle
import gzip
import argparse
from pathlib import Path
import pickle
import os
import pandas as pd
import numpy as np

MRMS_VARIABLES = ['MergedZdr_00.50',#0
                'MergedZdr_00.75',#1
                'MergedZdr_01.00',#2
                'MergedZdr_01.25',#3
                'MergedZdr_01.50',#4
                'MergedZdr_01.75',#5
                'MergedZdr_02.00',#6
                'MergedZdr_02.25',#7
                'MergedZdr_02.50',#8
                'MergedZdr_02.75',#9
                'MergedZdr_03.00',#10
                'MergedZdr_03.50',#11
                'MergedZdr_04.00',#12
                'MergedZdr_04.50',#13
                'MergedZdr_05.00',#14
                'MergedZdr_05.50',#15
                'MergedZdr_06.00',#16
                'MergedZdr_06.50',#17
                'MergedZdr_07.00',#18
                'MergedZdr_07.50',#19
                'MergedZdr_08.00',#20
                'MergedZdr_08.50',#21
                'MergedZdr_09.00',#22
                'MergedZdr_10.00',#23
                'MergedZdr_11.00',#24
                'MergedZdr_12.00',#25
                'MergedZdr_13.00',#26
                'MergedZdr_14.00',#27
                'MergedZdr_15.00',#28
                'MergedZdr_16.00',#29
                'MergedZdr_17.00',#30
                'MergedZdr_18.00',#31
                'Reflectivity_-10C_00.50',#32
                'Reflectivity_-15C_00.50',#33
                'Reflectivity_-20C_00.50',#34
                'Reflectivity_-5C_00.50',#35
                'Reflectivity_0C_00.50',#36
                'VII_00.50',#37
                'VIL_00.50']#38

def create_parser():
    parser = argparse.ArgumentParser(description='Download', fromfile_prefix_chars='@')
    parser.add_argument('--idx',type=int,default=0,help='0-38')
    args = parser.parse_args()
    return args.idx

def years_dict():
    """
    This code generates a dictionary for the date and hour information. 
    """
    years = ['2018','2019','2020','2021','2022','2023','2024']
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']


    yrs_dict = {}
    for yr in years:#for each year
        mos_dict = {}
        for mo in mos: #for each month
            days = []
            mo_jul = []
            if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
                for t in range(1,32):
                    days.append(f"{t:02}")
            elif mo=='02':
                if yr=='2020':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                elif yr=='2024':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                else:
                    for t in range(1,29):
                        days.append(f"{t:02}")
            else:
                for t in range(1,31):
                    days.append(f"{t:02}")
            mos_dict.update({mo:days})
        yrs_dict.update({yr:mos_dict})
    return yrs_dict

def build_variable_files_lists(var_idx=0):
    

    variable = variables[idx]
    files_list=[]
    date_dict = years_dict()    
    years = ['2021','2022','2023','2024']
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for year in years:
        for mo in mos:
            for day in date_dict[year][mo]:
                data_dir = Path(
                    f"/ourdisk/hpc/ai2es/datasets/MRMS/"
                    f"{year}/{year}{mo}{day}/{variable}/compressed/"
                )

                if not data_dir.exists():
                    continue

                files_list.extend(str(p) for p in data_dir.iterdir() if p.is_file())

    print(variable, len(files_list), flush=True)

    save_dir = "/scratch/bmac87/mrms_parallel_files"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/parallel_files_{variable}.pkl", "wb") as f:
        pickle.dump(files_list, f)

def check_for_remaining_gz():
    """
    This is from chatGPT. 
    """
    directory = "/ourdisk/hpc/ai2es/datasets/MRMS/"
    gz_files = []

    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".gz"):
                gz_files.append(os.path.join(root, f))
    print(f"Total .gz files (recursive): {len(gz_files)}")
    pickle.dump(gz_files,open('./remaining_gz_files.pkl','wb'))

def extract_gz():
    start_idx,end_idx = create_parser()
    print(start_idx,end_idx)
    files = pickle.load(open('remaining_gz_files.pkl','rb'))
    files = files[start_idx:end_idx]
    print(len(files))
    for f,file in enumerate(files):
        if f%100==0:
            print(f,len(files))
        try:
            file_out = file.rstrip('.gz') 
            with gzip.open(file, 'rb') as f_in:
                with open(file_out, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    os.remove(file)
        except Exception as e:
            print("Error occurred while extracting the file:", e)

def convert_fname_to_date(var_idx=0):
    variable = MRMS_VARIABLES[var_idx]
    print(variable)
    fname = "/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/4_MRMS_Data/regridding_files/parallel_files_%s.pkl"%variable
    files_list = pickle.load(open(fname,'rb'))
    files_dict = {'fname_dir':files_list}
    files_df = pd.DataFrame(files_dict)
    files_df['datetime'] = pd.to_datetime(files_df['fname_dir'].str.extract(r'(\d{8}-\d{6})')[0],format='%Y%m%d-%H%M%S')
    files_df['fname'] = files_df['fname_dir'].str.rsplit('/', n=1).str[-1]
    files_df = files_df.sort_values('datetime')
    files_df = files_df.set_index('datetime', drop=False)
    files_df.to_pickle(open('/scratch/bmac87/mrms_parallel_files_df/%s_df.pkl'%variable,'wb'))

def build_variable_files(var_idx=0):
    variable = MRMS_VARIABLES[var_idx]
    files_df = pickle.load(open('/scratch/bmac87/mrms_parallel_files_df/%s_df.pkl'%variable,'rb'))
    for year in ['2021','2022','2023','2024']:
        files_list = []
        print(year,variable)
        samples_df = pickle.load(open('../5_Merge/%s_samples_w_ltg_counts_w_efm_flag.pkl'%(year),'rb'))
        efm_dist_samples = samples_df.loc[samples_df['withinEFM_5nm']==1.0].copy() 
        files_yr_df = files_df.loc[files_df['datetime'].dt.year==int(year)].copy()
        files_yr_df = files_yr_df.sort_index(ascending=True)
        radar_columns = ['R60245', 'R45230', 'R30215', 'R15200']
        radar_df = efm_dist_samples[radar_columns]
        radar_dt = pd.Timedelta(minutes=15) - pd.Timedelta(nanoseconds=1)
        all_times = pd.to_datetime(radar_df.values.ravel())
        needed_files = []
        for t in all_times:
            subset = files_yr_df.loc[t : t + radar_dt]
            needed_files.append(subset)
        needed_files = pd.concat(needed_files).drop_duplicates(subset='fname')
        print(year,variable,'num_files',len(needed_files))
        print(needed_files.head(5))
        needed_files.to_pickle('/scratch/bmac87/mrms_files_to_process/%s_%s.pkl'%(year,variable))

def concat_file_list():
    df_list = []
    # for variable in MRMS_VARIABLES[36]:
    variable = MRMS_VARIABLES[36]
    for year in ['2021','2022','2023','2024']:
        temp_df = pickle.load(open('/scratch/bmac87/%s_%s_target_mask.pkl'%(year,variable),'rb'))
        temp_df['variable'] = variable
        temp_df['year'] = temp_df.index.year
        temp_df['month'] = temp_df.index.month
        temp_df['day'] = temp_df.index.day
        temp_df['hour'] = temp_df.index.hour
        temp_df['minute'] = temp_df.index.minute
        temp_df['save_dir'] = (f'/scratch/bmac87/MRMS_on_HRRR_grid/target_mask/'+temp_df.index.strftime('%Y%m')+'/')
        temp_df['fsave'] = temp_df.index.strftime('%Y%m%d_%H%M.nc')
        df_list.append(temp_df)
    needed_files = pd.concat(df_list).copy()
    needed_files.to_pickle('/scratch/bmac87/MRMS_target_mask_files_needed.pkl')


def make_dirs_for_processing():
    base_dir = '/scratch/bmac87/MRMS_on_HRRR_grid/'
    for variable in MRMS_VARIABLES:
        for year in ['2021','2022','2023','2024']:
            for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
                os.makedirs('%s%s/%s%s/'%(base_dir,variable,year,month))

def make_z_mask_dirs():
    base_dir = '/scratch/bmac87/MRMS_on_HRRR_grid/target_mask/'
    for year in ['2021','2022','2023','2024']:
        for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            os.makedirs('%s/%s%s/'%(base_dir,year,month))

def files_z_mask(var_idx=36):
    files_list_dir = '/scratch/bmac87/mrms_parallel_files_df/'
    z_var = MRMS_VARIABLES[var_idx]
    print(z_var)
    fname = '%s_df.pkl'%z_var
    files_df = pickle.load(open(files_list_dir+fname,'rb'))
    print(files_df.head(10))

    for year in ['2021','2022','2023','2024']:
        samples_df = pickle.load(open('../5_Merge/%s_samples_w_ltg_counts_w_efm_flag.pkl'%(year),'rb'))
        efm_dist_samples = samples_df.loc[samples_df['withinEFM_5nm']==1.0].copy() 
        files_yr_df = files_df.loc[files_df['datetime'].dt.year==int(year)].copy()
        files_yr_df = files_yr_df.sort_index(ascending=True)
        target_columns = ['M00215','M15230','M30245','M45260']
        merlin_df = efm_dist_samples[target_columns]
        merlin_dt = pd.Timedelta(minutes=15) - pd.Timedelta(nanoseconds=1)
        all_times = pd.to_datetime(merlin_df.values.ravel())
        needed_files = []
        for t in all_times:
            subset = files_yr_df.loc[t : t + merlin_dt]
            needed_files.append(subset)
        needed_files = pd.concat(needed_files).drop_duplicates(subset='fname')
        print(year,z_var,'num_files',len(needed_files))
        print(needed_files.head(5))
        needed_files.to_pickle('/scratch/bmac87/%s_%s_target_mask.pkl'%(year,z_var))

def check_files_processed():

    #chat gpt recommendation
    # files_df = pickle.load(open('/scratch/bmac87/MRMS_needed_files.pkl','rb'))
    # paths = files_df['save_dir']+files_df['fsave']
    # paths = paths.values
    # missing = [p for p in paths if not os.path.exists(p[0:-2]+'pkl')]
    # if len(missing)>0:
    #     print(len(missing),'# of missing MRMS files')
    #     pickle.dump(missing,open('MRMS2HRRR_missing.pkl','wb'))
    #     print('missing files list saved successfully')
    # del files_df, paths, missing

    files_df = pickle.load(open('/scratch/bmac87/MRMS_target_mask_files_needed.pkl','rb'))
    paths = files_df['save_dir']+files_df['fsave']
    paths = paths.values
    missing = [p for p in paths if not os.path.exists(p[0:-2]+'pkl')]
    print(missing)
    if len(missing)>0:
        print(len(missing),'# of missing target MRMS files')
        pickle.dump(missing,open('MRMS2HRRR_target_missing.pkl','wb'))
        print('missing files list saved successfully')
    del files_df, paths, missing

def main():
    # for idx in range(39):
    #     build_variable_files(var_idx=idx)
    # make_dirs_for_processing()
    # files_z_mask()
    # concat_file_list()
    # make_z_mask_dirs()
    # check_files_processed()

    # missing = pickle.load(open('./MRMS2HRRR_missing.pkl','rb'))
    # print(missing)
    # files_df = pickle.load(open('/scratch/bmac87/MRMS_needed_files.pkl','rb'))
    # files_df['full_save_path'] = files_df['save_dir']+files_df['fsave']
    # files_df = files_df.loc[files_df['full_save_path'].isin(missing)]
    # print(files_df)
    # files_df.to_pickle('missing_files_df.pkl')
    del files_df

    
if __name__=='__main__':

    files_df = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/MRMS_needed_files.pkl','rb'))
    num_files = len(files_df)
    start_idx = 0
    d_idx = int(np.floor(num_files/950))
    print(d_idx)
    print(num_files)
    for exp in range(951):
        if exp==950:
            end_idx = num_files
        else:
            end_idx = start_idx+d_idx
        print(exp,start_idx,end_idx,num_files)
        start_idx = end_idx
        batch_df = files_df.iloc[start_idx:end_idx].copy()
        batch_df.to_pickle(open('/scratch/bmac87/exp_list/exp_%s.pkl'%exp,'wb'))
        

