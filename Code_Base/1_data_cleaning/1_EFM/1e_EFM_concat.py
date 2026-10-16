import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import xarray as xr

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
    import argparse
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--year',type=int,default=4,help='SLURM ARRAY')
    args = parser.parse_args()

    if args.year==0:
        year='2018'
    elif args.year==1:
        year='2019'
    elif args.year==2:
        year='2020'
    elif args.year==3:
        year='2021'
    elif args.year==4:
        year='2022'
    elif args.year==5:
        year='2023'
    else:
        year='2024'
    return year

def one_concat_months(year='2022'):
    efm_keys = ['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for m,month in enumerate(months):
        save_dir = '/scratch/bmac87/efm_v2/pkl/%s/month_concat/'%(year)
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)

        start_clock = time.time()
        pd_list = []
        load_dir ='/scratch/bmac87/efm_v2/pkl/%s/%s%s/'%(year,year,month)
        files = os.listdir(load_dir)
        for f,file in enumerate(files):
            if f%100==0:
                print(year,month,f,len(files))
            pd_list.append(pickle.load(open(load_dir+file,'rb')))
        dfs_clean = [df.loc[:, df.columns.get_level_values(0).isin(efm_keys)] for df in pd_list]
        print('concating')
        df_all = pd.concat(dfs_clean)
        print('concat complete')

        print(year,month)
        print('sorting index')
        df_all = df_all.sort_index(ascending=True)
        print('sort complete')
        print('len(df)',len(df_all))
        print('columns')
        print(df_all.columns.get_level_values(0))

        print('saving')
        df_all.to_pickle(open(save_dir+year+month+'.pkl','wb'))
        print('save done')

        end_clock = time.time()
        print('time to run: year, month: %s, %s:'%(year,month), (end_clock-start_clock)/60,' minutes')
        
        del df_all, pd_list, save_dir, start_clock, end_clock,dfs_clean

def two_check_concat_months():
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    years = ['2018','2019','2020','2021','2022','2023','2024']
    for year in years:
        for month in months:
            save_dir='/scratch/bmac87/efm_v2/pkl/%s/month_concat/'%(year)
            fsave = '%s%s.pkl'%(year,month)
            if os.path.isfile(save_dir+fsave)==False:
                print(fsave)

def three_visualize_months():
    stats = ['median','min','max','std']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    years = ['2018','2019','2020','2021','2022','2023','2024']
    for stat in stats:
        for year in years:
            for month in months:
                load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_EFM_concat_annual/%s/'%(stat)
                fload = '%s%s.pkl'%(year,month)
                df = pickle.load(open(load_dir+fload,'rb'))
                fig, axes = plt.subplots(1,1,figsize=(10,10))
                im = axes.pcolormesh(df.values)
                plt.savefig('./monthly_stats/%s%s_%s.png'%(year,month,stat))
                plt.close()

def four_concat_2_annual():
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    years = ['2018','2019','2020','2021','2022','2023','2024']
    for year in years:
        df_list=[]
        print(year)
        load_dir = '/scratch/bmac87/efm_v2/pkl/%s/month_concat/'%(year)
        for month in months:
            df_list.append(pickle.load(open(load_dir+'%s%s.pkl'%(year,month),'rb')))
        df = pd.concat(df_list)
        df = df.sort_index(ascending=True)

        save_dir = '/scratch/bmac87/efm_v2/pkl/annual_concat/'
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        df.to_pickle(open(save_dir+'%s.pkl'%year,'wb'))
        del df, df_list

def five_annual_pkl_2_nc(year='2021'):
    load_dir='/scratch/bmac87/efm_v2/pkl/annual_concat/'
    fname = '%s.pkl'%year

    df = pickle.load(open(load_dir+fname,'rb'))
    df.columns.names = ['station', 'stat']
    df.index.name = 'time'
    ds = xr.Dataset()
    for stat in df.columns.get_level_values('stat').unique():
        sub = df.xs(stat, axis=1, level='stat')
        ds[stat] = xr.DataArray(
            sub.values,
            dims=('time', 'station'),
            coords={'time': df.index, 'station': sub.columns}
        )
    ds['mask'] = ds['mask'].astype('float32')
    ds.to_netcdf('/scratch/bmac87/efm_v2/pkl/annual_concat/%s.nc'%year)
    print(ds)

def main():
    print('happy EFMing')
    five_annual_pkl_2_nc(year='2018')
    five_annual_pkl_2_nc(year='2019')
    five_annual_pkl_2_nc(year='2020')
    five_annual_pkl_2_nc(year='2022')
    five_annual_pkl_2_nc(year='2023')
    five_annual_pkl_2_nc(year='2024')
    
if __name__=='__main__':
    main()
    print('END OF 1e_EFM_concat_annual.py')
    print()
    print()