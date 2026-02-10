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
    parser.add_argument('--stat',type=str,default='median')
    parser.add_argument('--year',type=int,default=5,help='SLURM ARRAY')
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
    return year, args.stat

def one_concat_months():
    print('1e_EFM_concat_annual.py')
    slurm_vars = extract_slurm_env()
    year,stat = parse_args()
    print(year,stat,' YEAR and STAT')

    months = months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for m,month in enumerate(months):
        start_time = time.time()
        pd_list = []
        load_dir ='/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1d_EFM_resample_qc/%s/%s%s/'%(stat,year,month)
        files = os.listdir(load_dir)
        for file in files:
            pd_list.append(pickle.load(open(load_dir+file,'rb')))
        save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_EFM_concat_annual/%s/'%stat
        df = pd.concat(pd_list)
        df.to_pickle(open(save_dir+year+month+'.pkl','wb'))
        print('len(df)',len(df))
        end_time = time.time()
        print('time to run: %s, year, month: %s, %s:'%(stat,year,month), (end_time-start_time)/60,' minutes')
        del df, pd_list

def two_check_concat_months():
    stats = ['median','max','min','std']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    years = ['2018','2019','2020','2021','2022','2023','2024']
    for stat in stats:
        save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_EFM_concat_annual/%s/'%stat
        for year in years:
            for month in months:
                fsave = '%s%s.pkl'%(year,month)
                if os.path.isfile(save_dir+fsave)==False:
                    print(stat,fsave)

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

    stats = ['median','min','max','std']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    years = ['2018','2019','2020','2021','2022','2023','2024']

    for stat in stats:
        print(stat)
        for year in years:
            df_list=[]
            print(year)
            for month in months:
                print(stat,year,month)
                load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_concat_monthly/%s/'%(stat)
                df_list.append(pickle.load(open(load_dir+'%s%s.pkl'%(year,month),'rb')))
            df = pd.concat(df_list)
            print(df.head(10))
            save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_concat_annual/%s/'%stat
            df.to_pickle(open(save_dir+'%s.pkl'%year,'wb'))
            del df, df_list

def five_annual_pkl_2_nc():
    stats = ['median','min','max','std']
    years = ['2018','2019','2020','2021','2022','2023','2024']

    for year in years:
        for s,stat in enumerate(stats):
            print(year,stat)
            load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_concat_annual/%s/'%stat
            fname = '%s.pkl'%year
            df = pickle.load(open(load_dir+fname,'rb'))
            ds = df.to_xarray()
            ds.to_netcdf(load_dir+'%s.nc'%(year),engine='netcdf4')
            del ds, df

def main():
    print('happy EFMing')
    
if __name__=='__main__':
    main()
    print('END OF 1e_EFM_concat_annual.py')
    print()
    print()