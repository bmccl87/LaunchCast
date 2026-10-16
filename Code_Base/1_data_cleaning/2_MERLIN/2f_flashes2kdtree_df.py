import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
import pygrib
import cartopy.crs as ccrs
import pickle
import argparse
import glob

def parse_args():

    print('parsing args')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',type=int,default=67,help='1-84 int for year month')
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

def get_group_indices_kdtree(df):
    print(df)
    print('trying the kd tree')
    glm_lats = df['Lat_Decimal'].values
    glm_lons = df['Lon_Decimal'].values

    # hrrr indices for 66x66 downselection - faster processing for lightning location
    LC_hrrr_indexes = pickle.load(open('../LC_hrrr_indexes.pkl','rb'))
    x1 = LC_hrrr_indexes['x'].start-1
    x2 = LC_hrrr_indexes['x'].stop+1
    y1 = LC_hrrr_indexes['y'].start-1
    y2 = LC_hrrr_indexes['y'].stop+1
    
    #get the lat lon grid
    og_hrrr = pickle.load(open('../0_HRRR/hrrr_og_grid.pkl','rb'))
    hrrr_lat = og_hrrr['lat']
    hrrr_lon = og_hrrr['lon']
    
    hrrr_lat = hrrr_lat[y1:y2,x1:x2]
    hrrr_lon = hrrr_lon[y1:y2,x1:x2]

    # Flatten HRRR grid
    points_hrrr = np.column_stack((hrrr_lat.ravel(), hrrr_lon.ravel()))

    # Build tree
    tree = cKDTree(points_hrrr)

    # Query nearest HRRR point for each GLM flash
    _, idx = tree.query(np.column_stack((glm_lats, glm_lons)), k=1)

    # Map indices back to HRRR grid
    hrrr_shape = hrrr_lon.shape
    glm_i, glm_j = np.unravel_index(idx, hrrr_shape)
    df['r_idx_kdtree'] = glm_i
    df['c_idx_kdtree'] = glm_j
    return df

def assign_r_c_2flash():

    flash_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/'
    cc_file = 'MERLIN_CC_all_flashes.pkl'
    cg_file = 'MERLIN_CG_all_flashes.pkl'
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2g_MERLIN_df_rc/'
    year,month = parse_args()
    
    print('running the kdtree for glm groups, year, month:',year,month)
    cc_df = pickle.load(open(flash_dir+cc_file,'rb'))
    cg_df = pickle.load(open(flash_dir+cg_file,'rb'))

    cc_yr_df = cc_df[cc_df.index.year==int(year)]
    cc_mo_df = cc_yr_df[cc_yr_df.index.month==int(month)]
    cc_mo_df2 = get_group_indices_kdtree(cc_mo_df)
    cc_fsave = 'cc_%s%s.pkl'%(year,month)
    pickle.dump(cc_mo_df2,open(save_dir+cc_fsave,'wb'))
    print(cc_mo_df2)

    cg_yr_df = cg_df[cg_df.index.year==int(year)]
    cg_mo_df = cg_yr_df[cg_yr_df.index.month==int(month)]
    cg_mo_df2 = get_group_indices_kdtree(cg_mo_df)
    cg_fsave = 'cg_%s%s.pkl'%(year,month)
    print(cg_mo_df2)
    pickle.dump(cg_mo_df2,open(save_dir+cg_fsave,'wb'))

def concat_r_c_files():

    load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2g_MERLIN_df_rc/'
    glob_call = load_dir+'cc_*.pkl'
    cc_files = glob.glob(glob_call)

    glob_call = load_dir+'cg_*.pkl'
    cg_files = glob.glob(glob_call)

    print(cc_files)
    print(cg_files)

    cc_pd_list = []
    for file in cc_files:
        cc_pd_list.append(pickle.load(open(file,'rb')))
    
    cc_rc_pd = pd.concat(cc_pd_list)
    cc_rc_pd = cc_rc_pd.sort_index(ascending=True)
    pickle.dump(cc_rc_pd,open('cc_rc_pd.pkl','wb'))

    cg_pd_list = []
    for file in cg_files:
        cg_pd_list.append(pickle.load(open(file,'rb')))
    cg_rc_pd = pd.concat(cg_pd_list)
    cg_rc_pd = cg_rc_pd.sort_index(ascending=True)
    pickle.dump(cg_rc_pd,open('cg_rc_pd.pkl','wb'))

def main():
    assign_r_c_2flash()
    # concat_r_c_files()

if __name__=='__main__':
    main()