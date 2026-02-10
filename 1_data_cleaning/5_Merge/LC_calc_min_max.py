import xarray as xr
from LC_parser import *
import numpy as np
import os
import pickle

def calc_min_max(args,
                data_dir='/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/6_final_xr/',
                year='2021'):

    print('calculating the min max')
    hrrr_features = args.hrrr_features
    efm_keys = args.efm_ts_keys
    efm_stats = args.efm_stats
    glm_keys = args.GLM_keys
    Z_keys = args.Z_keys
    Zdr_keys = args.Zdr_keys
    VI_keys = args.VI_keys
    radar_keys = Z_keys+Zdr_keys+VI_keys

    ds = xr.open_dataset(data_dir+'LC_linear_zdr_%s.nc'%year,engine='netcdf4')
    
    min_max_dict = {}
    for fm in efm_keys:
        efm_values = ds[fm].values
        efm_values[efm_values==-20000] = np.nan#(batch, time, ts, stats)
        data_min=np.nanmin(efm_values,axis=(0,1,2))
        data_max=np.nanmax(efm_values,axis=(0,1,2))

        for s,stat in enumerate(efm_stats):
            min_max_dict.update({'%s_%s_max'%(fm,stat):data_max[s]})
            min_max_dict.update({'%s_%s_min'%(fm,stat):data_min[s]})
    del data_min, data_max, efm_values
    
    hrrr_data = ds['hrrr_data'].values
    for f,ftr in enumerate(hrrr_features):
        temp_data = hrrr_data[:,:,:,:,f]
        data_max = np.nanmax(temp_data,axis=(0,1,2,3))
        data_min = np.nanmin(temp_data,axis=(0,1,2,3))

        min_max_dict.update({'%s_min'%ftr:data_min})
        min_max_dict.update({'%s_max'%ftr:data_max})
    del data_max, data_min, hrrr_data, temp_data

    for g,glm_key in enumerate(glm_keys):
        temp_data = ds[glm_key].values
        temp_data[temp_data<=0.0]=np.nan
        data_max = np.nanmax(temp_data,axis=(0,1,2,3))
        data_min = np.nanmin(temp_data,axis=(0,1,2,3))

        min_max_dict.update({'%s_min'%glm_key:data_min})
        min_max_dict.update({'%s_max'%glm_key:data_max})
    del data_max, data_min, temp_data

    for r,radar_key in enumerate(radar_keys):
        temp_data = ds[radar_key].values
        temp_data[temp_data<=0.0]=np.nan
        data_max = np.nanmax(temp_data,axis=(0,1,2,3))
        data_min = np.nanmin(temp_data,axis=(0,1,2,3))

        min_max_dict.update({'%s_min'%radar_key:data_min})
        min_max_dict.update({'%s_max'%radar_key:data_max})
    del data_max, data_min, temp_data

    for key in min_max_dict:
        print(key,min_max_dict[key])
    return min_max_dict

def main():
    print('main min_max')
    parser = create_parser()
    args = parser.parse_args()
    years = ['2021','2022','2023','2024']
    dict_list = []

    for year in years:
        yr_dict = calc_min_max(args=args,year=year)
        dict_list.append(yr_dict)
    
    final_dict={}
    for key in yr_dict:
        if 'min' in key:
            print(key)
            final_dict.update({key:np.min([dict_list[0][key],
                                            dict_list[1][key],
                                            dict_list[2][key],
                                            dict_list[3][key]])})
        if 'max' in key:
            print(key)
            final_dict.update({key:np.max([dict_list[0][key],
                                            dict_list[1][key],
                                            dict_list[2][key],
                                            dict_list[3][key]])})

    for key in final_dict:
        print(key,final_dict[key])
    
    pickle.dump(final_dict,open('forecast_min_maxes.pkl','wb'))
    pickle.dump(final_dict,open('../../2_model_training/forecast_min_maxes.pkl','wb'))
    pickle.dump(final_dict,open('../../3_model_analysis/forecast_min_maxes.pkl','wb'))

if __name__=='__main__':
    main()