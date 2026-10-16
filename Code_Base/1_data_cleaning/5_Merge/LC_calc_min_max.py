import xarray as xr
from LC_parser import *
import numpy as np
import os
import pickle

EFM_MASK_VALUE=-999999.0
ZDR_MASK_VALUE=-99
    
def calc_overall_min_max(args):

    overall_dict = {}
    nc_dir ='/scratch/bmac87/'
    fname = 'LC_Forecast_total_ds.nc'
    ds = xr.open_dataset(nc_dir+fname,engine='netcdf4')
    
    VI_keys = args.VI_keys
    Z_keys = args.Z_keys
    Zdr_keys = args.Zdr_keys
    hrrr_features = args.hrrr_features
    glm_keys = args.GLM_keys

    for k,key in enumerate(glm_keys+VI_keys+Z_keys+Zdr_keys+hrrr_features):
        print(key)
        da = ds[key]
        data_np = da.values
        print(data_np.shape)
        if key in Z_keys:
            data_np[data_np<0.0] = np.nan
        elif key in Zdr_keys:
            data_np[data_np==ZDR_MASK_VALUE]=np.nan
        elif key in VI_keys:
            data_np[data_np<0.0] = np.nan
        else:
            pass

        data_mean = np.nanmean(data_np)
        print('data_mean',data_mean)
        data_std = np.nanstd(data_np)
        print('data_std',data_mean)
        data_min = np.nanmin(data_np)
        print('data_min',data_min)
        data_max = np.nanmax(data_np)
        print('data_max',data_max)
        p95,p99,p995 = np.nanpercentile(data_np.flatten(), [95, 99, 99.5])
        print(p95,p99,p995)
        temp_dict = {'min':data_min,
                    'max':data_max,
                    'mean':data_mean,
                    'std':data_std,
                    'p95':p95,
                    'p99':p99,
                    'p995':p995}
        overall_dict.update({key:temp_dict})
        del da, data_np,temp_dict, data_min, data_max, data_std, data_mean

    pickle.dump(overall_dict,open('./forecast_min_maxes.pkl','wb'))
    pickle.dump(overall_dict,open('../../2_model_training/forecast_min_maxes.pkl','wb'))
    pickle.dump(overall_dict,open('../../3_model_analysis/forecast_min_maxes.pkl','wb'))

def calc_efm_min_max_mean_std(args):
    nc_dir ='/scratch/bmac87/'
    fname = 'LC_Forecast_total_ds.nc'
    ds = xr.open_dataset(nc_dir+fname,engine='netcdf4')

    efm_keys = args.efm_keys
    efm_stats = args.efm_stats
    fm_dict={}

    for k,key in enumerate(efm_keys):
        print(key)
        da = ds[key]
        data_np = da.values
        print(data_np.shape)
        data_np[data_np==EFM_MASK_VALUE] = np.nan
        ftr_dict={}
        for f,ftr in enumerate(efm_stats):
            data_mean = np.nanmean(data_np[:,0:120,f])
            data_std = np.nanstd(data_np[:,0:120,f])
            data_min = np.nanmin(data_np[:,0:120,f])
            data_max = np.nanmax(data_np[:,0:120,f])

            p95,p99,p995 = np.nanpercentile(np.abs(data_np[:,0:120,f].flatten()),[95, 99, 99.5])

            eps=1e-6
            log_mean = np.nanmean(np.log(data_np[:,0:120,f]+eps))
            log_std = np.nanstd(np.log(data_np[:,0:120,f]+eps))
            log_min = np.nanmin(np.log(data_np[:,0:120,f]+eps))
            log_max = np.nanmax(np.log(data_np[:,0:120,f]+eps))

            p95_log,p99_log,p995_log = np.nanpercentile(np.log(data_np[:,0:120,f]+eps).flatten(),[95, 99, 99.5])
            temp_dict = {'min':data_min,'max':data_max,'mean':data_mean,'std':data_std,
                            'p95':p95,'p99':p99,'p995':p995,
                            'log_min':log_min,'log_max':log_max,'log_mean':log_mean,'log_std':log_std,
                            'p95_log':p95_log,'p99_log':p99_log,'p995_log':p995_log}

            ftr_dict.update({ftr:temp_dict})
        print(ftr_dict)
        fm_dict.update({key:ftr_dict})
    print(fm_dict)
    pickle.dump(fm_dict,open('./efm_forecast_min_max.pkl','wb'))
    pickle.dump(fm_dict,open('../../2_model_training/efm_forecast_min_max.pkl','wb'))
    pickle.dump(fm_dict,open('../../3_model_analysis/efm_forecast_min_max.pkl','wb'))

def main(args):
    # calc_efm_min_max_mean_std(args=args)
    calc_overall_min_max(args=args)

if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args=args)