import os
import shutil
import glob
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--job_id',type=int,default=1)
    parser.add_argument('--year',type=str,default='2018')
    args = parser.parse_args()
    return f"{args.job_id:02}", args.year

def build_features(year='2022',month='06'):
    
    hrrr_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/0_HRRR_Data/0_HRRR_downselect/f00/%s%s/'%(year,month)
    files = sorted(os.listdir(hrrr_dir))
    level = 50
    levels = [50]
    l=0
    while level<1000:
        level = levels[l]
        level = level+25
        levels.append(level)
        l=l+1

    idx250 = 8 
    idx500 = 18
    idx700 = 26
    
    idxs1000_850 = range(32,39)
    idxs1000_700 = range(26,39)
    idxs1000_500 = range(18,39)
    idxs1000_250 = range(8,39)

    for f,file in enumerate(files):
        
        if f>=0:
            print(f,file)
            hrrr_dict = pickle.load(open(hrrr_dir+file,'rb'))
            #accumulate the mixing ratios
            graupel = np.expand_dims(np.sum(hrrr_dict['graupel_q_3d'],axis=-1),axis=-1)
            cloud = np.expand_dims(np.sum(hrrr_dict['cloud_q_3d'],axis=-1),axis=-1)
            rain = np.expand_dims(np.sum(hrrr_dict['rain_q_3d'],axis=-1),axis=-1)
            snow = np.expand_dims(np.sum(hrrr_dict['snow_q_3d'],axis=-1),axis=-1)
            w_max = np.expand_dims(np.max(hrrr_dict['w_3d'],axis=-1),axis=-1)

            u_winds = np.expand_dims(hrrr_dict['u_10m'],axis=-1)
            v_winds = np.expand_dims(hrrr_dict['v_10m'],axis=-1)
            temp = np.expand_dims(hrrr_dict['temp_3d'][:,:,38],axis=-1)
            vort_500 = np.expand_dims(hrrr_dict['vort_3d'][:,:,idx500],axis=-1)

            z = np.expand_dims(hrrr_dict['reflectivity'],axis=-1)
            z10 = np.expand_dims(hrrr_dict['refl_neg10C'],axis=-1)
            cape = np.expand_dims(hrrr_dict['cape'],axis=-1)

            phi0 = np.expand_dims(hrrr_dict['gph_zero'],axis=-1)
            phi20 = np.expand_dims(hrrr_dict['gph_neg20C'],axis=-1)
            land_sea = np.expand_dims(hrrr_dict['land_sea'],axis=-1)
            mslp = np.expand_dims(hrrr_dict['mslp'],axis=-1)

            hrrr_features = np.concatenate([graupel,
                                        cloud,
                                        rain,
                                        snow,
                                        u_winds,
                                        v_winds,
                                        w_max,
                                        temp,
                                        vort_500,
                                        z,
                                        z10,
                                        cape,
                                        phi0,
                                        phi20,
                                        land_sea,
                                        mslp],axis=-1)
            feature_list = ['graupel','cloud','rain','snow','u10m','v10m','wmax','temp1000mb','vort500','Z','Zneg10','cape','phi0','phi20','land_sea','mslp']
            save_dict = {'x':hrrr_features,'valid_time':hrrr_dict['valid_time'],'feature_list':feature_list}
            year = file[5:9]
            month = file[10:12]
            day = file[13:15]
            hour = file[16:18]
            save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/0_HRRR_Data/2_16_HRRR_features/%s%s/'%(year,month)
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            fsave = '%s%s%s%s.pkl'%(year,month,day,hour)
            pickle.dump(save_dict,open(save_dir+fsave,'wb'))
            del hrrr_dict, graupel, cloud, rain, snow, w_max, u_winds, v_winds, temp, vort_500, z, z10, cape, phi0, phi20, land_sea, mslp, save_dict, year, month, day, hour, feature_list

def main():
    print('generating the hrrr features')
    month, year = parse_args()
    build_features(year=year,month=month)

if __name__=='__main__':
    main()