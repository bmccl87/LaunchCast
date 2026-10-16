import sys
print(sys.executable)

import xarray as xr
import numpy as np
import os
import shutil
import pygrib
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xesmf as xe
import argparse
import time
import pandas as pd

def create_parser():
    parser = argparse.ArgumentParser(description='Download', fromfile_prefix_chars='@')
    parser.add_argument('--exp', type=int, default=0,help='0-998')
    args = parser.parse_args()
    exp = args.exp
    start_idx = exp*1212
    end_idx = start_idx+1212
    return start_idx, end_idx, exp

def save_mrms_2_hrrr_grids():

    #set the box to get the groups from
    west_lon_box = -87.0+360
    east_lon_box = -77.0+360
    north_lat_box = 31.0
    south_lat_box = 24.0

    #load the mrms grid
    mrms_grid = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/4_MRMS_Data/regridding_files/mrms_grid_whole.pkl','rb'))
    pickle.dump(mrms_grid,open('./mrms_grid_whole.pkl','wb'))

    mrms_lat = mrms_grid['lat']
    mrms_lon = mrms_grid['lon']

    print(mrms_lat.shape)
    print(mrms_lon.shape)
    print(mrms_lon.max())
    print(mrms_lon.min())
    
    # Find lat/lon within bounds
    lat_mask = (mrms_lat >= south_lat_box) & (mrms_lat <= north_lat_box)
    lon_mask = (mrms_lon >= west_lon_box) & (mrms_lon <= east_lon_box)
    valid_points = np.where(lat_mask & lon_mask)
    
    y_min, y_max = valid_points[0].min(), valid_points[0].max()
    x_min, x_max = valid_points[1].min(), valid_points[1].max()
    idx_dict = {'y_min':y_min,'y_max':y_max,'x_min':x_min,'x_max':x_max}
    pickle.dump(idx_dict,open('idx_dict.pkl','wb'))

    mrms_cropped_lat = mrms_lat[y_min:y_max, x_min:x_max]
    mrms_cropped_lon = mrms_lon[y_min:y_max, x_min:x_max]
    mrms_cropped_dict = {'cropped_lat':mrms_cropped_lat,'cropped_lon':mrms_cropped_lon}
    pickle.dump(mrms_cropped_dict,open('mrms_cropped_grid.pkl','wb'))

    # hrrr indices for 64x64 downselection - faster processing for lightning location
    #version 3 idxs

    hrrr_slices_dict = pickle.load(open('../LC_hrrr_indexes.pkl','rb'))
    xi = hrrr_slices_dict['x']
    yi = hrrr_slices_dict['y']

    num_pixels_outside_LC = 128
    x_hrrr_idxs_regrid = [xi.start-num_pixels_outside_LC,xi.stop+num_pixels_outside_LC]
    y_hrrr_idxs_regrid = [yi.start-num_pixels_outside_LC,yi.stop+num_pixels_outside_LC] 
    
    #get the lat lon grid
    og_hrrr_grid = pickle.load(open('../0_HRRR/hrrr_og_grid.pkl','rb'))
    hrrr_lat = og_hrrr_grid['lat']
    hrrr_lon = og_hrrr_grid['lon']
    hrrr_lon=hrrr_lon+360
    
    hrrr_cropped_lat = hrrr_lat[y_hrrr_idxs_regrid[0]:y_hrrr_idxs_regrid[1],x_hrrr_idxs_regrid[0]:x_hrrr_idxs_regrid[1]]
    hrrr_cropped_lon = hrrr_lon[y_hrrr_idxs_regrid[0]:y_hrrr_idxs_regrid[1],x_hrrr_idxs_regrid[0]:x_hrrr_idxs_regrid[1]]
    hrrr_grid_dict2 = {'cropped_lat':hrrr_cropped_lat,'cropped_lon':hrrr_cropped_lon}
    pickle.dump(hrrr_grid_dict2,open('hrrr_grid_cropped.pkl','wb'))
    print('regridding meta data complete')

def execute_task(file='/scratch/bmac87/',
                    mrms_cropped_lon=[],
                    mrms_cropped_lat=[],
                    hrrr_cropped_lon=[],
                    hrrr_cropped_lat=[],
                    source_grid_mrms={},
                    target_grid_hrrr={},
                    regridder=(),
                    y_min=0,
                    y_max=1,
                    x_min=0,
                    x_max=1,
                    variable='reflectivity'):
    try:
        load_file = file['fname_dir']
        save_dir = file['save_dir']
        file_str = file['fsave'][0:-2]+'pkl'
        radar_time = file['datetime']
        print(save_dir)
        print(file_str)
        num_pixels_outside_LC = 128
        mrms_radar_data = pygrib.open(load_file)[1].values[y_min:y_max, x_min:x_max]
        hrrr_radar_data = regridder(np.ascontiguousarray(mrms_radar_data))
        hrrr_radar_data_LC = hrrr_radar_data[num_pixels_outside_LC:-num_pixels_outside_LC,num_pixels_outside_LC:-num_pixels_outside_LC]
        pickle.dump({'data':hrrr_radar_data_LC,'time':radar_time},open(save_dir+file_str,'wb'))
        del hrrr_radar_data_LC, hrrr_radar_data, mrms_radar_data, save_dir, file_str
        print('pickle file saved successfully')

    except Exception as e:
        print(e)


def run_files(files_df=pd.DataFrame()):

    idx_dict = pickle.load(open('idx_dict.pkl','rb'))
    y_min = idx_dict['y_min']
    y_max = idx_dict['y_max']
    x_min = idx_dict['x_min']
    x_max = idx_dict['x_max']

    mrms_cropped_dict = pickle.load(open('mrms_cropped_grid.pkl','rb'))
    mrms_cropped_lon = mrms_cropped_dict['cropped_lon']
    mrms_cropped_lat = mrms_cropped_dict['cropped_lat']

    hrrr_cropped_dict = pickle.load(open('hrrr_grid_cropped.pkl','rb'))
    hrrr_cropped_lon = hrrr_cropped_dict['cropped_lon']
    hrrr_cropped_lat = hrrr_cropped_dict['cropped_lat']

    # Define source and target grids
    source_grid_mrms = {'lon': np.ascontiguousarray(mrms_cropped_lon), 
                        'lat': np.ascontiguousarray(mrms_cropped_lat)}
    target_grid_hrrr = {'lon': np.ascontiguousarray(hrrr_cropped_lon), 
                        'lat': np.ascontiguousarray(hrrr_cropped_lat)}

    weights_file = 'regridder.nc'
    # reuse = os.path.exists(weights_file)

    # Create regridder (bilinear interpolation for smooth fields)
    regridder = xe.Regridder(source_grid_mrms, target_grid_hrrr, method='bilinear',reuse_weights=True,filename=weights_file)
    print('regridder.nc loaded')
    
    for i in range(len(files_df)):
        print(i,len(files_df))
        execute_task(file=files_df.iloc[i],
                    mrms_cropped_lon=mrms_cropped_lon,
                    mrms_cropped_lat=mrms_cropped_lat,
                    hrrr_cropped_lon=hrrr_cropped_lon,
                    hrrr_cropped_lat=hrrr_cropped_lat,
                    y_min=y_min,
                    y_max=y_max,
                    x_min=x_min,
                    x_max=x_max,
                    source_grid_mrms=source_grid_mrms,
                    target_grid_hrrr=target_grid_hrrr,
                    regridder=regridder)

def visualize_one_day_variable(month='06',day='30',year='2022',variable='Reflectivity_-20C_00.50'):
    
    print('visualizing', year,month,day,variable)
    data_dir = '/scratch/bmac87/MRMS_on_HRRR_grid/%s/%s%s%s/'%(variable,year,month,day)
    files = sorted(os.listdir(data_dir))
    for file in files:
        data = pickle.load(open(data_dir+file,'rb'))['data']
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(20,20))
        ax.imshow(data)
        save_dir = './MRMS_test_images/%s/%s%s%s/'%(variable,year,month,day)
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        plt.savefig(save_dir+file[0:-4]+'.png')
        plt.close()

def main():
    
    files_df = pickle.load(open('./missing_files_df.pkl','rb'))
    # n_files = len(files_df)
    # start_idx, end_idx,exp = create_parser()
    # if exp==997:
    #     end_idx=len(files_df)-1
    # print(start_idx,end_idx,exp)
    # # save_mrms_2_hrrr_grids()
    # files_subset = files_df.iloc[start_idx:end_idx]
    run_files(files_df = files_df)
    

if __name__=='__main__':
    start_clock = time.time()
    main()
    end_clock = time.time()
    run_time_mins = (end_clock-start_clock)/60
    print('run_time_mins:',run_time_mins)