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

def create_parser():
    parser = argparse.ArgumentParser(description='Download', fromfile_prefix_chars='@')
    parser.add_argument('--exp', type=int, default=0,help='0-500')
    parser.add_argument('--var',type=str,default='MergedZdr_00.50',help='variable to process')
    args = parser.parse_args()
    exp = args.exp
    start_idx = exp*886
    end_idx = start_idx+885
    variable = args.var
    return start_idx, end_idx, exp, variable

def save_mrms_2_hrrr_grids():

    #set the box to get the groups from
    west_lon_box = -87.0+360
    east_lon_box = -77.0+360
    north_lat_box = 31.0
    south_lat_box = 24.0

    #load the mrms grid
    mrms_grid = pickle.load(open('mrms_grid_whole.pkl','rb'))
    mrms_lat = mrms_grid['lat']
    mrms_lon = mrms_grid['lon']

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
    x_idxs = [1422,1486]
    y_idxs = [176,240]

    num_pixels_outside_LC = 128
    x_hrrr_idxs_regrid = [x_idxs[0]-num_pixels_outside_LC,x_idxs[1]+num_pixels_outside_LC]
    y_hrrr_idxs_regrid = [y_idxs[0]-num_pixels_outside_LC,y_idxs[1]+num_pixels_outside_LC] 

    #target grid post 64x64 downselection
    #x__target_idxs for slicein: 23:39
    #y__target_idxs for slicin:26:42
    
    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    hrrr_lon=hrrr_lon+360
    hrrr_grid_dict = {'lat':hrrr_lat,'lon':hrrr_lon}
    pickle.dump(hrrr_grid_dict,open('hrrr_grid_whole.pkl','wb'))
    
    hrrr_cropped_lat = hrrr_lat[y_hrrr_idxs_regrid[0]:y_hrrr_idxs_regrid[1],x_hrrr_idxs_regrid[0]:x_hrrr_idxs_regrid[1]]
    hrrr_cropped_lon = hrrr_lon[y_hrrr_idxs_regrid[0]:y_hrrr_idxs_regrid[1],x_hrrr_idxs_regrid[0]:x_hrrr_idxs_regrid[1]]
    hrrr_grid_dict2 = {'cropped_lat':hrrr_cropped_lat,'cropped_lon':hrrr_cropped_lon}
    pickle.dump(hrrr_grid_dict2,open('hrrr_grid_cropped.pkl','wb'))

def execute_task(file='/scratch/bmac87/',
                    mrms_cropped_lon=[],
                    mrms_cropped_lat=[],
                    hrrr_cropped_lon=[],
                    hrrr_cropped_lat=[],
                    y_min=0,
                    y_max=1,
                    x_min=0,
                    x_max=1,
                    variable='reflectivity'):
    try:
        fyr = file[-21:-17]
        fmo = file[-17:-15]
        fday = file[-15:-13]
        fhr = file[-12:-10]
        fmin = file[-10:-8]
        fsec = file[-8:-6]

        time_str = '%s-%s-%sT%s:%s:%s'%(fyr,fmo,fday,fhr,fmin,fsec)
        file_str = '%s%s%s_%s%s%s.pkl'%(fyr,fmo,fday,fhr,fmin,fsec)
        radar_time = np.datetime64(time_str,'ns')
        save_dir = '/scratch/bmac87/MRMS_on_HRRR_grid/%s/%s%s%s/'%(variable,fyr,fmo,fday)
        if os.path.isfile(save_dir+file_str)==False:

            # Define source and target grids
            source_grid_mrms = {'lon': np.ascontiguousarray(mrms_cropped_lon), 
                                'lat': np.ascontiguousarray(mrms_cropped_lat)}
            target_grid_hrrr = {'lon': np.ascontiguousarray(hrrr_cropped_lon), 
                                'lat': np.ascontiguousarray(hrrr_cropped_lat)}

            weights_file = 'regridder.nc'
            reuse = os.path.exists(weights_file)

            # Create regridder (bilinear interpolation for smooth fields)
            regridder = xe.Regridder(source_grid_mrms, target_grid_hrrr, method='bilinear',reuse_weights=reuse,filename=weights_file)
            num_pixels_outside_LC = 128
    
            mrms_radar_data = pygrib.open(file)[1].values[y_min:y_max, x_min:x_max]
            hrrr_radar_data = regridder(np.ascontiguousarray(mrms_radar_data))
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            hrrr_radar_data_LC = hrrr_radar_data[num_pixels_outside_LC:-num_pixels_outside_LC,num_pixels_outside_LC:-num_pixels_outside_LC]
            pickle.dump({'data':hrrr_radar_data_LC,'time':radar_time},open(save_dir+file_str,'wb'))
            del hrrr_radar_data_LC, hrrr_radar_data, mrms_radar_data

    except Exception as e:
        print(e)
        print('bad file:',file)

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

    start_idx, end_idx, exp, variable = create_parser()
    fname = 'parallel_files_%s.pkl'%variable
    files = pickle.load(open(fname,'rb'))
    print(len(files))
    if exp==500:
        end_idx=len(files)-1
    if start_idx>(len(files)-1):
        print('start_idx is greater than file list size')
        return
    print(start_idx, end_idx, exp, variable)
    files = files[start_idx:end_idx]

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

    for f,file in enumerate(files):
        if f%25==0:
            print(f,len(files))
        execute_task(file=file,
                    mrms_cropped_lon=mrms_cropped_lon,
                    mrms_cropped_lat=mrms_cropped_lat,
                    hrrr_cropped_lon=hrrr_cropped_lon,
                    hrrr_cropped_lat=hrrr_cropped_lat,
                    y_min=y_min,
                    y_max=y_max,
                    x_min=x_min,
                    x_max=x_max,
                    variable=variable)

if __name__=='__main__':
    start_clock = time.time()
    main()
    end_clock = time.time()
    run_time_mins = (end_clock-start_clock)/60
    print('run_time_mins:',run_time_mins)

def parallelize_files_code():
    # Creating a queue with all the files that we want to download
    files_queue = [(file) for file in files]

    # Maximum number of parallel processes (number of threads)
    num_processes = 100

    # Create a list to hold the process objects
    processes = []

    # Starting the parallelization processes
    while files_queue or processes:

        # Start new processes up to the maximum number of threads
        while len(processes) < num_processes and files_queue:
            file = files_queue.pop(0)
            process = mp.Process(target=execute_task, args=(file, 
                                                            mrms_cropped_lon, 
                                                            mrms_cropped_lat, 
                                                            hrrr_cropped_lon,
                                                            hrrr_cropped_lat,
                                                            y_min,
                                                            y_max,
                                                            x_min,
                                                            x_max,
                                                            variable))
            processes.append(process)
            process.start()

        # Removing the completed processes from the list, keeping only the active ones
        processes = [process for process in processes if process.is_alive()]

        # Wait for a short time before checking again
        time.sleep(1)

    # Wait for the remaining training processes to complete
    for process in processes:
        process.join()
    print("All threads have completed their tasks.")