import pickle
from helper import *
from LC_util import *
import sys
import os
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_hrrr_grid(grbs):

    #set the extent of LaunchCast bounds over KSC
    west_extent = -81.61
    east_extent = -79.82
    south_extent = 27.7
    north_extent = 29.3

    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018011609_f000.grib'
    grbs = pygrib.open(og_hrrr)
    hrrr_lat,hrrr_lon = grbs[1].latlons()

    print('lat/lon mask')
    lat_mask = np.where(((hrrr_lat>=south_extent) & (hrrr_lat<=north_extent)),1,np.nan) 
    lon_mask = np.where(((hrrr_lon>=west_extent) & (hrrr_lon<=east_extent)),1,np.nan)
    print(lat_mask.shape)
    print(lon_mask.shape)

    print('the mask information')
    hrrr_lat_masked = hrrr_lat*lat_mask*lon_mask
    hrrr_lon_masked = hrrr_lon*lat_mask*lon_mask
    print(hrrr_lat_masked.shape)
    print(hrrr_lon_masked.shape)


    print(' ~np.isnan(hrrr_lat_masked)')
    lat_notnan = ~np.isnan(hrrr_lat_masked)
    print(lat_notnan.shape)
    lat_nonans = hrrr_lat_masked[lat_notnan]
    print(lat_nonans.shape)

    lon_notnan = ~np.isnan(hrrr_lon_masked)
    lon_nonans = hrrr_lon_masked[lon_notnan]
    print(lon_nonans.shape)

    # ltg_distances = np.sqrt((hrrr_x_1d-ltg_x)**2+(hrrr_y_1d-ltg_y)**2)
    # min_distance_index = np.argmin(ltg_distances)
    # row_index,col_index=np.unravel_index(min_distance_index, hrrr_lat.shape)

    # print(hrrr_lat[row_index,col_index])
    # print(hrrr_lon[row_index,col_index])
    # print(row_index,col_index)

def build_merlin2hrrr_grid():
    """
    This function is not complete. Use with caution. 
    """
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018011609_f000.grib'
    grbs = pygrib.open(og_hrrr)
    get_hrrr_grid(grbs)

    


    
    # ltg_distances = np.sqrt((hrrr_x_1d-ltg_x)**2+(hrrr_y_1d-ltg_y)**2)
    # min_distance_index = np.argmin(ltg_distances)
    # row_index,col_index=np.unravel_index(min_distance_index, hrrr_lat.shape)

    

def downselect_2mrms_latlon():
    
    print('putting merlin data on mrms grid')
    cc_file = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/merlin_cc_df.pkl'
    cg_file = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/merlin_cg_df.pkl'

    cc_df = pickle.load(open(cc_file,'rb'))
    cg_df = pickle.load(open(cg_file,'rb'))

    print('loading the mrms grid')
    mrms_dir = '/ourdisk/hpc/ai2es/datasets/MRMS/2020/20201014/MergedReflectivityQCComposite_00.50/compressed/'
    fname = 'MRMS_MergedReflectivityQCComposite_00.50_20201014-235033.grib2'
    
    print('getting the projection information for the mrms grid')
    grbs = pygrib.open(mrms_dir+fname)
    message = grbs[1]
    projection_params = message.projparams
    proj_a = projection_params['a']
    proj_b = projection_params['b']
    mrms_proj = ccrs.PlateCarree(globe=ccrs.Globe(semimajor_axis=proj_a,semiminor_axis=proj_b))

    print('loading the mrms latlons()')
    mrms_lat,mrms_lon = message.latlons()
    mrms_lat_1d = mrms_lat[:,0]
    mrms_lon_1d = mrms_lon[0,:]

    print('finding the min/max MERLIN ltg lat/lons')
    #get the lat/lon spread of the cc lightning flashes
    cc_max_lat = np.max(cc_df['Lat_Decimal'].values)
    cc_min_lat = np.min(cc_df['Lat_Decimal'].values)
    cc_max_lon = np.max(cc_df['Lon_Decimal'].values)
    cc_min_lon = np.min(cc_df['Lon_Decimal'].values)
    
    print('finding the min/max MERLIN ltg lat/lons')
    #get the lat/lon spread of the cg lightning flashes
    cg_max_lat = np.max(cg_df['Lat'].values)
    cg_min_lat = np.min(cg_df['Lat'].values)
    cg_max_lon = np.max(cg_df['Lon'].values)
    cg_min_lon = np.min(cg_df['Lon'].values)
    
    print('downselecting the MRMS grid based on the min/max ltg lat/lon')
    #use the cg box-spread, since it is closer together
    #find the mrms indices
    found_min=False
    found_max=False
    for i in range(len(mrms_lat_1d)):
        if found_min==False:
            if mrms_lat_1d[i]<=cg_min_lat:
                min_lat_idx=i
                found_min=True

        if found_max==False:
            if mrms_lat_1d[i]<=cg_max_lat:
                max_lat_idx=i
                found_max=True
    found_min=False
    found_max=False

    cg_min_lon = cg_min_lon+360
    cg_max_lon = cg_max_lon+360
    for i in range(len(mrms_lon_1d)):
        if found_min==False:
            if mrms_lon_1d[i]>=cg_min_lon:
                min_lon_idx=i
                found_min=True

        if found_max==False:
            if mrms_lon_1d[i]>=cg_max_lon:
                max_lon_idx=i
                found_max=True

    cc_df['Lon'] = cc_df['Lon_Decimal']+360
    cg_df['Lon'] = cg_df['Lon']+360
    cc_df['Lat'] = cc_df['Lat_Decimal']

    print('making the mrms grid 64x64, adjust if needed')
    #get the grid centered on the cape
    max_lat_idx = max_lat_idx+230
    min_lat_idx = min_lat_idx-250
    min_lon_idx = min_lon_idx+330
    max_lon_idx = max_lon_idx-323

    #get the minimum/maximum latitudes and longitudes
    mrms_min_lon = mrms_lon_1d[min_lon_idx]
    mrms_max_lon = mrms_lon_1d[max_lon_idx]
    mrms_min_lat = mrms_lat_1d[min_lat_idx]
    mrms_max_lat = mrms_lat_1d[max_lat_idx]

    print('getting the 2D grid of the downselected MRMS grid. Both lat/lon and x/y')
    #get the 2D grid
    lc_mrms_lat = mrms_lat[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]
    lc_mrms_lon = mrms_lon[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]

    print('downselecting the CG lightning')
    cg_df = cg_df[cg_df['Lon']>=mrms_min_lon]
    cg_df = cg_df[cg_df['Lon']<=mrms_max_lon]
    cg_df = cg_df[cg_df['Lat']>=mrms_min_lat]
    cg_df = cg_df[cg_df['Lat']<=mrms_max_lat]
    cg_x = np.zeros(len(cg_df))
    cg_y = np.zeros(len(cg_df))

    print('downselecting the CC lightning')
    cc_df = cc_df[cc_df['Lon']>=mrms_min_lon]
    cc_df = cc_df[cc_df['Lon']<=mrms_max_lon]
    cc_df = cc_df[cc_df['Lat']>=mrms_min_lat]
    cc_df = cc_df[cc_df['Lat']<=mrms_max_lat]

    ltg_dict = {'cc':cc_df,
                'cg':cg_df}
    pickle.dump(ltg_dict,
                open('/home/bmac87/LaunchCast/1_data_cleaning/3_MERLIN/ltg_dict.pkl','wb'))
    
    grid_dict = {'lc_mrms_lat':lc_mrms_lat,
                'lc_mrms_lon':lc_mrms_lon}
    pickle.dump(grid_dict,
                open('/home/bmac87/LaunchCast/1_data_cleaning/3_MERLIN/grid_dict.pkl','wb'))
"""
The grid_df function takes in an increasing, 1-D list of the latitudes and longitudes.   
"""
def grid_df2mrms(df=pd.DataFrame(),ltg_type='CC',grid_lat=[],grid_lon=[]):

    print(len(df),' # of %s Flashes'%(ltg_type))

    start_time = np.datetime64('2018-01-01T00:00:00.000000000')
    end_time = np.datetime64('2025-01-01T00:00:00.000000000')
    dt = np.timedelta64(5,'m')

    df['date_time'] = df.index
    ds = xr.Dataset(data_vars=dict(
                        flash_id=(["time"], df['flash_id']),
                        lat=(["time"], df['Lat']),
                        lon=(["time"],df['Lon'])),
                        coords=dict(time=df['date_time']),
                        attrs=dict(description="Cloud to Cloud lightning flashes from the MERLIN lightning location system. "))
    ds = ds.sortby('time')

    data_list = []
    time_list = []

    t=0
    while start_time<end_time:
        int_time = start_time+dt
        min5_ds = ds.sel(time=slice(start_time,int_time))
        num_flashes = len(min5_ds['flash_id'].values)
        if num_flashes==0:
            flash_bins=np.zeros((64,64))
        else:
            flash_lats = min5_ds['lat'].values
            flash_lons = min5_ds['lon'].values
            flash_bins=np.zeros((64,64))
            for l in range(len(flash_lats)):
                lat_diff = np.abs(grid_lat-flash_lats[l])
                lat_idx = np.argmin(lat_diff)
                lon_diff = np.abs(grid_lon-flash_lons[l])
                lon_idx = np.argmin(lon_diff)
                flash_bins[lat_idx,lon_idx]+=1
        data_list.append(flash_bins)
        time_list.append(start_time)
        start_time = int_time
        t+=1

        if t%10000==0:
            print(start_time)

    data_np = np.stack(data_list,axis=2)
    ds2 = xr.Dataset(data_vars=dict(fed=(["lat","lon","time"], data_np)),
                        coords=dict(
                            lat=grid_lat,
                            lon=grid_lon,
                            time=time_list))
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/'
    fsave = '%s_mrms.nc'%(ltg_type)
    ds2.to_netcdf(save_dir+fsave,engine='netcdf4')

def build_merlin2mrms_grid(cc=False,cg=False):
    print('build_merlin2mrms_grid(), gridding the lightning data')
    grid_dict = pickle.load(open('/home/bmac87/LaunchCast/1_data_cleaning/3_MERLIN/grid_dict.pkl','rb'))
    ltg_dict = pickle.load(open('/home/bmac87/LaunchCast/1_data_cleaning/3_MERLIN/ltg_dict.pkl','rb'))
    
    print('loading the grid_dict.pkl data')
    lc_mrms_lat = grid_dict['lc_mrms_lat']
    lc_mrms_lon = grid_dict['lc_mrms_lon']
    lc_mrms_lat_1d = lc_mrms_lat[:,0]
    lc_mrms_lon_1d = lc_mrms_lon[0,:]
    if cc==True:
        grid_df(df=ltg_dict['cc'],ltg_type='CC',grid_lat=lc_mrms_lat_1d,grid_lon=lc_mrms_lon_1d)

    if cg==True:
        grid_df(df=ltg_dict['cg'],ltg_type='CG',grid_lat=lc_mrms_lat_1d,grid_lon=lc_mrms_lon_1d)

def print_ds():
    stor_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/'
    file = 'CC_mrms.nc'
    ds = xr.open_dataset(stor_dir+file,engine='netcdf4')
    print(ds)

def test_output():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(1,2,2,projection=ccrs.PlateCarree())
    
    stor_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/'
    file = 'CC_mrms.nc'
    ds = xr.open_dataset(stor_dir+file,engine='netcdf4')
    lat = ds['lat'].values
    lon = ds['lon'].values
    cc_np = ds['fed'].values
    cc_np_sum = np.sum(cc_np,axis=2)
    print(np.max(np.max(cc_np_sum)))
    print(cc_np_sum.shape)
    del ds,cc_np

    file = 'CG_mrms.nc'
    ds = xr.open_dataset(stor_dir+file,engine='netcdf4')
    cg_np = ds['fed'].values
    cg_np_sum = np.sum(cg_np,axis=2)
    print(np.max(np.max(cg_np_sum)))
    del ds, cg_np

    cb_cg = ax.pcolormesh(lon,lat,cg_np_sum,cmap='coolwarm')
    plt.colorbar(cb_cg,ax=ax,label='# Flashes')
    ax.coastlines()
    ax.set_title('CG')
    cb_cc = ax1.pcolormesh(lon,lat,cc_np_sum,cmap='coolwarm')
    plt.colorbar(cb_cc,ax=ax1,label='# Flashes')
    ax1.coastlines()
    ax1.set_title('CC')
    plt.savefig('temp.png')
    plt.close()

def main():
    build_merlin2hrrr_grid()
    
if __name__=='__main__':
    main()