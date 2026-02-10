import xarray as xr
import numpy as np
from LC_parser import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import pygrib
from matplotlib.lines import Line2D

def generate_binary_mask(args,dist=10):
    
    print('in generate_binary_mask function')
    data_dir='/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/88_LC_observed_datasets/6_final_xr_observed/'
    fname = 'LC_observed_2024_w_EFM_dist.nc'

    ds = xr.open_dataset(data_dir+fname,engine='netcdf4')
    lat = ds['lat'].values
    lon = ds['lon'].values

    efm_keys = args.efm_keys
    efm_distances = np.squeeze(ds['efm_distances'].values[0,0,:,:,:])
    dist_thresh=10*1000
    print(efm_distances.shape)
    
    mask_list = []
    for i in range(efm_distances.shape[-1]):
        temp_dist = efm_distances[:,:,i]
        temp_dist[temp_dist<=dist_thresh] = 1.0
        temp_dist[temp_dist>dist_thresh] = 0.0
        mask_list.append(temp_dist)
        del temp_dist

    stacked_distances = np.stack(mask_list,axis=0)
    print(stacked_distances.shape)
    full_mask = np.sum(stacked_distances,axis=0)
    print(full_mask.shape)
    full_mask[full_mask>0] = 1.0
    full_mask[full_mask==0.0] = 0.0
    total_pixels = np.sum(full_mask)
    print('total pixels',total_pixels,'out of',64*64)

    efm_sites = pd.read_excel('EFM_Locations.xlsx')
    efm_sites = efm_sites[efm_sites['IsActive']==True]

    efm_site_lats = efm_sites['Latitude'].values
    efm_site_lons = efm_sites['Longitude'].values+360

    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)

    hrrr_lat,hrrr_lon = grbs[1].latlons()
    print(hrrr_lon.shape)
    box_x_idxs = [1422,1486]
    box_y_idxs = [176,240]
    hrrr_lat_box = hrrr_lat[box_y_idxs[0]:box_y_idxs[1],box_x_idxs[0]:box_x_idxs[1]]
    hrrr_lon_box = hrrr_lon[box_y_idxs[0]:box_y_idxs[1],box_x_idxs[0]:box_x_idxs[1]]
    hrrr_lon_box = hrrr_lon_box+360

    lon_box = [hrrr_lon_box[0,0],hrrr_lon_box[0,-1],hrrr_lon_box[-1,-1],hrrr_lon_box[-1,0],hrrr_lon_box[0,0]]
    lat_box = [hrrr_lat_box[0,0],hrrr_lat_box[0,-1],hrrr_lat_box[-1,-1],hrrr_lat_box[-1,0],hrrr_lat_box[0,0]]

    contour_proxy = Line2D([0], [0],color="red",linewidth=2,label="w/in 10-km")
    fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(5,5),subplot_kw={'projection': ccrs.PlateCarree()})
    cs = axes.contour(lon,lat,full_mask,levels=[0.5],linewidths=2,colors='red')
    sc=axes.scatter(efm_site_lons,efm_site_lats,marker='o',s=6.0,color='green',label='EFM Sites')
    pl=axes.plot(lon_box,lat_box,transform=ccrs.PlateCarree(),color='black',linewidth=2,label='LaunchCast Boundary')[0]
    axes.coastlines()
    axes.legend(handles=[contour_proxy,sc,pl],fontsize=12)
    plt.savefig('efm_eval_mask.png')
    plt.close()

    return full_mask

def main():
    print('generating the binary mask for the EFM eval')
    parser = create_parser()
    args =  parser.parse_args()

    generate_binary_mask(args=args,dist=10)#distance in km away from the EFM
if __name__=='__main__':
    main()