import numpy as np
import os
import xarray as xr
import shutil
import cartopy.crs as ccrs
import pickle
import pandas as pd
import pygrib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

def diurnal_seasonal_climo_calc():
    print('quickly calculating the climatologies')

    cc_file = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CC_all_flashes.pkl'
    cg_file = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CG_all_flashes.pkl'
    years_int = np.arange(2021,2025)

    cc_df = pickle.load(open(cc_file,'rb'))
    cc_df['Date'] = cc_df.index    
    cc_df = cc_df[cc_df['Date'].dt.year>=2021]
    
    cg_df = pickle.load(open(cg_file,'rb'))
    cg_df['Date'] = cg_df.index
    cg_df = cg_df[cg_df['Date'].dt.year>=2021]

    cc_hrly = cc_df.resample('h').count()
    cc_hrly['count'] = cc_hrly['Lat_Decimal']
    cc_hrly = cc_hrly.drop(columns=['Lat_Decimal','Lon_Decimal'])
    cc_hrly['hr'] = cc_hrly.index.hour

    cg_hrly = cg_df.resample('h').count()
    cg_hrly['count'] = cg_hrly['Lat_Decimal']
    cg_hrly = cg_hrly.drop(columns=['Lat_Decimal','Lon_Decimal'])
    cg_hrly['hr'] = cg_hrly.index.hour

    cc_hr_count = np.zeros(24)
    cc_hr_max = np.zeros(24)
    cc_hr_mean = np.zeros(24)
    cc_hr_min = np.zeros(24)

    cg_hr_count = np.zeros(24)
    cg_hr_max = np.zeros(24)
    cg_hr_mean = np.zeros(24)
    cg_hr_min = np.zeros(24)

    for hr in range(24):
        cc_hr_count[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].sum()
        cc_hr_max[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].max()
        cc_hr_mean[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].mean()
        cc_hr_min[hr] = cc_hrly[cc_hrly['hr']==hr]['count'].min()

        cg_hr_count[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].sum()
        cg_hr_max[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].max()
        cg_hr_mean[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].mean()
        cg_hr_min[hr] = cg_hrly[cg_hrly['hr']==hr]['count'].min()

    cc_moly = cc_df.resample('ME').count()
    cc_moly['count'] = cc_moly['Lat_Decimal']
    cc_moly = cc_moly.drop(columns=['Lat_Decimal','Lon_Decimal'])
    cc_moly['mo'] = cc_moly.index.month

    cg_moly = cg_df.resample('ME').count()
    cg_moly['count'] = cg_moly['Lat_Decimal']
    cg_moly = cg_moly.drop(columns=['Lat_Decimal','Lon_Decimal'])
    cg_moly['mo'] = cg_moly.index.month

    cg_mo_count = np.zeros(12)
    cg_mo_max = np.zeros(12)
    cg_mo_mean = np.zeros(12)
    cg_mo_min = np.zeros(12)

    cc_mo_count = np.zeros(12)
    cc_mo_max = np.zeros(12)
    cc_mo_mean = np.zeros(12)
    cc_mo_min = np.zeros(12)

    for mo in range(12):
        cc_mo_count[mo] = cc_moly[cc_moly['mo']==mo]['count'].sum()
        cc_mo_max[mo] = cc_moly[cc_moly['mo']==mo]['count'].max()
        cc_mo_mean[mo] = cc_moly[cc_moly['mo']==mo]['count'].mean()
        cc_mo_min[mo] = cc_moly[cc_moly['mo']==mo]['count'].min()
        
        cg_mo_count[mo] = cg_moly[cg_moly['mo']==mo]['count'].sum()
        cg_mo_max[mo] = cg_moly[cg_moly['mo']==mo]['count'].max()
        cg_mo_mean[mo] = cg_moly[cg_moly['mo']==mo]['count'].mean()
        cg_mo_min[mo] = cg_moly[cg_moly['mo']==mo]['count'].min()
    cc_dict = {'cc_mo_count':cc_mo_count,
                'cc_mo_max':cc_mo_max,
                'cc_mo_min':cc_mo_min,
                'cc_mo_mean':cc_mo_min,
                'cc_hr_count':cc_hr_count,
                'cc_hr_max':cc_hr_max,
                'cc_hr_mean':cc_hr_mean,
                'cc_hr_min':cc_hr_min}

    cg_dict = {'cg_mo_count':cg_mo_count,
                'cg_mo_max':cg_mo_max,
                'cg_mo_min':cg_mo_min,
                'cg_mo_mean':cg_mo_min,
                'cg_hr_count':cg_hr_count,
                'cg_hr_max':cg_hr_max,
                'cg_hr_mean':cg_hr_mean,
                'cg_hr_min':cg_hr_min}
    
    climo_dict = {'cc':cc_dict,'cg':cg_dict}
    pickle.dump(climo_dict,open('./pickles/Figure_1_climo_dict.pkl','wb'))

def grid_flashes(flash_lats,flash_lons,flash_grid,hrrr_x_1d,hrrr_y_1d,hrrr_z_1d,hrrr_xyz,hrrr_proj,hrrr_lon):

    #turn the lat lon of the flashes into an xyz coordingate system
    flash_xyz = hrrr_xyz.transform_points(hrrr_proj,flash_lons,flash_lats)#brilliant!!

    #for each flash, calculate the distance to each hrrr point
    for fl in range(flash_xyz.shape[0]):

        fl_x = flash_xyz[fl,0]
        fl_y = flash_xyz[fl,1]
        fl_z = flash_xyz[fl,2]

        dx = hrrr_x_1d-fl_x
        dy = hrrr_y_1d-fl_y
        dz = hrrr_z_1d-fl_z

        #calculate the distance
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        #find the index of the closest distance
        idx1, idx2 = np.unravel_index(np.argmin(dist, axis=None), hrrr_lon.shape)
        flash_grid[idx1,idx2]+=1
    return flash_grid

def grid_ltg_hrrr():

    # hrrr indices for spatial climatology
    x_idxs = [1200,1550]
    y_idxs = [100,300]

    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    print(hrrr_lon.shape)
    hrrr_lat = hrrr_lat[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]
    hrrr_lon = hrrr_lon[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]
    hrrr_lon=hrrr_lon+360
    print(hrrr_lon.shape)
    hrrr_lat_1d = np.ravel(hrrr_lat)
    hrrr_lon_1d = np.ravel(hrrr_lon)

    projection_params = grbs[1].projparams
    proj_a = projection_params['a']
    proj_b = projection_params['b']
    lon_0 = projection_params['lon_0']
    lat_0 = projection_params['lat_0']
    lat_parallel = projection_params['lat_1']

    print('creating the hrrr ccrs projection')
    hrrr_proj = ccrs.LambertConformal(central_longitude=lon_0, 
                                        central_latitude=lat_0,
                                        globe=ccrs.Globe(semimajor_axis=proj_a,
                                                            semiminor_axis=proj_b))

    print('creating the plot transform')                                                      
    plot_proj = ccrs.PlateCarree()

    print('creating the hrrr_xy transform')
    hrrr_xyz = hrrr_proj.as_geocentric()

    print('transforming the hrrr_lat/lon to hrrr_xyz2')
    hrrr_xyz2 = hrrr_xyz.transform_points(hrrr_proj,hrrr_lon,hrrr_lat)
    hrrr_x = hrrr_xyz2[:,:,0]
    hrrr_x_1d = np.ravel(hrrr_x)
    
    hrrr_y = hrrr_xyz2[:,:,1]
    hrrr_y_1d = np.ravel(hrrr_y)
    
    hrrr_z = hrrr_xyz2[:,:,2]
    hrrr_z_1d = np.ravel(hrrr_z)
    
    cc_df = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CC_all_flashes.pkl','rb'))
    cc_df['Date'] = cc_df.index    
    cc_df = cc_df[cc_df['Date'].dt.year>=2021]
    
    cc_lats = cc_df['Lat_Decimal'].values
    cc_lons = cc_df['Lon_Decimal'].values+360
    print(len(cc_lats))
    print(len(cc_lons))

    cc_grid = np.zeros(hrrr_lon.shape)
    print(cc_grid.shape)
    cc_grid = grid_flashes(flash_lats=cc_lats,
                            flash_lons=cc_lons,
                            flash_grid=cc_grid,
                            hrrr_xyz=hrrr_xyz,
                            hrrr_proj=hrrr_proj,
                            hrrr_x_1d=hrrr_x_1d,
                            hrrr_y_1d=hrrr_y_1d,
                            hrrr_z_1d=hrrr_z_1d,
                            hrrr_lon=hrrr_lon)
    pickle.dump(cc_grid,open('./pickles/cc_spatial_climo.pkl','wb'))

    cg_df = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CG_all_flashes.pkl','rb'))
    cg_df['Date'] = cg_df.index
    cg_df = cg_df[cg_df['Date'].dt.year>=2021]

    cg_lats = cg_df['Lat_Decimal'].values
    cg_lons = cg_df['Lon_Decimal'].values+360
    cg_grid = np.zeros(hrrr_lon.shape)
    cg_grid = grid_flashes(flash_lats=cg_lats,
                            flash_lons=cg_lons,
                            flash_grid=cg_grid,
                            hrrr_xyz=hrrr_xyz,
                            hrrr_proj=hrrr_proj,
                            hrrr_x_1d=hrrr_x_1d,
                            hrrr_y_1d=hrrr_y_1d,
                            hrrr_z_1d=hrrr_z_1d,
                            hrrr_lon=hrrr_lon)
    pickle.dump(cg_grid,open('./pickles/cg_spatial_climo.pkl','wb'))

def make_figure():
    climo_dict = pickle.load(open('./pickles/Figure_1_climo_dict.pkl','rb'))
    print('making the figure')
    fig = plt.figure(figsize=(8.5, 11))#(width, height)

    axes = []
    axes.append(fig.add_subplot(3, 2, 1))#IC diurnal climo
    axes.append(fig.add_subplot(3, 2, 2))#CG diurnal climo
    axes.append(fig.add_subplot(3, 2, 3))#IC seasonal climo
    axes.append(fig.add_subplot(3, 2, 4))#CG seasonal climo
    axes.append(fig.add_subplot(3, 2, 5, projection=ccrs.PlateCarree()))#IC spatial climatology  
    axes.append(fig.add_subplot(3, 2, 6, projection=ccrs.PlateCarree()))#CG spatial climatology

    x_ticks = np.arange(1,25)
    x_tick_labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12',
                        '13','14','15','16','17','18','19','20','21','22','23']
    axes[0].bar(x_ticks,climo_dict['cc']['cc_hr_count']/1e4,color='red',zorder=3)
    axes[0].set_xticks(x_ticks[::3])
    axes[0].set_xticklabels(x_tick_labels[::3],fontsize=12,rotation=45)
    y_ticks = np.linspace(0,45,10)
    y_tick_labels = []
    for y_tick in y_ticks:
        y_tick_labels.append(f"{int(y_tick):02d}")
    
    axes[0].set_xlabel('Time (UTC)',fontsize=12)
    axes[0].set_title('IC Flashes',fontsize=18)
    axes[0].set_ylim([0,45])
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_tick_labels,fontsize=12)
    axes[0].grid()
    axes[0].set_ylabel('Flash Count (1e4; #/hr)',fontsize=12)

    axes[1].bar(x_ticks,climo_dict['cg']['cg_hr_count']/1e4,color='blue',zorder=3)
    axes[1].set_xticks(x_ticks[::3])
    axes[1].set_xticklabels(x_tick_labels[::3],fontsize=12,rotation=45)
    axes[1].set_xlabel('Time (UTC)',fontsize=12)
    axes[1].set_title('CG Flashes',fontsize=18)
    y_ticks = np.linspace(0,16,9)
    y_tick_labels = []
    for y_tick in y_ticks:
        y_tick_labels.append(f"{int(y_tick):02d}")
    axes[1].set_ylim([0,16])
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(y_tick_labels,fontsize=12)
    axes[1].grid(zorder=0)

    x_ticks = np.arange(0,12)
    x_tick_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    axes[2].bar(x_ticks,climo_dict['cc']['cc_mo_count']/1e4,color='red',zorder=3)
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(x_tick_labels,fontsize=12,rotation=45)
    axes[2].set_xlabel('Month',fontsize=12)
    y_ticks = np.linspace(0,80,9)
    y_tick_labels = []
    for y_tick in y_ticks:
        y_tick_labels.append(f"{int(y_tick):02d}")
    axes[2].set_ylabel('Flash Count (1e4; #/yr)',fontsize=12)
    axes[2].set_yticks(y_ticks)
    axes[2].set_yticklabels(y_tick_labels,fontsize=12)
    axes[2].grid(zorder=0)


    axes[3].bar(x_ticks,climo_dict['cg']['cg_mo_count']/1e4,color='blue',zorder=3)
    axes[3].set_xticks(x_ticks)
    axes[3].set_xticklabels(x_tick_labels,fontsize=12,rotation=45)
    axes[3].set_xlabel('Month',fontsize=12)
    y_ticks = np.linspace(0,40,5)
    y_tick_labels = []
    for y_tick in y_ticks:
        y_tick_labels.append(f"{int(y_tick):02d}")
    axes[3].set_yticks(y_ticks)
    axes[3].set_yticklabels(y_tick_labels,fontsize=12)
    axes[3].grid(zorder=0)

    # hrrr indices for spatial climatology
    x_idxs = [1200,1550]
    y_idxs = [100,300]

    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    print(hrrr_lon.shape)
    hrrr_lat_climo = hrrr_lat[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]
    hrrr_lon_climo = hrrr_lon[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]
    hrrr_lon_climo = hrrr_lon_climo+360

    # hrrr indices for 64x64 downselection
    box_x_idxs = [1422,1486]
    box_y_idxs = [176,240]
    hrrr_lat_box = hrrr_lat[box_y_idxs[0]:box_y_idxs[1],box_x_idxs[0]:box_x_idxs[1]]
    hrrr_lon_box = hrrr_lon[box_y_idxs[0]:box_y_idxs[1],box_x_idxs[0]:box_x_idxs[1]]
    hrrr_lon_box = hrrr_lon_box+360

    lon_box = [hrrr_lon_box[0,0],hrrr_lon_box[0,-1],hrrr_lon_box[-1,-1],hrrr_lon_box[-1,0],hrrr_lon_box[0,0]]
    lat_box = [hrrr_lat_box[0,0],hrrr_lat_box[0,-1],hrrr_lat_box[-1,-1],hrrr_lat_box[-1,0],hrrr_lat_box[0,0]]

    cc_climo = pickle.load(open('./pickles/cc_spatial_climo.pkl','rb'))
    cc_climo[cc_climo==0]=np.nan
    im=axes[4].pcolormesh(hrrr_lon_climo,hrrr_lat_climo,cc_climo,cmap='Reds')
    axes[4].plot(lon_box,lat_box,
                transform=ccrs.PlateCarree(),color='black',linewidth=2)
    axes[4].set_extent([lon_box[0]-2.0, lon_box[2]+2.0, lat_box[0]-2.0, lat_box[2]+2.0])
    axes[4].coastlines()
    cb = plt.colorbar(im,ax=axes[4],label='IC Flash Count')
    
    cg_climo = pickle.load(open('./pickles/cg_spatial_climo.pkl','rb'))
    cg_climo[cg_climo==0]=np.nan
    im=axes[5].pcolormesh(hrrr_lon_climo,hrrr_lat_climo,cg_climo,cmap='Blues')
    axes[5].plot(lon_box,lat_box,
                transform=ccrs.PlateCarree(),
                color='black',linewidth=2)
    axes[5].set_extent([lon_box[0]-2.0, lon_box[2]+2.0, lat_box[0]-2.0, lat_box[2]+2.0])
    axes[5].coastlines()
    cb = plt.colorbar(im,ax=axes[5],label='CG Flash Count')


    plt.tight_layout()
    plt.savefig('Figure_1_Merlin_Climo.png')
    plt.savefig('Figure_1_Merlin_Climo.pdf')
    plt.close()

def main():
    # grid_ltg_hrrr()
    make_figure()
    # test_grid()

if __name__=='__main__':
    main()