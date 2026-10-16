import cartopy.crs as ccrs
import numpy as np
import os
import xarray as xr
import pandas as pd
import pygrib
import matplotlib.pyplot as plt
from LC_parser import create_parser
from matplotlib.lines import Line2D
import pickle
from shapely.geometry import box
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.geometry import shape

def nearest_hrrr_ij(center_lat, center_lon,
                    hrrr_lat=[], hrrr_lon=[]):

    # handle HRRR longitudes if 0–360
    lon2 = hrrr_lon.copy()
    center_lon2 = center_lon
    if np.nanmax(lon2) > 180 and center_lon < 0:
        center_lon2 = center_lon + 360

    dist2 = (hrrr_lat - center_lat)**2 + (lon2 - center_lon2)**2
    iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)

    return iy, ix

def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius (m)

        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (np.sin(dlat/2)**2 +
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)

        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

def centered_slice(idx, size, max_size):
    half = size // 2
    start = idx - half
    end = idx + half

    if start < 0:
        start = 0
        end = size
    if end > max_size:
        end = max_size
        start = max_size - size

    return slice(start, end)

def build_figure(args):
    
    efm_keys = args.efm_keys
    # 2245-2345 June 06
    # 202406302230.pkl
    # 202406302300.pkl
    # 202406302330.pkl

    df_list = []
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1b_EFM_50HZ_single_file_pkl/202406/'
    df1 = pickle.load(open(data_dir+'202406062230.pkl','rb'))
    fm071 = df1[7.0]
    df2 = pickle.load(open(data_dir+'202406062300.pkl','rb'))
    fm072 = df2[7.0]
    df3 = pickle.load(open(data_dir+'202406062330.pkl','rb'))
    fm073 = df3[7.0]
    
    df = pd.concat([fm071,fm072,fm073])
    df = df['2024-06-06T22:45:00.000':'2024-06-06T23:45:00.000']
    df['unix_seconds'] = df.index.astype('int64') // 10**9
    
    
    merlin_cc = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CC_all_flashes.pkl','rb'))
    merlin_cc['unix_seconds'] = merlin_cc.index.astype('int64') // 10**9
    merlin_cc = merlin_cc['2024-06-06T22:45:00.000':'2024-06-06T23:45:00.000']
    cc_lats = merlin_cc['Lat_Decimal'].values
    cc_lons = merlin_cc['Lon_Decimal'].values

    merlin_cg = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CG_all_flashes.pkl','rb'))
    merlin_cg['unix_seconds'] = merlin_cg.index.astype('int64') // 10**9
    merlin_cg = merlin_cg['2024-06-06T22:45:00.000':'2024-06-06T23:45:00.000']
    cg_lats = merlin_cg['Lat_Decimal'].values
    cg_lons = merlin_cg['Lon_Decimal'].values

    dist = 5 #nautical miles
    dist_thresh=dist*1852

    efm_locs = pd.read_excel('../1_data_cleaning/1_EFM/xlsx_EFM_Locations.xlsx')
    efm_locs = efm_locs[efm_locs['IsActive']==True]
    efm_locs = efm_locs.loc[efm_locs['SiteName'].isin(args.efm_keys)]

    efm_locs.to_pickle(open('./efm_loc_df.pkl','wb'))
    efm_locs.to_pickle(open('../3_model_analysis/efm_loc_df.pkl','wb'))
    efm_locs.to_pickle(open('../2_model_training/efm_loc_df.pkl','wb'))
    efm_locs.to_pickle(open('../1_data_cleaning/1_EFM/efm_loc_df.pkl','wb'))
    efm_lats = efm_locs['Latitude'].values
    efm_lons = efm_locs['Longitude'].values
    mean_efm_lat = np.mean(efm_lats)
    mean_efm_lon = np.mean(efm_lons)

    efm_07 = efm_locs.loc[efm_locs['SiteName']=='FM07']
    lat07 = efm_07['Latitude'].values[0]
    lon07 = efm_07['Longitude'].values[0]

    cc_win5 = np.zeros(len(cc_lats))
    for i in range(len(cc_lats)):
        dist = haversine(lat1=lat07, lon1=lon07, lat2=cc_lats[i], lon2=cc_lons[i])
        if dist<=dist_thresh:
            cc_win5[i]=1.0
    merlin_cc['win5'] = cc_win5

    cg_win5 = np.zeros(len(cg_lats))
    for i in range(len(cg_lats)):
        dist = haversine(lat1=lat07, lon1=lon07, lat2=cg_lats[i], lon2=cg_lons[i])
        if dist<=dist_thresh:
            cg_win5[i]=1.0
    merlin_cg['win5'] = cg_win5

    merlin_locs = pd.read_excel('../1_data_cleaning/1_EFM/xlsx_MERLIN_Locs.xlsx')
    merlin_lats = merlin_locs['Lat'].values
    merlin_lons = merlin_locs['Lon'].values

    # # hrrr indices for spatial climatology
    # x_idxs = [1200,1550]
    # y_idxs = [100,300]

    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    hrrr_og_dict = {'lat':hrrr_lat,'lon':hrrr_lon,'lon_360':hrrr_lon+360}
    pickle.dump(hrrr_og_dict,open('./hrrr_og_grid.pkl','wb'))
    pickle.dump(hrrr_og_dict,open('../3_model_analysis/hrrr_og_grid.pkl','wb'))
    pickle.dump(hrrr_og_dict,open('../2_model_training/hrrr_og_grid.pkl','wb'))
    pickle.dump(hrrr_og_dict,open('../1_data_cleaning/0_HRRR/hrrr_og_grid.pkl','wb'))
    pickle.dump(hrrr_og_dict,open('../0_data_acq/hrrr_og_grid.pkl','wb'))
    cy, cx=nearest_hrrr_ij(center_lat=mean_efm_lat, center_lon=mean_efm_lon,
                    hrrr_lat=hrrr_lat, hrrr_lon=hrrr_lon)
    
    # -------------------------
    # 3. 64x64 domain centered there
    # -------------------------
    y64 = centered_slice(cy, 64, hrrr_lat.shape[0])
    x64 = centered_slice(cx, 64, hrrr_lat.shape[-1])
    slices_dict = {'x':x64,'y':y64}
    pickle.dump(slices_dict,open('./LC_hrrr_indexes.pkl','wb'))
    pickle.dump(slices_dict,open('../3_model_analysis/LC_hrrr_indexes.pkl','wb'))
    pickle.dump(slices_dict,open('../2_model_training/LC_hrrr_indexes.pkl','wb'))
    pickle.dump(slices_dict,open('../1_data_cleaning/LC_hrrr_indexes.pkl','wb'))
    pickle.dump(slices_dict,open('../0_data_acq/LC_hrrr_indexes.pkl','wb'))
    
    # hrrr indices for 64x64 downselection
    box_x_idxs = [x64.start,x64.stop]
    box_y_idxs = [y64.start,y64.stop]
    hrrr_lat_box = hrrr_lat[box_y_idxs[0]:box_y_idxs[1],box_x_idxs[0]:box_x_idxs[1]]
    hrrr_lon_box = hrrr_lon[box_y_idxs[0]:box_y_idxs[1],box_x_idxs[0]:box_x_idxs[1]]
    hrrr_lon_box = hrrr_lon_box+360

    lon_box = [hrrr_lon_box[0,0],hrrr_lon_box[0,-1],hrrr_lon_box[-1,-1],hrrr_lon_box[-1,0],hrrr_lon_box[0,0]]
    lat_box = [hrrr_lat_box[0,0],hrrr_lat_box[0,-1],hrrr_lat_box[-1,-1],hrrr_lat_box[-1,0],hrrr_lat_box[0,0]]

    lat16x16=hrrr_lat_box[24:40,24:40]
    lon16x16=hrrr_lon_box[24:40,24:40]

    lon_box16 = [lon16x16[0,0],lon16x16[0,-1],lon16x16[-1,-1],lon16x16[-1,0],lon16x16[0,0]]
    lat_box16 = [lat16x16[0,0],lat16x16[0,-1],lat16x16[-1,-1],lat16x16[-1,0],lat16x16[0,0]]

    #load the presaved coastline data from build_save_coast() function
    with open("./gshhs_full_slc40_clipped.pkl", "rb") as f:
        payload = pickle.load(f)
    gshhs_geoms = [shape(d) for d in payload["geoms"]]
    extent = payload["extent"]

    dist_ds = xr.open_dataset('./efm_distance_proximity.nc',engine='netcdf4')
    full_mask5 = dist_ds['mask_5nm'].values
    contour_proxy1 = Line2D([0], [0],color="red",linewidth=2,label="w/in 5-nm")
    fig = plt.figure(figsize=(20, 8))#(width, height)
    axes = []
    axes.append(fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree()))#locations map
    axes.append(fig.add_subplot(1, 2, 2))#efm time series
    axes[0].set_facecolor("lightblue")
    axes[0].add_geometries(gshhs_geoms,crs=ccrs.PlateCarree(),facecolor="tan",edgecolor="0.25",linewidth=0.6,zorder=0)
    b=axes[0].plot(lon_box,lat_box,transform=ccrs.PlateCarree(),color='black',linewidth=2,label='LaunchCast Boundary',zorder=1)[0]
    b1=axes[0].plot(lon_box16,lat_box16,transform=ccrs.PlateCarree(),color='green',linestyle='--',linewidth=2,label='Target Boundary',zorder=1)[0]
    e=axes[0].scatter(efm_lons,efm_lats,marker='x',color='red',s=10,transform=ccrs.PlateCarree(),label='EFM Sensors',zorder=2)
    axes[0].scatter(lon07,lat07,marker='*',color='red',s=50,transform=ccrs.PlateCarree(),zorder=3)
    axes[0].text(lon07-.62,lat07+.07,'FM07',fontsize=12,zorder=3)
    axes[0].plot([lon07-.4,lon07],[lat07+.07,lat07],color='black',linewidth=1,zorder=3)
    m=axes[0].scatter(merlin_lons,merlin_lats,marker='o',s=20,color='blue',transform=ccrs.PlateCarree(),label='MERLIN Sensors',zorder=1)
    axes[0].set_extent([lon_box[0]-0.5, lon_box[2]+0.5, lat_box[0]-0.5, lat_box[2]+0.5])
    cs1 = axes[0].contour(lon16x16,lat16x16,full_mask5,levels=[0.5],transform=ccrs.PlateCarree(),linewidths=2,colors='red',zorder=12)
    axes[0].legend(handles=[contour_proxy1,e,m,b,b1],fontsize=12)
    axes[0].text(lon07-1.38,lat07+.95,'(a)',fontsize=18,fontweight='bold',zorder=3)

    ez = df['ez'].values
    mean = np.nanmean(ez)
    mean_str = f"{mean:.02f}"
    
    count_gt_1000 = np.sum((np.abs(ez)>=1000))
    count_str = f"{count_gt_1000:.02f}"
    
    var = np.nanvar(ez)
    var_str = f"{var:.02f}"
    
    std = np.nanstd(ez)
    std_str = f"{std:.02f}"
    
    max_abs = np.nanmax(np.abs(ez))
    max_abs_str = f"{max_abs:.02f}"
    
    rms = np.sqrt(np.nanmean(ez**2))
    rms_str = f"{rms:.02f}"

    us = df['unix_seconds'].values

    cc5 = merlin_cc[merlin_cc["win5"] == 1]
    cg5 = merlin_cg[merlin_cg["win5"] == 1]
    axes[1].plot(us,np.zeros((len(us))),color='black',linewidth=3.0,zorder=0)
    axes[1].vlines(x=cc5['unix_seconds'],ymin=-10000,ymax=10000,color='blue',linestyle='dashed',label='IC',alpha=1,zorder=1)
    axes[1].vlines(x=cg5['unix_seconds'],ymin=-10000,ymax=10000,color='blue',linestyle='dotted',label='CG',alpha=1,zorder=1)
    axes[1].plot(us,ez,linewidth=2,color='red',zorder=2)
    
    xticks=[us[0],us[-1]]
    xt_labels=['22:45','23:45']

    axes[1].legend(fontsize=18)

    axes[1].text(us[2500],-2990,'Mean: '+mean_str,fontsize=18)
    axes[1].text(us[2500],-3490,'Std: '+std_str,fontsize=18)
    axes[1].text(us[2500],-3990,'Var: '+var_str,fontsize=18)
    axes[1].text(us[2500],-4490,'Count>=1000: '+count_str,fontsize=18)
    axes[1].text(us[2500],-4990,'RMS: '+rms_str,fontsize=18)
    axes[1].text(us[2500],-5490,'Max. Abs.: '+max_abs_str,fontsize=18)
    
    axes[1].set_title('FM07 - 50 HZ EFM Time Series',fontsize=24)
    axes[1].set_xticks(xticks,xt_labels,fontsize=18)
    axes[1].set_xlabel('06 Jun 2024 - Time (UTC; HH:MM)',fontsize=18)
    axes[1].set_xlim([xticks[0],xticks[-1]])
    axes[1].set_ylabel(r'$\nabla \Phi$ (kV/m)',fontsize=18)
    axes[1].text(us[450],2600,'(b)',fontweight='bold',fontsize=18)


    ytick=-7000
    yticks=[]
    yt_labels=[]
    while ytick<=3000:
        yticks.append(ytick)
        yt_labels.append(str(ytick/1000))
        ytick+=500
    axes[1].set_ylim([-7100,3000])
    axes[1].set_yticks(yticks,yt_labels,fontsize=18)
    axes[1].grid(axis='y')
    plt.savefig('./Figure_3_efm_merlin_loc.png',dpi=1200)
    plt.savefig('./Figure_3_efm_merlin_loc.pdf',dpi=600)
    plt.close()


def main():

    print('main function')
    parser=create_parser()
    args=parser.parse_args()
    build_figure(args=args)

if __name__=='__main__':
    main()