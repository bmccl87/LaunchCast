import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib as mpl
from sklearn.metrics import average_precision_score
import os
import xarray as xr
from LC_model_64216 import Stack, FlattenStations, UnflattenStations
from LC_parser import *
import pandas as pd
import copy
import tensorflow as tf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
from shapely.geometry import box
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.geometry import shape

EFM_MASK_VALUE=-999999.0
ZDR_MASK_VALUE=-99
ZDR_FILL_VALUE=-5


def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")
    return slurm_vars

def preamble():
    print('paring args')
    slurm_vars = extract_slurm_env()
    parser = create_parser()
    args = parser.parse_args()

    print('CPU check')
    num_cpus = os.cpu_count()
    print(num_cpus, 'on node')
    print(args.cpus_per_task,'requested in slurm')

    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print('GPUS:', visible_devices)

    if args.gpu==False:
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')
    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    else:
        print('cpus are')
        print(args.cpus_per_task)
        print('cpus not allocated to tensorflow op_parallelism_threads')

    return args

def build_save_coast():

    # ---- your fixed plot domain around SLC-40 ----
    lat0 = 28.56852255125016
    lon0 = -80.56663647112914
    pad = 2  # degrees padding; tune as needed

    extent = (lon0 - pad, lon0 + pad, lat0 - pad, lat0 + pad)  # (minlon, maxlon, minlat, maxlat)
    clip_poly = box(extent[0], extent[2], extent[1], extent[3])

    # ---- build feature once ----
    feat = cfeature.GSHHSFeature(scale="full")  # slow ONCE here

    # ---- extract and clip geometries ----
    clipped = []
    for g in feat.geometries():
        if g.intersects(clip_poly):
            gg = g.intersection(clip_poly)
            if not gg.is_empty:
                clipped.append(gg)

    # (optional) merge to fewer geometries (often speeds drawing)
    clipped_merged = unary_union(clipped)

    # ---- serialize to pickle (GeoJSON-like dicts) ----
    payload = {
        "extent": extent,
        "geoms": [mapping(clipped_merged)] if clipped_merged.geom_type != "GeometryCollection"
                else [mapping(x) for x in clipped_merged.geoms]
    }

    with open("gshhs_full_slc40_clipped.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("Saved:", len(payload["geoms"]), "geoms to gshhs_full_slc40_clipped.pkl")

def build_map(args,
                ds=xr.Dataset(),
                targets=[],
                ltg_probs=[],
                lons=[],
                lats=[],
                stime=np.datetime64(),
                lc_y1=0,
                lc_x1=0,
                lc_y2=0,
                lc_x2=0):

    # Create geodesic object
    geod = Geod(ellps="WGS84")
    lon0 = -80.5774#SLC 40 coordinates
    lat0 = 28.5619
    radius_m = 5 * 1852 #5 nm

    # Generate circle points
    azimuths = np.linspace(0, 360, 361)
    lons_circle = []
    lats_circle = []

    for az in azimuths:
        lon, lat, _ = geod.fwd(lon0, lat0, az, radius_m)
        lons_circle.append(lon)
        lats_circle.append(lat)

    #load the presaved coastline data from build_save_coast() function
    with open("./gshhs_full_slc40_clipped.pkl", "rb") as f:
        payload = pickle.load(f)
    gshhs_geoms = [shape(d) for d in payload["geoms"]]
    extent = payload["extent"]

    # Explicit norms (use your true expected ranges)
    contour_levels = [10,20,30,40,50,60,70,80,90]
    contour_linewidths = [1,1.1,1.3,1.5,1.7,1.9,2.0,2.5,3.0]
    subplot_labels  = ['(a)','(b)','(c)','(d)']

    lons = ((lons + 180) % 360) - 180
    lon_box = [lons[0,0],lons[0,-1],lons[-1,-1],lons[-1,0],lons[0,0]]
    lat_box = [lats[0,0],lats[0,-1],lats[-1,-1],lats[-1,0],lats[0,0]]

    dt = np.timedelta64(15,'m')

    subtitle_dict = {}
    for t in range(4):
        ts = pd.Timestamp(stime)
        minute = ts.minute
        hour = ts.hour
        day = ts.day
        month = ts.month
        year = ts.year
        month_str = args.months_str[month-1]
        temp_dt = stime+dt
        temp_ts = pd.Timestamp(temp_dt)
        temp_hour = temp_ts.hour
        temp_minute = temp_ts.minute
        if t==0:
            file_str ='%s%s%s%s%s'%(f"{year:04}",f"{month:02}",f"{day:02}",f"{hour:02}",f"{minute:02}")
        subtitle_str = '%s:%s - %s:%s UTC %s %s %s'%(f"{hour:02}",f"{minute:02}",f"{temp_hour:02}",f"{temp_minute:02}",f"{day:02}",month_str,f"{year:04}")
        print(subtitle_str)
        subtitle_dict.update({t:subtitle_str})
        stime=temp_dt
    
    efm_loc_df = pickle.load(open('./efm_loc_df.pkl','rb'))
    site_names = efm_loc_df['SiteName'].values
    efm_lats = efm_loc_df['Latitude'].values
    efm_lons = efm_loc_df['Longitude'].values
    efm_maxs = []
    for r,site_name in enumerate(site_names):
        print(r,site_name)
        efm_data = ds[site_name].values
        efm_max_abs = efm_data[:,1]#max_abs
        print(efm_max_abs.shape)
        efm_max_abs[efm_max_abs==EFM_MASK_VALUE]=np.nan
        print(np.nanmax(efm_max_abs))
        efm_maxs.append(np.nanmax(efm_max_abs))
    
    efm_cmap = plt.cm.plasma
    efm_norm = mcolors.Normalize(vmin=np.nanmin(efm_maxs),vmax=np.nanmax(efm_maxs))

    label_idx = 0
    ltg_probs = np.squeeze(ltg_probs)
    targets = np.squeeze(targets)
    fig, axes = plt.subplots(nrows=1,ncols=4,figsize=(30,10),subplot_kw={'projection':ccrs.PlateCarree()})
    for t,t_key in enumerate(['t0','t1','t2','t3']):
        axes[t].set_facecolor("lightblue")
        axes[t].add_geometries(gshhs_geoms,crs=ccrs.PlateCarree(),facecolor="tan",edgecolor="0.25",linewidth=0.6,zorder=4)
        plot_targets = np.ma.masked_where(targets[t,:,:] == 0, targets[t,:,:])
        plot_probs = np.squeeze(ltg_probs[t,:,:])
        pr_auc = average_precision_score(np.ravel(targets[t,:,:]), np.ravel(plot_probs))
        max_prob = np.max(plot_probs)*100
        axes[t].pcolormesh(lons,lats,plot_targets,alpha=.6,cmap='gray',transform=ccrs.PlateCarree(),zorder=5)
        axes[t].contour(lons,lats,np.squeeze(ltg_probs[t,:,:]*100),cmap='viridis',linewidths=contour_linewidths,levels=contour_levels,transform=ccrs.PlateCarree(),zorder=6)
        axes[t].plot(lon_box,lat_box,transform=ccrs.PlateCarree(),color='black',linewidth=1,linestyle='dashed',zorder=7)
        axes[t].plot(lons_circle,lats_circle,color='black',linewidth=0.5,transform=ccrs.PlateCarree(),zorder=10)
        axes[t].scatter(lon0,lat0,s=20,color='blue',transform=ccrs.PlateCarree(),zorder=12)
        if t==0:
            sc=axes[t].scatter(efm_lons,efm_lats,c=efm_maxs,s=120,cmap=efm_cmap,norm=efm_norm,edgecolor='white',linewidth=1.5,transform=ccrs.PlateCarree(),zorder=10)
        axes[t].text(lon0-.04,lat0+.04,'SLC-40',fontsize=18,rotation=0,weight='bold',transform=ccrs.PlateCarree(),zorder=14)
        axes[t].text(lon_box[3]-.1,lat_box[3]-.015,subplot_labels[t],fontsize=18,weight='bold',transform=ccrs.PlateCarree(),zorder=15)
        axes[t].set_title(subtitle_dict[t],fontsize=18)
        axes[t].text(lon_box[0],lat_box[0]-.07,'Max: %s'%(f"{max_prob:.02f}"),weight='bold',fontsize=18,transform=ccrs.PlateCarree(),zorder=17)
        axes[t].text(lon_box[2]-.2,lat_box[2]+.05,'AUC-PR: %s'%f"{pr_auc:.02f}",weight='bold',fontsize=18,transform=ccrs.PlateCarree(),zorder=18)

        if t==0:
            fig1, axes1 = plt.subplots(nrows=1,ncols=1,figsize=(10,10),subplot_kw={'projection':ccrs.PlateCarree()})
            cf = axes1.contourf(lons,lats,np.squeeze(ltg_probs[t,:,:]*100),cmap='viridis',levels=contour_levels,transform=ccrs.PlateCarree(),zorder=6)
            contour_ref = cf
            plt.close()
    
    # Give the bottom of the figure more room
    fig.subplots_adjust(bottom=0.28)

    # ---- Lightning probability colorbar ----
    prob_cmap = plt.cm.viridis
    prob_norm = mcolors.BoundaryNorm(contour_levels, prob_cmap.N)

    prob_sm = mpl.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
    prob_sm.set_array([])

    # centered colorbars
    cax_prob = fig.add_axes([0.30, 0.17, 0.40, 0.025])
    cax_efm  = fig.add_axes([0.30, 0.08, 0.40, 0.025])

    cbar = fig.colorbar(
        prob_sm,
        cax=cax_prob,
        orientation="horizontal",
        ticks=contour_levels
    )
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Lightning Probability (%)", fontsize=18, labelpad=14)
    cbar.ax.tick_params(labelsize=18)

    # ---- EFM colorbar ----
    efm_cbar = fig.colorbar(
        sc,
        cax=cax_efm,
        orientation="horizontal"
    )
    efm_cbar.set_label(r"Max. $|\nabla \Phi|$ Input (V/m)", fontsize=18, labelpad=14)
    efm_cbar.ax.tick_params(labelsize=18)

    plt.savefig('./case_study_images/%s.png'%file_str,dpi=200)
    plt.savefig('./case_study_images/%s.pdf'%file_str,dpi=200)
    plt.close()

            
def main():

    """
    This method is modified hourly. For more stagnant code, reference the functions above. 
    """

    

    args = preamble()
    proj_dir = args.results_path+args.project+'/'
    exp_name = args.exp_name
    model_dir = proj_dir+exp_name+'/'
    model_path = model_dir+'model.keras'

    y_dict = pickle.load(open(model_dir+'labels_prob_outputs.pkl','rb'))
    y_true = y_dict['y_true']
    y_pred_probs = y_dict['y_pred']
    cal_dict = pickle.load(open(model_dir+'calibration_dict.pkl','rb'))

    nc_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/'
    fname = 'LC_Forecast_Final_log1p_GLM_VII_2024.nc'
    test_ds = xr.open_dataset(nc_dir+fname,engine='netcdf4')
    sample_times = test_ds['sample_time'].values

    lon = test_ds['lon'].values[24:40,24:40]
    lat = test_ds['lat'].values[24:40,24:40]
    
    for s,st in enumerate(sample_times):
        if s>=1:
            case_1_ds = test_ds.sel(sample_time=st)
            y1_true = y_true[s,:,:,:]
            y1_pred = y_pred_probs[s,:,:,:]

            st = st+np.timedelta64(1,'h')
            build_map(args=args,
                        ds=case_1_ds,
                        targets=y1_true,
                        ltg_probs=y1_pred,
                        lons=lon,
                        lats=lat,
                        stime=st)
            del case_1_ds, y1_true, y1_pred
    
def model_arch_targets(args):
    # Create geodesic object
    geod = Geod(ellps="WGS84")
    lon0 = -80.5774#SLC 40 coordinates
    lat0 = 28.5619
    radius_m = 5 * 1852 #5 nm

    # Generate circle points
    azimuths = np.linspace(0, 360, 361)
    lons_circle = []
    lats_circle = []

    for az in azimuths:
        lon, lat, _ = geod.fwd(lon0, lat0, az, radius_m)
        lons_circle.append(lon)
        lats_circle.append(lat)

    case_1_dt = np.datetime64('2024-06-06T22:45')
    dt = np.timedelta64(15,'m')

    stime=case_1_dt
    subtitle_dict = {}
    for t in range(4):
        ts = pd.Timestamp(stime)
        minute = ts.minute
        hour = ts.hour
        day = ts.day
        month = ts.month
        year = ts.year
        month_str = args.months_str[month-1]
        temp_dt = stime+dt
        temp_ts = pd.Timestamp(temp_dt)
        temp_hour = temp_ts.hour
        temp_minute = temp_ts.minute
        if t==0:
            file_str ='%s%s%s%s%s'%(f"{year:04}",f"{month:02}",f"{day:02}",f"{hour:02}",f"{minute:02}")
        subtitle_str = '%s:%s - %s:%s UTC %s %s %s'%(f"{hour:02}",f"{minute:02}",f"{temp_hour:02}",f"{temp_minute:02}",f"{day:02}",month_str,f"{year:04}")
        print(subtitle_str)
        subtitle_dict.update({t:subtitle_str})
        stime=temp_dt

    nc_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/'
    fname = 'LC_Forecast_Final_log1p_GLM_VII_2024.nc'
    test_ds = xr.open_dataset(nc_dir+fname,engine='netcdf4')
    ds2 = test_ds.sel(sample_time=case_1_dt)
    target = ds2['targets'].values[:,24:40,24:40]
    print(target.shape)
    with open("./gshhs_full_slc40_clipped.pkl", "rb") as f:
        payload = pickle.load(f)
    gshhs_geoms = [shape(d) for d in payload["geoms"]]
    extent = payload["extent"]
    lon = ds2['lon'].values[24:40,24:40]
    lat = ds2['lat'].values[24:40,24:40]

    fig, axes = plt.subplots(nrows=1,ncols=4,figsize=(30,10),subplot_kw={'projection':ccrs.PlateCarree()})
    for t,t_key in enumerate(['t0','t1','t2','t3']):
        axes[t].set_facecolor("lightblue")
        axes[t].add_geometries(gshhs_geoms,crs=ccrs.PlateCarree(),facecolor="tan",edgecolor="0.25",linewidth=0.6,zorder=4)
        plot_targets = np.ma.masked_where(target[t,:,:] == 0, target[t,:,:])
        axes[t].pcolormesh(lon,lat,plot_targets,alpha=.6,cmap='gray',transform=ccrs.PlateCarree(),zorder=5)
        axes[t].set_title(subtitle_dict[t],fontsize=18)
        lon_box = [lon[0,0],lon[0,-1],lon[-1,-1],lon[-1,0],lon[0,0]]
        lat_box = [lat[0,0],lat[0,-1],lat[-1,-1],lat[-1,0],lat[0,0]]
        axes[t].plot(lon_box,lat_box,transform=ccrs.PlateCarree(),color='black',linewidth=1,linestyle='dashed',zorder=7)
        axes[t].plot(lons_circle,lats_circle,color='black',linewidth=0.5,transform=ccrs.PlateCarree(),zorder=10)
        axes[t].text(lon0-.04,lat0+.04,'SLC-40',fontsize=18,rotation=0,weight='bold',transform=ccrs.PlateCarree(),zorder=14)
        axes[t].scatter(lon0,lat0,s=20,color='blue',transform=ccrs.PlateCarree(),zorder=12)
    plt.tight_layout()
    plt.savefig('./model_arch_targets.png')
    plt.close()



if __name__=='__main__':
    parser=create_parser()
    args = parser.parse_args()
    main()
    # model_arch_targets(args=args)