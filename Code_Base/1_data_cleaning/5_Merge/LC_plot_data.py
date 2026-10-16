import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
from LC_parser import *
import seaborn as sns

EFM_MASK_VALUE=-999999.0
ZDR_MASK_VALUE=-99

def MERLIN_visualize(year='2022'):
    
    print('visualizing the MERLIN_data')
    load_dir = '/scratch/bmac87/'
    ds = xr.open_dataset(load_dir+'MERLIN_Observed_%s.nc'%year)
    ds_gt100 = ds.where(ds['cg'].sum(dim=['merlin_t','x','y'])>=100,drop=True)
    ds_final = ds_gt100.where(ds_gt100['cc'].sum(dim=['merlin_t','x','y'])>0,drop=True)
    lat = ds['lat'].values
    lon = ds['lon'].values
    rand_idx = np.random.choice(ds_final.sizes["sample_time"], size=20, replace=False)
    ds_sampled = ds_final.isel(sample_time=rand_idx)
    valid_times = ds_sampled['sample_time'].values

    for v,vt in enumerate(valid_times):
        ltg_ds = ds_sampled.sel(sample_time=vt)
        cc_np = ltg_ds['cc'].values
        cc_binary = np.copy(cc_np)
        cc_binary[cc_binary>=1.0] = 1.0
        cc_binary[cc_binary<1.0] = 0.0

        cg_np = ltg_ds['cg'].values
        cg_binary = np.copy(cg_np)
        cg_binary[cg_binary>=1.0] = 1.0
        cg_binary[cg_binary<1.0] = 0.0

        fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(20,10),subplot_kw={'projection':ccrs.PlateCarree()})
        for t in range(4):
            axes[0,t].pcolormesh(lon,lat,cc_binary[t,:,:])
            axes[0,t].coastlines()
            axes[1,t].pcolormesh(lon,lat,cg_binary[t,:,:])
            axes[1,t].coastlines()
        plt.savefig('./sample_images/MERLIN_%s_%s.png'%(year,v))
        plt.close()

def visualize_samples(year='2022'):

    print('visualizing the data for year:',year)
    ds = xr.open_dataset('/scratch/bmac87/All_LC_Inputs_%s.nc'%year)
    rand_idx = np.random.choice(ds.sizes["sample_time"], size=5, replace=False)
    ds_sampled = ds.isel(sample_time=rand_idx)
    print(ds_sampled)

    sample_times = ds_sampled['sample_time'].values
    lat = ds_sampled['lat'].values
    lon = ds_sampled['lon'].values
    hrrr_features = ds_sampled['hrrr_feature'].values
    efm_stats = ds_sampled['stats'].values

    for s,st in enumerate(sample_times):
        temp_ds = ds_sampled.sel(sample_time=st)
        prior_times = temp_ds['prior_times'].values

        plot_2d = ['hrrr_data','group_energy','group_area',
            'low_zdr','mid_zdr','high_zdr','Reflectivity_-10C_00.50',
            'Reflectivity_-15C_00.50','Reflectivity_-20C_00.50','Reflectivity_-5C_00.50',
            'Reflectivity_0C_00.50','VII_00.50','VIL_00.50']

        for p in range(len(plot_2d)):
            if p==0:
                hrrr_vts = temp_ds['hrrr_valid'].values
                plot_data = temp_ds[plot_2d[p]].values
                fig,axes = plt.subplots(nrows=9,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(20,90))
                for r in range(9):#feature
                    for c in range(4):#forecast time
                        axes[r,c].pcolormesh(lon,lat,plot_data[c,:,:,r])
                        axes[r,c].coastlines()
                        if c==0:
                            axes[r,c].set_ylabel(hrrr_features[r])
                        if r==0:
                            ts = pd.Timestamp(hrrr_vts[c])
                            minute = f"{ts.minute:02}"
                            hour = f"{ts.hour:02}"
                            day = f"{ts.day:02}"
                            month = ts.month
                            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                            month = months[month-1]
                            year = f"{ts.year:04}"
                            title_str = '%s:%s UTC %s %s %s'%(hour,minute,day,month,year)
                            axes[r,c].set_title(title_str,fontsize=24)
                            del ts, minute, hour, day, month, months, year
                plt.tight_layout()
                plt.savefig('./sample_images/hrrr_forecast_%s.png'%s)
                plt.close()
            else:
                fig,axes = plt.subplots(nrows=1,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(20,5))
                plot_data = temp_ds[plot_2d[p]].values
                for c in range(4):
                    axes[c].pcolormesh(lon,lat,plot_data[c,:,:])
                    axes[c].coastlines()
                    ts = pd.Timestamp(prior_times[c])
                    minute = f"{ts.minute:02}"
                    hour = f"{ts.hour:02}"
                    day = f"{ts.day:02}"
                    month = ts.month
                    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                    month = months[month-1]
                    year = f"{ts.year:04}"
                    title_str = '%s:%s UTC %s %s %s'%(hour,minute,day,month,year)
                    axes[c].set_title(title_str,fontsize=24)
                    del ts, minute, hour, day, month, months, year
                plt.savefig('./sample_images/%s_%s.png'%(plot_2d[p],s))
                plt.close()

        plot_time = ['FM01','FM02','FM04','FM05','FM06','FM07',
                    'FM08','FM09','FM10','FM11','FM12','FM14',
                    'FM15','FM16','FM17','FM18','FM19','FM20',
                    'FM21','FM22','FM24','FM25','FM26','FM27',
                    'FM28','FM29','FM30','FM31','FM32','FM34','FM35']
        
        for l in range(len(plot_time)):#plot each EFMs four stats
            plot_data = temp_ds[plot_time[l]].values
            print(plot_data.shape)

            fig,axes=plt.subplots(nrows=4,ncols=4,figsize=(30,30))
            for r in range(4):#stats
                ymin = np.min(np.min(plot_data[:,:,r]))
                ymax = np.max(np.max(plot_data[:,:,r]))

                for c in range(4):#time
                    axes[r,c].plot(plot_data[c,:,r])
                    axes[r,c].set_ylim([ymin,ymax])
                    ts = pd.Timestamp(prior_times[c])
                    minute = f"{ts.minute:02}"
                    hour = f"{ts.hour:02}"
                    day = f"{ts.day:02}"
                    month = ts.month
                    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                    month = months[month-1]
                    year = f"{ts.year:04}"
                    label_str = '%s:%s UTC %s %s %s'%(hour,minute,day,month,year)
                    if c==0:
                        axes[r,c].set_ylabel(efm_stats[r],fontsize=24)
                    if r==3:
                        axes[r,c].set_xlabel('# of 5s intervals')
                    if r==0:
                        axes[r,c].set_title(label_str,fontsize=24)
                    axes[r,c].set_xlim([0,181])
                    axes[r,c].set_xticks([0,50,100,150],['0','50','100','150'],fontsize=24)
                    axes[r,c].tick_params(axis='y', labelsize=24)
                    axes[r,c].grid('on')
            plt.suptitle(plot_time[l],fontsize=36)
            plt.tight_layout()
            plt.savefig('./sample_images/%s_%s'%(plot_time[l],s))
            plt.close()

def corr_mat_plot(args,year='2021'):
    print('generating the correlation plot for %s'%year)
    load_fname = '/scratch/bmac87/GLM_MRMS_all_%s.nc'%year
    ds = xr.open_dataset(load_fname,engine='netcdf4')

    vars = args.GLM_keys+args.Z_keys+args.Zdr_keys+args.VI_keys
    stacked = ds.stack(all_points=("sample_time", "tt", "y", "x"))
    stacked_sel = stacked.where(stacked['Reflectivity_0C_00.50'] > 0, drop=True)
    stacked2 = stacked_sel[vars]
    del stacked, stacked_sel
    
    # 2. Convert to DataFrame (variables become columns)
    df = stacked2.to_dataframe()
    df = df[vars]

    # 3. Drop NaNs (important)
    df = df.dropna()

    # 4. Compute correlation matrix
    corr = df.corr()

    # 5. Plot
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()
    plt.savefig('sns_corr_radar_glm.png',dpi=800)
    plt.close()

def hist_input_2_target(args):

    nc_dir ='/scratch/bmac87/'
    fname = 'LC_Forecast_total_ds.nc'
    ds = xr.open_dataset(nc_dir+fname,engine='netcdf4')
    print(ds)
    for key in ds:
        print(key)
    target_np = ds['targets'].isel(merlin_t=0).values
    target_np = target_np[24:40,24:40].ravel()

    hrrr_vars = ['mslp','td_2m','temp_2m','precip_rate','divergence_10m']
    for hv in hrrr_vars:
        data_np = ds[hv].isel(ht=1).values[24:40,24:40].ravel()
        print(hv)
        print(np.nanmax(data_np),np.nanmin(data_np))
        vals0 = data_np[target_np==0]
        vals1 = data_np[target_np==1]

        plt.figure(figsize=(10,10))
        plt.hist(vals0, bins=50, alpha=0.5, label='No Lightning', density=True,zorder=0)
        plt.hist(vals1, bins=50, alpha=0.5, label='Lightning', density=True,zorder=1)
        plt.grid('on')
        plt.xlabel(hv,fontsize=18)
        plt.ylabel('Count Density',fontsize=18)
        plt.title('Histogram of %s'%(key),fontsize=24)
        plt.legend()
        plt.show()
        plt.savefig('./test_images/%s_hist_2_target.png'%(hv),dpi=800)
        del data_np
    
    for key in args.Z_keys:

        data_np = ds[key].isel(tt=3).values[24:40,24:40].ravel()
        data_np[data_np<0] = np.nan

        vals0 = data_np[target_np==0]
        vals1 = data_np[target_np==1]

        plt.figure(figsize=(10,10))
        plt.hist(vals0, bins=50, alpha=0.5, label='No Lightning', density=True,zorder=0)
        plt.hist(vals1, bins=50, alpha=0.5, label='Lightning', density=True,zorder=1)
        plt.grid('on')
        plt.xlabel(key,fontsize=18)
        plt.ylabel('Count Density',fontsize=18)
        plt.title('Histogram of %s'%(key),fontsize=24)
        plt.legend()
        plt.show()
        plt.savefig('./test_images/%s_hist_2_target.png'%(key),dpi=800)
        del data_np
    
    for key in args.VI_keys:
        print(key)
        data_np = ds[key].isel(tt=3).values[24:40,24:40].ravel()
        data_np[data_np<=0]=np.nan

        print(data_np.shape)
        print(np.sum(np.isnan(data_np)))
        print(np.nanmax(data_np))
        print(np.nanmin(data_np))

        vals0 = data_np[target_np==0]
        vals1 = data_np[target_np==1]

        plt.figure(figsize=(10,10))
        plt.hist(vals0, bins=50, alpha=0.5, label='No Lightning', density=True,zorder=0)
        plt.hist(vals1, bins=50, alpha=0.5, label='Lightning', density=True,zorder=1)
        plt.grid('on')
        plt.xlabel(key,fontsize=18)
        plt.ylabel('Count Density',fontsize=18)
        plt.title('Histogram of %s'%(key),fontsize=24)
        plt.legend()
        plt.show()
        plt.savefig('./test_images/%s_hist_2_target.png'%(key),dpi=800)
        del data_np
    
    for key in args.GLM_keys:
        print(key)
        data_np = ds[key].isel(tt=3).values[24:40,24:40].ravel()
        data_np[data_np<=0]=np.nan

        print(np.nanmin(data_np), np.nanmax(data_np))
        print(np.nanmin(np.log1p(data_np)), np.nanmax(np.log1p(data_np)))
        if key=='group_energy_log':
            ge = ds['group_energy'].values
            print(np.nanmin(ge), np.nanmax(ge))
        vals0 = data_np[target_np==0]
        vals1 = data_np[target_np==1]

        plt.figure(figsize=(10,10))
        plt.hist(vals0, bins=50, alpha=0.5, label='No Lightning', density=True,zorder=0)
        plt.hist(vals1, bins=50, alpha=0.5, label='Lightning', density=True,zorder=1)
        plt.grid('on')
        plt.xlabel(key,fontsize=18)
        plt.ylabel('Count Density',fontsize=18)
        plt.title('Histogram of %s'%(key),fontsize=24)
        plt.legend()
        plt.show()
        plt.savefig('./test_images/%s_hist_2_target.png'%(key),dpi=800)
        del data_np

    for key in args.Zdr_keys:
        print(key)
        data_np = ds[key].isel(tt=3).values[24:40,24:40].ravel()
        data_np[data_np==ZDR_MASK_VALUE] = np.nan

        vals0 = data_np[target_np==0]
        vals1 = data_np[target_np==1]

        plt.figure(figsize=(10,10))
        plt.hist(vals0, bins=50, alpha=0.5, label='No Lightning', density=True,zorder=0)
        plt.hist(vals1, bins=50, alpha=0.5, label='Lightning', density=True,zorder=1)
        plt.grid('on')
        plt.xlabel(key,fontsize=18)
        plt.ylabel('Count Density',fontsize=18)
        plt.title('Histogram of %s'%(key),fontsize=24)
        plt.legend()
        plt.show()
        plt.savefig('./test_images/%s_hist_2_target.png'%(key),dpi=800)
        del data_np
    
def plot_efm_histos(args):
    efm_keys = args.efm_keys
    efm_distances = xr.open_dataset('../1_EFM/efm_distance_proximity.nc',engine='netcdf4')
    print(efm_distances)
    site_names = efm_distances['site_names'].values
    print(site_names)
    
    distances = efm_distances['distances'].values
    lon = efm_distances['lon'].values
    lat = efm_distances['lat'].values
    for e in range(len(site_names)):
        print(e)
        one_dist = distances[:,:,e]
        
    #get the minimum distance idx pairs on the target domain
    
def aggressive_data_test():
    print('aggressive data test')
    hrrr_features = args.hrrr_features
    data_dir = '/scratch/bmac87/'
    ds = xr.open_dataset(data_dir+'LC_Forecast_total_ds.nc',engine='netcdf4')
    lat = ds['lat'].values
    lon = ds['lon'].values
    sample_times = ds['sample_time'].values
    
    glm_ds = ds[args.GLM_keys]
    zdr_ds = ds[args.Zdr_keys]
    z_ds = ds[args.Z_keys]
    efm_ds = ds[args.efm_keys]

    
    hrrr_ds = ds[hrrr_features]
    fig,axes = plt.subplots(nrows=len(hrrr_features),ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,11))
    for f in range(len(hrrr_features)):
        plot_data=hrrr_ds[hrrr_features[f]]
        for img in range(len(sample_times)):
            if img==2:
                sample_data=np.squeeze(plot_data[img,:,:,:])
                for t in range(4):
                    axes[f,t].pcolormesh(lon,lat,sample_data[t,:,:])
                    axes[f,t].coastlines()
                    if t==0:
                        axes[f,t].set_ylabel(hrrr_features[f])
                del sample_data
        del plot_data
    plt.tight_layout()
    plt.suptitle(sample_times[img])

    save_dir = '/scratch/bmac87/aggressive_data_test/hrrr/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    plt.savefig('%shrrr_%s.png'%(save_dir,img))
    plt.close()

    all_area_data = glm_ds['group_area'].values
    print(all_area_data.shape)
    all_area_data[all_area_data<=0] = np.nan
    for img in range(len(sample_times)):
        if img==2:
            area_data = np.squeeze(all_area_data[img,:,:,:])
            print(area_data.shape)
            fig,axes = plt.subplots(nrows=1,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,5))
            for t in range(4):
                axes[t].pcolormesh(lon,lat,area_data[t,:,:])
                axes[t].coastlines()
            plt.tight_layout()
            plt.suptitle(sample_times[img])
            save_dir = '/scratch/bmac87/aggressive_data_test/glm/'
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig('%sglm_%s.png'%(save_dir,img))
            plt.close()
            del all_area_data,area_data

            # temp_ds = zdr_ds.sel(sample_time=sample_times[img])
            # low_data = temp_ds['low_zdr'].values
            # low_data[low_data<=0] = np.nan

            # mid_data = temp_ds['mid_zdr'].values
            # mid_data[mid_data<=0] = np.nan

            # high_data = temp_ds['high_zdr'].values
            # high_data[high_data<=0] = np.nan

            # fig,axes = plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,7))
            # for t in range(4):
            #     axes[0,t].pcolormesh(lon,lat,low_data[t,:,:])
            #     axes[0,t].coastlines()
            #     axes[1,t].pcolormesh(lon,lat,mid_data[t,:,:])
            #     axes[1,t].coastlines()
            #     axes[2,t].pcolormesh(lon,lat,high_data[t,:,:])
            #     axes[2,t].coastlines()
            # plt.tight_layout()
            # plt.suptitle(sample_times[img])
            # save_dir = '/scratch/bmac87/aggressive_data_test/%s/zdr/'%year
            # if os.path.isdir(save_dir)==False:
            #     os.makedirs(save_dir)
            # plt.savefig('%szdr_%s_%s.png'%(save_dir,year,img))
            # plt.close()
            # del low_data, mid_data, high_data, temp_ds

            # temp_ds = z_ds.sel(sample_time=sample_times[img])
            # fig,axes = plt.subplots(nrows=len(args.Z_keys),ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,11))
            # for k,key in enumerate(args.Z_keys):
            #     plot_data = temp_ds[key].values
            #     plot_data[plot_data<=0] = np.nan
            #     for t in range(4):
            #         axes[k,t].pcolormesh(lon,lat,plot_data[t,:,:])
            #         axes[k,t].coastlines()
            # plt.suptitle(sample_times[img])
            # save_dir = '/scratch/bmac87/aggressive_data_test/%s/isothermal_z/'%year
            # if os.path.isdir(save_dir)==False:
            #     os.makedirs(save_dir)
            # plt.savefig('%sisothermal_z_%s_%s.png'%(save_dir,year,img))
            # plt.close()
    
            # temp_ds = efm_ds.sel(sample_time=sample_times[img])
            # fig,axes = plt.subplots(nrows=len(args.efm_ts_keys),ncols=4,figsize=(8.5,44))
            # for k,key in enumerate(args.efm_ts_keys):
            #     plot_data = temp_ds[key].values
            #     for t in range(4):
            #         axes[k,t].plot(plot_data[t,:,0])
            # plt.tight_layout()
            # plt.suptitle(sample_times[img])
            # save_dir = '/scratch/bmac87/aggressive_data_test/%s/efm_median/'%year
            # if os.path.isdir(save_dir)==False:
            #     os.makedirs(save_dir)
            # plt.savefig('%smedian_efm_%s_%s.png'%(save_dir,year,img))
            # plt.close()
            # del plot_data

            # fig,axes = plt.subplots(nrows=len(args.efm_ts_keys),ncols=4,figsize=(8.5,44))
            # for k,key in enumerate(args.efm_ts_keys):
            #     plot_data = temp_ds[key].values
            #     for t in range(4):
            #         axes[k,t].plot(plot_data[t,:,3])
            # plt.tight_layout()
            # plt.suptitle(sample_times[img])
            # save_dir = '/scratch/bmac87/aggressive_data_test/%s/efm_std/'%year
            # if os.path.isdir(save_dir)==False:
            #     os.makedirs(save_dir)
            # plt.savefig('%sstd_efm_%s_%s.png'%(save_dir,year,img))
            # plt.close()
            # del temp_ds, plot_data


if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    hist_input_2_target(args=args)
    # plot_efm_histos(args)