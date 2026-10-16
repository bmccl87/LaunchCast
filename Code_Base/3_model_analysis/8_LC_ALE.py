import tensorflow as tf
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from LC_parser import create_parser
from LC_model_64216 import Stack, FlattenStations, UnflattenStations
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

EFM_MASK_VALUE=-999999.0
ZDR_MASK_VALUE=-99
ZDR_FILL_VALUE=-5


def collect_efm_feature_values(args,tfds, efm_key, efm_stat, mask_value=-999999.0, max_batches=None):
    vals = []
    efm_stats = args.efm_stats
    stat_idxs = {}
    for s,stat in enumerate(efm_stats):
        stat_idxs.update({stat:s})

    for b, (X, y) in enumerate(tfds):
        print('x.shape before getting stat:',efm_stat)
        x = X[efm_key].numpy()
        print(x.shape)
        stat_idx=stat_idxs[efm_stat]
        x = x[:,:,stat_idx]
        print('x.shape after getting stat')
        print(x.shape)
        print('len(x.ravel)')
        print(len(x.ravel()))
        valid = np.isfinite(x) & (x != mask_value)
        print('sum of valid')
        print(np.sum(valid))
        x = x[valid]
        print('len(x[valid])')
        print(len(x))
        vals.append(x)

        if max_batches is not None and b + 1 >= max_batches:
            break
    
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    return vals, np.nanmin(vals),np.nanmax(vals),mask_value

def collect_feature_values(args,tfds, feature_key, mask_value=-999999.0, max_batches=None):
    vals = []
    for b, (X, y) in enumerate(tfds):
        x = X[feature_key].numpy()
        valid = np.isfinite(x) & (x != mask_value)
        x = x[valid]
        vals.append(x)

        if max_batches is not None and b + 1 >= max_batches:
            break
    
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    return vals, np.nanmin(vals),np.nanmax(vals)

def compute_ale_for_feature(args,model,tfds,feature_key,quantile_edges,mask_value=-999999.0,):

    n_bins = len(quantile_edges) - 1
    ale_sum = np.zeros((n_bins, 4, 16, 16), dtype=np.float64)
    ale_count = np.zeros(n_bins, dtype=np.int64)
    batch_count=0
    for X, y in tfds:
        x_np = X[feature_key].numpy()

        valid = np.isfinite(x_np) & (x_np != mask_value)

        for k in range(n_bins):
            x_lo_full = X[feature_key].numpy().copy()
            x_hi_full = X[feature_key].numpy().copy()

            lo = quantile_edges[k]
            hi = quantile_edges[k + 1]

            in_bin = valid & (x_np >= lo) & (x_np < hi)
            if k == n_bins - 1:
                in_bin = valid & (x_np >= lo) & (x_np <= hi)
            
            if not np.any(in_bin):
                continue
            
            x_lo = x_np.copy()
            x_hi = x_np.copy()

            x_lo[in_bin] = lo
            x_hi[in_bin] = hi

            x_lo_full = x_lo
            x_hi_full = x_hi

            X_lo = {key1: val for key1, val in X.items()}
            X_hi = {key1: val for key1, val in X.items()}

            X_lo[feature_key] = tf.convert_to_tensor(x_lo_full,dtype=X[feature_key].dtype)
            X_hi[feature_key] = tf.convert_to_tensor(x_hi_full,dtype=X[feature_key].dtype)

            pred_lo = tf.nn.sigmoid(model(X_lo,training=False,)).numpy()*100
            pred_hi = tf.nn.sigmoid(model(X_hi,training=False)).numpy()*100

            diff_bin = np.squeeze(pred_hi - pred_lo)

            ale_sum[k] += np.sum(diff_bin, axis=0)
            ale_count[k] += diff_bin.shape[0]

        batch_count+=1

    save_dict = {'quantile_edges':quantile_edges,'ale_sum':ale_sum,'ale_count':ale_count}
    fsave = '%s_ALE.pkl'%(feature_key)
    pickle.dump(save_dict,open('/scratch/bmac87/ALE_v3.2/%s'%fsave,'wb'))

def compute_ale_for_efm_feature(args,model,tfds,efm_key,efm_stat,quantile_edges,mask_value=-999999.0,):

    prox_ds = xr.open_dataset('./efm_distance_proximity.nc',engine='netcdf4')
    efm_distances = prox_ds['distances']
    site_idx = np.where(efm_distances['site_names'].values==efm_key)[0]
    one_site_dist = efm_distances.sel(e=site_idx).values
    mask_5nm = 1852*5
    one_site_5nm_mask = (one_site_dist<=mask_5nm)
    one_site_5nm_mask = np.squeeze(one_site_5nm_mask)

    efm_stats = args.efm_stats
    print(efm_stats)
    stat_idxs = {}
    for s,stat in enumerate(efm_stats):
        stat_idxs.update({stat:s})
    stat_idx=stat_idxs[efm_stat]

    n_bins = len(quantile_edges) - 1
    ale_sum = np.zeros((n_bins, 4, 16, 16), dtype=np.float64)
    ale_count = np.zeros(n_bins, dtype=np.int64)
    ale_sum_mask = np.zeros(n_bins)
    ale_count_mask = np.zeros(n_bins)

    batch_count=0
    for X, y in tfds:
        print(batch_count)
        x_np = X[efm_key].numpy()
        x_np = x_np[:,:,stat_idx]
        print('compute_ale: x_np.shape')
        print(x_np.shape)
        valid = np.isfinite(x_np) & (x_np != mask_value)

        for k in range(n_bins):
            x_lo_full = X[efm_key].numpy().copy()
            x_hi_full = X[efm_key].numpy().copy()

            lo = quantile_edges[k]
            hi = quantile_edges[k + 1]

            in_bin = valid & (x_np >= lo) & (x_np < hi)
            if k == n_bins - 1:
                in_bin = valid & (x_np >= lo) & (x_np <= hi)
            
            if not np.any(in_bin):
                continue
            
            x_lo = x_np.copy()
            x_hi = x_np.copy()

            x_lo[in_bin] = lo
            x_hi[in_bin] = hi

            #needs adjusted for EFM. 
            x_lo_full[:, :, stat_idx][in_bin] = lo
            x_hi_full[:, :, stat_idx][in_bin] = hi

            X_lo = {key2: val for key2, val in X.items()}
            X_hi = {key2: val for key2, val in X.items()}

            X_lo[efm_key] = tf.convert_to_tensor(x_lo_full,dtype=X[efm_key].dtype)
            X_hi[efm_key] = tf.convert_to_tensor(x_hi_full,dtype=X[efm_key].dtype)

            pred_lo = tf.nn.sigmoid(model(X_lo,training=False,)).numpy()*100
            pred_hi = tf.nn.sigmoid(model(X_hi,training=False)).numpy()*100

            diff_bin = np.squeeze(pred_hi - pred_lo)
            
            ale_sum[k] += np.sum(diff_bin, axis=0)
            ale_count[k] += diff_bin.shape[0]

            diff_t0 = diff_bin[:, 0, :, :]      # (batch, 16, 16)
            print('diff_t0.shape')
            print(diff_t0.shape)
            masked_diff = diff_t0[:, one_site_5nm_mask]  # (batch, n_mask_pixels)
            print('masked_diff.shape')
            print(masked_diff.shape)
            print(masked_diff.size)

            ale_sum_mask[k] += np.sum(masked_diff)
            ale_count_mask[k] += masked_diff.size

        batch_count+=1

    save_dict = {'quantile_edges':quantile_edges,'ale_sum':ale_sum,'ale_count':ale_count}
    fsave = '%s_%s_ALE.pkl'%(efm_key,efm_stat)
    pickle.dump(save_dict,open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/ALE_v3.2/%s'%fsave,'wb'))

    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/ALE_v3.2_win5nm/'
    save_dict2 = {'quantile_edges':quantile_edges,'ale_sum_mask':ale_sum_mask,'ale_count_mask':ale_count_mask}
    pickle.dump(save_dict2,open(save_dir+fsave,'wb'))



def drive_ale_all(args):
    print('working on the ALE')

    # load test dataset
    load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/'
    test_tfds = tf.data.Dataset.load(load_dir+'LC_All_2024.tfds')
    test_tfds = test_tfds.batch(32)

    exp_name = args.exp_name
    exp_dir = '%s%s/'%(args.results_path,args.project)

    print('loading the model')
    model_dir = exp_dir+exp_name+'/'
    model_fname = 'model.keras'
    print(model_dir+model_fname)
    model = tf.keras.models.load_model(model_dir+model_fname,custom_objects={"Stack": Stack,"FlattenStations":FlattenStations,"UnflattenStations":UnflattenStations},compile=False)
    model.trainable = False

    t_keys = args.t_keys
    Z_keys = args.Z_keys
    Zdr_keys = args.Zdr_keys
    VI_keys = args.VI_keys

    input_names = []
    for t_key in t_keys:
        for key in (Z_keys+Zdr_keys+VI_keys):
            input_names.append('MRMS_%s_%s'%(key,t_key))

    hrrr_keys = args.hrrr_features
    for t_key in t_keys:
        for key in hrrr_keys:
            input_names.append('HRRR_%s_%s'%(key,t_key))


    GLM_keys = args.GLM_keys
    for t_key in t_keys:
        for key in GLM_keys:
            input_names.append('GLM_%s_%s'%(key,t_key))

    #load efm keys
    efm_keys = args.efm_keys
    for key in efm_keys:
        input_names.append(key)
    
    for n,name in enumerate(input_names):
        print(n,name)

    n_bins = 20
    max_batches = None
    name = input_names[args.exp]

    if 'zdr' in name:
        e_vals,e_min,e_max = collect_feature_values(args=args,tfds=test_tfds, feature_key=name, mask_value=-5.0, max_batches=max_batches)
        quantile_edges = np.nanquantile(e_vals, np.linspace(0, 1, n_bins + 1))
        quantile_edges = np.unique(quantile_edges)
        if len(quantile_edges) < 3:
            print("Skipping because too few unique bin edges:", name)
        else:
            compute_ale_for_feature(args=args,model=model,tfds=test_tfds,feature_key=name,quantile_edges=quantile_edges,mask_value=-5.0,)
    
    elif 'Reflectivity' in name:
        print(name)
        dict_key = name[5:-3]
        print(dict_key)
        norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
        mu = norm_stats_dict[dict_key]['mean']
        std = norm_stats_dict[dict_key]['std']
        temp_mask_value = np.float32((0.0-mu)/std)
        print('temp_mask_value',temp_mask_value)
        e_vals,e_min,e_max = collect_feature_values(args=args,tfds=test_tfds, feature_key=name, mask_value=temp_mask_value, max_batches=max_batches)
        quantile_edges = np.nanquantile(e_vals, np.linspace(0, 1, n_bins + 1))
        quantile_edges = np.unique(quantile_edges)

        if len(quantile_edges) < 3:
            print("Skipping because too few unique bin edges:", name)
        else:
            compute_ale_for_feature(args=args,model=model,tfds=test_tfds,feature_key=name, quantile_edges=quantile_edges,mask_value=temp_mask_value,)
    
    elif 'VII_00.50_log' in name:
        print(name)
        norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
        for key in norm_stats_dict:
            print(key)
        mu = norm_stats_dict['VII_00.50_log']['mean']
        std = norm_stats_dict['VII_00.50_log']['std']
        temp_mask_value = np.float32((np.log1p(0.0)-mu)/std)
        print('temp_mask_value',temp_mask_value)
        e_vals,e_min,e_max = collect_feature_values(args=args,tfds=test_tfds, feature_key=name, mask_value=temp_mask_value, max_batches=max_batches)
        quantile_edges = np.nanquantile(e_vals, np.linspace(0, 1, n_bins + 1))
        print(quantile_edges)
        print(e_min,e_max)
        quantile_edges = np.unique(quantile_edges)
        if len(quantile_edges) < 3:
            print("Skipping because too few unique bin edges:", name)
        else:
            compute_ale_for_feature(args=args,model=model,tfds=test_tfds,feature_key=name, quantile_edges=quantile_edges,mask_value=temp_mask_value,)
    
    elif 'group_area_log' in name:
        norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
        mu = norm_stats_dict['group_area_log']['mean']
        std = norm_stats_dict['group_area_log']['std']
        temp_mask_value = (np.log1p(0.0)-mu)/std
        e_vals,e_min,e_max = collect_feature_values(args=args,tfds=test_tfds, feature_key=name, mask_value=temp_mask_value, max_batches=max_batches)
        quantile_edges = np.nanquantile(e_vals, np.linspace(0, 1, n_bins + 1))
        quantile_edges = np.unique(quantile_edges)
        print(quantile_edges)
        print(e_min,e_max)
        if len(quantile_edges) < 3:
            print("Skipping because too few unique bin edges:", name)
        else:
            compute_ale_for_feature(args=args,model=model,tfds=test_tfds,feature_key=name,quantile_edges=quantile_edges,mask_value=temp_mask_value)
    
    elif 'HRRR' in name:
        e_vals,e_min,e_max = collect_feature_values(args=args,tfds=test_tfds, feature_key=name, mask_value=-999999, max_batches=max_batches)
        quantile_edges = np.nanquantile(e_vals, np.linspace(0, 1, n_bins + 1))
        quantile_edges = np.unique(quantile_edges)
        if len(quantile_edges) < 3:
            print("Skipping because too few unique bin edges:", name)
        else:
            compute_ale_for_feature(args=args,model=model,tfds=test_tfds,feature_key=name,quantile_edges=quantile_edges,mask_value=-999999.0,)
    
    elif 'FM' in name:
        efm_stats = args.efm_stats
        for s,stat in enumerate(efm_stats):
            print('efm',stat)
            e_vals,e_min,e_max,mask_value = collect_efm_feature_values(args=args,tfds=test_tfds, efm_key=name, efm_stat=stat, mask_value=-999999.0, max_batches=max_batches)
            print('min','max')
            print(e_min,e_max,mask_value)
            quantile_edges = np.nanquantile(e_vals, np.linspace(0, 1, n_bins + 1))
            quantile_edges = np.unique(quantile_edges)
            print(quantile_edges)
            if len(quantile_edges) < 3:
                print("Skipping because too few unique bin edges:", name)
                continue
            compute_ale_for_efm_feature(args=args,model=model,tfds=test_tfds,efm_key=name,efm_stat=stat,quantile_edges=quantile_edges,mask_value=-999999.0,)

def make_ale_maps(args):
    """
    I multiplied the predictions by 100 to avoid the issue here. 
    """
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/LC_Forecast_Final_log1p_GLM_VII_2024.nc',engine='netcdf4')
    efm_loc_df = pickle.load(open('./efm_loc_df.pkl','rb'))
    lat = ds['lat'].values[24:40,24:40]
    lon = ds['lon'].values[24:40,24:40]
    print('making the ale plots')
    ale_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/ALE/'
    for pkl_name in os.listdir(ale_dir):
        print(pkl_name)
        ale_dict = pickle.load(open('%s%s'%(ale_dir,pkl_name),'rb'))
        for key in ale_dict:
            print(key)
        ale_sum = ale_dict['ale_sum']
        print(ale_sum.shape)

        ale_count = ale_dict['ale_count']
        print(ale_count.shape)

        ale_effect = np.ones(ale_sum.shape)
        for b in range(ale_sum.shape[0]):
            ale_effect[b,:,:,:] = ale_sum[b,:,:,:]/ale_count[b]
        
        ale_cum_effect = np.cumsum(ale_effect,axis=0)
        ale_centered = ale_cum_effect - np.mean(ale_cum_effect,axis=0,keepdims=True)
        v = np.nanmax(np.abs(ale_centered))

        fig,axes = plt.subplots(nrows=1,ncols=4,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(40,10))
        for t in range(4):
            if 'FM' in pkl_name:
                one_efm = efm_loc_df.loc[efm_loc_df['SiteName']==pkl_name[0:4]]
                efm_lat = one_efm['Latitude']
                efm_lon = one_efm['Longitude']
                axes[t].scatter(efm_lon,efm_lat,s=50,color='black',marker='*',transform=ccrs.PlateCarree())
            im=axes[t].pcolormesh(lon,lat,ale_centered[0,t,:,:],cmap='coolwarm',vmin=-v,vmax=v,transform=ccrs.PlateCarree())
            axes[t].coastlines()
            plt.colorbar(im,ax=axes[t],label='Centered ALE')
        plt.savefig('./ALE_MAPS/ALE_%s.png'%(pkl_name[0:-4]),dpi=300)
        plt.close()

def drive_ALE_line_plots(args):

    # load_dir = '/scratch/bmac87/'
    # test_tfds = tf.data.Dataset.load(load_dir+'LC_All_2024.tfds')
    # test_tfds = test_tfds.batch(32)


    print('making the ALE line plots')
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/LC_Forecast_Final_log1p_GLM_VII_2024.nc',engine='netcdf4')
    efm_loc_df = pickle.load(open('./efm_loc_df.pkl','rb'))
    lat = ds['lat'].values[24:40,24:40]
    lon = ds['lon'].values[24:40,24:40]

    
    Z_keys = args.Z_keys
    Zdr_keys = args.Zdr_keys
    VI_keys = args.VI_keys
    radar_keys = Z_keys+Zdr_keys+VI_keys
    t_keys = args.t_keys
    input_times = ['60 - 45 min','45 - 30 min','30 - 15 min','15 - 00 min']

    for r,radar_key in enumerate(Z_keys):
        if r>=-1:
            print(r,radar_key)
            data_np = ds[radar_key].values
            for t,t_key in enumerate(t_keys):
                data_np_unique = np.ravel(data_np[:,t,:,:])
                # MRMS_Reflectivity_0C_00.50_t0_ALE.pkl
                file = 'MRMS_%s_%s_ALE.pkl'%(radar_key,t_key)
                tf_dict_key = 'MRMS_%s_%s'%(radar_key,t_key)
                norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
                mu = norm_stats_dict[radar_key]['mean']
                std = norm_stats_dict[radar_key]['std']
                temp_mask_value = np.float32((0.0-mu)/std)
                
                if '10C' in radar_key:
                    title = 'Isothermal Reflectivity (-10C) - %s Prior'%(input_times[t])
                elif '20C' in radar_key:
                    title = 'Isothermal Reflectivity (-20C) - %s Prior'%(input_times[t])
                else:
                    title = 'Isothermal Reflectivity (0C) - %s Prior'%(input_times[t])

                plot_ale_lines(args=args,file=file,
                                mu=mu,std=std,
                                unique_values=data_np_unique,xfactor=1,
                                yfactor=1,xlabel='(dBZ)',
                                ylabel='ALE (%)',title=title)
    
    for r,radar_key in enumerate(VI_keys):
        if r>=-1:
            data_np = ds[radar_key].values
            for t,t_key in enumerate(t_keys):
                data_np_unique = np.ravel(data_np[:,t,:,:])
                # MRMS_Reflectivity_0C_00.50_t0_ALE.pkl
                file = 'MRMS_%s_%s_ALE.pkl'%(radar_key,t_key)
                tf_dict_key = 'MRMS_%s_%s'%(radar_key,t_key)
                norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
                mu = norm_stats_dict[radar_key]['mean']
                std = norm_stats_dict[radar_key]['std']
                temp_mask_value = np.float32((0.0-mu)/std)
                title = 'VII - %s Prior'%(input_times[t])
                plot_ale_lines(args=args,file=file,
                                    mu=mu,std=std,
                                    unique_values=data_np_unique,xfactor=1,
                                    yfactor=1,xlabel='($kg m^{-2}$)',
                                    ylabel='ALE (%)',title=title)
    
    for r,radar_key in enumerate(Zdr_keys):
        if r>=-1:
            data_np = ds[radar_key].values
            for t,t_key in enumerate(t_keys):
                print(r,radar_key,t,t_key)
                data_np_unique = np.ravel(data_np[:,t,:,:])
                # MRMS_Reflectivity_0C_00.50_t0_ALE.pkl
                file = 'MRMS_%s_%s_ALE.pkl'%(radar_key,t_key)
                tf_dict_key = 'MRMS_%s_%s'%(radar_key,t_key)
                norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
                mu = norm_stats_dict[radar_key]['mean']
                std = norm_stats_dict[radar_key]['std']
                temp_mask_value = np.float32((0.0-mu)/std)
                
                if 'low' in file:
                    title = 'Zdr (low) - %s Prior'%(input_times[t])
                elif 'mid' in file:
                    title = 'Zdr (mid) - %s Prior'%(input_times[t])
                else:
                    title = 'Zdr (high) - %s Prior'%(input_times[t])
                
                plot_ale_lines(args=args,file=file,
                                    mu=mu,std=std,
                                    unique_values=data_np_unique,xfactor=1,
                                    yfactor=1,xlabel='(dB)',
                                    ylabel='ALE (%)',title=title)

    GLM_keys = args.GLM_keys
    for g,glm_key in enumerate(GLM_keys):
        if g>=-1:
            print(glm_key)
            data_np = ds[glm_key].values
            for t,t_key in enumerate(t_keys):
                data_np_unique = np.ravel(np.expm1(data_np[:,t,:,:]))
                file = 'GLM_%s_%s_ALE.pkl'%(glm_key,t_key)
                tf_dict_key = 'GLM_%s_%s'%(glm_key,t_key)
                norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
                mu = norm_stats_dict[glm_key]['mean']
                std = norm_stats_dict[glm_key]['std']
                temp_mask_value = np.float32((0.0-mu)/std)
                plot_ale_lines(args,file=file,mu=mu,std=std,
                                unique_values=data_np_unique,xfactor=1e9,
                                yfactor=1e-3,xlabel='Group Area/1e9 $m^{2}$',
                                ylabel = 'ALE/(1e-3) (%)',
                                title='GLM Group Area - %s Prior'%(input_times[t]))
    
    hrrr_keys = args.hrrr_features
    fcst_times = ['00-15 min Forecast','15-30 min Forecast','30-45 min Forecast','45-60 min Forecast']
    for h,hrrr_key in enumerate(hrrr_keys):
        if h>=-1:
            print(h,hrrr_key)
            data_np = ds[hrrr_key].values

            if hrrr_key=='precip_rate':
                t_keys = ['t1','t2','t3']
            else:
                t_keys = ['t0','t1','t2','t3']
            
            for t,t_key in enumerate(t_keys):
                data_np_unique = np.ravel(data_np[:,t,:,:])
                file = 'HRRR_%s_%s_ALE.pkl'%(hrrr_key,t_key)
                tf_dict_key = 'HRRR_%s_%s'%(hrrr_key,t_key)
                norm_stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
                mu = norm_stats_dict[hrrr_key]['mean']
                std = norm_stats_dict[hrrr_key]['std']
                
                if 'temp' in hrrr_key:
                    xlabel = '(K)'
                    title = 'HRRR - Temperature 2m - %s'%(fcst_times[t])
                elif 'td' in hrrr_key:
                    xlabel = '(K)'
                    title = 'HRRR - Dew Point 2m - %s'%(fcst_times[t])
                elif 'mslp' in hrrr_key:
                    xlabel = '(Pa)'
                    title = 'MSLP - %s'%(fcst_times[t])
                elif 'divergence' in hrrr_key:
                    xlabel = '($s^{-1}$)'
                    title = 'Divergence - %s'%(fcst_times[t])
                else:
                    xlabel = '$(kg m^{-2} s^{-1})$'
                    title ='Precipitation Rate - %s'%(fcst_times[t])
                
                plot_ale_lines(args=args,file=file,
                                        mu=mu,std=std,
                                        unique_values=data_np_unique,xfactor=1,
                                        yfactor=1,xlabel=xlabel,
                                        ylabel='ALE (%)',title=title)

    efm_keys = args.efm_keys
    efm_stats = args.efm_stats
    for e,efm_key in enumerate(efm_keys):
        data_np = ds[efm_key].values
        for s,stat in enumerate(efm_stats):
            print(e,efm_key,s,stat)
            one_stat_np = data_np[:,:,s]
            file = '%s_%s_ALE.pkl'%(efm_key,stat)
            norm_stats_dict = pickle.load(open('./efm_forecast_min_max.pkl','rb'))
            fm_stats_dict = norm_stats_dict[efm_key][stat]

            if stat=='max_abs' or stat=='var' or stat=='std':
                mu = fm_stats_dict['log_mean']
                std = fm_stats_dict['log_std']
            else:
                mu = fm_stats_dict['mean']
                std = fm_stats_dict['std']
            
            title = '%s - %s'%(efm_key,stat)
            if stat=='mean' or stat=='std' or stat=='max_abs' or stat=='rms':
                xlabel='$(V m^{-1})$'
            elif stat=='var':
                xlabel='$(V^2 m^{-2})$'
            else:
                xlabel=('#')
            plot_ale_lines_efm(args=args,file=file,mu=mu,std=std,unique_values=one_stat_np,
                    yfactor=1.0,xfactor=1.0,xlabel=xlabel,ylabel='ALE - (%)',title=title)

def plot_ale_lines(args,file='',mu=0.0,std=0.0,unique_values=[],
                    yfactor=1.0,xfactor=1.0,xlabel='',ylabel='',title=''):

    ale_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/ALE_v3.2/'
    ale_dict = pickle.load(open(ale_dir+file,'rb'))
    ale_sum = ale_dict['ale_sum']
    ale_bins = ale_dict['quantile_edges']
    ale_bin_centers = 0.5 * (ale_bins[:-1] + ale_bins[1:])

    if ('GLM' in file) or ('VII' in file) or ('max_abs' in file) or ('std' in file) or ('var' in file):
        bin_centers_og = np.expm1(ale_bin_centers * std + mu)
    else:
        bin_centers_og = ale_bin_centers*std + mu

    ale_count = ale_dict['ale_count']
    ale_effect = np.ones(ale_sum.shape)
    for b in range(ale_sum.shape[0]):
        ale_effect[b,:,:,:] = ale_sum[b,:,:,:]/ale_count[b]
    
    ale_cum_effect = np.cumsum(ale_effect,axis=0)
    ale_centered = ale_cum_effect - np.mean(ale_cum_effect,axis=0,keepdims=True)
    v = np.nanmax(np.abs(ale_centered))/yfactor

    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly, same as BC
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']
    fcst_times = ['00-15 min Forecast','15-30 min Forecast','30-45 min Forecast','45-60 min Forecast']

    axes.grid(zorder=0)
    hist_counts, hist_bins = np.histogram(unique_values / xfactor,bins=20,range=(bin_centers_og[0]/xfactor, bin_centers_og[-1]/xfactor))
    hist_counts = hist_counts / np.max(hist_counts)
    axes.bar(hist_bins[:-1],.8*v*hist_counts,width=np.diff(hist_bins),
                bottom=-v,align='edge',color='gray',alpha=.5,zorder=1,edgecolor='black',
                linewidth=0.25)
    for t in range(4):
        ale_line = np.mean(ale_centered[:,t,:,:],axis=(1,2))
        axes.plot(bin_centers_og/xfactor,ale_line/yfactor,color=colors[t],marker=markers[t],
                    linestyle=linestyles[t],linewidth=3.0,markeredgecolor='black',
                    label=fcst_times[t],zorder=2)
    axes.plot(bin_centers_og/xfactor,np.zeros(ale_bin_centers.shape),color='black',linewidth=2,zorder=0)
    axes.set_ylabel(ylabel,fontsize=18)
    axes.set_xlabel(xlabel,fontsize=18)
    axes.set_ylim([-v,v])
    axes.set_xlim([bin_centers_og[0]/xfactor,bin_centers_og[-1]/xfactor])
    axes.tick_params(axis='y', labelsize=18)
    axes.tick_params(axis='x', labelsize=18)
    axes.set_title(title,fontsize=24)
    axes.legend(fontsize=18)
    plt.savefig('./ALE_LINES/%s.png'%(file[0:-3]))
    plt.close()

def plot_ale_lines_efm(args,file='',mu=0.0,std=0.0,unique_values=[],
                    yfactor=1.0,xfactor=1.0,xlabel='',ylabel='',title=''):
    print(file)
    ale_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/ALE_v3.2/'
    ale_dict = pickle.load(open(ale_dir+file,'rb'))
    ale_sum = ale_dict['ale_sum']
    print(ale_sum.shape)
    ale_bins = ale_dict['quantile_edges']
    ale_bin_centers = 0.5 * (ale_bins[:-1] + ale_bins[1:])
    print(ale_bin_centers)
    if ('GLM' in file) or ('VII' in file) or ('max_abs' in file) or ('std' in file) or ('var' in file):
        bin_centers_og = np.expm1(ale_bin_centers * std + mu)
    else:
        bin_centers_og = ale_bin_centers*std + mu
    print(bin_centers_og)
    print(std,mu)

    ale_count = ale_dict['ale_count']
    ale_effect = np.ones(ale_sum.shape)
    for b in range(ale_sum.shape[0]):
        ale_effect[b,:,:,:] = ale_sum[b,:,:,:]/ale_count[b]
    
    ale_cum_effect = np.cumsum(ale_effect,axis=0)
    ale_centered = ale_cum_effect - np.mean(ale_cum_effect,axis=0,keepdims=True)
    v = np.nanmax(np.abs(ale_centered))/yfactor


    dict_5nm = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/ALE_v3.2_win5nm/'+file,'rb'))
    ale_count_5nm = dict_5nm['ale_count_mask']
    ale_sum_5nm = dict_5nm['ale_sum_mask']

    ale_effect_5nm = ale_sum_5nm / np.maximum(ale_count_5nm, 1)
    ale_centered_5nm = np.cumsum(ale_effect_5nm)
    ale_centered_5nm -= np.mean(ale_centered_5nm)

    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    axes.set_facecolor('#f0f0f0')
    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly, same as BC
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']
    fcst_times = ['00-15 min Forecast','15-30 min Forecast','30-45 min Forecast','45-60 min Forecast']

    axes.grid(zorder=0)
    hist_counts, hist_bins = np.histogram(unique_values / xfactor,bins=20,range=(bin_centers_og[0]/xfactor, bin_centers_og[-1]/xfactor))
    hist_counts = hist_counts / np.max(hist_counts)
    axes.bar(hist_bins[:-1],.8*v*hist_counts,width=np.diff(hist_bins),
                bottom=-v,align='edge',color='gray',alpha=.5,zorder=1,edgecolor='black',
                linewidth=0.25)
    for t in range(4):
        ale_line = np.mean(ale_centered[:,t,:,:],axis=(1,2))
        axes.plot(bin_centers_og/xfactor,ale_line/yfactor,color=colors[t],marker=markers[t],
                    linestyle=linestyles[t],linewidth=3.0,markeredgecolor='black',
                    label=fcst_times[t],zorder=2)
    axes.plot(bin_centers_og,ale_centered_5nm,linewidth=4.0,marker=markers[0],color='red',markeredgecolor='white',linestyle='solid',label='00-15 min Forecast w/in 5nm')
    axes.plot(bin_centers_og/xfactor,np.zeros(ale_bin_centers.shape),color='black',linewidth=2,zorder=0)
    axes.set_ylabel(ylabel,fontsize=18)
    axes.set_xlabel(xlabel,fontsize=18)
    axes.set_ylim([-v,v])
    axes.set_xlim([bin_centers_og[0]/xfactor,bin_centers_og[-1]/xfactor])
    axes.tick_params(axis='y', labelsize=18)
    axes.tick_params(axis='x', labelsize=18)
    axes.set_title(title,fontsize=24)
    axes.legend(fontsize=18)
    plt.savefig('./ALE_LINES/%s.png'%(file[0:-3]))
    plt.close()

def plot_ale_subplot(args):

    stats_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/LC_Forecast_Final_log1p_GLM_VII_2024.nc',engine='netcdf4')

    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly, same as BC
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']

    sb_labels = ['(a)','(b)','(c)','(d)',
                '(e)','(f)','(g)','(h)',
                '(i)','(j)','(k)','(l)',
                '(m)','(n)','(o)','(p)',
                '(q)','(r)','(s)','(t)',
                '(u)','(v)','(w)','(x)',
                '(y)','(z)']
    sb_count=0

    fcst_times = ['00-15 min Forecast','15-30 min Forecast','30-45 min Forecast','45-60 min Forecast']
    input_times = ['60-45 min','45-30 min','30-15 min','15-00 min']
    ale_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/ALE_v3.2/'
    fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(13,15.5))
    for r in range(2):
        for c in range(2):
            if r==0 and c==0:
                key = 'MRMS_Reflectivity_-10C_00.50'
                units = '($dBZ$)'
                ylabel = '-10C Z - (%)'
                mu = stats_dict['Reflectivity_-10C_00.50']['mean']
                std = stats_dict['Reflectivity_-10C_00.50']['std']
                data_np = ds['Reflectivity_-10C_00.50'].values
                title1 = '-10C Z: 15 - 00 min prior'
                t=3

            if r==0 and c==1:
                key = 'MRMS_mid_zdr_dB'
                units = '($dB$)'
                ylabel = 'ALE - (%)'
                mu = stats_dict['mid_zdr_dB']['mean']
                std = stats_dict['mid_zdr_dB']['std']
                data_np = ds['mid_zdr_dB'].values
                title1 = 'Mid $Z_{dr}$: 15 - 00 min prior ' 
                t=3
            
            if r==1 and c==1:
                key = 'GLM_group_area_log'
                units = '($m^{2}$)'
                ylabel = 'ALE - (%)'
                mu = stats_dict['group_area_log']['mean']
                std = stats_dict['group_area_log']['std']
                data_np = ds['group_area_log'].values
                title1= 'Group Area: 15 - 00 min prior'
                t=3

            if r==1 and c==0:
                key = 'HRRR_mslp'
                units = '($Pa$)'
                ylabel = 'ALE - (%)'
                mu = stats_dict['mslp']['mean']
                std = stats_dict['mslp']['std']
                data_np = ds['mslp'].values
                t=3
                title1 = 'MSLP: 45 - 60 min forecast'
            
            t_key = args.t_keys[t]
            unique_values = np.ravel(data_np[:,t,:,:])
            
            file = '%s_%s_ALE.pkl'%(key,t_key)
            print(file)
            ale_dict = pickle.load(open(ale_dir+file,'rb'))
            ale_sum = ale_dict['ale_sum']
            ale_bins = ale_dict['quantile_edges']
            ale_bin_centers = 0.5 * (ale_bins[:-1] + ale_bins[1:])
            if ('GLM' in file):
                bin_centers_og = np.expm1(ale_bin_centers * std + mu)
                unique_values = np.expm1(unique_values)
            else:
                bin_centers_og = ale_bin_centers*std + mu
            
            hist_counts, hist_bins = np.histogram(unique_values,bins=20,range=(bin_centers_og[0], bin_centers_og[-1]))
            hist_counts = hist_counts / np.max(hist_counts)

            ale_count = ale_dict['ale_count']
            ale_effect = np.ones(ale_sum.shape)
            for b in range(ale_sum.shape[0]):
                ale_effect[b,:,:,:] = ale_sum[b,:,:,:]/ale_count[b]
            ale_cum_effect = np.cumsum(ale_effect,axis=0)
            ale_centered = ale_cum_effect - np.mean(ale_cum_effect,axis=0,keepdims=True)
            v = np.nanmax(np.abs(ale_centered))
            for t1 in range(4):
                ale_line = np.mean(ale_centered[:,t1,:,:],axis=(1,2))
                axes[r,c].plot(bin_centers_og,ale_line,color=colors[t1],marker=markers[t1],
                            linestyle=linestyles[t1],linewidth=3.0,markeredgecolor='black',
                            label=fcst_times[t1],zorder=2)
            axes[r,c].bar(hist_bins[:-1],.8*v*hist_counts,width=np.diff(hist_bins),
                        bottom=-v,align='edge',color='gray',alpha=.5,zorder=1,
                        edgecolor='black',linewidth=0.25)
            axes[r,c].plot(bin_centers_og,np.zeros(ale_bin_centers.shape),color='black',linewidth=4,zorder=0)
            axes[r,c].set_title(title1,fontsize=12)
            if c==0:
                axes[r,c].set_ylabel(ylabel,fontsize=12)
            if r==1 and c==0:
                axes[r,c].legend(fontsize=12)
            axes[r,c].set_ylim([-v,v])
            axes[r,c].set_xlabel(units,fontsize=12)
            axes[r,c].tick_params(axis='x', labelsize=10)
            axes[r,c].tick_params(axis='y', labelsize=10)
            axes[r,c].grid()
            axes[r,c].text(bin_centers_og[0],(v-.15*v),sb_labels[sb_count],fontsize=12,weight='heavy')
            sb_count+=1
            del ale_dict, unique_values
        del data_np

    r=2
    efm_key='FM07'
    data_np = ds[efm_key].values
    norm_stats_dict = pickle.load(open('./efm_forecast_min_max.pkl','rb'))
    for c,stat in enumerate(['mean','max_abs']):
        if c==0 :
            units = '($Vm^{-1}$)'
            stat_idx=2
            file = '%s_%s_ALE.pkl'%(efm_key,stat)
            fm_stats_dict = norm_stats_dict[efm_key][stat]
            mu = fm_stats_dict['mean']
            std = fm_stats_dict['std']
            one_stat_np = data_np[:,:,stat_idx]
            title = 'FM07 - Mean'

        if c==1:
            units = '($Vm^{-1}$)'
            stat_idx=1
            file = '%s_%s_ALE.pkl'%(efm_key,stat)
            fm_stats_dict = norm_stats_dict[efm_key][stat]
            mu = fm_stats_dict['log_mean']
            std = fm_stats_dict['log_std']
            one_stat_np = data_np[:,:,stat_idx]
            title = 'FM07 - Max. Abs. Value'
        
        unique_values = np.ravel(one_stat_np)
        ale_dict = pickle.load(open(ale_dir+file,'rb'))
        ale_sum = ale_dict['ale_sum']
        ale_bins = ale_dict['quantile_edges']
        ale_bin_centers = 0.5 * (ale_bins[:-1] + ale_bins[1:])
        if ('GLM' in file) or ('VII' in file) or ('max_abs' in file) or ('std' in file) or ('var' in file):
            bin_centers_og = np.expm1(ale_bin_centers * std + mu)
        else:
            bin_centers_og = ale_bin_centers*std + mu

        ale_count = ale_dict['ale_count']
        ale_effect = np.ones(ale_sum.shape)
        for b in range(ale_sum.shape[0]):
            ale_effect[b,:,:,:] = ale_sum[b,:,:,:]/ale_count[b]
        ale_cum_effect = np.cumsum(ale_effect,axis=0)
        ale_centered = ale_cum_effect - np.mean(ale_cum_effect,axis=0,keepdims=True)
        v = np.nanmax(np.abs(ale_centered))
        axes[r,c].grid(zorder=0)
        axes[r,c].set_title(title,fontsize=12)
        for t1 in range(4):
            ale_line = np.mean(ale_centered[:,t1,:,:],axis=(1,2))
            axes[r,c].plot(bin_centers_og,ale_line,color=colors[t1],marker=markers[t1],
                    linestyle=linestyles[t1],linewidth=3.0,markeredgecolor='black',
                    label=fcst_times[t1],zorder=2)
        axes[r,c].plot(bin_centers_og,np.zeros(ale_bin_centers.shape),color='black',linewidth=4,zorder=0)
        axes[r,c].set_ylim([-v,v])

        dict_5nm = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3.2/ALE_v3.2_win5nm/'+file,'rb'))
        ale_count_5nm = dict_5nm['ale_count_mask']
        ale_sum_5nm = dict_5nm['ale_sum_mask']

        ale_effect_5nm = ale_sum_5nm / np.maximum(ale_count_5nm, 1)
        ale_centered_5nm = np.cumsum(ale_effect_5nm)
        ale_centered_5nm -= np.mean(ale_centered_5nm)
        h5, = axes[r,c].plot(bin_centers_og,ale_centered_5nm,linewidth=4.0,marker=markers[0],color='red',markeredgecolor='white',linestyle='solid',label='00-15 min Forecast w/in 5nm')
        if c==0:
            axes[r,c].legend(handles=[h5],fontsize=12)

        hist_counts, hist_bins = np.histogram(unique_values,bins=20,range=(bin_centers_og[0], bin_centers_og[-1]))
        hist_counts = hist_counts / np.max(hist_counts)
        axes[r,c].bar(hist_bins[:-1],.8*v*hist_counts,width=np.diff(hist_bins),
                bottom=-v,align='edge',color='gray',alpha=.5,zorder=1,edgecolor='black',
                linewidth=0.25)
        axes[r,c].set_xlabel(units,fontsize=12)
        axes[r,c].tick_params(axis='x', labelsize=12)
        axes[r,c].tick_params(axis='y', labelsize=12)
        axes[r,c].text(bin_centers_og[0],(v-.15*v),sb_labels[sb_count],fontsize=12,weight='heavy')
        if c==0:
            axes[r,c].set_ylabel('ALE (%)',fontsize=12)
        sb_count+=1

    plt.savefig('./temp_ALE_subplot.png',dpi=300)
    plt.savefig('./Figure_XX_ALE.pdf',dpi=300)
    plt.close()


if __name__=='__main__':
    """
    Irrelevant lines for EFM are commented out for other variables.
    """
    parser = create_parser()
    args = parser.parse_args()
    # drive_ale_all(args=args)
    # # make_ale_maps(args=args)
    # drive_ALE_line_plots(args=args)
    plot_ale_subplot(args=args)

