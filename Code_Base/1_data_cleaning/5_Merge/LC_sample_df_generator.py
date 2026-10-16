import os
import shutil
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from LC_parser import create_parser
import pickle
import pandas as pd

def make_sample_list():

    input_start = np.datetime64('2021-01-01T00:00:00','ns')
    input_dt = np.timedelta64(5,'m')
    input_end = np.datetime64('2024-12-31T23:00:00','ns')

    hrrr_start = np.datetime64('2021-01-01T00:00:00','ns')
    hrrr_dt = np.timedelta64(1,'h')

    glm_start = np.datetime64('2020-12-31T23:00:00','ns')
    glm_dt = np.timedelta64(15,'m')

    efm_start = glm_start
    efm_dt = glm_dt

    mrms_start = glm_start
    mrms_dt = glm_dt

    merlin_start = input_start
    merlin_dt = np.timedelta64(15,'m')

    merlin_headers = ['M00215','M15230','M30245','M45260']
    hrrr_headers = ['H_INIT']
    glm_headers = ['G60245','G45230','G30215','G15200']
    efm_headers = ['E60245','E45230','E30215','E15200']
    radar_headers = ['R60245','R45230','R30215','R15200']

    all_headers = np.concatenate([hrrr_headers,radar_headers,glm_headers,efm_headers,merlin_headers])
    hrrr_one = [hrrr_start]
    hrrr_dt = np.timedelta64(1,'h')
    sample_times = []

    while input_start<=input_end:
        mrms_one = []
        glm_one = []
        efm_one = []
        merlin_one = []

        for t in range(4):#15 minute intervals
            mrms_one.append(mrms_start)
            mrms_start+=mrms_dt

            glm_one.append(glm_start)
            glm_start+=glm_dt

            efm_one.append(efm_start)
            efm_start+=efm_dt

            merlin_one.append(merlin_start)
            merlin_start+=merlin_dt

        del mrms_start, glm_start, efm_start, merlin_start

        #increment the initial sample times by 5 minutes
        mrms_start = mrms_one[0]+input_dt
        glm_start = glm_one[0]+input_dt
        efm_start = efm_one[0]+input_dt
        merlin_start = merlin_one[0]+input_dt

        one_sample_times = np.concatenate([hrrr_one,mrms_one,glm_one,efm_one,merlin_one])
        sample_times.append(one_sample_times)
        del one_sample_times
        input_start+=input_dt
        sample_minute = pd.Timestamp(input_start).minute
        if sample_minute==0:
            hrrr_start += hrrr_dt
            del hrrr_one
            hrrr_one = [hrrr_start]

    sample_times = np.stack(sample_times,axis=0)
    sample_df = pd.DataFrame(data=sample_times, columns=all_headers)
    return sample_df

def count_first_flash(args):
    print(ds)
    print('cc_ltg')
    cc_ltg = ds['cc_forecast'].values
    print(cc_ltg.shape)

    print('cg_ltg')
    cg_ltg = ds['cg_forecast'].values
    print(cg_ltg.shape)
    total_ltg = np.copy(cc_ltg+cg_ltg)
    total_ltg[total_ltg>0] = 1.0
    total_ltg[total_ltg==0] = 0.0
    del cc_ltg, cg_ltg

    for i in range(total_ltg.shape[0]):
        if i%500==0:
            plt.figure()
            plt.imshow(total_ltg[i,0,:,:])
            plt.savefig('test_total_ltg_%s.png'%(i))
            plt.close()

    targets = np.copy(total_ltg)
    targets00215 = np.copy(targets[:,0,:,:])
    print(targets00215.shape)
    del total_ltg, targets

    group_area = ds['group_area'].values
    group_area[group_area>0] = 1.0
    group_area[group_area==0.0] = 0.0
    group_mask = np.copy(group_area)
    del group_area
    groups15200 = np.copy(group_mask[:,-1,:,:])
    print(groups15200.shape)
    for i in range(groups15200.shape[0]):
        if i%500==0:
            plt.figure()
            plt.imshow(groups15200[i,:,:])
            plt.savefig('test_group_%s.png'%(i))
            plt.close()

    targets00215_wltg = np.sum(np.sum(targets00215,axis=2),axis=1)
    plt.figure()
    plt.plot(targets00215_wltg)
    plt.savefig('test1.png')
    plt.close()

    print(targets00215_wltg.shape)
    groups15200_wltg = np.sum(np.sum(groups15200,axis=2),axis=1)
    print(groups15200_wltg.shape)
    target_mask2 = (targets00215_wltg>=1.0)
    group_mask2 = (groups15200_wltg>=1.0)
    total_mask = target_mask2*group_mask2
    print(np.sum(total_mask))
    
    first_flash_count=0
    total_pixel_count=0
    for i in range(targets00215_wltg.shape[0]):
        t00215 = targets00215[i,:,:]
        g15200 = groups15200[i,:,:]

        for r in range(64):
            for c in range(64):
                if t00215[r,c]==1.0 and g15200[r,c]==0.0:
                    first_flash_count+=1
                total_pixel_count+=1

    print(first_flash_count)
    print(total_pixel_count)
    print(first_flash_count/total_pixel_count*100,'percent pixels have first flash')


    del ds

def plot_efm_histogram(tfds,ds=xr.Dataset(),fm_key='FM01', year='2024'):
    title_str = '%s - %s'%(fm_key,year)
    
    efm_data = ds[fm_key].values
    fm_data = tf.concat([x[fm_key] for x, y in tfds], axis=0)
    fm_np = fm_data.numpy()

    efm_std = efm_data[:,:,:,-1]
    efm_std[efm_std==-20000]=np.nan
    efm_med = efm_data[:,:,:,0]
    efm_med[efm_med==-20000]=np.nan
    num_bins=int(20000/100)
    med_bins = np.linspace(-10000,10000,num_bins+1)

    
    ymin=10*0
    ymax=10**6

    fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(20,20))
    for t in range(4):
        efm_med_to_hist = np.ravel(efm_med[:,t,:])
        counts, bins = np.histogram(efm_med_to_hist, bins=med_bins)
        axes[0,t].bar(bins[:-1], counts, width=np.diff(bins), align='edge')
        if t==0:
            axes[0,t].set_ylabel('Count')
        axes[0,t].grid()
        # axes[0,t].set_ylim([ymin,ymax])
        axes[0,t].set_xlabel('Median')
        axes[0,t].set_yscale('log')
        axes[0,t].text(-10000,10**4,'Min: %s'%(f"{np.nanmin(efm_med_to_hist):.02f}"))
        axes[0,t].text(3000,10**4,'Max: %s'%(f"{np.nanmax(efm_med_to_hist):.02f}"))

    for t in range(4):
        print('norm med',t)
        if t==0:
            i0=0
            i1=180
            axes[1,t].set_ylabel('Count')
        elif t==1:
            i0=180
            i1=360
        elif t==2:
            i0=360
            i1=540
        else:
            i0=540
            i1=720
        
        norm_fm_med = fm_np[:,i0:i1,0]
        med2bin = np.ravel(norm_fm_med)
        med_norm_bins = np.linspace(0,1,num_bins+1)
        counts, bins = np.histogram(med2bin, bins=med_norm_bins)
        axes[1,t].bar(bins[:-1], counts, width=np.diff(bins), align='edge')
        axes[1,t].grid()
        # axes[1,t].set_ylim([ymin,ymax])
        axes[1,t].set_xlabel('Scaled Median')
        axes[1,t].set_yscale('log')
        print(np.max(med2bin))
        print(np.min(med2bin))

    ymin=0
    
    num_bins2=100
    std_bins = np.linspace(0,100,num_bins2+1)
    for t in range(4):
        efm_std_to_hist = np.ravel(efm_std[:,t,:])
        neg_std = np.sum(efm_std_to_hist<0)
        label = 'Num Neg Stds: %s'%(f"{neg_std:04}")
        print(t)
        print(np.min(efm_std_to_hist))
        print(np.max(efm_std_to_hist))
        counts, bins = np.histogram(efm_std_to_hist, bins=std_bins)
        axes[2,t].bar(bins[:-1], counts, width=np.diff(bins), align='edge',label=label)
        if t==0:
            axes[2,t].set_ylabel('Count')
        axes[2,t].grid()
        # axes[2,t].set_ylim([ymin,ymax])
        axes[2,t].set_xlabel('Standard Deviation')
        axes[2,t].legend()
        axes[2,t].set_yscale('log')
        axes[2,t].text(20,10**4,'Min: %s'%(f"{np.nanmin(efm_std_to_hist):.02f}"))
        axes[2,t].text(60,10**4,'Max: %s'%(f"{np.nanmax(efm_std_to_hist):.02f}"))

    
    for t in range(4):
        print('norm std',t)
        if t==0:
            i0=0
            i1=180
            axes[3,t].set_ylabel('Count')
        elif t==1:
            i0=180
            i1=360
        elif t==2:
            i0=360
            i1=540
        else:
            i0=540
            i1=720
        norm_fm_std = fm_np[:,i0:i1,1]
        std2bin = np.ravel(norm_fm_std)
        std_norm_bins = np.linspace(0,np.max(std2bin),num_bins2+1)
        counts, bins = np.histogram(std2bin, bins=std_norm_bins)
        axes[3,t].bar(bins[:-1], counts, width=np.diff(bins), align='edge')
        axes[3,t].grid()
        # axes[3,t].set_ylim([ymin,ymax])
        axes[3,t].set_xlabel('Scaled Std')
        axes[3,t].set_yscale('log')
        print(np.min(std2bin))
        print(np.max(std2bin))
    axes[0,0].set_title('60-45 min prior')
    axes[0,1].set_title('45-30 min prior')
    axes[0,2].set_title('30-15 min prior')
    axes[0,3].set_title('15-00 min prior')
    plt.suptitle(title_str)
    plt.savefig('./efm_hist_%s_%s.png'%(fm_key,year))
    plt.close()
    
def drive_efm_hist_w_ltg():
    parser = create_parser()
    args = parser.parse_args()

    years = ['2021','2022','2023','2024']
    for y,year in enumerate(years):
        if y>=0:
            nc_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/6_final_xr/'
            fname = 'LC_linear_zdr_%s.nc'%(year)
            ds = xr.open_dataset(nc_dir+fname,engine='netcdf4')

            print('loading the tensorflow dataset')
            tfds_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/7_forecast_tfds/'
            fname = 'LC_All_%s.tfds'%(year)
            tfds = tf.data.Dataset.load(tfds_dir+fname)
            tfds = tfds.batch(2)

            for f,fm_site in enumerate(args.efm_keys):
                if f>=0:
                    print(year,fm_site)
                    try:
                        make_efm_histogram(ds=ds,tfds=tfds,fm_key=fm_site, year=year)
                    except Exception as e:
                        print('EXCEPTION',e)
                        continue
            del tfds, ds

def flash_count_2_list():

    years = ['2021','2022','2023','2024']
    cc_pd = pickle.load(open('../2_MERLIN/cc_rc_pd.pkl','rb'))
    cg_pd = pickle.load(open('../2_MERLIN/cg_rc_pd.pkl','rb'))

    samples_meta_pd = make_sample_list()

    for y,year in enumerate(years):
        if y>=0:
            samples_pd2 = samples_meta_pd[samples_meta_pd['H_INIT'].dt.year==int(year)].copy()
            glm_pd = pickle.load(open('/scratch/bmac87/glm_v3/annual_rc/%s.pkl'%year,'rb'))

            #drop the rows and columns that are on the edge
            glm_pd = glm_pd[~glm_pd['r_idx_kdtree'].isin([0,65])]
            glm_pd = glm_pd[~glm_pd['c_idx_kdtree'].isin([0,65])]
            glm_pd = glm_pd.sort_index()

            #drop the rows and columns that are on the edge, and not in the 16x16 target.
            #range is off by one to accomodate inclusivity, and the 66x66 r/c values
            cc_pd2 = cc_pd[cc_pd.index.year==int(year)]
            cc_pd2 = cc_pd2[~cc_pd2['r_idx_kdtree'].isin([0,65])]
            cc_pd2 = cc_pd2[cc_pd2['r_idx_kdtree'].isin(range(21,42))]
            cc_pd2 = cc_pd2[~cc_pd2['c_idx_kdtree'].isin([0,65])]
            cc_pd2 = cc_pd2[cc_pd2['c_idx_kdtree'].isin(range(21,42))]
            cc_pd2 = cc_pd2.sort_index()

            #drop the rows and columns on the edge
            cg_pd2 = cg_pd[cg_pd.index.year==int(year)]
            cg_pd2 = cg_pd2[~cg_pd2['r_idx_kdtree'].isin([0,65])]
            cg_pd2 = cg_pd2[cg_pd2['r_idx_kdtree'].isin(range(21,42))]
            cg_pd2 = cg_pd2[~cg_pd2['c_idx_kdtree'].isin([0,65])]
            cg_pd2 = cg_pd2[cg_pd2['c_idx_kdtree'].isin(range(21,42))]
            cg_pd2 = cg_pd2.sort_index()

            ltg_dt = np.timedelta64(15,'m')
            merlin_starts = samples_pd2[['M00215','M15230','M30245','M45260']].values
            merlin_ends = merlin_starts+ltg_dt

            samples_pd2['M00215_num_CC'] = np.zeros(len(samples_pd2))
            samples_pd2['M15230_num_CC'] = np.zeros(len(samples_pd2))
            samples_pd2['M30245_num_CC'] = np.zeros(len(samples_pd2))
            samples_pd2['M45260_num_CC'] = np.zeros(len(samples_pd2))

            samples_pd2['M00215_num_CG'] = np.zeros(len(samples_pd2))
            samples_pd2['M15230_num_CG'] = np.zeros(len(samples_pd2))
            samples_pd2['M30245_num_CG'] = np.zeros(len(samples_pd2))
            samples_pd2['M45260_num_CG'] = np.zeros(len(samples_pd2))

            samples_pd2['M00215_num_total'] = np.zeros(len(samples_pd2))
            samples_pd2['M15230_num_total'] = np.zeros(len(samples_pd2))
            samples_pd2['M30245_num_total'] = np.zeros(len(samples_pd2))
            samples_pd2['M45260_num_total'] = np.zeros(len(samples_pd2))

            glm_starts = samples_pd2[['G60245','G45230','G30215','G15200']].values
            glm_ends = glm_starts+ltg_dt

            samples_pd2['G60245_num'] = np.zeros(len(samples_pd2))
            samples_pd2['G45230_num'] = np.zeros(len(samples_pd2))
            samples_pd2['G30215_num'] = np.zeros(len(samples_pd2))
            samples_pd2['G15200_num'] = np.zeros(len(samples_pd2))

            for tt in range(len(merlin_starts)):
                if tt>=0:
                    if tt%1000==0:
                        print(tt,len(merlin_starts))
                    
                    m_starts = merlin_starts[tt]
                    m_ends = merlin_ends[tt]

                    g_starts = glm_starts[tt]
                    g_ends = glm_ends[tt]

                    for t1 in range(4):
                        m_start_time = m_starts[t1]
                        m_end_time = m_ends[t1]

                        g_start_time = g_starts[t1]
                        g_end_time = g_ends[t1]

                        num_cc = len(cc_pd2.loc[m_start_time:m_end_time])
                        num_cg = len(cg_pd2.loc[m_start_time:m_end_time])
                        num_total = num_cc+num_cg
                        
                        num_glm = len(glm_pd.loc[g_start_time:g_end_time])
                        if t1==0:
                            samples_pd2.loc[samples_pd2.index[tt], 'G60245_num'] = num_glm
                            samples_pd2.loc[samples_pd2.index[tt], 'M00215_num_CC'] = num_cc
                            samples_pd2.loc[samples_pd2.index[tt], 'M00215_num_CG'] = num_cg
                            samples_pd2.loc[samples_pd2.index[tt], 'M00215_num_total'] = num_total

                        elif t1==1:
                            samples_pd2.loc[samples_pd2.index[tt], 'G45230_num'] = num_glm
                            samples_pd2.loc[samples_pd2.index[tt], 'M15230_num_CC'] = num_cc
                            samples_pd2.loc[samples_pd2.index[tt], 'M15230_num_CG'] = num_cg
                            samples_pd2.loc[samples_pd2.index[tt], 'M15230_num_total'] = num_total

                        elif t1==2:
                            samples_pd2.loc[samples_pd2.index[tt], 'G30215_num'] = num_glm
                            samples_pd2.loc[samples_pd2.index[tt], 'M30245_num_CC'] = num_cc
                            samples_pd2.loc[samples_pd2.index[tt], 'M30245_num_CG'] = num_cg
                            samples_pd2.loc[samples_pd2.index[tt], 'M30245_num_total'] = num_total

                        else:
                            samples_pd2.loc[samples_pd2.index[tt], 'G15200_num'] = num_glm
                            samples_pd2.loc[samples_pd2.index[tt], 'M45260_num_CC'] = num_cc
                            samples_pd2.loc[samples_pd2.index[tt], 'M45260_num_CG'] = num_cg
                            samples_pd2.loc[samples_pd2.index[tt], 'M45260_num_total'] = num_total
            
            ltg_columns = ['M00215_num_CG','M00215_num_CC','M00215_num_total',
                'M15230_num_CG','M15230_num_CC','M15230_num_total',
                'M30245_num_CG','M30245_num_CC','M30245_num_total',
                'M45260_num_CG','M45260_num_CC','M45260_num_total',
                'G60245_num','G45230_num','G30215_num','G15200_num']

            #call a sample when either GLM or MERLIN have 5 flashes within 15 minutes
            samples_pd3 = samples_pd2.loc[(samples_pd2[ltg_columns] >= 5).any(axis=1)].copy()
            samples_pd3.to_pickle(open('%s_samples_w_ltg_counts_v3.pkl'%year,'wb'))
            print(year,len(samples_pd3))

def isWithin_efm_distance(args,year='2021'):

    samples_df = pickle.load(open('%s_samples_w_ltg_counts_v3.pkl'%year,'rb'))
    cc_pd = pickle.load(open('../2_MERLIN/cc_rc_pd.pkl','rb'))
    cg_pd = pickle.load(open('../2_MERLIN/cg_rc_pd.pkl','rb'))
    samples_df = samples_df.reindex()
    print(samples_df.head(10))

    #drop the rows and columns that are on the edge, and not in the 16x16 target.
    #range is off by one to accomodate inclusivity, and the 66x66 r/c values
    cc_pd2 = cc_pd[cc_pd.index.year==int(year)]
    cc_pd2 = cc_pd2[~cc_pd2['r_idx_kdtree'].isin([0,65])]
    cc_pd2 = cc_pd2[cc_pd2['r_idx_kdtree'].isin(range(21,42))]
    cc_pd2 = cc_pd2[~cc_pd2['c_idx_kdtree'].isin([0,65])]
    cc_pd2 = cc_pd2[cc_pd2['c_idx_kdtree'].isin(range(21,42))]
    cc_pd2 = cc_pd2.sort_index()

    #drop the rows and columns on the edge
    cg_pd2 = cg_pd[cg_pd.index.year==int(year)]
    cg_pd2 = cg_pd2[~cg_pd2['r_idx_kdtree'].isin([0,65])]
    cg_pd2 = cg_pd2[cg_pd2['r_idx_kdtree'].isin(range(21,42))]
    cg_pd2 = cg_pd2[~cg_pd2['c_idx_kdtree'].isin([0,65])]
    cg_pd2 = cg_pd2[cg_pd2['c_idx_kdtree'].isin(range(21,42))]
    cg_pd2 = cg_pd2.sort_index()

    dist_ds = xr.open_dataset('../1_EFM/efm_distance_proximity.nc',engine='netcdf4')
    full_mask5 = dist_ds['mask_5nm'].values
    mask5_64x64 = np.zeros((64,64))
    mask5_64x64[24:40,24:40] = full_mask5
    
    target_cols = ['M00215','M15230','M30245','M45260']
    samples_df['withinEFM_5nm'] = np.zeros(len(samples_df))
    target_sample_times = samples_df[target_cols].values.copy()
    num_samples = len(samples_df)

    for tt in range(num_samples):
        # if tt%100==0:
        #     print(tt,num_samples)
        m_start_times = target_sample_times[tt]
        dt1 = np.timedelta64(15,'m')
        for t1 in range(4):
            mst = m_start_times[t1]
            met = mst+dt1

            temp_cc_pd = cc_pd2[mst:met]
            if len(temp_cc_pd)>0:
                r_idxs = temp_cc_pd['r_idx_kdtree'].values-1#subtract 1 due to kd tree grid was 66x66
                c_idxs = temp_cc_pd['c_idx_kdtree'].values-1
                for q in range(len(temp_cc_pd)):
                    if mask5_64x64[r_idxs[q],c_idxs[q]]==1:
                        samples_df.loc[samples_df.index[tt], 'withinEFM_5nm'] = 1.0

            temp_cg_pd = cg_pd2[mst:met]
            if len(temp_cg_pd)>0:
                r_idxs = temp_cg_pd['r_idx_kdtree'].values-1
                c_idxs = temp_cg_pd['c_idx_kdtree'].values-1
                for z in range(len(temp_cg_pd)):
                    if mask5_64x64[r_idxs[z],c_idxs[z]]==1:
                        samples_df.loc[samples_df.index[tt], 'withinEFM_5nm'] = 1.0
            del temp_cg_pd, temp_cc_pd

    print(year)
    print('samples within 5nm')
    print(samples_df['withinEFM_5nm'].sum())

    pickle.dump(samples_df,open('%s_samples_w_ltg_counts_w_efm_flag.pkl'%year,'wb'))
    samples_df.to_csv('%s_samples_w_ltg_counts_w_efm_flag.csv'%year, sep='\t', index=False)

def first_flash_last_flash(year='2021'):

    df = pickle.load(open('%s_samples_w_ltg_counts_w_efm_flag.pkl'%year,'rb'))
    df_5nm = df.loc[df['withinEFM_5nm']==1.0].copy()
    print('total samples within 5nm')
    print(len(df_5nm))
    glm_columns = ['G60245_num','G45230_num','G30215_num','G15200_num']
    first_flash_5nm = df_5nm.loc[(df_5nm[glm_columns] == 0.0).all(axis=1)]
    
    target_columns = ['M00215_num_total','M15230_num_total','M30245_num_total','M45260_num_total']
    last_flash_5nm = df_5nm.loc[(df_5nm[target_columns]==0.0).any(axis=1)]
    
    print(year)
    print('first','5nm',len(first_flash_5nm))
    print('last time step')
    last_flash_5nm = df_5nm.loc[(df_5nm[target_columns[-1]]==0.0)]
    print('last','5nm',len(last_flash_5nm))

    print('last two time steps no lightning')
    last_flash_5nm = df_5nm.loc[(df_5nm[target_columns[2:4]]==0.0).all(axis=1)]
    print('last','5nm',len(last_flash_5nm))

    print('last three time steps no lightning')
    last_flash_5nm = df_5nm.loc[(df_5nm[target_columns[1:4]]==0.0).all(axis=1)]
    print('last','5nm',len(last_flash_5nm))

def main():
    
    parser = create_parser()
    args = parser.parse_args()

    # flash_count_2_list()
    # isWithin_efm_distance(args=args,year='2021')
    # isWithin_efm_distance(args=args,year='2022')
    # isWithin_efm_distance(args=args,year='2023')
    # isWithin_efm_distance(args=args,year='2024')

    first_flash_last_flash(year='2021')
    first_flash_last_flash(year='2022')
    first_flash_last_flash(year='2023')
    first_flash_last_flash(year='2024')


if __name__=='__main__':
    main()