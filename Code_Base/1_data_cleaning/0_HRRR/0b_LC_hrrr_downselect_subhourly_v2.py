import pygrib
import os
import numpy as np
import xarray as xr
import pickle
import argparse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--exp', type=int, default=0)
    args = parser.parse_args()
    return args.exp

def build_file_list():
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']

    years=['2021','2022','2023','2024']
    yrs_dict = {}
    mos_dict = {}
    for yr in years:
        for mo in mos: #for each month
            days = []
            mo_jul = []
            if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
                for t in range(1,32):
                    day_str = f"{t:02}"
                    days.append(day_str)
            elif mo=='02':
                if yr=='2020' or yr=='2024':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                else:
                    for t in range(1,29):
                        days.append(f"{t:02}")
            else:
                for t in range(1,31):
                    days.append(f"{t:02}")
            mos_dict.update({mo:days})
        yrs_dict.update({yr:mos_dict})

    f00_list = []
    f_hour = 'f000'
    for yr in years:
        for mo in yrs_dict[yr]:
            f00_dir = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR-Subhourly/%s%s/'%(yr,mo)
            days = yrs_dict[yr][mo]
            for day in days:
                for hr in hrs:
                    fname = 'hrrr-subh_%s%s%s%s_%s.grib'%(yr,mo,day,hr,f_hour)
                    f00_list.append(f00_dir+fname)

    f01_list = []
    f_hour = 'f001'
    for yr in years:
        for mo in yrs_dict[yr]:
            f01_dir = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_Subhourly_f01/%s%s/'%(yr,mo)
            days = yrs_dict[yr][mo]
            for day in days:
                for hr in hrs:
                    fname = 'hrrr-subh_%s%s%s%s_%s.grib'%(yr,mo,day,hr,f_hour)
                    f01_list.append(f01_dir+fname)
    file_dict = {'f00':f00_list,'f01':f01_list}
    return file_dict

def get_indices(exp=0):
    start_idx=exp*35
    if exp==998:
        end_idx = 35136
    else:
        end_idx=start_idx+35
    return start_idx,end_idx

def process_pair(file_pair):
    f00_file = file_pair[0]
    f01_file = file_pair[1]
    
    downselect_hrrr(f00_file=f00_file,f01_file=f01_file)

def makedirs():
    years = ['2021','2022','2023','2024']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for year in years:
        for month in months:
            os.makedirs('/scratch/bmac87/HRRR_Subhourly_Downselect_v3/%s%s/'%(year,month),exist_ok=True)
        
def downselect_hrrr(f00_file='',f01_file=''):
    
    # hrrr indices for 66x66 downselection - faster processing for lightning location
    LC_hrrr_indexes = pickle.load(open('../LC_hrrr_indexes.pkl','rb'))
    x1 = LC_hrrr_indexes['x'].start-1
    x2 = LC_hrrr_indexes['x'].stop+1
    y1 = LC_hrrr_indexes['y'].start-1
    y2 = LC_hrrr_indexes['y'].stop+1
    
    #get the lat lon grid
    og_hrrr = pickle.load(open('./hrrr_og_grid.pkl','rb'))
    hrrr_lat = og_hrrr['lat']
    hrrr_lon = og_hrrr['lon']
    
    hrrr_lat = hrrr_lat[y1:y2,x1:x2]
    hrrr_lon = hrrr_lon[y1:y2,x1:x2]
    ksc_idxs = [y1,y2,x1,x2]
    
    try:    
        grbs00 = pygrib.open(f00_file)
        grbs01 = pygrib.open(f01_file)

        u_winds_10m = np.zeros((5,66,66))
        v_winds_10m = np.zeros((5,66,66))
        u_winds_80m = np.zeros((5,66,66))
        v_winds_80m = np.zeros((5,66,66))
        u_mean_10m = np.zeros((5,66,66))
        v_mean_10m = np.zeros((5,66,66))
        temps_2m = np.zeros((5,66,66))
        tds_2m = np.zeros((5,66,66))
        surface_pressures = np.zeros((5,66,66))
        comp_refls = np.zeros((5,66,66))
        vils = np.zeros((5,66,66))
        refls_1000m = np.zeros((5,66,66))
        refls_4000m = np.zeros((5,66,66))
        precip_rates = np.zeros((5,66,66))

        u_wnd_grbs = grbs01.select(name='10 metre U wind component')#m/s, instant
        date_list = []
        u_winds_10m[0,:,:] = grbs00.select(name='10 metre U wind component')[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        date_list.append(np.datetime64(grbs00[1].validDate))
        for u,u_grb in enumerate(u_wnd_grbs):
            date_list.append(np.datetime64(u_grb.validDate))
            u_winds_10m[u+1,:,:] = u_grb.values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

        v_winds_10m[0,:,:] = grbs00.select(name='10 metre V wind component')[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        v_wnd_grbs = grbs01.select(name='10 metre V wind component')#m/s, instant
        for v,v_grb in enumerate(v_wnd_grbs):
            v_winds_10m[v+1,:,:] = v_grb.values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        
        
        temps_2m[0,:,:] = grbs00.select(name='2 metre temperature')[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        temp_2m_grbs = grbs01.select(name='2 metre temperature')#Kelvin
        for t,temp_2m_grb in enumerate(temp_2m_grbs):
            temps_2m[t+1,:,:]= temp_2m_grb.values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]  

        tds_2m[0,:,:] = grbs00.select(name='2 metre dewpoint temperature')[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]] #Kelvin, instant
        td_2m_grbs = grbs01.select(name='2 metre dewpoint temperature')
        for td, td_2m_grb in enumerate(td_2m_grbs):
            tds_2m[td+1,:,:] = td_2m_grb.values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        
        surface_pressures[0,:,:] = grbs00.select(name='Surface pressure')[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        sfc_pres_grbs = grbs01.select(name='Surface pressure')#Pa
        for s,sfc_pres_grb in enumerate(sfc_pres_grbs):
            surface_pressures[s+1,:,:] = sfc_pres_grb.values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

        precip_rates[0,:,:] = grbs00.select(name='Precipitation rate')[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]] #kg m**-2 s**-1, instant
        pr_grbs = grbs01.select(name='Precipitation rate')
        for pr, pr_grb in enumerate(pr_grbs):
            precip_rates[pr+1,:,:] = pr_grb.values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

        ds = xr.Dataset(
            data_vars={
                "u_10m":(["ht","y","x"],u_winds_10m),
                "v_10m":(["ht","y","x"],v_winds_10m),
                "temp_2m": (["ht", "y", "x"],temps_2m),
                "td_2m":(["ht","y","x"],tds_2m),
                "mslp":(["ht","y","x"],surface_pressures),
                "precip_rate":(["ht","y","x"],precip_rates)
            },
            coords={
                "valid_times": date_list,
                "ht": np.arange(temps_2m.shape[0]),  # or your actual prior_times
                "y": np.arange(temps_2m.shape[1]),
                "x": np.arange(temps_2m.shape[2]),
                "lat": (["y", "x"], hrrr_lat),
                "lon": (["y", "x"], hrrr_lon),
            }
        )
        save_dir = '/scratch/bmac87/HRRR_Subhourly_Downselect_v3/'
        fsave = f00_file[-20:-10]+'.nc'
        yr = fsave[0:4]
        mo = fsave[4:6]
        save_dir2 = save_dir+yr+mo+'/'
        print('saving:',save_dir2+fsave)
        ds.to_netcdf(save_dir2+fsave,engine='netcdf4')
        print('saved successfully')

        grbs00.close()
        grbs01.close()

    except Exception as e:
        print('bad gribs:',f00_file,f01_file)
        print(e)

def file_check(f00_files=[]):

    missing_files = []
    idx_list = []
    for f,file in enumerate(f00_files):
        if f>=0:
            fsave = file[-20:-10]+'.pkl'
            yr = fsave[0:4]
            mo = fsave[4:6]
            save_dir = '/scratch/bmac87/HRRR_Subhourly_Downselect_v3/%s%s/'%(yr,mo)
            if os.path.isfile(save_dir+fsave)==False:
                idx_list.append(f)
    print(len(idx_list), ' # of files missing')
    return idx_list

def visualize_one_file():
    data_dict = pickle.load(open('/scratch/bmac87/HRRR_Subhourly_Downselect_v3/202101/2021010118.pkl','rb'))
    fig, axes = plt.subplots(nrows=6,ncols=5,figsize=(80,20),subplot_kw={'projection': ccrs.PlateCarree()})
    for f in range(6):
        for t in range(5):
            axes[f,t].pcolormesh(data_dict['lon'],data_dict['lat'],data_dict['x'][t,:,:,f])
            axes[f,t].coastlines()
    plt.savefig('test_one_file.png')
    plt.close()

def main():
    
    print('Main function of 0b_LC_hrrr_downselect_subhourly.py')
    exp = parse_args()#exp = 0-998
    print(exp)
    files_dict = pickle.load(open('/scratch/bmac87/exp_files/exp_%s.pkl'%(exp),'rb'))
    f00_files = files_dict['f00_files']
    f01_files = files_dict['f01_files']
    print(len(f00_files))
    print(len(f01_files))
    print(f00_files)
    print(f01_files)
    file_pairs = list(zip(f00_files,f01_files))
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print("CPUs per task:", n_cpus)
    with Pool(processes=n_cpus) as pool:
        results = pool.map(process_pair, file_pairs)
    
    
def save_exp_files():
    files_dict = build_file_list()
    f00_files = files_dict['f00']
    f01_files = files_dict['f01']
    print(len(f00_files))
    print(len(f01_files))

    for exp in range(999):
        print(exp)
        start_idx = exp*35
        end_idx = start_idx+35
        if exp==998:
            end_idx=len(f00_files)
        print(start_idx,end_idx)
        f00_temp = f00_files[start_idx:end_idx]
        print(len(f00_temp))
        f01_temp = f01_files[start_idx:end_idx]
        print(len(f01_temp))
        files_dict={'f00_files':f00_temp,'f01_files':f01_temp}
        pickle.dump(files_dict,open('/scratch/bmac87/exp_files/exp_%s.pkl'%exp,'wb'))    

if __name__=='__main__':
    # makedirs()
    # save_exp_files()
    main()
    
    # visualize_one_file()
    



# idx_list = file_check(f00_files=f00_files)
    # f00_selected = [f00_files[i] for i in idx_list]
    # f01_selected = [f01_files[j] for j in idx_list]
    # print(len(f00_selected))
    # print(len(f01_selected))