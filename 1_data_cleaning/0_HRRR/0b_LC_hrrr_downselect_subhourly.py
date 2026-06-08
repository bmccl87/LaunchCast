import pygrib
import os
import numpy as np
import xarray as xr
import pickle
import argparse
import cartopy.crs as ccrs

def parse_args():
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--exp', type=int, default=0)
    args = parser.parse_args()
    return args.exp

def build_file_list():
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']

    years=['2018','2019','2020','2021','2022','2023','2024']
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
    if exp==0:
        start_idx=0
    else:
        start_idx=(exp*60)+1
    if exp==998:
        end_idx = 61487
    else:
        end_idx=(exp+1)*60
    return start_idx,end_idx

def main():
    print('Main function of 0b_LC_hrrr_downselect_subhourly.py')
    exp = parse_args()
    print('experiment: ',exp)
    start_idx,end_idx = get_indices(exp=exp)
    files_dict = build_file_list()
    f00_files = files_dict['f00']
    f01_files = files_dict['f01']

    idx_list = []
    for f,file in enumerate(f00_files):
        save_dir = '/scratch/bmac87/HRRR_Subhourly_Downselect/'
        fsave = file[-20:-10]+'.pkl'
        yr = fsave[0:4]
        mo = fsave[4:6]
        save_dir2 = save_dir+yr+mo+'/'
        if os.path.isfile(save_dir2+fsave)==False:
            idx_list.append(f)
    print(len(idx_list), ' # of files missing')
    for i in idx_list:
        if i>=0:
            f00_file = f00_files[i]
            f01_file = f01_files[i]
            
            try:
                grbs00 = pygrib.open(f00_file)
                grbs01 = pygrib.open(f01_file)
            

                u_wnd_grbs = grbs01.select(name='10 metre U wind component')#m/s
                u_winds = np.zeros((1059,1799,4))
                v_winds = np.zeros((1059,1799,4))
                temps_2m = np.zeros((1059,1799,4))
                surface_pressures = np.zeros((1059,1799,4))

                date_list = []
                u_winds[:,:,0] = grbs00.select(name='10 metre U wind component')[0].values
                date_list.append(np.datetime64(grbs00[1].validDate))
                for u,u_grb in enumerate(u_wnd_grbs):
                    if u<=2:
                        date_list.append(np.datetime64(u_grb.validDate))
                        u_winds[:,:,u+1] = u_grb.values

                v_winds[:,:,0] = grbs00.select(name='10 metre V wind component')[0].values
                v_wnd_grbs = grbs01.select(name='10 metre V wind component')#m/s
                for v,v_grb in enumerate(v_wnd_grbs):
                    if v<=2:
                        v_winds[:,:,v+1] = v_grb.values

                surface_pressures[:,:,0] = grbs00.select(name='Surface pressure')[0].values
                sfc_pres_grbs = grbs01.select(name='Surface pressure')#Pa
                for s,sfc_pres_grb in enumerate(sfc_pres_grbs):
                    if s<=2:
                        surface_pressures[:,:,s+1] = sfc_pres_grb.values

                temps_2m[:,:,0] = grbs00.select(name='2 metre temperature')[0].values
                temp_2m_grbs = grbs01.select(name='2 metre temperature')#Kelvin
                for t,temp_2m_grb in enumerate(temp_2m_grbs):
                    if t<=2:
                        temps_2m[:,:,t+1]= temp_2m_grb.values

                x_idxs = [1422,1486]
                y_idxs = [176,240]
                hrrr_lat, hrrr_lon = grbs00[1].latlons()
                projection_params = grbs00[1].projparams
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

                ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]
                LC_lats = hrrr_lat[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
                LC_lons = hrrr_lon[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]+360

                LC_u = u_winds[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3],:]
                LC_v = v_winds[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3],:]
                LC_sfc_pres = surface_pressures[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3],:]
                LC_temp = temps_2m[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3],:]

                data_dict = {'u':LC_u,'v':LC_v,'sfc_pres':LC_sfc_pres,'LC_temp':temps_2m,'valid_times':date_list,'lat':LC_lats,'lon':LC_lons,'hrrr_proj':hrrr_proj}
                save_dir = '/scratch/bmac87/HRRR_Subhourly_Downselect/'
                fsave = f00_file[-20:-10]+'.pkl'
                yr = fsave[0:4]
                mo = fsave[4:6]
                print(fsave,yr,mo)
                save_dir2 = save_dir+yr+mo+'/'
                if os.path.isdir(save_dir2)==False:
                    os.makedirs(save_dir2)
                pickle.dump(data_dict,open(save_dir2+fsave,'wb'))
                del data_dict, save_dir, save_dir2, LC_u, LC_v, LC_sfc_pres, LC_temp
                del surface_pressures, u_winds, v_winds, temps_2m
                del ksc_idxs, LC_lats, LC_lons, hrrr_proj, proj_a, proj_b, lon_0, lat_0, lat_parallel
                del temp_2m_grbs, sfc_pres_grbs, u_wnd_grbs, v_wnd_grbs
                grbs00.close()
                grbs01.close()
            except Exception as e:
                print('bad file:',f00_file,f01_file)
                continue

    #exp = 0-998


if __name__=='__main__':
    main()