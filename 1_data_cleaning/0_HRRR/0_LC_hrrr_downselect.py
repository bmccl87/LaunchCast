import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pygrib
import argparse 
import pickle
import glob
import cartopy.crs as ccrs
from helper import *

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")

def parse_args():
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--year', type=str, default='2024')
    parser.add_argument('--job_id',type=int,default=188)
    args = parser.parse_args()
    yr = args.year
    job_id = args.job_id
    return yr, job_id

def get_variable(label, levels, grbindx, ksc_idxs):
    for i,level in enumerate(levels):
        try:
            if i==0:
                data_3d = grbindx.select(name=label,typeOfLevel='isobaricInhPa',level=level)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
            else:
                data_3d = np.dstack([data_3d, 
                                        grbindx.select(name=label,typeOfLevel='isobaricInhPa',level=level)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]])
        except Exception as e:
            print(e)
    return data_3d


def get_hrrr_grid(grbs):
    #get the lat lon grid
    x_idxs = [1422,1486]
    y_idxs = [176,240]
    hrrr_lat, hrrr_lon = grbs[1].latlons()
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

    ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]
    LC_lats = hrrr_lat[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    LC_lons = hrrr_lon[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    return ksc_idxs, LC_lats, LC_lons, hrrr_proj


def downselect_grib2dict(yr,job_id):
    mo,day = get_day_mo(yr,job_id)
    hrrr_dir = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/'+yr+mo+'/'
    glob_call = 'hrrr_%s%s%s*_f000.grib'%(yr,mo,day)
    glob_files = sorted(glob.glob(hrrr_dir+glob_call))
    print(len(glob_files))
    #generate a list of levels from 50hPa to 1000hPa
    level = 50
    levels = [50]
    l=0
    while level<1000:
        level = levels[l]
        level = level+25
        levels.append(level)
        l=l+1
    grbs = pygrib.open(glob_files[0])
    ksc_idxs, LC_lats, LC_lons, hrrr_proj = get_hrrr_grid(grbs)
    print(LC_lons)
    grid_dict = {'ksc_idxs':ksc_idxs,'LC_lats':LC_lats,'LC_lons':LC_lons,'hrrr_proj':hrrr_proj}
    pickle.dump(grid_dict,open('./grid_dict_2d.pkl','wb'))
    grbs.close()
    downselect_glob(yr=yr, mo=mo, day=day, glob_files=glob_files, levels=levels)  

def downselect_glob(yr,#year,string
                    mo,#month,string
                    day,#day,string
                    glob_files,#list of files from the glob call
                    levels):#list of pressure levels 

    for i, file in enumerate(glob_files):
        print(file)
        grbs = pygrib.open(file)
        valid_time = grbs[1].validDate
        ksc_idxs, LC_lats, LC_lons, hrrr_proj = get_hrrr_grid(grbs)
        
        #get the variables across all of the levels
        grbindx = pygrib.index(file,'name','typeOfLevel','level')

        #declare the file name and check to see if the file already exists
        hr = file[-12:-10]
        pkl_fname = 'hrrr_%s_%s_%s_%s.pkl'%(yr,mo,day,hr)
        out_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/0_HRRR_downselect/f00/%s%s/'%(yr,mo)
        if os.path.isdir(out_dir)==False:
            os.makedirs(out_dir)
        if os.path.isfile(out_dir+pkl_fname)==True:
            continue
        
        # 528:Cloud mixing ratio:kg kg**-1 (instant):lambert:isobaricInhPa:level 97500 Pa:fcst time 0 hrs:from 201806151900
        # 530:Rain mixing ratio:kg kg**-1 (instant):lambert:isobaricInhPa:level 97500 Pa:fcst time 0 hrs:from 201806151900
        # 531:Snow mixing ratio:kg kg**-1 (instant):lambert:isobaricInhPa:level 97500 Pa:fcst time 0 hrs:from 201806151900
        # 518:Graupel (snow pellets):kg kg**-1 (instant):lambert:isobaricInhPa:level 95000 Pa:fcst time 0 hrs:from 201806151900
        # 18:Vertical velocity:Pa s**-1 (instant):lambert:isobaricInhPa:level 25000 Pa:fcst time 0 hrs:from 20230615190
        # 113:Geopotential height:gpm (instant):lambert:isobaricInhPa:level 25000 Pa:fcst time 0 hrs:from 202306151900
        # 114:Temperature:K (instant):lambert:isobaricInhPa:level 25000 Pa:fcst time 0 hrs:from 202306151900
        # 115:Relative humidity:% (instant):lambert:isobaricInhPa:level 25000 Pa:fcst time 0 hrs:from 202306151900
        # 116:Dew point temperature:K (instant):lambert:isobaricInhPa:level 25000 Pa:fcst time 0 hrs:from 202306151900
        # 117:Specific humidity:kg kg**-1 (instant):lambert:isobaricInhPa:level 25000 Pa:fcst time 0 hrs:from 202306151900
        # 483:U component of wind:m s**-1 (instant):lambert:isobaricInhPa:level 90000 Pa:fcst time 0 hrs:from 201806151900
        # 484:V component of wind:m s**-1 (instant):lambert:isobaricInhPa:level 90000 Pa:fcst time 0 hrs:from 201806151900
        # 485:Absolute vorticity:s**-1 (instant):lambert:isobaricInhPa:level 90000 Pa:fcst time 0 hrs:from 201806151900
        # 645:Surface lifted index:K (instant):lambert:isobaricLayer:levels 50000-100000 Pa:fcst time 0 hrs:from 202306151900
        # 646:Convective available potential energy:J kg**-1 (instant):lambert:surface:level 0:fcst time 0 hrs:from 202306151900
        # 647:Convective inhibition:J kg**-1 (instant):lambert:surface:level 0:fcst time 0 hrs:from 202306151900
        # 580:Vertically-integrated liquid:kg m**-1 (instant):lambert:atmosphere:level 0 -:fcst time 0 hrs:from 202306151900
        # 584:Derived radar reflectivity:dB (instant):lambert:isothermal:level 263 K:fcst time 0 hrs:from 202306151900
        # 628:Precipitation rate:kg m**-2 s**-1 (instant):lambert:surface:level 0:fcst time 0 hrs:from 202306151900
        # 645:Surface lifted index:K (instant):lambert:isobaricLayer:levels 50000-100000 Pa:fcst time 0 hrs:from 202306151900
        # 622:10 metre U wind component:m s**-1 (instant):lambert:heightAboveGround:level 10 m:fcst time 0 hrs:from 202306151900
        # 623:10 metre V wind component:m s**-1 (instant):lambert:heightAboveGround:level 10 m:fcst time 0 hrs:from 202306151900
        # 677:Geopotential height:gpm (instant):lambert:isothermZero:level 0:fcst time 0 hrs:from 202306151900
        # 683:Geopotential height:gpm (instant):lambert:isothermal:level 263 K:fcst time 0 hrs:from 202306151900
        # 684:Geopotential height:gpm (instant):lambert:isothermal:level 253 K:fcst time 0 hrs:from 202306151900
        # 604:Lightning:dimensionless (instant):lambert:atmosphere:level 0 -:fcst time 0 hrs:from 202306151900
        # 703:Land-sea mask:(0 - 1) (instant):lambert:surface:level 0:fcst time 0 hrs:from 202306151900
        # 670:Boundary layer height:m (instant):lambert:surface:level 0:fcst time 0 hrs:from 201806151900
        # 589:MSLP (MAPS System Reduction):Pa (instant):lambert:meanSea:level 0:fcst time 0 hrs:from 202306151900

        #get the cloud physics mixing ratios
        graupel_q_3d = get_variable('Graupel (snow pellets)',levels,grbindx,ksc_idxs)
        cloud_q_3d = get_variable('Cloud mixing ratio',levels,grbindx,ksc_idxs)
        rain_q_3d = get_variable('Rain mixing ratio',levels,grbindx,ksc_idxs)
        snow_q_3d = get_variable('Snow mixing ratio',levels,grbindx,ksc_idxs)

        #get the traditional state variables
        u_3d = get_variable('U component of wind',levels,grbindx,ksc_idxs)
        v_3d = get_variable('V component of wind',levels,grbindx,ksc_idxs)
        w_3d = get_variable('Vertical velocity',levels,grbindx,ksc_idxs)
        vort_3d = get_variable('Absolute vorticity',levels,grbindx,ksc_idxs)
        temp_3d = get_variable('Temperature',levels,grbindx,ksc_idxs)
        gph_3d = get_variable('Geopotential height',levels,grbindx,ksc_idxs)
        rh_3d = get_variable('Relative humidity',levels,grbindx,ksc_idxs)
        td_3d = get_variable('Dew point temperature',levels,grbindx,ksc_idxs)
        spec_h_3d = get_variable('Specific humidity',levels,grbindx,ksc_idxs)
        u_10m = grbs.select(name='10 metre U wind component',typeOfLevel='heightAboveGround',level=10)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        v_10m = grbs.select(name='10 metre V wind component',typeOfLevel='heightAboveGround',level=10)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

        #get the accumulated atmospheric variables
        reflectivity = grbs.select(name='Maximum/Composite radar reflectivity',typeOfLevel='atmosphere',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        refl_neg10C = grbs.select(name='Derived radar reflectivity',typeOfLevel='isothermal',level=263)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        cape = grbs.select(name='Convective available potential energy',typeOfLevel='surface',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        cinh = grbs.select(name='Convective inhibition',typeOfLevel='surface',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        vil = grbs.select(name='Vertically-integrated liquid',typeOfLevel='atmosphere',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        precip_rate = grbs.select(name='Precipitation rate',typeOfLevel='surface',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        gph_zero = grbs.select(name='Geopotential height',typeOfLevel='isothermZero',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        gph_neg10 = grbs.select(name='Geopotential height',typeOfLevel='isothermal',level=263)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        gph_neg20 = grbs.select(name='Geopotential height',typeOfLevel='isothermal',level=253)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        land_sea = grbs.select(name='Land-sea mask', typeOfLevel='surface',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        lightning = grbs.select(name='Lightning',typeOfLevel='atmosphere',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        bl_hgt = grbs.select(name='Boundary layer height',typeOfLevel='surface',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        mslp = grbs.select(name='MSLP (MAPS System Reduction)',typeOfLevel='meanSea',level=0)[0].values[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        
        #store the data in a dict
        hrrr_dict = {
            'graupel_q_3d':graupel_q_3d,
            'cloud_q_3d':cloud_q_3d,
            'rain_q_3d':rain_q_3d,
            'snow_q_3d':snow_q_3d,
            'u_3d':u_3d,
            'v_3d':v_3d,
            'w_3d':w_3d,
            'vort_3d':vort_3d,
            'temp_3d':temp_3d,
            'gph_3d':gph_3d,
            'rh_3d':rh_3d,
            'td_3d':td_3d,
            'spec_h_3d':spec_h_3d,
            'u_10m':u_10m,
            'v_10m':v_10m,
            'reflectivity':reflectivity,
            'refl_neg10C':refl_neg10C,
            'cape':cape,
            'cinh':cinh,
            'vil':vil,
            'precip_rate':precip_rate,
            'gph_zero':gph_zero,
            'gph_neg10C':gph_neg10,
            'gph_neg20C':gph_neg20,
            'land_sea':land_sea,
            'lightning':lightning,
            'bl_hgt':bl_hgt,
            'mslp':mslp,
            'valid_time':valid_time,
            'lat':LC_lats,
            'lon':LC_lons
        }

        with open(out_dir+pkl_fname, 'wb') as pkl_file:
            pickle.dump(hrrr_dict,pkl_file)

def main():
    print("0_LC_hrrr_downselect.py")
    extract_slurm_env()
    yr,job_id = parse_args()
    downselect_grib2dict(yr,job_id)
    
if __name__=="__main__":
    main()