import os
import shutil
import numpy as np
import wget
import argparse

def main():

    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--year', type=str, default='2018')
    parser.add_argument('--f_hour',type=str,default='f001')
    args = parser.parse_args()
    yr = args.year
    f_hour = args.f_hour

    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']

    yrs_dict = {}
    mos_dict = {}
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

    for mo in yrs_dict[yr]:
        days = yrs_dict[yr][mo]
        for day in days:
            for hr in hrs:
                hrrr_dir = '/scratch/bmac87/HRRR/%s%s/'%(yr,mo)
                hrrr_fname = 'hrrr_%s%s%s%s_%s.grib'%(yr,mo,day,hr,f_hour)
                print(hrrr_fname)
                if os.path.isfile(hrrr_dir+hrrr_fname)==False:
                    missing_url = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.%s%s%s/conus/hrrr.t%sz.wrfprsf%s.grib2'%(yr,mo,day,hr,f_hour[2:4])
                    print(missing_url)
                    if os.path.isdir(hrrr_dir)==False:
                        os.makedirs(hrrr_dir)
                    wget.download(missing_url,out=hrrr_dir+hrrr_fname)
                # https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20140912/conus/hrrr.t00z.wrfprsf02.grib2

if __name__=='__main__':
    main()