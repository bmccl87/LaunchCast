import os
import shutil
import wget

def main():
    
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']
    f_hour = 'f001'

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
    
    for yr in years:
        for mo in yrs_dict[yr]:
            days = yrs_dict[yr][mo]
            for day in days:
                for hr in hrs:
                    archive_dir = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_Subhourly_f01/%s%s/'%(yr,mo)
                    hrrr_fname = 'hrrr-subh_%s%s%s%s_%s.grib'%(yr,mo,day,hr,f_hour)
                    if os.path.isfile(archive_dir+hrrr_fname)==False:
                        print(hrrr_fname)
                        missing_url = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.%s%s%s/conus/hrrr.t%sz.wrfsubhf%s.grib2'%(yr,mo,day,hr,f_hour[-2:])
                        # print(missing_url)
                        # if os.path.isdir(archive_dir)==False:
                        #     os.makedirs(archive_dir)
                        # try:
                        #     wget.download(missing_url,out=archive_dir+hrrr_fname)
                        # except Exception as e:
                        #     print('bad url:',missing_url)
                        #     print(e)
                        #     print()
                        #     print()
# url = https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20180517/conus/hrrr.t23z.wrfsubhf00.grib2

if __name__=='__main__':
    main()