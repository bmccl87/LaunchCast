import os
from helper import *
from LC_hrrr_downselect_0831 import *
def main():
    print("0a_LC_hrrr_downselect_check.py")

    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/0_HRRR_downselect/'
    years = ['2018','2019','2020','2021','2022','2023','2024']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    yrs_dict = years_dict()

    for year in years:
        dir2 = data_dir+'%s/'%(year)
        files2 = os.listdir(dir2)
        print(year,len(files2))

        for month in months:
            mos_dict = yrs_dict[year]
            days = mos_dict[month]
            for day in days:
                for hr in hrs:
                    fname = 'hrrr_%s_%s_%s_%s.pkl'%(year,month,day,hr)
                    if os.path.isfile(dir2+fname)==False:
                        print('downselecting,',fname)
                        # downselect_grib2dict(year,month,day,hr)
                

if __name__=='__main__':
    main()