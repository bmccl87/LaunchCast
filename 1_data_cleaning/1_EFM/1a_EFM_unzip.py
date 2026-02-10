import os
import shutil
import glob
import numpy as np

def unzip_mo_yr_2_days_folder():
    
    #unzip the data and remove the zip files
    archive_format = 'zip'

    unpack_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/Field_Mill_50HZ_Feb2025/2024/'
    zip_files = glob.glob(unpack_dir+'*.zip')
    for zip_file in zip_files:
        print(zip_file)
        extract_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/Field_Mill_50HZ_2024_zip/'
        if os.path.isdir(extract_dir)==False:
            os.makedirs(extract_dir) 
        try:
            shutil.unpack_archive(zip_file,extract_dir,archive_format)
        except Exception as e:
            print(e)
            print(str(f)+' '+unpack_dir+'\n')

def unzip():
    days_int = np.arange(1,32)
    days = []
    for day in days_int:
        days.append(f'{day:02}')
    print(days)

    archive_format = 'zip'

    extract_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/Field_Mill_50HZ_txt/2018/'
    for day in days:
        unpack_dir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/Field_Mill_50HZ_2018_zip/%s/'%day
        zip_files = glob.glob(unpack_dir+'*.zip')
        for zip_file in zip_files:
            try:
                shutil.unpack_archive(zip_file,extract_dir,archive_format)
            except Exception as e:
                print(e)
                print(str(f)+' '+unpack_dir+'\n')

def main():
    print('In the main 1_EFM_unzip.py main folder')
    unzip()
    # unzip_mo_yr_2_days_folder(year='2020')
    # unzip_mo_yr_2_days_folder(year='2021')
    # unzip_mo_yr_2_days_folder(year='2022')
    
    # unzip_to_text(year='2018')
    # unzip_to_text(year='2024')
    


if __name__=='__main__':
    main()
