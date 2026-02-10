import os
import shutil
import pickle
import gzip
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Download', fromfile_prefix_chars='@')
    parser.add_argument('--idx',type=int,default=0,help='0-38')
    args = parser.parse_args()
    return args.idx

def years_dict():
    """
    This code generates a dictionary for the date and hour information. 
    """
    years = ['2018','2019','2020','2021','2022','2023','2024']
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']


    yrs_dict = {}
    for yr in years:#for each year
        mos_dict = {}
        for mo in mos: #for each month
            days = []
            mo_jul = []
            if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
                for t in range(1,32):
                    days.append(f"{t:02}")
            elif mo=='02':
                if yr=='2020':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                elif yr=='2024':
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
    return yrs_dict

def build_variable_files_lists(var_idx=0):
    variables = ['MergedZdr_00.50',#0
                'MergedZdr_00.75',#1
                'MergedZdr_01.00',#2
                'MergedZdr_01.25',#3
                'MergedZdr_01.50',#4
                'MergedZdr_01.75',#5
                'MergedZdr_02.00',#6
                'MergedZdr_02.25',#7
                'MergedZdr_02.50',#8
                'MergedZdr_02.75',#9
                'MergedZdr_03.00',#10
                'MergedZdr_03.50',#11
                'MergedZdr_04.00',#12
                'MergedZdr_04.50',#13
                'MergedZdr_05.00',#14
                'MergedZdr_05.50',#15
                'MergedZdr_06.00',#16
                'MergedZdr_06.50',#17
                'MergedZdr_07.00',#18
                'MergedZdr_07.50',#19
                'MergedZdr_08.00',#20
                'MergedZdr_08.50',#21
                'MergedZdr_09.00',#22
                'MergedZdr_10.00',#23
                'MergedZdr_11.00',#24
                'MergedZdr_12.00',#25
                'MergedZdr_13.00',#26
                'MergedZdr_14.00',#27
                'MergedZdr_15.00',#28
                'MergedZdr_16.00',#29
                'MergedZdr_17.00',#30
                'MergedZdr_18.00',#31
                'Reflectivity_-10C_00.50',#32
                'Reflectivity_-15C_00.50',#33
                'Reflectivity_-20C_00.50',#34
                'Reflectivity_-5C_00.50',#35
                'Reflectivity_0C_00.50',#36
                'VII_00.50',#37
                'VIL_00.50']#38

    date_dict = years_dict()    
    years = ['2018','2019','2020','2021','2022','2023','2024']
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']
    file_dict = {}
    for v,variable in enumerate(variables):
        if v==var_idx:
            files_list = []
            for year in years:
                mos_dict = date_dict[year]
                for mo in mos:
                    days = mos_dict[mo]
                    for day in days:
                        try:
                            data_dir = '/ourdisk/hpc/ai2es/datasets/MRMS/%s/%s%s%s/%s/compressed/'%(year,year,mo,day,variable)
                            files = os.listdir(data_dir)
                            for file in files:
                                files_list.append(os.path.join(data_dir,file))
                        except Exception as e:
                            print(e)
            print(variable,len(files_list))
            pickle.dump(files_list,open('parallel_files_%s.pkl'%(variable),'wb'))
            del files_list

def check_for_remaining_gz():
    """
    This is from chatGPT. 
    """
    directory = "/ourdisk/hpc/ai2es/datasets/MRMS/"
    gz_files = []

    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".gz"):
                gz_files.append(os.path.join(root, f))
    print(f"Total .gz files (recursive): {len(gz_files)}")
    pickle.dump(gz_files,open('./remaining_gz_files.pkl','wb'))

def extract_gz():
    start_idx,end_idx = create_parser()
    print(start_idx,end_idx)
    files = pickle.load(open('remaining_gz_files.pkl','rb'))
    files = files[start_idx:end_idx]
    print(len(files))
    for f,file in enumerate(files):
        if f%100==0:
            print(f,len(files))
        try:
            file_out = file.rstrip('.gz') 
            with gzip.open(file, 'rb') as f_in:
                with open(file_out, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    os.remove(file)
        except Exception as e:
            print("Error occurred while extracting the file:", e)

if __name__=='__main__':
    idx = create_parser()
    build_variable_files_lists(var_idx=idx)