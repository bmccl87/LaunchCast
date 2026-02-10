import os
import pickle

def main():
    print('doing some file qc yo')

def check_zdr_download():
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
                'MergedZdr_18.00']
    
    for variable in variables:
        files = sorted(pickle.load(open('parallel_files_%s.pkl'%variable,'rb')))
        print(len(files))

def check_scratch2ourdisk():
    scratch_dir = '/scratch/bmac87/MRMS_on_HRRR_grid/'
    ourdisk_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/MRMS_on_HRRR_grid/'
    
if __name__=='__main__':
    main()
