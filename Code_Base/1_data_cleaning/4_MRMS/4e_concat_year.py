import xarray as xr
import os
def annual_concat(year='2021',variable='MergedZdr_00.50'):
    print(year,variable)

def main():
    years = ['2021','2022','2023','2024']
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

    for variable in variables:
        for year in years:
            print(variable,year)

if __name__=='__main__':
    main()