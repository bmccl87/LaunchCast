import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np

def main():

    efm_dir='/scratch/bmac87/efm_v2/pkl/annual_concat/'
    file = '2023.nc'
    stat='rms'
    ds = xr.open_dataset(efm_dir+file,engine='netcdf4')

    df = ds.to_dataframe().reset_index()
    stats_df = df.drop(columns=['time', 'station'])
    corr = stats_df.corr()

    plt.figure(figsize=(10,8))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title('Correlation Matrix of EFM Statistics')
    plt.tight_layout()
    plt.show()
    plt.savefig('efm_corr.png')
    plt.close()

    import seaborn as sns

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of EFM Statistics')
    plt.show() 
    plt.savefig('corr_sns.png')   
    
if __name__=='__main__':
    main()