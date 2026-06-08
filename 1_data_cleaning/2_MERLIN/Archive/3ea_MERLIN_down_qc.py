import os
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pickle
import pandas as pd
import matplotlib.colors as mcolors

def main():
    print('3ea_MERLIN_down_qc.py')
    ltg_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/3_LC_ltg/'

    # Define the color segments and corresponding values
    colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]
    bounds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    # Create a colormap and norm
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for f,file in enumerate(os.listdir(ltg_dir)):
        if f>=2:
            ds = xr.open_dataset(ltg_dir+file, engine='netcdf4')
            
            lat = ds['lat'].values
            lon = ds['lon'].values
            valid_times = ds['valid_times'].values

            cc_np = ds['cc'].values
            cg_np = ds['cg'].values
            print('cg_np')
            print(cg_np.shape)
            print(np.max(np.max(np.max((cg_np)))))

            grid_dict = pickle.load(open('grid_dict_2d.pkl','rb'))
            hrrr_proj = grid_dict['hrrr_proj']

            for t, vt in enumerate(valid_times):
                if t>=0:
                    fig = plt.figure(figsize=(30,30))
                    ax1 = fig.add_subplot(1,2,1,projection=hrrr_proj)
                    cc_np = cc_np.astype('float')
                    cc_np[ cc_np==0 ] = np.nan
                    cb = ax1.pcolormesh(lon,lat,cc_np[:,:,t],transform=ccrs.PlateCarree(),cmap=cmap)
                    plt.colorbar(cb,ax=ax1)
                    ax1.set_title('CC',fontsize=24)
                    ax1.coastlines()

                    ax2 = fig.add_subplot(1,2,2,projection=hrrr_proj)
                    cg_np = cg_np.astype('float')
                    cg_np[cg_np==0] = np.nan
                    cb = ax2.pcolormesh(lon,lat,cg_np[:,:,t],transform=ccrs.PlateCarree(),cmap=cmap)
                    cbh = plt.colorbar(cb,ax=ax2)
                    cbh.set_label('# of Flashes',size=24,weight='bold')
                    ax2.set_title('CG',fontsize=24)
                    ax2.coastlines()

                    ts = pd.Timestamp(vt)
                    day = ts.day
                    mo = ts.month
                    yr = ts.year
                    hr = ts.hour
                    title_str = '%s/%s/%s %sZ'%(mo,day,yr,hr)
                    fsave = '%s_%s_%s_%sZ'%(mo,day,yr,hr)
                    plt.suptitle(title_str,fontsize=36)
                    save_fig = '/scratch/bmac87/LaunchCast_scratch/MERLIN_output_pngs/'
                    plt.savefig(save_fig+'%s.png'%fsave)
                    plt.close()

if __name__=='__main__':
    main()