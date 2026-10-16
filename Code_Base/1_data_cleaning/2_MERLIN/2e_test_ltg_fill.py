import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
import random
from scipy.ndimage import binary_fill_holes, binary_closing

def load_ltg_images():

    print('loading the lightning data')
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/88_LC_observed_datasets/6_final_xr_observed/'
    fname = 'LC_observed_2022_w_EFM_dist.nc'
    ds = xr.open_dataset(data_dir+fname,engine='netcdf4')
    cc_data = ds['cc_forecast'].values
    cg_data = ds['cg_forecast'].values
    rand_idxs = random.sample(range(0, cc_data.shape[0]), 100)
    
    cc_data = cc_data[rand_idxs,:,:,:]
    cc_data[cc_data>=1] = 1.0
    cc_data[cc_data==0] = 0.0

    cg_data = cg_data[rand_idxs,:,:,:]
    cg_data[cg_data>=1] = 1.0
    cg_data[cg_data==0] = 0.0

    cc_filled = np.zeros(cc_data.shape)
    cg_filled = np.zeros(cg_data.shape)

    for i in range(1):
        
        fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(20,20))
        temp_cc_data = cc_data[i,:,:,:]
        temp_cg_data = cg_data[i,:,:,:]

        for t in range(4):
            #from chatgpt
            # filled = binary_fill_holes(lightning_binary)
            # filled = binary_closing(filled, structure=np.ones((3,3)))
            
            filled_cc_data = binary_fill_holes(temp_cc_data[t,:,:])
            filled_cc_data = binary_closing(filled_cc_data,structure=np.ones((4,4)))

            filled_cg_data = binary_fill_holes(temp_cg_data[t,:,:])
            filled_cg_data = binary_closing(filled_cg_data,structure=np.ones((4,4)))

            axes[t,0].imshow(temp_cc_data[t,:,:])
            axes[t,0].set_xticks([],[])
            axes[t,0].set_yticks([],[])
            if t==0:
                axes[t,0].set_title('CC True Labels')

            axes[t,1].imshow(filled_cc_data)
            axes[t,1].set_xticks([],[])
            axes[t,1].set_yticks([],[])
            if t==0:
                axes[t,1].set_title('Filled CC')

            axes[t,2].imshow(temp_cg_data[t,:,:])
            axes[t,2].set_xticks([],[])
            axes[t,2].set_yticks([],[])
            if t==0:
                axes[t,2].set_title('CG True Labels')

            axes[t,3].imshow(filled_cg_data)
            axes[t,3].set_xticks([],[])
            axes[t,3].set_yticks([],[])
            if t==0:
                axes[t,3].set_title('CG Filled')

        plt.savefig('test_fill_%s.png'%i)
        plt.close()


    return

def main():
    print('trying the new lightning fill')
    ltg_data = load_ltg_images()
if __name__=='__main__':
    main()