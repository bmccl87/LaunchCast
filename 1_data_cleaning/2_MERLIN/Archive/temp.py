import pygrib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

print('loading the mrms grid')
mrms_dir = '/ourdisk/hpc/ai2es/datasets/MRMS/2020/20201014/MergedReflectivityQCComposite_00.50/compressed/'
fname = 'MRMS_MergedReflectivityQCComposite_00.50_20201014-235033.grib2'

print('getting the projection information for the mrms grid')
grbs = pygrib.open(mrms_dir+fname)
message = grbs[1]
projection_params = message.projparams
proj_a = projection_params['a']
proj_b = projection_params['b']

print('loading the mrms latlons()')
mrms_lat,mrms_lon = message.latlons()
mrms_lat_1d = mrms_lat[:,0]
mrms_lon_1d = mrms_lon[0,:]

print('converting the mrms latlons to geocentric()')
mrms_proj = ccrs.PlateCarree(globe=ccrs.Globe(semimajor_axis=proj_a,semiminor_axis=proj_b))
mrms_2xy = mrms_proj.as_geocentric()
mrms_xy = mrms_2xy.transform_points(mrms_proj,mrms_lon,mrms_lat)#2D
mrms_x = mrms_xy[:,:,0]
mrms_y = mrms_xy[:,:,1]
mrms_z = mrms_xy[:,:,2]

plt.figure()
cb = plt.imshow(mrms_z)
plt.colorbar()
plt.savefig('temp.png')
plt.close()