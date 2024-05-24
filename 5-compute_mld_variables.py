#!/usr/bin/env python

import xarray as xr
import isas_io as io
import numpy as np
import profile_tools as pftools


#----------------------------------------------------------
# Set batch parameters
#----------------------------------------------------------

dim = 'depth'

def compute_mixed_layer_variables(month):
    
    ds = xr.open_zarr("/home/serazin/Data/UOP/N2_PROFILES/n2_m%02i_profiles.zarr" % month)
    ds_peaks = xr.open_dataset('/home/serazin/Data/UOP/N2_PEAKS/n2_peaks_m%02i.nc' % month).isel(peak=0)
    
    time_shared = np.intersect1d(ds.time, ds_peaks.time)
    z_left = ds_peaks['width_left'].sel(time=time_shared)
    ds = ds.sel(time=time_shared)  
    
    # Compute mean mixed layer variables
    variables = ['SIGMA0', 'T', 'SP', 'N2']
    mean_ml = ds[variables].where(ds.depth <= z_left).mean('depth').compute()
    variables_ml = {variable: '%s_ml' % variable for variable in variables}
    mean_ml = mean_ml.rename(variables_ml) 
    
    # Compute variable at 10m and the mixed layer base 
    variables = ['SIGMA0', 'T', 'SP']
    ds_10m = ds[variables].isel(depth=2).dropna('time').load()
    ds_mlb = pftools.var_at_zref(ds[variables], z_left).compute()
    delta_ml = ds_mlb - ds_10m
    variables_delta_ml = {variable: '%s_delta_ml' % variable for variable in variables}
    delta_ml = delta_ml.rename(variables_delta_ml) 
    
    ds_out = xr.merge([mean_ml, delta_ml])
    ds_out.to_netcdf("/home/serazin/Data/UOP/N2_PEAKS/ml_variables_m%02i.nc" % month, mode='w')
    

if __name__ == "__main__":
    # Open and run a local cluster with multi-threading
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client)
    for month in range(1, 13):
        compute_mixed_layer_variables(month)
