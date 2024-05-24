#!/usr/bin/env python

import xarray as xr
import isas_io as io
import numpy as np
import profile_tools as pftools


#----------------------------------------------------------
# Set batch parameters
#----------------------------------------------------------

dim = 'depth'

def compute_variable_at_peak_edges(month):
    
    ds = xr.open_zarr("/home/serazin/Data/UOP/N2_PROFILES/n2_m%02i_profiles.zarr" % month)
    ds_peaks = xr.open_dataset('/home/serazin/Data/UOP/N2_PEAKS/n2_peaks_m%02i.nc' % month)
    
    time_shared = np.intersect1d(ds.time, ds_peaks.time)
    z_left = ds_peaks['width_left'].sel(time=time_shared)
    z_right = ds_peaks['width_right'].sel(time=time_shared)
    ds = ds.sel(time=time_shared)
    ds_left = pftools.var_at_zref(ds, z_left).compute()
    ds_right = pftools.var_at_zref(ds, z_right).compute()
    
    variables = ['SIGMA0', 'N2', 'SP', 'T', 'KAPPA', 'BC']
    variables_left = {variable: '%s_left' % variable for variable in variables}
    variables_right = {variable: '%s_right' % variable for variable in variables}
    
    ds_left = ds_left.rename(variables_left)
    ds_right = ds_right.rename(variables_right)
    
    ds_out = xr.merge([ds_left, ds_right])
    ds_out.to_netcdf("/home/serazin/Data/UOP/N2_PEAKS/variables_at_peak_edges_m%02i.nc" % month, mode='w')
    

if __name__ == "__main__":
    # Open and run a local cluster with multi-threading
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client)
    for month in range(1, 13):
        compute_variable_at_peak_edges(month)
