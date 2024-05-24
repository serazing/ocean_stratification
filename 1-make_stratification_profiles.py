#!/usr/bin/env python

import xarray as xr
import isas_io as io
import profile_tools as pftools

#----------------------------------------------------------
# Set batch parameters
#----------------------------------------------------------

# Filtering dataset
io_kwargs = dict()
io_kwargs['lon_min'] = 0
io_kwargs['lon_max'] = 360
io_kwargs['lat_min'] = -90
io_kwargs['lat_max'] = 90
io_kwargs['qc_max'] = 4

# Interpolation
interp_kwargs = dict()
interp_kwargs['dim'] = 'depth'
interp_kwargs['max_depth'] = 2000
interp_kwargs['dz'] = 5.

# TEOS-10 computation
teos10_kwargs = dict()
teos10_kwargs['depth_dim'] = 'depth'
teos10_kwargs['output_variables'] = ['T', 'SP', 'SIGMA0', 'CT', 'SA']
#----------------------------------------------------------


def make_profiles(month):
    
    # Open temperature and salinity dataset
    ds_psal = io.open_isas_mfdata(month, variable='PSAL', **io_kwargs) 
    ds_temp = io.open_isas_mfdata(month, variable='TEMP',  **io_kwargs)
    ds = xr.merge([ds_temp, ds_psal], join='inner', compat='override')

    # Interpolation
    profiles = pftools.interp_on_constant_levels(ds, **interp_kwargs)

    # Computation of TEOS 10 variables 
    profiles_teos10 = pftools.teos10_variables(profiles, **teos10_kwargs)

    # Run a quality control
    sigma0 = pftools.quality_control(profiles_teos10['SIGMA0'])

    # Compute N2 profiles
    n2_profiles = pftools.nsquared(sigma0)
    curvature_profiles = pftools.curvature(sigma0)
    columnar_buoyancy =  pftools.columnar_buoyancy(sigma0)

    ds_out = profiles_teos10.assign({'N2': n2_profiles, 
                                     'KAPPA': curvature_profiles,
                                     'BC': columnar_buoyancy})
    ds_out = ds_out.chunk({'time': 10000, 'depth':-1})
    print(ds_out)

    io.save_stratification_profiles(ds_out, month)


if __name__ == "__main__":
    # Open and run a local cluster with multi-threading
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client)
    
    for month in range(1, 13):
        make_profiles(month)
    
    