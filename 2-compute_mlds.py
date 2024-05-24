#!/usr/bin/env python

import xarray as xr
import gsw
import isas_io as io
import profile_tools as pftools


#----------------------------------------------------------
# Set batch parameters
#----------------------------------------------------------

dim = 'depth'


def compute_mlds(month):
    
    ds = xr.open_zarr("/home/serazin/Data/UOP/N2_PROFILES/n2_m%02i_profiles.zarr" % month)
    ds = ds.chunk({'time': 1e4}).dropna('time', how='all')
    sigma0 = pftools.quality_control(ds['SIGMA0'], dim=dim)
    ct = pftools.quality_control(ds['CT'], dim=dim)
    pres = gsw.p_from_z(-ds.depth, ds.latitude)
    alpha = gsw.alpha(ds['SA'], ds['CT'], pres).sel(time=ct.time)
    
    output_dict = {}    
    
    # Calculation of estimates of the mixed layer depth
    # Density threshold 0.03 kg.m-3
    mld_dr003 = pftools.mld_dr(sigma0, 0.03).compute()
    mld_dr003.attrs['long_name'] = 'Mixed layer depth with density threshold 0.03 kg.m-3'
    mld_dr003.attrs['standard_name'] = 'mixed_layer_depth_sigma0_003'
    mld_dr003.attrs['units'] = 'm'
    output_dict['MLD_DR003'] = mld_dr003
    
    # Temperature threshold 0.2 deg C
    mld_dt02 = pftools.mld_dt(ct, 0.2).compute()
    mld_dt02.attrs['long_name'] = 'Mixed layer depth with temperature threshold 0.2 deg C'
    mld_dt02.attrs['standard_name'] = 'mixed_layer_depth_temperature_02'
    mld_dt02.attrs['units'] = 'm'
    output_dict['MLD_DT02'] = mld_dt02
    
    # Density threshold equivalent to temperature threshold 0.2 deg C
    mld_dreqdt02 = pftools.mld_dreqdt(sigma0, alpha, 0.2).compute()
    mld_dreqdt02.attrs['long_name'] = 'Mixed layer depth with density threshold equivalent to a temperature threshold 0.2 deg C'
    mld_dreqdt02.attrs['standard_name'] = 'mixed_layer_depth_denisty_equivalent_temperature_02'
    mld_dreqdt02.attrs['units'] = 'm'
    output_dict['MLD_DREQDT02'] = mld_dreqdt02
    
    # Density threshold equivalent to temperature threshold 0.2 deg C
    
    mld_min = xr.concat([mld_dr003, mld_dt02, mld_dreqdt02], dim='mld_variable').min(dim='mld_variable')
    mld_min.attrs['long_name'] = 'Shallowest mixed layer depth between computed variables' 
    mld_min.attrs['standard_name'] = 'minimum_mixed_layer_depth'
    mld_min.attrs['units'] = 'm'
    output_dict['MLD_MIN'] = mld_min
    
    # Density threshold 0.01 kg.m-3
    #mld_001 = pftools.mld_thres(sigma0, 0.01).compute()
    #mld_001.attrs['long_name'] = 'Mixed layer depth with density threshold 0.01 kg.m-3'
    #mld_001.attrs['standard_name'] = 'mixed_layer_depth_sigma0_003'
    #mld_001.attrs['units'] = 'm'
    #output_dict['MLD_001'] = mld_001
    # Relative variance method
    #mld_relvar = pftools.mld_rvar(sigma0).compute()
    #mld_relvar.attrs['long_name'] = 'Mixed layer depth with relative variance method'
    #mld_relvar.attrs['standard_name'] = 'mixed_layer_depth_rvar'
    #mld_relvar.attrs['units'] = 'm'
    #output_dict['MLD_RVAR'] = mld_relvar
    
    res = xr.Dataset(output_dict)
    
    res.to_netcdf("/home/serazin/Data/UOP/MLDS/mlds_m%02i.nc" % month, mode='w')



if __name__ == "__main__":
    # Open and run a local cluster with multi-threading
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client)
    for month in range(1, 13):
        compute_mlds(month)
    
