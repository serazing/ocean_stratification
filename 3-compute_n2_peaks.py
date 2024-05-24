#!/usr/bin/env python

import xarray as xr
import isas_io as io
import profile_tools as pftools

#----------------------------------------------------------
# Set batch parameters
#----------------------------------------------------------

dim = 'depth'


def find_peaks(month):
    
    ds = xr.open_zarr("/home/serazin/Data/UOP/N2_PROFILES/n2_m%02i_profiles.zarr" % month).load()
    print('\n')
    
    list_of_peaks = []
    for i in range(0, ds.sizes['time']):
        prof = ds['N2'].isel(time=i)
        peaks = pftools.find_profile_peaks(prof, rel_height=0.707)
        #print(peaks)
        list_of_peaks.append(peaks)
    output = xr.concat(list_of_peaks, dim='time')
    print(output)
    output.to_netcdf("/home/serazin/Data/UOP/N2_PEAKS/n2_peaks_m%02i.nc" % month, mode='w')
    #def peaks(i):
    #    prof = ds['N2'].isel(time=i)
    #    res = pftools.find_profile_peaks(prof)
    #    return res
    
    
    #import dask
    #import dask.bag as db
    #b = db.from_sequence(range(0, ds.sizes['time']), npartitions=100)
    #b = b.map(peaks)
    #res = dask.compute(b)
    



if __name__ == "__main__":
    # Open and run a local cluster with multi-threading
    #from dask.distributed import Client, LocalCluster
    #cluster = LocalCluster(processes=False, n_workers=8)
    #client = Client(cluster)
    #print(client)
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nb_proc = comm.Get_size()
    proc_indx = comm.Get_rank()

    for month in range(1, 13):
        if (month % nb_proc) == proc_indx:
            print('Processing month %s on core %s \n' % (month, proc_indx))
            print('-------------------------------\n')
            find_peaks(month)
    
