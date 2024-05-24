import xarray as xr
import pandas as pd
import numpy as np

# Set variables
ISAS_PATH = '/home/serazin/Data/ISAS'
UOP_PATH = '/home/serazin/Data/UOP'


MONTH_LIST = ['January', 'February', 'March',
              'April', 'May', 'June',
              'July', 'August', 'September', 
              'October', 'November', 'December']

def decode_isas_time(ds):
    """
    Decode ISAS timestamps using the correct origin and the Julian calendar
    """
    origin = pd.Timestamp(ds.REFERENCE_DATE_TIME.data.astype(str).item()[:8], tz=None)
    time = pd.to_timedelta(ds.JULD.data, unit='D') + origin
    ds = ds.assign_coords(time=xr.IndexVariable('time', time))
    return ds


def preprocess_isas(ds):
    # Rename dimensions for easier interfacing with xarray
    ds = ds.rename_dims({'N_PROF': 'time'})
    # Manually decode time vector
    ds = decode_isas_time(ds)
    ds = ds.sortby('time')
    # Remove duplicated values
    ds = ds.sel(time=~ds.indexes['time'].duplicated())
    
    # Rename some future coordinates
    ds = ds.rename({'DEPH' : 'depth'})
    ds = ds.swap_dims({'N_LEVELS': 'depth'})
    # Force longitude values to sit between 0° and 360°
    lon_attrs = ds['LONGITUDE'].attrs.copy()
    ds['LONGITUDE'] = (ds['LONGITUDE'] + 360) % 360
    ds['LONGITUDE'].attrs = lon_attrs
    ds = ds.set_coords(('LATITUDE', 'LONGITUDE'))
    ds = ds.rename({'LONGITUDE': 'longitude',
                    'LATITUDE': 'latitude'})
    return ds


def filter_isas(ds, variable, lon_min=0, lon_max=360, 
                lat_min=-90, lat_max=90, 
                qc_max=4, delayed_mode=True):
    # Select only region and depth of interest
    ds = ds.where((ds['LONGITUDE'] >= lon_min) &
                  (ds['LONGITUDE'] <= lon_max) &
                  (ds['LATITUDE'] >= lat_min) &
                  (ds['LATITUDE'] <= lat_max), drop=True)
    #ds = ds.where((ds['z'] >= z_min) &
    #              (ds['z'] <= z_max), drop=True)
    ds = ds.where(ds[variable + '_QC'] <= qc_max, drop=True)
    if delayed_mode:
        ds = ds.where(ds['DATA_MODE'] == b'D')
    ds = ds.dropna('time', how='all')
    return ds


def open_isas_data(year, month, variable='PSAL', lon_min=0, lon_max=360, lat_min=-90, lat_max=90, z_min=0, z_max=5000):
    path = '%s/data/%04i/' % (ISAS_PATH, year)
    ds = xr.open_dataset(path + 'ISAS20_ARGO_%04i%02i15_dat_%s.nc' % (year, month, variable),
                         decode_times=False)
    # Preprocess data
    ds = preprocess_isas(ds, month)
    # Filter data
    ds = filter_isas(ds, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, z_min=z_min, z_max=z_max)
    return ds


def open_isas_mfdata(month, variable='PSAL', 
                     qc_max=4, delayed_mode=True,
                     lon_min=0, lon_max=360, 
                     lat_min=-90, lat_max=90):
    import glob
    list_of_file = sorted(glob.glob('%s/data/????/ISAS20_ARGO_????%02i15_dat_%s.nc' % (ISAS_PATH, month, variable)))
    ds_monthly = xr.open_mfdataset(list_of_file, decode_times=False, preprocess=preprocess_isas, parallel=True)
    ds_monthly = dict(ds_monthly.groupby('time.month'))[month]
    ds_monthly = filter_isas(ds_monthly, variable, 
                             lon_min=lon_min, lon_max=lon_max, 
                             lat_min=lat_min, lat_max=lat_max, 
                             qc_max=qc_max, delayed_mode=delayed_mode)
    ds_monthly = ds_monthly.where(ds_monthly[variable + '_QC'] <= qc_max, drop=True)
    return ds_monthly


def open_isas_mffield(variable='PSAL', qc_max=4,
                     lon_min=0, lon_max=360, 
                     lat_min=-90, lat_max=90, 
                     z_min=0, z_max=5000):
    import glob
    list_of_file = sorted(glob.glob('%s/field/????/ISAS20_ARGO_??????15_fld_%s.nc' % (ISAS_PATH, variable)))
    ds_monthly = xr.open_mfdataset(list_of_file, decode_times=True, parallel=True)
    #ds_monthly = ds_monthly.rename({'time': 'year'})
    #ds_monthly = filter_isas(ds_monthly, 
    #                         lon_min=lon_min, lon_max=lon_max, 
    #                         lat_min=lat_min, lat_max=lat_max, 
    #                         z_min=z_min, z_max=z_max)
    #ds_monthly = ds_monthly.where(ds_monthly[variable + '_QC'] <= qc_max, drop=True)
    return ds_monthly


def save_stratification_profiles(ds, month):
    path = '%s/N2_PROFILES/' % UOP_PATH
    ds.to_zarr(path + 'n2_m%02i_profiles.zarr' % month, mode='w')

    
def open_full_profiles():
    n2_peaks = {month: xr.open_dataset('/home/serazin/Data/UOP/N2_PEAKS/N2_peaks_m%02i.nc' % (i + 1))
           for i, month in enumerate(MONTH_LIST)}

    profiles = {month: xr.open_zarr('/home/serazin/Data/UOP/N2_PROFILES/N2_m%02i_profiles.zarr/' % (i + 1))
           for i, month in enumerate(MONTH_LIST)}

    full_profiles = {month: xr.merge([n2_peaks[month], profiles[month]]) 
                 for i, month in enumerate(MONTH_LIST)}
    
    return full_profiles