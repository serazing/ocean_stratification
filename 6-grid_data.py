#!/usr/bin/env python

import xarray as xr
import numpy as np
import pandas as pd

def hist_data(da, statistics='quantile', quantile=0.5, 
              lon_min=0., lon_max=360., lon_step=2,
              lat_min=-80., lat_max=80., lat_step=2):
    import pyinterp
    import pyinterp.backends.xarray
    hist2d = pyinterp.Histogram2D(pyinterp.Axis(np.arange(lon_min, lon_max, lon_step), is_circle=True),
                                  pyinterp.Axis(np.arange(lat_min, lat_max, lat_step)))
    hist2d.clear()
    hist2d.push(da['longitude'], da['latitude'], da)
    
    if statistics == 'quantile':
        nearest = hist2d.variable('quantile', quantile)
    else: 
        nearest = hist2d.variable(statistics)
    #nobs = hist2d.variable('count')
    
    lon, lat = np.meshgrid(hist2d.x, hist2d.y, indexing='ij')
    
    res = xr.DataArray(nearest, dims=('longitude', 'latitude'),
                       coords={'longitude': xr.IndexVariable('longitude', hist2d.x),
                               'latitude': xr.IndexVariable('latitude', hist2d.y)})
    #res = res.to_dataset(name='binned_data').assign(nobs=xr.DataArray(nobs, dims=('longitude', 'latitude')))
    
    return res

def bin_data(da, how='variance',
             lon_min=0., lon_max=363, lon_step=2, 
             lat_min=-80., lat_max=80., lat_step=2):
    import pyinterp
    import pyinterp.backends.xarray
    binning = pyinterp.Binning2D(pyinterp.Axis(np.arange(lon_min, lon_max, lon_step), is_circle=True),
                                 pyinterp.Axis(np.arange(lat_min, lat_max, lat_step)))
    binning.clear()
    binning.push(da['longitude'], da['latitude'], da, False)
    nearest = binning.variable(how)
    
    lon, lat = np.meshgrid(binning.x, binning.y, indexing='ij')
    res = xr.DataArray(nearest, dims=('longitude', 'latitude'),
                       coords={'longitude': xr.IndexVariable('longitude', binning.x),
                               'latitude': xr.IndexVariable('latitude', binning.y)})
    res = res.to_dataset(name='binned_data').assign(nobs=xr.DataArray(nobs, dims=('longitude', 'latitude')))
    return res


def find_max_intensity(ds):
    intensity = ds['intensity']
    max_intensity = intensity.dropna('time', how='all').max('peak')
    ds_max = ds.where(intensity == max_intensity).mean('peak')
    return ds_max


def mask_upper_vs_max(ds_upper, ds_max):
    ds_upper = ds_upper.dropna('time')
    ds_max = ds_max.dropna('time')
    mask = (ds_max['depth'] != ds_upper['depth'])  
    return mask

def peak_assymetry(ds):
    assymetry = ((ds['width_right'] - ds['depth']) - 
                 (ds['depth'] - ds['width_left']))
    return assymetry

#def max_curvature(ds, sigma0):
    #z_max = ds['width_right']
    #ds['KAPPA'].where(ds['z'] <= z_max).argmax('depth')

def delta_variables(full_dataset):    
    month_vector = xr.IndexVariable('month', range(1, 13))    
    for month in month_vector:
        ds = full_dataset[int(month)] 
        for var in ['T', 'SP', 'SIGMA0']:
            delta_var = ds[var + '_left'] - ds[var + '_right']
            ds = ds.assign(**{'delta_' + var: delta_var})
        full_dataset[int(month)] = ds
    return full_dataset

def distance_mld_uop(full_dataset):
    month_vector = xr.IndexVariable('month', range(1, 13))    
    for month in month_vector:
        ds = full_dataset[int(month)] 
        dist = ds['MLD_MIN'] - ds['width_left']
        ds = ds.assign(**{'dist_mld_min_uop': dist})
        full_dataset[int(month)] = ds
    return full_dataset

def grid_statistics(full_dataset, variable):
    month_vector = xr.IndexVariable('month', range(1, 13))
    statistics = ['mean', 'variance', 'skewness', 'kurtosis', 'min', 'max', 'count']
    list_of_stats = []
    for stat in statistics:
            stat_res = xr.concat([hist_data(full_dataset[int(month)][variable], stat) for month in month_vector],
                                 dim=month_vector)
            list_of_stats.append(stat_res) 
    
    quantiles = {'q1': 0.25, 'median': 0.5, 'q3': 0.75}
    for q in quantiles:
        stat_res = xr.concat([hist_data(full_dataset[int(month)][variable], statistics='quantile', quantile=quantiles[q]) for month in month_vector],
                                 dim=month_vector)
        list_of_stats.append(stat_res) 
    full_statistics = statistics + list(quantiles)
    return xr.concat(list_of_stats, dim=xr.IndexVariable('stat', full_statistics))



def make_gridded_dataset():
    month_vector = xr.IndexVariable('month', range(1, 13))

    mlds = {int(month): xr.open_mfdataset('/home/serazin/Data/UOP/MLDS/mlds_m%02i.nc' % month)
            for month in month_vector}
    
    ml_variables = {int(month): xr.open_mfdataset('/home/serazin/Data/UOP/N2_PEAKS/ml_variables_m%02i.nc' % month)
                    for month in month_vector}

    n2_peaks = {int(month): xr.open_mfdataset('/home/serazin/Data/UOP/N2_PEAKS/n2_peaks_m%02i.nc' % month) 
                for month in month_vector}
        
    peak_edges = {int(month): xr.open_mfdataset('/home/serazin/Data/UOP/N2_PEAKS/variables_at_peak_edges_m%02i.nc' % month) 
                  for month in month_vector}

    full_dataset = {int(month): xr.merge([mlds[int(month)], 
                                          n2_peaks[int(month)], 
                                          peak_edges[int(month)]]).dropna(dim='time', subset=['MLD_DR003'])
                    for month in month_vector}
    
    full_dataset = delta_variables(full_dataset)
    
    full_dataset = distance_mld_uop(full_dataset)

    n2_upper_peak = {int(month): full_dataset[int(month)].isel(peak=0) 
                     for month in month_vector}

    n2_max_peak = {int(month): find_max_intensity(full_dataset[int(month)]) for month in month_vector}
    
    # Mixed layer depth
    mld_dr003  = grid_statistics(n2_upper_peak, 'MLD_DR003')
    mld_dt02  = grid_statistics(n2_upper_peak, 'MLD_DT02')
    mld_dreqdt02  = grid_statistics(n2_upper_peak, 'MLD_DREQDT02')
    mld_min =  grid_statistics(n2_upper_peak, 'MLD_MIN')
    
    # UOP intensity  
    uop_intensity = grid_statistics(n2_upper_peak, 'intensity')
 
    # UOP depth  
    uop_depth = grid_statistics(n2_upper_peak, 'depth')
    
    # UOP upper boundary
    uop_upper_boundary = grid_statistics(n2_upper_peak, 'width_left')
    
    dist_mld_min_uop = grid_statistics(n2_upper_peak, 'dist_mld_min_uop')
    
    # UOP lower boundary
    uop_lower_boundary = grid_statistics(n2_upper_peak, 'width_right')
        
    # UOP thickness  
    uop_width = grid_statistics(n2_upper_peak, 'width')
    
    # UOP columnar buoyancy
    uop_bc = grid_statistics(n2_upper_peak, 'BC_right')
    
    # UOP delta T
    uop_delta_t = grid_statistics(n2_upper_peak, 'delta_T')
    
   # UOP delta T
    uop_delta_s = grid_statistics(n2_upper_peak, 'delta_SP')
    
    # UOP delta sigma0
    uop_delta_sigma0 = grid_statistics(n2_upper_peak, 'delta_SIGMA0')
    
    # Mixed layer N2
    ml_n2 = grid_statistics(ml_variables, 'N2_ml')
    
    # Mixed layer S
    ml_s = grid_statistics(ml_variables, 'SP_ml')
    
    # Mixed layer T
    ml_t = grid_statistics(ml_variables, 'T_ml')
    
    # Mixed layer SIGMAO
    ml_sigma0 = grid_statistics(ml_variables, 'SIGMA0_ml')
    
    # Mixed layer delta SIGMAO
    delta_ml_sigma0 = grid_statistics(ml_variables, 'SIGMA0_delta_ml')
    
    # Mixed layer delta temperature
    delta_ml_t = grid_statistics(ml_variables, 'T_delta_ml')
    
    # Mixed layer delta salinity
    delta_ml_s = grid_statistics(ml_variables, 'SP_delta_ml')
    
        
    
    #n2_upper_peak_assymetry = xr.concat([hist_data(peak_assymetry(n2_upper_peak[int(month)])) for month in month_vector],
    #                                     dim=month_vector)
    
    #n2_upper_peak_bc = xr.concat([hist_data(n2_upper_peak[int(month)]['BC_right']) for month in month_vector],
    #                                        dim=month_vector)
    
    #n2_upper_peak_bc_var = xr.concat([bin_data(n2_upper_peak[int(month)]['BC_right'], how='variance') for month in month_vector],
    #                                        dim=month_vector)
    
    
    #n2_upper_peak_delta_sigma0 = xr.concat([hist_data(delta_var(n2_upper_peak[int(month)], var='SIGMA0')) for month in month_vector],
    #                                        dim=month_vector)
    
    #n2_upper_peak_delta_T = xr.concat([hist_data(delta_var(n2_upper_peak[int(month)], var='T')) for month in month_vector],
    #                                        dim=month_vector)
        
    #n2_upper_peak_delta_S = xr.concat([hist_data(delta_var(n2_upper_peak[int(month)], var='SP')) for month in month_vector],
    #                                        dim=month_vector)  
    

    #mask = {int(i): mask_upper_vs_max(n2_upper_peak[int(i)], n2_max_peak[int(i)]) for i in month_vector}

    #ratio = xr.concat([hist_data(mask[int(month)]) for month in month_vector],
    #                                     dim=month_vector)
    
    #nb_peaks = xr.concat([hist_data((n2_peaks[int(month)].intensity / n2_peaks[int(month)].intensity).sum('peak')) for month in month_vector], dim=month_vector) 
    
    #delta_uop_mld_rvar = xr.concat([hist_data(n2_upper_peak[int(month)]['depth'] - full_dataset[int(month)]['MLD_RVAR']) for month in month_vector],
    #                                     dim=month_vector)
    
    #delta_uop_mld_003 = xr.concat([hist_data(n2_upper_peak[int(month)]['depth'] - full_dataset[int(month)]['MLD_003']) for month in month_vector],
    #                                     dim=month_vector)

    var_dict = {'mld_dr003': mld_dr003,
                'mld_dt02': mld_dt02, 
                'mld_dreqdt02': mld_dreqdt02, 
                'mld_min': mld_min, 
                'dist_mld_min_uop': dist_mld_min_uop,
                'uop_intensity': uop_intensity,
                'uop_depth': uop_depth,
                'uop_thickness': uop_width,
                'uop_bc': uop_bc,
                'uop_delta_t': uop_delta_t,
                'uop_delta_s': uop_delta_s,
                'uop_delta_sigma0': uop_delta_sigma0,
                'uop_upper_boundary': uop_upper_boundary,
                'uop_lower_boundary': uop_lower_boundary,
                'ml_t': ml_t,
                'ml_s': ml_s,
                'ml_sigma0': ml_sigma0, 
                'ml_n2': ml_n2,
                'ml_delta_t': delta_ml_t, 
                'ml_delta_s': delta_ml_s,
                'ml_delta_sigma0': delta_ml_sigma0,
               }
                

                #'uop_delta_sigma0': n2_upper_peak_delta_sigma0.binned_data,
                #'uop_delta_temperature': n2_upper_peak_delta_T.binned_data,
                #'uop_delta_salinity': n2_upper_peak_delta_S.binned_data,
                #'n2_uop_intensity_var': n2_upper_peak_intensity_var.binned_data, 
        
                #'n2_uop_depth': n2_upper_peak_depth.binned_data,
                #'n2_uop_depth_var': n2_upper_peak_depth_var.binned_data,
        
                #'n2_uop_thickness': n2_upper_peak_width.binned_data,
                #'n2_uop_thickness_var': n2_upper_peak_width_var.binned_data,
                #'n2_uop_columnar_buoyancy': n2_upper_peak_bc.binned_data,

                #'n2_uop_columnar_buoyancy_var': n2_upper_peak_bc_var.binned_data,
                #'n2_uop_assymetry': n2_upper_peak_assymetry.binned_data,

                #'n2_max_intensity': n2_max_peak_intensity.binned_data, 
                #'n2_max_depth': n2_max_peak_depth.binned_data, 
                #'n2_max_thickness': n2_max_peak_width.binned_data,
                #'nobs': n2_upper_peak_intensity.nobs,
                #'ratio': ratio.binned_data,
                #'delta_uop_mld_rvar':  delta_uop_mld_rvar.binned_data,
                #'delta_uop_mld_003':  delta_uop_mld_003.binned_data,
                #'nb_peaks': nb_peaks.binned_data    

    ds = xr.Dataset(var_dict).transpose('month', 'latitude', 'longitude', 'stat')
    print(ds)

    ds.to_netcdf('/home/serazin/Data/UOP/GRIDDED_FIELDS/full_dataset.nc', mode='w')

    
if __name__ == "__main__":
    make_gridded_dataset()