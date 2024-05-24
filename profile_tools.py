import isas_io as io
import xarray as xr
import numpy as np
from numba import guvectorize
import math

def teos10_variables(profile, 
                     salinity_variable='PSAL', 
                     temperature_variable='TEMP',
                     depth_dim='depth',
                     output_variables=['T', 'SP', 'P',
                                       'SA', 'CT', 'SIGMA0'],
                     anomaly=False):
    
    import gsw
    # In situ temperature and practical salinity
    output_dict = {}
    psal = profile[salinity_variable]
    temp = profile[temperature_variable]
    if anomaly:
        psal = psal -  profile[salinity_variable + '_CLMN']
        temp = temp -  profile[temperature_variable + '_CLMN']
    if 'T' in output_variables:
        output_dict['T'] = temp
    if 'SP' in output_variables:
        output_dict['SP'] = psal
    
    # Longitude and latitude coordinates
    lon = profile['longitude'] * xr.ones_like(temp)
    lat = profile['latitude'] * xr.ones_like(temp)
    
    # Vertical coordinates
    z = - profile[depth_dim]    
    pres = ((gsw.p_from_z(z, lat) *       
             xr.ones_like(temp)).transpose(*temp.dims))
    pres.attrs['long_name'] = 'Pressure'
    pres.attrs['standard_name'] = 'pressure'
    pres.attrs['units'] = 'db'
    if 'P' in output_variables:
        output_dict['P'] = pres
    
    # Absolute Salinity
    sa = gsw.SA_from_SP(psal, pres, lon, lat)
    sa.attrs = psal.attrs
    sa.attrs['long_name'] = 'Absolute Salinity (TEOS-10)'
    sa.attrs['standard_name'] = 'absolute_salinity'
    if 'SA' in output_variables:
        output_dict['SA'] = sa
        
    # Potential temperature with a reference pressure of zero dbar
    pt0 = gsw.pt0_from_t(psal, temp, pres)
    pt0.attrs = temp.attrs
    pt0.attrs['long_name'] = 'Potential temperature referenced to surface pressure'
    pt0.attrs['standard_name'] = 'potential_temperature_0db'
    if 'PT0' in output_variables:
        output_dict['PT0'] = pt0
        
    # Conservative Temperature from Potential Temperature
    ct = gsw.CT_from_pt(sa, pt0)
    ct.attrs = temp.attrs
    ct.attrs['long_name'] = 'Conservative Temperature (TEOS-10)'
    ct.attrs['standard_name'] = 'conservative_temperature'
    if 'CT' in output_variables:
        output_dict['CT'] = ct
        
    # Potential Density with reference at the surface
    if 'SIGMA0' in output_variables:
        sigma0 = gsw.rho(sa, ct, 0)
        sigma0.attrs['long_name'] = 'Potential density referenced to surface pressure'
        sigma0.attrs['standard_name'] = 'potential_density_0db'
        sigma0.attrs['units'] = 'kg/m3'
        output_dict['SIGMA0'] = sigma0
        
    #if 'ALPHA' in output_variables:
    #    alpha.attrs['long_name'] = 'Thermal expansion coefficient with respect to Conservative Temperature'
    #    alpha.attrs['standard_name'] = 'thermal_expansion_coefficient'
    #    alpha.attrs['units'] = '1/K'
    #    output_dict['ALPHA'] = alpha
    
    #if 'BETA' in output_variables:
    #    beta.attrs['long_name'] = 'Haline contraction coefficient with respect to Absolute Salinity'
    #    beta.attrs['standard_name'] = 'haline_contraction_coefficient'
    #    beta.attrs['units'] = '1/K'
    #    output_dict['BETA'] = beta
    
    # Specific entropy
    if 'S' in output_variables:
        s = gsw.entropy_from_CT(sa, ct)
        s.attrs['long_name'] = 'Specific entropy'
        s.attrs['standard_name'] = 'specific_entropy'
        s.attrs['units'] = 'J/(kg.K)'
        output_dict['S'] = s   

    return xr.Dataset(output_dict)    


def quality_control(phi, nb_min=10, dim='depth'):
    
    nb_valid_200 = (1 - np.isnan(phi.sel(**{dim: slice(0, 200)}))).sum(dim)               
    #nb_valid_points = (1 - np.isnan(phi)).sum(dim)
    phi = phi.where(nb_valid_200 >= 20, drop=True)
    #np.isnan(phi.sel(depth=slice(0, 200))).sum(dim)
    return phi


def expfit(phi, dim, phi_min, phi_max):
    
    def exp_func(z, phi_bottom, delta_phi, z0):
        return phi_bottom - delta_phi * np.exp(- z / z0)
    
    #mean_phi_max = phi.max(dim).mean()
    #mean_phi_min = phi.min(dim).mean()
    delta_phi = phi_max - phi_min
    
    fit = phi.curvefit(phi[dim], 
                       exp_func,  
                       p0={'phi_bottom': phi_max,
                           'delta_phi': delta_phi,
                           'z0': 1000})
    phi_fit = exp_func(phi[dim], 
               fit.curvefit_coefficients.sel(param='phi_bottom'),
               fit.curvefit_coefficients.sel(param='delta_phi'),
               fit.curvefit_coefficients.sel(param='z0'))
    return phi_fit


def mld_dr(phi, phi_step, ref_depth=10, dim='depth'):
    
    # Get the depth vector
    z = phi[dim]
    # Find the index closest to the reference depth
    k_ref = int((np.abs(z - ref_depth)).argmin())
    
    from numba import guvectorize
    @guvectorize(['void(float64[:], float64[:], intp, float64, float64[:])'],
              '(n),(n),(),()->()', nopython=True)
    def mld_gufunc(phi, z, k_ref, phi_step, res):
        phi_ref = phi[k_ref]
        phi_mlb = phi_ref + phi_step
        for k in range(k_ref, phi.shape[0]):
            if phi[k] >= phi_mlb:
                k_mlb = k
                break
        # Make a linear interpolation to find the approximate MLD
        delta_z = z[k_mlb - 1] - z[k_mlb]
        delta_phi = phi[k_mlb - 1] - phi[k_mlb]
        alpha = delta_z / delta_phi
        beta = z[k_mlb] - alpha * phi[k_mlb]
        res[0] = alpha * phi_mlb + beta
        
    mld = xr.apply_ufunc(mld_gufunc, phi, z, k_ref, phi_step, 
                         input_core_dims=[[dim], [dim], [], []],
                         dask='parallelized')
    return mld


def mld_dt(ct, ct_step, ref_depth=10, dim='depth'):
    
    # Get the depth vector
    z = ct[dim]
    # Find the index closest to the reference depth
    k_ref = int((np.abs(z - ref_depth)).argmin())
    
    from numba import guvectorize
    @guvectorize(['void(float64[:], float64[:], intp, float64, float64[:])'],
              '(n),(n),(),()->()', nopython=True)
    def mld_gufunc(phi, z, k_ref, phi_step, res):
        phi_ref = phi[k_ref]
        phi_mlb = phi_ref - phi_step
        for k in range(k_ref, phi.shape[0]):
            if phi[k] <= phi_mlb:
                k_mlb = k
                break
        # Make a linear interpolation to find the approximate MLD
        delta_z = z[k_mlb - 1] - z[k_mlb]
        delta_phi = phi[k_mlb - 1] - phi[k_mlb]
        alpha = delta_z / delta_phi
        beta = z[k_mlb] - alpha * phi[k_mlb]
        res[0] = alpha * phi_mlb + beta
        
    mld = xr.apply_ufunc(mld_gufunc, ct, z, k_ref, ct_step, 
                         input_core_dims=[[dim], [dim], [], []],
                         dask='parallelized')
    return mld

def mld_dreqdt(phi, alpha, ct_step, ref_depth=10, dim='depth'):
    
    # Get the depth vector
    z = phi[dim]
    # Find the index closest to the reference depth
    k_ref = int((np.abs(z - ref_depth)).argmin())
        
    #import gsw
    #gsw.alpha([sa, sa], [ct, ct + 0.02], 0)
    
    from numba import guvectorize
    @guvectorize(['void(float64[:], float64[:], float64[:], intp, float64, float64[:])'],
                  '(n),(n),(n),(),()->()', nopython=True)
    def mld_gufunc(phi, alpha, z, k_ref, ct_step, res):
        phi_ref = phi[k_ref]
        alpha_ref = alpha[k_ref]
        phi_mlb = phi_ref + alpha * phi * ct_step
        for k in range(k_ref, phi.shape[0]):
            if phi[k] >= phi_mlb[k]:
                k_mlb = k
                break
        # Make a linear interpolation to find the approximate MLD
        delta_z = z[k_mlb - 1] - z[k_mlb]
        delta_phi = phi[k_mlb - 1] - phi[k_mlb]
        a = delta_z / delta_phi
        b = z[k_mlb] - a * phi[k_mlb]
        res[0] = a * phi_mlb[k_mlb] + b
        
    mld = xr.apply_ufunc(mld_gufunc, phi, alpha, z, k_ref, ct_step, 
                         input_core_dims=[[dim], [dim], [dim], [], []],
                         dask='parallelized')
    return mld


def mld_dreqdt_v2(ct, sa, ct_step, ref_depth=10, dim='depth'):
    
    # Get the depth vector
    z = ct[dim]
    # Find the index closest to the reference depth
    k_ref = int((np.abs(z - ref_depth)).argmin())
        
    import gsw
    sigma0 = gsw.rho(sa, ct, 0)
    sigma0_ct = gsw.rho(sa, ct - ct_step, 0)
    sigma0_step = (sigma0_ct - sigma0).sel(depth=ref_depth)
    print(sigma0_step.load())
    
    from numba import guvectorize
    @guvectorize(['void(float64[:], float64[:], intp, float64, float64[:])'],
              '(n),(n),(),()->()', nopython=True)
    def mld_gufunc(phi, z, k_ref, phi_step, res):
        phi_ref = phi[k_ref]
        phi_mlb = phi_ref + phi_step
        for k in range(k_ref, phi.shape[0]):
            if phi[k] >= phi_mlb:
                k_mlb = k
                break
        # Make a linear interpolation to find the approximate MLD
        delta_z = z[k_mlb - 1] - z[k_mlb]
        delta_phi = phi[k_mlb - 1] - phi[k_mlb]
        alpha = delta_z / delta_phi
        beta = z[k_mlb] - alpha * phi[k_mlb]
        res[0] = alpha * phi_mlb + beta
        
    mld = xr.apply_ufunc(mld_gufunc, sigma0, z, k_ref, sigma0_step, 
                         input_core_dims=[[dim], [dim], [], []],
                         dask='parallelized')
    return mld



def mld_curv(kappa, kappa_step, ref_depth=10, dim='depth'):
    
    # Get the depth vector
    z = phi[dim]
    # Find the index closest to the reference depth
    k_ref = int((np.abs(z - ref_depth)).argmin())
    
    from numba import guvectorize
    @guvectorize(['void(float64[:], float64[:], intp, float64, float64[:])'],
              '(n),(n),(),()->()', nopython=True)
    def mld_gufunc(phi, z, k_ref, phi_step, res):
        phi_ref = phi[k_ref]
        phi_mlb = phi_ref + phi_step
        for k in range(k_ref, phi.shape[0]):
            if phi[k] >= phi_mlb:
                k_mlb = k
                break
        # Make a linear interpolation to find the approximate MLD
        delta_z = z[k_mlb - 1] - z[k_mlb]
        delta_phi = phi[k_mlb - 1] - phi[k_mlb]
        alpha = delta_z / delta_phi
        beta = z[k_mlb] - alpha * phi[k_mlb]
        res[0] = alpha * phi_mlb + beta
        
    mld = xr.apply_ufunc(mld_gufunc, phi, z, k_ref, phi_step, 
                         input_core_dims=[[dim], [dim], [], []],
                         dask='parallelized')
    return mld


def var_at_zref(bc, z_ref, dim='depth'):
    
    from numba import guvectorize
    @guvectorize(['void(float64[:], float64[:], float64, float64[:])'],
                  '(n),(n),()->()', nopython=True)
    
    def find_gufunc(phi, z, z_ref, res):
        for k in range(0, phi.shape[0]):
            if z[k] >= z_ref:
                k_ref = k
                break
        # Make a linear interpolation to find the approximate MLD
        delta_z = z[k_ref - 1] - z[k_ref]
        delta_phi = phi[k_ref - 1] - phi[k_ref]
        alpha = delta_phi / delta_z 
        beta = phi[k_ref] - alpha * z[k_ref]
        res[0] = alpha * z_ref + beta
        
    # Depth vector
    z = bc[dim]
    bc_ref = xr.apply_ufunc(find_gufunc, bc, z, z_ref,  
                            input_core_dims=[[dim], [dim], []],
                            dask='parallelized')
    return bc_ref

   
@guvectorize(['void(float64[:], float64[:], intp, intp, float64[:])'],
              '(n),(n),(),()->()', nopython=True)
def rvar(phi, z, k_min, k_max, mld):
    # Initialise variables
    phi_min = 9e6 
    phi_max = -9e6
    phi_mean = 0
    nb_nans = 0
    chi_min = 0.5
    delta = np.zeros_like(phi)
    
    # First part: find zn1       
    # Loop on depth levels
    for k in range(k_min, phi.shape[0] - 1):
        if z[k] > (2 * z[k_max]):
            break
        if math.isnan(phi[k]):
            nb_nans += 1
            continue
        # Get the number of points
        n = k - k_min + 1 - nb_nans
        # Compute the cumulative amplitude
        if phi[k] < phi_min:
            phi_min = phi[k]
        if phi[k] > phi_max:
            phi_max = phi[k]
        sigma = phi_max - phi_min
        # Compute the cumulative mean
        phi_mean = 1. / n * ((n - 1) * phi_mean + phi[k])
        # Compute the cumulative variance
        phi_var = 0
        for l in range(k_min, k + 1):
            if not math.isnan(phi[l]):
                phi_var = (phi[l] - phi_mean) ** 2 + phi_var
        phi_var = 1. / n * phi_var
        delta[k] = phi_var ** 0.5 
        # Get the relative variance
        if sigma != 0:
            chi = delta[k] / sigma
        else: 
            chi = np.nan
        if chi <= chi_min :
            chi_min = chi
            n1 = k
            
    # Second part: find zn2   
    # Do this second loop only if there are at least 3 points        
    if n >= 3:        
        delta_n1 = delta[n1 + 1] - delta[n1]

        l = n1 - 1
        while l >= k_min:
            r = (delta[l + 1] - delta[l]) / delta_n1
            if r <= 0.3:
                n2 = l
                break
            l -= 1
        else:
            n2 = 0
        mld[0] = z[n2]
    else: 
        mld[0] = -9e6

        
def mld_rvar(phi, k_min=0, k_max=None, dim='depth'):
    z = phi[dim]
    if k_max is None:
        linfit = phi.polyfit(dim, 3, skipna=True)['polyfit_coefficients']
        trend = xr.polyval(z, linfit)
        phi_dtr = (phi - trend)
        k_max = phi_dtr.argmin('depth')
    
    new_dims = [di for di in phi.dims if di != dim]
    new_coords = [co for co in phi.coords if co != dim]

    mld = xr.apply_ufunc(rvar, phi, z, k_min, k_max, 
                         input_core_dims=[[dim], [dim], [], []], 
                         dask='parallelized')
    
    return mld.where(mld > 0)
    

def nsquared(sigma0, dim='depth'):
    # Use the Savitzky-Golay filter to compute a smoothed version of first density 
    #derivative with a 5-point window and second order polynomials
    from scipy.signal import savgol_filter
    dz = sigma0[dim].diff(dim).isel(**{dim: 0}).data
    n2 = (9.81 / 1025 
          * savgol_filter(sigma0, 5, 2, deriv=1, delta=dz, axis=-1) 
          * xr.ones_like(sigma0))       
    n2.attrs['long_name'] = 'Brunst Vaisala frequency squared'
    n2.attrs['standard_name'] = 'brunst_vaisaila_frequency_squared'
    n2.attrs['units'] = 's-2'
    return n2


def curvature(phi, dim='depth', phi0=1025, h=2000):
    from scipy.signal import savgol_filter
    dz = phi[dim].diff(dim).isel(**{dim: 0}).data / h
    drho_dz = savgol_filter(phi / phi0, 5, 2, deriv=1, delta=dz)
    d2rho_dz2 = savgol_filter(phi / phi0, 5, 2, deriv=2, delta=dz)
    # Curvature
    kappa = np.abs(d2rho_dz2) / (1 + drho_dz ** 2) ** (3/2) * xr.ones_like(phi)
    kappa.attrs['long_name'] = 'Profile curvature'
    kappa.attrs['standard_name'] = 'profile_curvature'
    kappa.attrs['units'] = 'n/a'
    return kappa

#def columnar_buoyancy(phi, dim='depth'):
#    dz = phi[dim].diff(dim).isel(**{dim: 0}).data
#    rolling = phi.rolling(**{dim: phi.sizes[dim]}, min_periods=1)
#    window = rolling.construct('l').chunk({'l': 1})
#    window = window.assign_coords(l=window[dim].rename({dim: 'l'}))
#    bc = 9.81 / 1025 * ((phi * (phi[dim] + dz / 2.) - window.fillna(0.).integrate('l')))
#    bc.attrs['long_name'] = 'Columnar buoyancy anomaly'
#    bc.attrs['standard_name'] = 'columnar_buoyancy_anomaly'
#    bc.attrs['units'] = 'm2.s-2'
#    return bc

def columnar_buoyancy(phi, dim='depth'):
    from scipy import integrate
    z = phi[dim]
    axis = phi.get_axis_num(dim)
    bc = 9.81 / 1025 * (phi * z - integrate.cumtrapz(phi, x=z, initial=0, axis=axis))
    bc.attrs['long_name'] = 'Columnar buoyancy anomaly'
    bc.attrs['standard_name'] = 'columnar_buoyancy_anomaly'
    bc.attrs['units'] = 'm2.s-2'
    return bc

def find_profile_peaks(x, dim='depth', distance=5, wlen=50, rel_height=0.5,
                       prominence=None, height=None):
    """
    Find the first maximum of the vertical profile using the function
    scipy.signal.find_peaks
    
    Parameters
    ----------
    x : 1d array_like
        The vertical profile on which the first maximum is evaluated
    distance : int, optional
        Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    width : int, optional
        Required width of peaks in samples. 
        
    Returns
    -------
    x_max : float or array_like
        The first maximum of the vertical profile 
    
    """
    from scipy.signal import find_peaks, peak_widths

    # Minimum prominence set to be half of the standard deviation of the vertical profile
    sigma = float(x.std())
    if prominence is None:
        prominence = sigma
    if height is None:
        height = 2 * sigma
    depth = x[dim]
    dz = depth.diff(dim).mean().data

    # Find peaks and corresponding widths at half height
    peaks, properties = find_peaks(x.data,
                                   height=height,
                                   prominence=prominence,
                                   distance=distance,
                                   rel_height=rel_height,
                                   width=1, 
                                   wlen=wlen)
    #peak_width_half_height = peak_widths(x.data, peaks, rel_height=0.5)

    # Get peak prominence 
    #peaks_width_heights= peak_width_half_height[1]
    
    # Get peak width in samples and convert results to meters 
    peaks_width_left = properties['left_ips'] * dz
    peaks_width_right = properties['right_ips'] * dz
    peaks_width = peaks_width_right - peaks_width_left

    # Get peak intensity 
    peaks_intensity = x.isel(**{dim: peaks}).rename({dim: 'peak'})
    
    ds_peaks = xr.Dataset({'intensity': peaks_intensity,
                           'prominence': properties['prominences'] * xr.ones_like(peaks_intensity),
                           'left': properties['left_bases'] * xr.ones_like(peaks_intensity) * dz,
                           'right': properties['right_bases'] * xr.ones_like(peaks_intensity) * dz,
                           'width_height':  properties['width_heights'] * xr.ones_like(peaks_intensity),
                           'width_left': peaks_width_left * xr.ones_like(peaks_intensity),
                           'width_right': peaks_width_right * xr.ones_like(peaks_intensity),
                           'width': peaks_width * xr.ones_like(peaks_intensity),
                           'depth': peaks_intensity['peak'],
                           })
    
    #ds_peaks = ds_peaks.rename({dim: 'peak'})
    ds_peaks['peak'] = range(0, ds_peaks.sizes['peak'])  
    
    return ds_peaks


def interp_on_constant_levels(ds, dim='depth', max_depth=2000., dz=5.):
    ds_interpolated = ds.interp(**{dim: np.arange(0, max_depth + dz, dz)}, kwargs={"fill_value": "extrapolate"})
    return ds_interpolated


def run_profile_analysis(month, interp=True, smooth=True, n2=True, mld=True, save=True,
                          interp_kwargs={}, teos10_kwargs={}):
    
    output_dict = {}    
    
    ds_psal = io.open_isas_mfdata(month,  variable='PSAL', 
                                  lon_min=0, lon_max=360, 
                                  lat_min=-90, lat_max=90, 
                                  qc_max=4)
    
    ds_temp = io.open_isas_mfdata(month,  variable='TEMP', 
                                  lon_min=0, lon_max=360, 
                                  lat_min=-90, lat_max=90, 
                                  qc_max=4)
    
    ds = xr.merge([ds_temp, ds_psal], join='inner', compat='override')
    
    if interp:
        profiles = interp_on_constant_levels(ds, **interp_kwargs)
    else: 
        profiles = ds
        
    profiles_teos10 = teos10_variables(profiles, 
                                       salinity_variable='PSAL', 
                                       temperature_variable='TEMP',
                                       depth_dim='depth',
                                       output_variables=['T', 'SP', 'SIGMA0'])
    
    sigma0 = profiles_teos10['SIGMA0'].dropna('time', how='all')
  
    
    if mld:
        # Calculation of estimates of the mixed layer depth
        # Density threshold 0.03 kg.m-3
        mld_003 = mld_thres(sigma0, 0.03)
        mld_003.attrs['long_name'] = 'Mixed layer depth with density threshold 0.03 kg.m-3'
        mld_003.attrs['standard_name'] = 'mixed_layer_depth_sigma0_003'
        mld_003.attrs['units'] = 'm'
        output_dict['MLD_003'] = mld_003
        # Density threshold 0.01 kg.m-3
        mld_001 = mld_thres(sigma0, 0.01)
        mld_001.attrs['long_name'] = 'Mixed layer depth with density threshold 0.01 kg.m-3'
        mld_001.attrs['standard_name'] = 'mixed_layer_depth_sigma0_003'
        mld_001.attrs['units'] = 'm'
        output_dict['MLD_003'] = mld_001
        # Relative variance method
        mld_rvar = mld_rvar(sigma0)
        mld_rvar.attrs['long_name'] = 'Mixed layer depth with relative variance method'
        mld_rvar.attrs['standard_name'] = 'mixed_layer_depth_rvar'
        mld_rvar.attrs['units'] = 'm'
        output_dict['MLD_RVAR'] = mld_rvar
    
    if n2:
        n2_profiles = nsquared(sigma0)
        
    output_dict['SIGMA0'] = sigma0
    
    #res = res.chunk({'time': 10000, 'depth':-1})

    return xr.Dataset(output_dict)
            

def compute_n2_peaks(n2_profiles, distance=10, dz=5.):
        
    list_of_n2_peaks = []
    for i in range(0, n2.sizes['time']):
        prof = n2.isel(time=i)
        try:
            list_of_n2_peaks.append(find_profile_peaks(prof, distance=distance))
        except IndexError:
            pass
    ds_n2_peaks = xr.concat(list_of_n2_peaks, dim='time')
    
    ds_n2_peaks.to_netcdf("/home/serazin/Data/UOP/uop_m%02i_profiles.nc" % month, mode='w')

    
def compute_and_save_upper_ocean_pycnocline(month, interp=True, smooth=True, mld=True, distance=10, dz=5.):
    """
    
    """
    
    
    list_of_n2_peaks = []
    for i in range(0, n2.sizes['time']):
        prof = n2.isel(time=i)
        try:
            list_of_n2_peaks.append(find_profile_peaks(prof, distance=distance))
        except IndexError:
            pass
    ds_n2_peaks = xr.concat(list_of_n2_peaks, dim='time')
    
    res = xr.merge([ds_n2_peaks, ds_mld])
    
    res.to_netcdf("/home/serazin/Data/UOP/uop_m%02i_profiles.nc" % month, mode='w')

