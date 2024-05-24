import numpy as np
import xarray as xr
import proplot as pplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd

def plot_n2_profile(ds, ax=None, zmin=0, zmax=2000, title=True, 
                    legend=True, xlabel=True, ylabel=True):
    
    xmin = 0
    xmax = 1.1 * ds.N2.max()
    z = ds.N2['z']
    
    if ax is None:
        ax = plt.gca()
    #Plot N2 vertical profile
    ax.plot(ds['N2'], ds['z'], lw=2.5, label='$N^2$ smoothed profile', color='C1')
    
    # Mark the detected stratification peaks
    ax.plot(ds['intensity'], ds['depth'], "x", markersize=15, lw=2, color='grey')
    
    #Plot the prominence of peaks 
    ax.hlines(y=ds['depth'], 
              #xmin=ds['intensity'] - ds['prominence'],
              xmin=0,
              xmax=ds['intensity'], color="grey", 
              linestyle='-.', label='Peak prominence')
    
    #Plot the width of peaks at half height 
    ax.vlines(x=ds['width_height'], 
              ymin=ds['width_left'], 
              ymax=ds['width_right'], color="grey", 
              label='Peak width')
    
    # Plot mixed layer depth
    plt.hlines(ds['MLD_DR003'], 0, xmax, ls='--', lw=2, 
               color='black', label=r'MLD $\Delta \sigma_0=0.03\,kg.m^{-3}$')
    
    # Plot mixed layer depth
    #plt.hlines(ds['MLD_RVAR'], 0, xmax, ls='-.', lw=2, 
    #           color='black', label=r'MLD $\chi_{min}$')
    
    #Format X axis
    ax.set_xlim([0, xmax])
    if xlabel:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0.01, 100))
        ax.set_xlabel('$N^2\,[s^{-2}]$')
    else:
        ax.set_xticklabels([])
    
    #Format Y axis
    ax.set_ylim([zmax, zmin])
    if ylabel:
        ax.set_ylabel('Depth [m]')
    else:
        ax.set_yticklabels([])
    
    #Plot peak width
    for peak in ds['peak']:
        ds_peak = ds.sel(peak=peak)
        zmax = ds_peak['width_right']
        zmin = ds_peak['width_left']
        ax.fill_between(x=[xmin, xmax],
                        y1=[zmin, zmin], 
                        y2=[zmax, zmax],
                        alpha=0.25)
    ax.grid()    
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc='lower right')   
        
        
def plot_curvature_profile(ds, ax=None, zmin=0, zmax=2000, title='', legend=True, xlabel=True, ylabel=True):
    kappa = ds.KAPPA
    z = kappa['z']
    xmin = 0.999 * kappa.min()
    xmax = 1.001 * kappa.max()
    if ax is None:
        ax = plt.gca()
    # Plots    
    ax.plot(kappa, z, lw=2, color='C4', label='Curvature profile')
    
    if xlabel:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0.01, 100))
        ax.set_xlabel('$\kappa$')
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel('Depth [m]')
    else:
        ax.set_yticklabels([])

    #Format Y axis
    ax.set_ylim([zmax, zmin])
  
        
    # Plot mixed layer depth
    plt.hlines(ds['MLD_003'], 0, xmax, ls='--', lw=2, 
               color='black', label=r'MLD $\Delta \sigma_0=0.03\,kg.m^{-3}$')
    
    for peak in ds['peak']:
        ds_peak = ds.sel(peak=peak)
        zmax = ds_peak['width_right']
        zmin = ds_peak['width_left']
        ax.fill_between(x=[xmin, xmax],
                        y1=[zmin, zmin], 
                        y2=[zmax, zmax],
                        alpha=0.25)
    
    #Format X axis
    ax.set_xlim([xmin, xmax])
    ax.grid()


    # Title and legends
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc='lower left')

        
def plot_profile_variable(ds, variable='SIGMA0', ax=None, zmin=0, zmax=2000, 
                          title='', legend=True, label='', loc='lower left',
                          xlabel='', ylabel=True, color='C0'):
    da = ds[variable]
    z = da['z']
    xmin = 0.99 * da.min()
    xmax = 1.01 * da.max()
    
    if ax is None:
        ax = plt.gca()
        
    # Plots    
    ax.plot(da, z, lw=2, color=color, label=label)
    
    #Format X axis
    if xlabel:
        #ax.ticklabel_format(axis='x', style='sci', scilimits=(0.01, 100))
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])
    ax.set_xlim([xmin, xmax])
    
    #Format Y axis
    if ylabel:
        ax.set_ylabel('Depth [m]')
    else:
        ax.set_yticklabels([])
    ax.set_ylim([zmax, zmin])
    
    for peak in ds['peak']:
        ds_peak = ds.sel(peak=peak)
        zmax = ds_peak['width_right']
        zmin = ds_peak['width_left']
        ax.fill_between(x=[xmin, xmax],
                        y1=[zmin, zmin], 
                        y2=[zmax, zmax],
                        alpha=0.25)
    
    # Title and legends
    #ax.grid()
    if title:
        ax.set_title(title, loc='left', fontweight='bold')
    if legend:
        ax.legend(loc=loc)
        
        
def plot_profile_v2(profile):
    
    fig = pplt.figure(sharex=False, facecolor='white')
    ncols = 3
    axs = fig.subplots(nrows=2, ncols=ncols, includepanels=True)
    suptitle = ('Time of profile: %s \n Lat=%02.1f, Lon=%02.1f \n' 
                     % (pd.to_datetime(profile.time.data),
                        profile.latitude.load().data, 
                        profile.longitude.load().data,))

    pplt.rc.cycle = 'default'


    variables = ['T', 'SP', 'SIGMA0',
                 'N2', 'KAPPA', 'BC']
    colors = ['C0', 'C3', 'C2', 'C1', 'C4', 'C5']

    labels = ['Temperature', 'Salinity', 'Potential density',  
              'Buoyancy frequency', 'Curvature', 'Columnar buoyancy']

    xlabels = [r'$T\,[^\circ C]$', r'$S\,[psu]$', r'$\sigma_0\,[kg.m^{-3}]$',
               r'$N^2\,[s^{-2}]$', r'$\kappa_\rho$', r'$b_c\,[m^{2}.s^{-2}]$']

    legend_loc = ['lr', 'lr', 'll', 'lr', 'lr', 'll']
    
    profile['SIGMA0'] =  profile['SIGMA0'] - 1000

    for i, var in enumerate(variables):

        da = profile[var]
        z = da['z']
        xmin = 0.99 * da.min()
        xmax = 1.01 * da.max()

        pxs = axs[i].panel('b', width='25em', space='0.5em')

        axs[i].plot(da.data, z, lw=2, color=colors[i])   
        pxs.plot(da.data, z, lw=2, color=colors[i], label=labels[i])

        for ax in [axs[i], pxs]:
            # Plot MLD threshold
            ax.hlines(profile['MLD_MIN'], xmin, xmax, 
                      ls='--', lw=2, color='black', 
                      label=r'MLD $\Delta \sigma_0=0.03\,kg.m^{-3}$')
            
            #ax.hlines(profile['left'], xmin, xmax,
            #          ls=':', lw=1, color='black')
            #ax.hlines(profile['right'], xmin, xmax,
            #          ls=':', lw=1, color='black')

            # Plot peak areas
            for peak in profile['peak']:
                ds_peak = profile.sel(peak=peak)
                zmax = ds_peak['width_right']
                zmin = ds_peak['width_left']
                ax.fill_between(x=[xmin, xmax],
                                y1=[zmin, zmin], 
                                y2=[zmax, zmax],
                                alpha=0.25)
            if var == 'N2':
                 # Mark the detected stratification peaks
                ax.plot(profile['intensity'], profile['depth'], 
                        "x", markersize=15, lw=2, color='grey', label='')

                #Plot the prominence of peaks 
                #ax.hlines(profile['depth'], 
                #          xmin * xr.ones_like(profile['intensity']),
                #          profile['intensity'], color="grey", 
                #          linestyle='-.', label='Peak prominence')

                #Plot the width of peaks at half height 
                ax.vlines(profile['width_height'], 
                          profile['width_left'], 
                          profile['width_right'], color="grey", 
                          label='Peak width')   
                ax.format(xformatter='%.1e')
                
        axs[i].format(xlim=[xmin, xmax], ylim=[250, 0], xlabel=xlabels[i])
        pxs.format(xlim=[xmin, xmax], ylim=[2000, 250], ylabel='Depth[m]')
        pxs.legend(ncols=1, loc=legend_loc[i])

    axs.format(abc='a)', abcloc='l',
               suptitle=suptitle, ylim=[250, 0], ylabel='')
        
    
        
def plot_profile(profile):
    #from matplotlib.gridspec import GridSpec
    
    fig = pplt.figure(figsize=(14, 18))
    
    fig.suptitle('Time of profile: %s \n Lat=%02.1f, Lon=%02.1f \n' 
                 % (pd.to_datetime(profile.time.data),
                    profile.latitude.load().data, 
                    profile.longitude.load().data,), 
                    fontsize=16, fontweight='bold')
        
    gs = pplt.GridSpec(nrows=4, ncols=3, hspace=0.1, wspace=0.1)  

    ax_sigma0 = fig.add_subplot(gs[0, 0])
    plot_profile_variable(profile, variable='SIGMA0', ax=ax_sigma0,
                          label='Potential density profile',
                          color='C2', 
                          zmin=0, zmax=250, title='a)',
                          legend=False, xlabel=False)
    
    ax_sigma0 = fig.add_subplot(gs[1, 0])
    plot_profile_variable(profile, variable='SIGMA0', ax=ax_sigma0,
                          label='Potential density profile',
                          color='C2', 
                          xlabel='$\sigma_0\,[kg.m^{-3}]$', 
                          zmin=250, zmax=2000, title=False,
                          legend=True,)

    ax_temp = fig.add_subplot(gs[0, 1])
    plot_profile_variable(profile, variable='T', ax=ax_temp,
                          label='Temperature profile',
                          color='C0', ylabel=False,
                          zmin=0, zmax=250, title='b)',
                          legend=False, xlabel=False)
    
    ax_temp = fig.add_subplot(gs[1, 1])
    plot_profile_variable(profile, variable='T', ax=ax_temp,
                          label='Temperature profile',
                          color='C0', ylabel=False,
                          xlabel='$T\,[^\circ C]$', 
                          zmin=250, zmax=2000, title=False,
                          legend=True, loc='lower right')

    ax_psal = fig.add_subplot(gs[0, 2])
    plot_profile_variable(profile, variable='SP', ax=ax_psal,
                          label='Salinity profile',
                          color='C3', ylabel=False,
                          zmin=0, zmax=250, title='c)',
                          legend=False, xlabel=False)
    
    ax_psal = fig.add_subplot(gs[1, 2])
    plot_profile_variable(profile, variable='SP', ax=ax_psal,
                          label='Salinity profile',
                          color='C3', ylabel=False,
                          xlabel='$S\,[psu]$', 
                          zmin=250, zmax=2000, title=False,
                          legend=True, loc='lower right') 
    
    ax_n2 = fig.add_subplot(gs[2, 0])
    plot_profile_variable(profile, variable='N2', ax=ax_n2,
                          label='Buoyancy frequency profile',
                          color='C1', 
                          zmin=0, zmax=250, title='d)',
                          legend=False, xlabel=False)
    
    ax_n2 = fig.add_subplot(gs[3, 0])
    plot_profile_variable(profile, variable='N2', ax=ax_n2,
                          label='Buoyancy frequency profile',
                          color='C1', 
                          xlabel=r'$N^2\,[s^{-2}]$', 
                          zmin=250, zmax=2000, title=False,
                          legend=True,)
    
    ax_curv = fig.add_subplot(gs[2, 1])
    plot_profile_variable(profile, variable='KAPPA', ax=ax_curv,
                          label='Curvature profile',
                          color='C4', ylabel=False,
                          zmin=0, zmax=250, title='e)',
                          legend=False, xlabel=False)
    
    ax_curv = fig.add_subplot(gs[3, 1])
    plot_profile_variable(profile, variable='KAPPA', ax=ax_curv,
                          label='$N^2$ profile',
                          color='C4', ylabel=False,
                          xlabel=r'$\kappa_\rho$', 
                          zmin=250, zmax=2000, title=False,
                          legend=True, loc='lower right')
    
    ax_bc = fig.add_subplot(gs[2, 2])
    plot_profile_variable(profile, variable='BC', ax=ax_bc,
                          label='Columnar buoyancy profile',
                          color='C5', ylabel=False,
                          zmin=0, zmax=250, title='f)',
                          legend=False, xlabel=False)
    
    ax_bc = fig.add_subplot(gs[3, 2])
    plot_profile_variable(profile, variable='BC', ax=ax_bc,
                          label='Columnar buoyancy profile',
                          color='C5', ylabel=False,
                          xlabel=r'$b_c\,[m^{2}.s^{-2}]$', 
                          zmin=250, zmax=2000, title=False,
                          legend=True, loc='lower right')
    
    plt.tight_layout()
    
        
def plot_random_profile(full_profiles, month=1, 
                        lat_min=-90, lat_max=90,
                        lon_min=0, lon_max=360):
    
    ds = full_profiles[month]

    ds = ds.where((ds['latitude'] >= lat_min) &
                  (ds['latitude'] <= lat_max) &
                  (ds['longitude'] >= lon_min) &
                  (ds['longitude'] <= lon_max), drop=True)
    
    # Select a random profile
    import random
    i = 0
    n = random.randint(0, ds.sizes['time'] - 1)
    profile_test = ds.isel(time=n).dropna('peak').load()
    plot_profile_v2(profile_test)

        
        
def format_lon_lat(ax, proj, lon_min, lon_max, lat_min, lat_max, delta_lon=10, delta_lat=10, title=''):
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    #from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    ax.set_extent([lon_min, lon_max, 
                   lat_min, lat_max], proj)
    # Add coastline
    ax.coastlines('50m')
    
    # Modify the title
    ax.set_title(title)
            
    # Set lon labels
    lon_labels = np.arange(lon_min, lon_max + 1, delta_lon)
    lon_labels[lon_labels > 180] -= 360
    ax.set_xticks(lon_labels, crs=proj)
    ax.set_xticklabels(lon_labels, rotation=45)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.set_xlabel('')
            
    # Set lat labels
    lat_labels = np.arange(lat_min, lat_max + 1, delta_lat)
    ax.set_yticks(lat_labels, crs=proj)
    ax.set_yticklabels(lat_labels)
    ax.yaxis.tick_left()
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_ylabel('')
                        
    # Plot the grid
    ax.grid(linewidth=2, color='black', alpha=0.5, linestyle='--')

            
def add_map(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90,
            central_longitude=0., scale='auto', ax=None):
    """
    Add the map to the existing plot using cartopy

    Parameters
    ----------
    lon_min : float, optional
        Western boundary, default is -180
    lon_max : float, optional
        Eastern boundary, default is 180
    lat_min : float, optional
        Southern boundary, default is -90
    lat_max : float, optional
        Northern boundary, default is 90
    central_longitude : float, optional
        Central longitude, default is 180
    scale : {?auto?, ?coarse?, ?low?, ?intermediate?, ?high, ?full?}, optional
        The map scale, default is 'auto'
    ax : GeoAxes, optional
        A new GeoAxes will be created if None

    Returns
    -------
    ax : GeoAxes
    Return the current GeoAxes instance
    """

    extent = (lon_min, lon_max, lat_min, lat_max)
    if ax is None:
        ax = plt.subplot(1, 1, 1,
                         projection=ccrs.PlateCarree(                                  central_longitude=central_longitude))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    land = cfeature.GSHHSFeature(scale=scale,
                                 levels=[1],
                                 facecolor=cfeature.COLORS['land'])
    ax.add_feature(land)
    gl = ax.gridlines(draw_labels=True, linestyle=':', color='black',
                      alpha=0.5)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


def plot_one_map(data, ax=None, lon_min=110, lon_max=180, 
                 lat_min=-45, lat_max=0, **kwargs):
    data['nav_lon'] = data['nav_lon'] % 360
    add_map(ax=ax, lon_min=lon_min, lon_max=lon_max, 
            lat_min=lat_min, lat_max=lat_max)
    mapped_data = data.plot(x='nav_lon', y='nav_lat', **kwargs)
    return mapped_data
    
