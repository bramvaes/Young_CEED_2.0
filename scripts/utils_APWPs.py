import numpy as np
import pandas as pd
from pmagpy import pmag, ipmag

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
from shapely.geometry import Polygon

from scripts.auxiliar import spherical2cartesian, shape, eigen_decomposition


def running_mean_APWP (data, plon_label, plat_label, age_label, window_length, time_step, max_age, min_age):
    """
    function to generate running mean APWP..
    """
    
    mean_pole_ages = np.arange(min_age, max_age + time_step, time_step)
    
    running_means = pd.DataFrame(columns=['age','N','n_studies','k','A95','csd','plon','plat'])
    
    for age in mean_pole_ages:
        window_min = age - (window_length / 2.)
        window_max = age + (window_length / 2.)
        poles = data.loc[(data[age_label] >= window_min) & (data[age_label] <= window_max)]
        number_studies = len(poles['Study'].unique())
        mean = ipmag.fisher_mean(dec=poles[plon_label].tolist(), inc=poles[plat_label].tolist())

        if mean: # this just ensures that dict isn't empty
            running_means.loc[age] = [age, mean['n'], number_studies, mean['k'],mean['alpha95'], mean['csd'], mean['dec'], mean['inc']]
    
    running_means.reset_index(drop=1, inplace=True)
    
    return running_means

def running_mean_APWP_shape(data, plon_label, plat_label, age_label, window_length, time_step, max_age, min_age):
    """
    function to generate running mean APWP..
    """
    
    mean_pole_ages = np.arange(min_age, max_age + time_step, time_step)
    
    running_means = pd.DataFrame(columns=['age','N','n_studies','k','A95','csd','plon','plat', 'foliation','lineation','collinearity','coplanarity'])
    
    for age in mean_pole_ages:
        window_min = age - (window_length / 2.)
        window_max = age + (window_length / 2.)
        poles = data.loc[(data[age_label] >= window_min) & (data[age_label] <= window_max)]
        number_studies = len(poles['Study'].unique())
        mean = ipmag.fisher_mean(dec=poles[plon_label].tolist(), inc=poles[plat_label].tolist())
        
        ArrayXYZ = np.array([spherical2cartesian([i[plat_label], i[plon_label]]) for _,i in poles.iterrows()])        
        if len(ArrayXYZ) > 3:
            shapes = shape(ArrayXYZ)       
        else:
            shapes = [np.nan,np.nan,np.nan,np.nan]
        
        if mean: # this just ensures that dict isn't empty
            running_means.loc[age] = [age, mean['n'], number_studies, mean['k'],mean['alpha95'], mean['csd'], mean['dec'], mean['inc'], 
                                      shapes[0], shapes[1], shapes[2], shapes[3]]
    
    running_means.reset_index(drop=1, inplace=True)
    
    return running_means



def plot_poles (df, plon, plat, A95, clr_scaling, size_scaling, extent, plot_A95s=True, connect_poles=False):
    """
    function to plot poles...could be broken into several simpler functions to make function passing simpler...
    """
    
    #plt.style.use('ggplot')
    fig = plt.figure(figsize=(20,10))
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-55) #30, -60
    ax = plt.axes(projection=proj)    
    ax.stock_img()
    ax.coastlines(linewidth=1, alpha=0.5)
    ax.gridlines(linewidth=1)
    
    cmap = mpl.cm.get_cmap('viridis')

    # plot the A95s
    if plot_A95s:
        norm = mpl.colors.Normalize(df[clr_scaling].min(), df[clr_scaling].max())
        df['geom'] = df.apply(lambda row: Polygon(Geodesic().circle(lon=row[plon], lat=row[plat], radius=row[A95]*111139, n_samples=360, endpoint=True)), axis=1)
        for idx, row in df.iterrows():
            ax.add_geometries([df['geom'][idx]], crs=ccrs.PlateCarree().as_geodetic(), facecolor='none', edgecolor=cmap(norm(df[clr_scaling][idx])), 
                              alpha=0.6, linewidth=1)
        df.drop(['geom'], axis=1)

    # plot the mean poles
    if not size_scaling == None:
        sns.scatterplot(x = df[plon], y = df[plat], hue = df[clr_scaling], palette=cmap, size = df[size_scaling], sizes=(50, 200),
                        transform = ccrs.PlateCarree(), zorder=4)
    else:
        sns.scatterplot(x = df[plon], y = df[plat], hue = df[clr_scaling], palette=cmap, s=50, transform = ccrs.PlateCarree(), zorder=4)
    
    if connect_poles:
        plt.plot(df[plon], df[plat], transform = ccrs.Geodetic(), color='red', linewidth=2.0)

    if extent != 'global':
        ax.set_extent(extent, crs = ccrs.PlateCarree())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))
    plt.show()
    

def plot_poles_and_stats (df, plon, plat, A95, clr_scaling, size_scaling, extent, plot_A95s=True, connect_poles=False):
    """
    function to plot poles...could be broken into several simpler functions to make function passing simpler...
    """
    
    #plt.style.use('ggplot')
    fig = plt.figure(figsize=(20,10))   
    
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-55) #30, -60
    # ax = plt.axes(projection=proj)    
    ax = fig.add_subplot(1,2,1, projection=proj)
    
    ax.stock_img()
    ax.coastlines(linewidth=1, alpha=0.5)
    ax.gridlines(linewidth=1)
    
    cmap = mpl.cm.get_cmap('viridis')

    # plot the A95s
    if plot_A95s:
        norm = mpl.colors.Normalize(df[clr_scaling].min(), df[clr_scaling].max())
        df['geom'] = df.apply(lambda row: Polygon(Geodesic().circle(lon=row[plon], lat=row[plat], radius=row[A95]*111139, n_samples=360, endpoint=True)), axis=1)
        for idx, row in df.iterrows():
            ax.add_geometries([df['geom'][idx]], crs=ccrs.PlateCarree().as_geodetic(), facecolor='none', edgecolor=cmap(norm(df[clr_scaling][idx])), 
                              alpha=0.6, linewidth=1)
        df.drop(['geom'], axis=1)

    # plot the mean poles
    if not size_scaling == None:
        sns.scatterplot(x = df[plon], y = df[plat], hue = df[clr_scaling], palette=cmap, size = df[size_scaling], sizes=(50, 200),
                        transform = ccrs.PlateCarree(), zorder=4)
    else:
        sns.scatterplot(x = df[plon], y = df[plat], hue = df[clr_scaling], palette=cmap, s=50, transform = ccrs.PlateCarree(), zorder=4)
    
    if connect_poles:
        plt.plot(df[plon], df[plat], transform = ccrs.Geodetic(), color='red', linewidth=2.0)

    if extent != 'global':
        ax.set_extent(extent, crs = ccrs.PlateCarree())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))
    
    
    ax2 = fig.add_subplot(1,2,2)  
    ax2.set_title('Moving average stats')
    
    df['kappa_norm'] = df['k'] / df['k'].max()
    df['N_norm'] = df['N'] / df['N'].max()
    
    dfm = df[['age', 'A95', 'n_studies', 'csd','kappa_norm']].melt('age', var_name='type', value_name='vals')
    
    
    ax2 = sns.lineplot(data  = dfm, x = dfm['age'], y = dfm['vals'], hue = dfm['type'],marker="o")

    plt.show()
    
    
def get_pseudo_vgps (df):  #column labels are presently hard-coded into this, if relevant.
    """
    function to parametrically resample poles from study-level statistics (to generate 'pseudo-vgps')...
    """

    poleIDs, age_draws, vgp_lon_draws, vgp_lat_draws = ([] for i in range(4))
    
    for _, ent in df.iterrows():
        
        bootstrap_vgps = ipmag.fishrot(k=ent.K, n=ent.N, dec=ent.Plon, inc=ent.Plat, di_block=False)
        vgp_lon_draws.append(bootstrap_vgps[0])
        vgp_lat_draws.append(bootstrap_vgps[1])
        N = len(bootstrap_vgps[0])
        
        if ent.uncer_dist == 'uniform':
            # first check if bounds of uniform range in fact have normally distributed errors
            
            if np.isnan(ent['2sig_min']): min_ages = [ent.min_age for _ in range(N)]
            else: min_ages = [np.random.normal(loc=ent.min_age, scale=(ent['2sig_min'])/2.) for _ in range(N)]   
            
            if np.isnan(ent['2sig_max']): max_ages = [ent.max_age for _ in range(N)]
            else: max_ages = [np.random.normal(loc=ent.max_age, scale=(ent['2sig_max'])/2.) for _ in range(N)]
            
            # grab uniform draws from range
            age_draws.append([np.random.uniform(min_ages[i], max_ages[i]) for i in range(N)])
            
        elif ent.uncer_dist == 'normal':
            # normal/gaussian draws
            age_draws.append([np.random.normal(loc=ent.mean_age, scale=(ent.max_age - ent.mean_age)/2.) for _ in range(N)]) 
        
        else: print ('unexpected age distribution type; cannot execute age bootstrap')
            
        poleIDs.append([ent.AgeIdx for _ in range(N)])

    bootstrap_data = {'pole_ID': [item for sublist in poleIDs for item in sublist],
                        'plat': [item for sublist in vgp_lat_draws for item in sublist],
                        'plon': [item for sublist in vgp_lon_draws for item in sublist],
                        'age':  [item for sublist in age_draws for item in sublist]}
    
    pseudo_vgps = pd.DataFrame(bootstrap_data)
    
    return pseudo_vgps


def resample_vgps (df, ignore_deterministic=False):  #column labels are presently hard-coded into this, if relevant.
    """
    function to parametrically resample vgps from site-level statistics (to generate 'pseudo-samples')...
    """

    poleIDs, age_draws, vgp_lon_draws, vgp_lat_draws = ([] for i in range(4))
    
    # identify any entries for which direction cannot be bootstrapped
    df['deterministic'] = df.apply(lambda row: True if (np.isnan(row.n) | (row.k==0)) else False, axis=1)
    
    if ignore_deterministic == True:
        df.drop(df[df.deterministic == True].index, inplace=True)
    
    # get new directions from all those sites which can be bootstrapped
    for _, ent in df.iterrows():
        
        if ent.deterministic == False:
            bootstrap_dirs = ipmag.fishrot(k=ent.k, n=int(ent.n), dec=ent.dec, inc=ent.inc, di_block=False) # don't need to cast n as int here if ensured above
            if ent.n > 1:
                mean_dir = ipmag.fisher_mean(dec=bootstrap_dirs[0], inc=bootstrap_dirs[1])
                new_vgp = pmag.dia_vgp(mean_dir['dec'], mean_dir['inc'], mean_dir['alpha95'], ent.slat, ent.slon)
            else: 
                new_vgp = pmag.dia_vgp(bootstrap_dirs[0], bootstrap_dirs[1], 0, ent.slat, ent.slon)
            
            if ent.polarity == 'R':  # should probably make a new column of re-assigned polarities in build_compilation notebook (else need to over-write the non-N/R options reported by authors...)?
                vgp_lon_draws.append(new_vgp[0])
                vgp_lat_draws.append(new_vgp[1])
            else:
                vgp_lon_draws.append((new_vgp[0]- 180.) % 360.)
                vgp_lat_draws.append(new_vgp[1] * -1)
        
        if ent.deterministic == True: # this only applies when ignore_deterministic == False
            vgp_lon_draws.append(ent.rev_VGP_lon)
            vgp_lat_draws.append(ent.rev_VGP_lat)
        
        if ent.uncer_dist == 'uniform':
            # first check if bounds of uniform range in fact have normally distributed errors
            
            if np.isnan(ent['2sig_min']): min_ages = ent.min_age
            else: min_ages = np.random.normal(loc=ent.min_age, scale=(ent['2sig_min'])/2.)   
            
            if np.isnan(ent['2sig_max']): max_ages = ent.max_age
            else: max_ages = np.random.normal(loc=ent.max_age, scale=(ent['2sig_max'])/2.)
            
            # grab uniform draws from range
            age_draws.append(np.random.uniform(min_ages, max_ages))
            
        elif ent.uncer_dist == 'normal':
            # normal/gaussian draws
            age_draws.append(np.random.normal(loc=ent.mean_age, scale=(ent.max_age - ent.mean_age)/2.)) 
        
        else: print ('unexpected age distribution type; cannot execute age bootstrap')
        
         # *** NEED TO IMPLEMENT AGE SORTING ACCORDING TO STRATIGRAPHIC ORDERING ***
            
        poleIDs.append(ent.AgeIdx)

    bootstrap_data = {'pole_ID': poleIDs,
                      'plat': vgp_lat_draws,
                      'plon': vgp_lon_draws,
                      'age': age_draws}
    
    resampled_vgps = pd.DataFrame(bootstrap_data)
    
    return resampled_vgps