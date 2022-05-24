import numpy as np
import pandas as pd
from cartopy.geodesic import Geodesic
from shapely.geometry import Polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns

def plot_poles (df, lon, lat, A95, clr_scaling, size_scaling, center_point, df_true, t_lon, t_lat, plot_A95s=True, connect_poles=False, plot_true=True):
    
    fig = plt.figure(figsize=(20, 10))
    proj = ccrs.Orthographic(central_longitude=center_point[0], central_latitude=center_point[1])
    ax = plt.axes(projection=proj)
    #ax.stock_img()
    #ax.coastlines(linewidth=1, alpha=0.5)
    ax.gridlines(linewidth=1)
    
    cmap = mpl.cm.get_cmap('viridis')

    # plot the A95s
    if plot_A95s:
        norm = mpl.colors.Normalize(df[clr_scaling].min(), df[clr_scaling].max())
        df['geom'] = df.apply(lambda row: Polygon(Geodesic().circle(lon=row[lon], lat=row[lat], radius=row[A95]*111139, n_samples=360, endpoint=True)), axis=1)
        for i, row in df.iterrows():
            ax.add_geometries([df['geom'][i]], crs=ccrs.PlateCarree().as_geodetic(), facecolor='none', edgecolor=cmap(norm(df[clr_scaling][i])), alpha=0.6, linewidth=1)
        df_poles.drop(['geom'], axis=1)

    # plot the mean poles
    if not size_scaling == None:
        sns.scatterplot(x = df[lon], y = df[lat], hue = df[clr_scaling], palette=cmap, size = df[size_scaling], sizes=(50, 200),
                        transform = ccrs.PlateCarree(), zorder=4)
    else:
        sns.scatterplot(x = df[lon], y = df[lat], hue = df[clr_scaling], palette=cmap, s=50, transform = ccrs.PlateCarree(), zorder=4)
    
    if connect_poles:
        plt.plot(df[lon], df[lat], transform = ccrs.Geodetic(), color = 'red', linewidth=2.0)
        
    if plot_true:
        sns.scatterplot(x = df_true[t_lon], y = df_true[t_lat], s=10, color='grey', transform = ccrs.PlateCarree(), zorder=3)

    ax.set_extent([-180, 180, 50, 90], crs = ccrs.PlateCarree())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))
    plt.show()
