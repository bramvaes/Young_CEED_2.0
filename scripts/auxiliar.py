import os
import numpy as np
from pmagpy import pmag, ipmag
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature 

def get_files_in_directory(path): 
    """
    Retrieves file names from a directory \
    \n\nInput: path = directory \
    \n\nOutput: list of subdirectories
    """

    # The last conditional here is in order to ignore the /DS_store file in macs 
    return [os.path.join(path, name) for name in os.listdir(path)
            if (os.path.isfile(os.path.join(path, name)) and (not name.startswith('.')))  ]

def cartesian2spherical(v):  
    """
    Take an array of lenght 3 correspoingt to a 3-dimensional vector and returns a array of lenght 2
    with co-latitade and longitude
    """
    theta = np.arcsin(v[2])         #facu     theta = np.arccos(v[2]) 
    phi = np.arctan2(v[1], v[0])
        
    return [theta, phi]

def spherical2cartesian(v):
    """
    v[0] = theta - Latitude
    """
    
    x = np.cos(v[0]) * np.cos(v[1])  # x = np.sin(v[0]) * np.cos(v[1])
    y = np.cos(v[0]) * np.sin(v[1])  # y = np.sin(v[0]) * np.sin(v[1])
    z = np.sin(v[0])                 # z = np.cos(v[0])
    
    return [x,y,z]

def GCD_cartesian(cartesian1, cartesian2):
    
    dot = np.dot(cartesian1, cartesian2)
    if abs(dot) > 1: dot = round(dot)    
    gcd =  np.arccos(dot)
    
    return gcd

def get_poles (df, name, slat, slon, dec, inc, plat, plon, verbose=True):
    """
    Seeks to fill in missing poles/vgp entries in dataframe.
    Input: dataframe + column labels for site name, site lat, site lon, dec, inc, pole (or vgp) lat, and pole (or vgp) lon.
    Output: the original dataframe with additional computed pole (or vgp) coordinates, where determinable
    """
    # first identify any entries missing pole / vgp information
    df['pole_exists'] = df.apply(lambda row: True if not (np.isnan(row[plat]) or np.isnan(row[plon])) else False, axis=1)
    df_missing_poles = df[df['pole_exists'] == False]
    
    if df_missing_poles.empty:
        if verbose: print ('no missing pole/vgp information')
    
    else:
        # now check that those which are missing vgp data have sufficient information to calculate it (dec/inc + site data)
        df_missing_poles['sufficient'] = df_missing_poles.apply(lambda row: True if not (np.isnan(row[slat]) or np.isnan(row[slon]) \
                                                                                        or np.isnan(row[dec]) or np.isnan(row[inc])) \
                                                                                        else False, axis=1)

        # report any sites where critical information is lacking 
        if not df_missing_poles['sufficient'].all():
            missing_idx = df_missing_poles.index[df_missing_poles['sufficient'] == False].tolist()
            if verbose:
                for i in missing_idx:
                    site = df[name][i]
                    print (f'Missing slat/slon and/or dec/inc at site {site} where no vgp is reported;'\
                           ' cannot calculate vgp -- dropping entry') 

            # drop entries with no vgp
            df.drop(labels=missing_idx, inplace=True)
                
        # calculate vgps. This adds columns: 'paleolatitude', 'vgp_lat', 'vgp_lon', 'vgp_lat_rev' and 'vgp_lon_rev'
        df_get_poles = df_missing_poles[df_missing_poles['sufficient'] == True]
        ipmag.vgp_calc(df_get_poles, site_lon=slon, site_lat=slat, dec_tc=dec, inc_tc=inc)

        # assign calculated vgps to original dataframe
        df[plat].fillna(df_get_poles.vgp_lat, inplace=True)
        df[plon].fillna(df_get_poles.vgp_lon, inplace=True)
    
    df.drop(['pole_exists'], axis=1, inplace=True)
    
    return df

def get_mean_age (df, name, mean_age, min_age, max_age, verbose=True):
    """
    Fills in missing mean age data from provided max and min reported ages.
    """
    # first identify any entries missing mean age information
    df['mean_age_exists'] = df.apply(lambda row: True if not np.isnan(row[mean_age]) else False, axis=1)
    df_missing_age = df[df['mean_age_exists'] == False]
    
    if not df_missing_age.empty:
        # check that those which are missing mean age data have sufficient information to calculate it (min and max age data)
        df_missing_age['sufficient'] = df_missing_age.apply(lambda row: True if not (np.isnan(row[min_age]) or np.isnan(row[max_age])) \
                                                                                        else False, axis=1)
        
        # report any sites where critical information is lacking 
        if not df_missing_age['sufficient'].all():
            missing_idx = df_missing_age.index[df_missing_age['sufficient'] == False].tolist()
            if verbose:
                for i in missing_idx:
                    location = df[name][i]
                    print (f'Missing min and/or max age at site {location} where no mean age is reported;' \
                           ' cannot calculate mean age -- dropping entry')
                    
            # drop entries with no vgp
            df.drop(labels=missing_idx, inplace=True)

        # calculate mean age.
        df_get_mean_age = df_missing_age[df_missing_age['sufficient'] == True]
        if not df_get_mean_age.empty:
            df_get_mean_age['mean_age'] = df_get_mean_age.apply(lambda row: (row[min_age]+row[max_age])/2., axis=1)
            
            # assign calculated a95s to original dataframe.
            df[mean_age].fillna(df_get_mean_age.mean_age, inplace=True)
        
    df.drop(['mean_age_exists'], axis=1, inplace=True)
    
    return df

def get_alpha95s (df, name, n, alpha95, k, verbose=True): 
    """
    Seeks to fill in missing alpha95s in dataframe.
    Input: dataframe + column labels for site name, sample count (n), alpha95, and precision parameter (k).
    Output: the original dataframe with additional computed alpha 95 estimates, where determinable
    """
    
    # first identify any entries missing alpha95 information
    df['a95_exists'] = df.apply(lambda row: True if not np.isnan(row[alpha95]) else False, axis=1)
    df_missing_a95s = df[df['a95_exists'] == False]
    
    if not df_missing_a95s.empty:
        # check that those which are missing alpha95 data have sufficient information to calculate it (n & k)
        df_missing_a95s['sufficient'] = df_missing_a95s.apply(lambda row: True if not (np.isnan(row[n]) or np.isnan(row[k])) \
                                                                                        else False, axis=1)

        # report any sites where critical information is lacking 
        if not df_missing_a95s['sufficient'].all():
            missing_idx = df_missing_a95s.index[df_missing_a95s['sufficient'] == False].tolist()
            if verbose:
                for i in missing_idx:
                    location = df[name][i]
                    print (f'Missing n and/or k at site {location} where no alpha95 is reported;' \
                           ' cannot calculate alpha95 -- setting to 999')

        # calculate alpha95s.
        df_get_a95s = df_missing_a95s[df_missing_a95s['sufficient'] == True]
        if not df_get_a95s.empty:
            df_get_a95s['a95'] = df_get_a95s.apply(lambda row: 140.0/np.sqrt(row[n] * row[k]), axis=1)

            # assign calculated a95s to original dataframe.
            df[alpha95].fillna(df_get_a95s.a95, inplace=True)
        
        # set those which could not be calculated to 999, and drop added column
        df[alpha95].fillna(value=999)
    
    df.drop(['a95_exists'], axis=1, inplace=True)
    
    return df

def xcheck_dirs_poles (df, name, slat, slon, dec, inc, plat, plon, verbose=True):
    """
    Cross checks combination of directions, poles and site coordinates to ensure they are consistent with one another
    Input: dataframe + column labels for site name, site lat, site lon, dec, inc, pole (or vgp) lat, and pole (or vgp) lon.
    Output: the original dataframe with poles (or vgps) inverted where they appear to have been reported 'upside down'. Alerts raised for otherwise
    spurious-looking poles (arbitrarily defined as a discrepancy of greater than 2 degrees between computed and reported pole / vgp)
    """
    
    # compute pole (or vgp) from dec/inc & slat/slon (this returns columns 'vgp_lon' and 'vgp_lat')
    ipmag.vgp_calc(df, site_lon=slon, site_lat=slat, dec_tc=dec, inc_tc=inc)
    
    # measure distance between recalculated pole (or vgp; note these are columns 'vgp_lon' and 'vgp_lat') and listed pole / vgp.
    df['GCD'] = df.apply(lambda row: pmag.angle([row[plon], row[plat]], [row.vgp_lon, row.vgp_lat]), axis=1)
    
    # if angle is greater than 178 degrees, assume it was inverted by original authors and re-invert
    invert_idx = df.index[df['GCD'] > 178.0].tolist()
    if verbose:
        for i in invert_idx:
            location = df[name][i]
            print (f'vgp from site {location} appears to be inverted. Flipping back (but perhaps check original reference).')
    
    df[plat] = np.where(df['GCD'] > 178., -df[plat], df[plat])
    df[plon] = np.where(df['GCD'] > 178., (df[plon]-180.) % 360., df[plon])
    
    # if any angle is between 2 and 178 degrees, flag it as spurious
    spurious_idx = df.index[(df['GCD'] > 2.0) & (df['GCD'] < 178.0)].tolist()
    if verbose:
        for i in spurious_idx:
            location = df[name][i]
            angle = int(df['GCD'][i])
            print (f'***SPURIOUS*** vgp from site {location};' \
                   f' reported pole differs from re-calculated by {angle} degrees. CHECK against original reference')
        
    # drop added columns
    df.drop(['GCD', 'paleolatitude', 'vgp_lat', 'vgp_lon', 'vgp_lat_rev', 'vgp_lon_rev'], axis=1, inplace=True)
    
    return df

def go_reverse (df, plat, plon, rev_plat, rev_plon, rev_mean=[0,-90]): 
    """
    Determines polarity and establishes a new series where the poles / vgps are all of reversed polarity
    Input: dataframe + column labels for pole (or vgp) lat, pole (or vgp) lon, and the desired labels for the new reversed plat / plon columns. A
    'guesstimate' of where the mean reverse pole [lon,lat] is also needed. For the last tens of Ma it is reasonable to leave this as [-90, 0], but this is
    not a safe assumption in deeper time (and thus this mean reverse pole may need to be set on a case-by-case basis).
    Output: the original dataframe with polarity specified (where not previously determined) and a new series with poles / vgps 
    reported in reverse polarity
    """
    
    # get principal component from (potentially dual-polarity) poles / vgps
    princ = pmag.doprinc(list(zip(df[plon].tolist(), df[plat].tolist())))
    mean = [princ['dec'], princ['inc']]
    
    # determine polarity of mean by comparison with user-provided 'guesstimation' and force to reverse polarity
    if pmag.angle(mean, rev_mean) > 90.0:
        mean[0] = (mean[0]-180.) % 360.
        mean[1] = -1 * mean[1]
        
    # determine polarity of each pole / vgp via distance to reversed mean
    df['GCD'] = df.apply(lambda row: pmag.angle([row[plon], row[plat]], mean), axis=1)
        
    # add series to dataframe where poles / vgps are forced into reverse polarity
    df[rev_plat] = np.where(df['GCD'] > 90, -df[plat], df[plat])
    df[rev_plon] = np.where(df['GCD'] > 90, (df[plon] - 180.) % 360., df[plon])
    
    df.drop(['GCD'], axis=1, inplace=True)
    
    return df

def print_pole_statistics(pole, vgp_mean):
    
    print(f"{'' : <20}{'Pole' : ^10}{'N' : ^10}{'Plat' : ^10}{'Plon' : ^10}{'A95' : >5}")        
    print(f"{'Reported paleopole' : <20}{pole.iloc[0]['pole'] : ^10}{pole.iloc[0]['N'] : ^10}{pole.iloc[0]['Plat'] : ^10}{pole.iloc[0]['Plon'] : ^10}{pole.iloc[0]['A95'] : >5}")    
    print(f"{'Recomputed paleopole' : <20}{pole.iloc[0]['pole'] : ^10}{vgp_mean['n'] : ^10}{vgp_mean['inc'] : ^10.1f}{vgp_mean['dec'] : ^10.1f}{vgp_mean['alpha95'] : >5.1f}")
    print(f"")

def test_fishqq(merged):
    if len(merged) <= 10: print (' - Not enough sites to conduct quantile-quantile test')
    else:                              
        plt.figure(1,figsize=(7,3))
        ipmag.fishqq(di_block=merged)               
        plt.show()
        
def reversal_test(mode1, mode2):
    if len(mode1) == 0 or len(mode2) == 0: 
        print (' ')
        print (' - Only one polarity; cannot conduct reversal test')
    elif len(mode1) < 3 or len(mode2) < 3: 
        print (' ')
        print (' - Not enough sites from one (or both) polarity populations to conduct reversal test')
    else: 
        flipped2 = np.delete(np.array(ipmag.do_flip(di_block=mode2)), -1, axis=1)
        ipmag.common_mean_bootstrap(mode1, flipped2)      
        
def invert_polarity(mode1, mode2):
    '''
    Direcctions are flipped to a single polarity and pooled into a single variable (merged)
    If mode2 is empty, the pooled group is equal to mode1
    '''
       
    if mode2.size == 0: merged = mode1
    else:
        flipped2 = np.array(ipmag.do_flip(di_block=mode2))[:,:2]
        #flipped2 = np.delete(np.array(ipmag.do_flip(di_block=mode2)), -1, axis=1)
        merged = np.concatenate((mode1, flipped2))
    
    return merged

def Plot_VgpsAndSites(df, pole, vgp_mean, mode1, mode2):
    '''
    Function to plot Site coordinates and directions in a stereonet along with the recomputed and reported pole  
    '''    
    
    fig = plt.figure(figsize=(12, 5))
    
    extent = [-135, -55,5, 90]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])    
    
    proj = ccrs.AlbersEqualArea(central_lon, central_lat)

    #ax = plt.axes(projection=proj)
    ax = fig.add_subplot(1,2,1, projection=proj)
    ax.set_extent(extent)

    # Put a background image on for nice sea rendering.
    ax.stock_img()

    ax.add_feature(cfeature.BORDERS)
    ax.gridlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)

    ax.scatter(x = df['slon'], y = df['slat'], color='r', s=20, marker='*', transform=ccrs.PlateCarree())
    
    ax2 = fig.add_subplot(1,2,2)    
    ax2.set_title('VGPs and Paleomagnetic poles')
    
    ax2 = ipmag.plot_net()
    ax2 = ipmag.plot_di([x[0] for x in mode1], [x[1] for x in mode1], 
                        label = "VGPs")
    
    if len(mode2) != 0:
        ax2 = ipmag.plot_di([x[0] for x in mode2], [x[1] for x in mode2],
                            label = "VGPs")
            
    if not pole.empty:
        pole_lat = pole['Plat'].values[0]
        pole_lon = pole['Plon'].values[0]
        pole_A95 = pole['A95'].values[0]
        ax2 = ipmag.plot_di_mean(pole_lon, pole_lat, pole_A95, 
                                 label="Reported Pole", color='y')    
    
    ax2 = ipmag.plot_di_mean(vgp_mean['dec'], vgp_mean['inc'], vgp_mean['alpha95'], 
                             label="Recalculated Pole", color='red')
    plt.legend(loc=3, fontsize=12)
    plt.show()
    

    
import scipy as sp #import scipy    
from scipy.integrate import quad #import 1D quadrature function
from scipy.integrate import nquad #import ND quadrature function
import SPD.lib.lib_directional_statistics as lib_direct



def bayes_probability_heslop(DI1,DI2):
    '''    
    Calculate the Bayes probability that the to group of directions
    are drawn from a single Fisher distributed poopulation following 
    Heslop and Roberts (2018). 
    
    input: two numpy arrays [dec, inc] flipped to the same polarity
    
    '''    
       
    if DI2.any():
        X1=pmag.dir2cart(DI1) #convert normal polarity sites
        X2=pmag.dir2cart(DI2) 
        X12=np.concatenate((X1,-X2), axis=0) #pool site directions
    else:
        return print("Single Polarity, cannot conduct Bayesian Reversal Test")
        
    def log_like(k,N,R):
        return N*sp.log(k/4/sp.pi)-N*log_sinh(k)+sp.log(4*sp.pi) \
                +log_sinh(k*R)-sp.log(k*R)

    def log_sinh(k):
        if k>700:
            s=k-sp.log(2.0)
        else:
            s=sp.log(sp.sinh(k))
        return s

    def log_prior(k):
        return np.log(4.0)+2.0*np.log(k)-np.log(np.pi)-2.0*np.log(1.+k**2)

    def integrand1(k,x):
        R=np.sqrt(np.sum(np.sum(x,axis=0)**2))
        N=np.size(x,axis=0)
        val = np.exp(log_like(k,N,R)+log_prior(k))
        return val

    def integrand2(k,x1,x2):
        R1=np.sqrt(np.sum(np.sum(x1,axis=0)**2))
        N1=np.size(x1,axis=0)

        R2=np.sqrt(np.sum(np.sum(x2,axis=0)**2))
        N2=np.size(x2,axis=0)

        val = np.exp(log_like(k,N1,R1)+log_like(k,N2,R2)+log_prior(k))

        return val        
    
    mL1 = quad(integrand1, 0, np.inf, args=(X12))[0] #Bayes factor numerator (Eqn 13)
    mL2 = quad(integrand2, 0, np.inf, args=(X1,X2))[0] #Bayes factor denominator (Eqn 14)
    
    BF0=mL1/mL2
    P=BF0/(1.+BF0)
    
    return BF0, P

def bayes_support(P):
    '''From Heslop & Roberts (2018)
     categories of support from the bayesian probability(float)'''
    
    support = ['Different means: very strong support','Different means: strong support','Different means: positive support','Ambiguous: weak support',
              'Common mean: positive support','Common mean: strong support','Common mean: very strong support']
    
    if P<0.01: return support[0]
    if P>=0.01 and P<0.05: return support[1]
    if P>=0.05 and P<0.25: return support[2]
    if P>=0.25 and P<0.75: return support[3]
    if P>=0.75 and P<0.95: return support[4]
    if P>=0.95 and P<0.99: return support[5]
    if P>=0.99: return support[6]