import os
import numpy as np
from pmagpy import pmag, ipmag

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