# Written by Lei Wu and Simon Williams with code snippets from a variety of sources

import numpy as np
import pandas as pd
import pygplates
import matplotlib.pyplot as plt
import math
import pmagpy.pmag as pmag

def distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    # arc = arc * 6373
    arc = arc * 180.0/math.pi # LW
    return arc

pd.options.mode.chained_assignment = None


def get_antipode(lat,long):
    # function to get antipodal lat/lon coordinates
    # return -lat,(long + 180.) % 360. # SW
    long += 180.
    if long > 180.:
        long += -360.
    elif long < -180.:
        long += 360.
    return -lat, long


def get_overlap(start1, end1, start2, end2):
    """how much does the range (start1, end1) overlap with (start2, end2)"""
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)


def wAPWP_weights(df_subset,windowIN,A95c,Wabc):
    # given a dataframe with VGPs, and a time window, find the weights
    # for each VGP within the time window
    weights = []
    for i,row in df_subset.iterrows():
        weight = wAPWP_weight(row,windowIN,A95c,Wabc)
        weights.append(weight)

    return weights


def wAPWP_weight(row,windowIN,A95c,Wabc,Wflag='AgeA95Q'):
    # similar to matlab function 'wAPWP_ageOverlap'
    # operates on a single VGP, returns the weight within defined time window
    # overlap = get_overlap(row.Youngage, row.Oldage, windowIN[0], windowIN[1])
    overlap = get_overlap(row.min_age, row.max_age, windowIN[0], windowIN[1])
    # W_dt = overlap / (row.Oldage - row.Youngage)
    W_dt = overlap / (row.max_age - row.min_age)

    if row.A95>A95c:
        W_A95 = A95c/row.A95
    else:
        W_A95=1.;

    W_Q = row.Q/7.

    if Wflag=='AgeA95Q':
        Wabc=Wabc
    elif Wflag=='Age':
        Wabc[1:]=0    # set second and third value to zero
    elif Wflag=='A95':
        Wabc[::2]=0  # set first and third value to zero
    elif Wflag=='Q':
        Wabc[:2]=0   # set first and second value to zero
    elif Wflag=='AgeA95':
        Wabc[2]=0
    elif Wflag=='AgeQ':
        Wabc[1]=0
    elif Wflag=='A95Q':
        Wabc[0]=0

    Wj = (Wabc[0]*W_dt + Wabc[1]*W_A95 + Wabc[2]*W_Q)/np.sum(Wabc)

    weight = {}
    weight['Wj'] = Wj
    weight['W_dt'] = [W_dt,overlap]
    weight['W_A95'] = W_A95
    weight['W_Q'] = W_Q
    weight['Wabc'] = Wabc

    return weight


def wAPWP_pole2R(phi, lmbda):
    # NB phi=long, lmbda=lat

    phi = np.radians(phi)
    lmbda = np.radians(lmbda)

    R11 = np.sin(lmbda) * np.cos(phi)
    R12 = np.sin(lmbda) * np.sin(phi)
    R13 = -np.cos(lmbda)

    R21 = -np.sin(phi)
    R22 = np.cos(phi)
    R23 = 0

    R31 = np.cos(lmbda) * np.cos(phi)
    R32 = np.cos(lmbda) * np.sin(phi)
    R33 = np.sin(lmbda)

    Rj=np.vstack(([R11,R12,R13],[R21,R22,R23],[R31,R32,R33]))

    return Rj


def wAPWP_pole2I(phi, lmbda, K, N):

    Rj = wAPWP_pole2R(phi, lmbda)

    Aj = K*N/(1+K)
    Bj = K*N/(1+K)
    Cj = 2*N/(1+K)

    Dj = np.vstack(([Aj,0,0],[0,Bj,0],[0,0,Cj]))

    Ij = np.dot(np.dot(Rj.T,Dj),Rj)

    return Ij


#% calculate fisher parameters
def FishM(D,I,a95):

    xyzs = []
    for pI,pD in zip(I,D):
        xyzs.append(pygplates.PointOnSphere(pI,pD).to_xyz())

    N = len(xyzs)

    x = np.array(xyzs)[:,0]
    y = np.array(xyzs)[:,1]
    z = np.array(xyzs)[:,2]

    if N>1:
        R2 = np.sum(x)**2+np.sum(y)**2+np.sum(z)**2
        R = np.sqrt(R2)

        m1 = (np.sum(x))/R
        m2 = (np.sum(y))/R
        m3 = (np.sum(z))/R

        # Fisherian Parameter: kappa(K); alpha95(A95)
        kappa = (N-1)/(N-R)
        alpha95 = np.degrees(np.arccos(1.-(N-R)/R*(((1./0.05)**(1./(N-1)))-1.)))

        # Convert back to (Im,Dm)
        ImDm = pygplates.PointOnSphere(m1,m2,m3).to_lat_lon()

    elif N==1:
        ImDm = (I[0],D[0]); alpha95 = a95[0]; kappa = 0; N = 1
    elif N==0:
        ImDm = None; kappa = np.nan; N = 0

    return ImDm, alpha95, kappa, N


def preprocess_pole_data(df,ageFilter):

    # remove rows with nans
    # df = df.dropna(subset=['Oldage','Youngage'])
    df = df.dropna(subset=['max_age','min_age', 'alpha95'])

    # sort all rows based on the 'MidAge' column
    # df = df.sort_values(by='MidAge', axis=0, ascending=False)
    df = df.sort_values(by='age', axis=0, ascending=False)

    vgps = []

    for i,row in df.iterrows():
        # vgp = pygplates.PointOnSphere(row.Plat,row.Plong)
        vgp = pygplates.PointOnSphere(row.plat,row.plon)
        # composed_rotation = pygplates.FiniteRotation((row.Elat,row.Elong),np.radians(row.Eangle))
        composed_rotation = pygplates.FiniteRotation((row.Euler_lat,row.Euler_lon),np.radians(row.Euler_ang))

        dist = np.degrees(pygplates.GeometryOnSphere.distance(vgp,composed_rotation.get_euler_pole_and_angle()[0]))
        if dist>90.:
            composed_rotation = composed_rotation.get_inverse()

        rotated_vgp = composed_rotation * vgp

        rotated_vgp_lat = rotated_vgp.to_lat_lon()[0] * -1
        rotated_vgp_lon = rotated_vgp.to_lat_lon()[1] - 180.

        # # ensure the rotated pole is south pole LW
        # if rotated_vgp_lat > 0:
        #     rotated_vgp_lat = - rotated_vgp_lat
        #     rotated_vgp_lon = (rotated_vgp_lon + 180.) % 360. - 180.
        if rotated_vgp_lon<-180.:
            rotated_vgp_lon = rotated_vgp_lon+360.
        vgps.append((rotated_vgp_lat,rotated_vgp_lon))

    # Does this get used??
    # df['polesR_lat'] = zip(*vgps)[0]
    # df['polesR_lon'] = zip(*vgps)[1]
    df['polesR_lat'] = list(zip(*vgps))[0] # LW
    df['polesR_lon'] = list(zip(*vgps))[1] # LW
    # df['polesR_lat'] = list(zip(*vgps))[0]
    # df['polesR_lon'] = list(zip(*vgps))[1]
    # remove this, since the subsequent steps will do the same windowing anyway
    #df = df[(df.Youngage>=np.array(ageFilter).min()) & (df.Oldage<=np.array(ageFilter).max())]

    return df


def get_windowed_pole_list(df, Tinv, ageFilter, window):

    winHalf = window/2.

    #young_end_of_series = np.floor((df.Youngage.min()-winHalf)/winHalf)*winHalf
    #old_end_of_series = np.ceil((df.Oldage.max()+winHalf)/winHalf)*winHalf
    #ageRW = np.arange(young_end_of_series,old_end_of_series+Tinv,Tinv)
    ageRW = np.arange(ageFilter[0],ageFilter[1],Tinv)
    young_end_of_series = ageRW[0]
    old_end_of_series = ageRW[-1]


    windowed_poles_list = []

    for ageRW_instance in ageRW:

        age_window_young_end = ageRW_instance-window/2.
        age_window_old_end = ageRW_instance+window/2.

        #https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-two-integer-ranges-for-overlap/25369187
        # if max(a2, b2) - min(a1, b1) < (a2 - a1) + (b2 - b1).....
        # term1 = np.max(np.vstack((np.ones(df.Oldage.shape)*age_window_old_end, df.Oldage)), axis=0)   # max(a2, b2)
        # term2 = np.min(np.vstack((np.ones(df.Youngage.shape)*age_window_young_end, df.Youngage)), axis=0)         # min(a1, b1)
        # term3 = np.ones(df.Oldage.shape)*age_window_old_end - np.ones(df.Youngage.shape)*age_window_young_end       # (a2 - a1)
        # term4 = np.array(df.Oldage - df.Youngage)    # (b2 - b1)

        term1 = np.max(np.vstack((np.ones(df.max_age.shape)*age_window_old_end, df.max_age)), axis=0)   # max(a2, b2)
        term2 = np.min(np.vstack((np.ones(df.min_age.shape)*age_window_young_end, df.min_age)), axis=0)         # min(a1, b1)
        term3 = np.ones(df.max_age.shape)*age_window_old_end - np.ones(df.min_age.shape)*age_window_young_end       # (a2 - a1)
        term4 = np.array(df.max_age - df.min_age)    # (b2 - b1)

        test_result = term1 - term2 < term3 + term4

        pole_indices = np.where(test_result)[0]

        windowed_poles_list.append((ageRW_instance,age_window_young_end,age_window_old_end,pole_indices))

    return windowed_poles_list


def weighted_APWP(df, windowed_poles_list, A95c, Wscale, Wabc):

    APWP_fish, APWP_weight = [], []
    for counter,windowed_poles in enumerate(windowed_poles_list):

        df_subset = df.iloc[windowed_poles[3]]

        # if there are no data points in the window, exit loop
        if df_subset.empty:
            APWP_fish.append([windowed_poles[0],np.nan,np.nan,np.nan,np.nan,np.nan])
#            print (windowed_poles)
            APWP_weight.append([windowed_poles[0],np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
            continue

        Npts = len(df_subset)

        windowIN = windowed_poles[1:3]

        # Get weights for the VGPs in this window
        weights = wAPWP_weights(df_subset,windowIN,A95c,Wabc)

        df_subset['Wj'] = [Wscale * weight['Wj'] for weight in weights]

        meanW_WjNj = 0
        meanW_IjWjSUM = 0   # works, but should arguably be a 3x3 array or zeros

        for i,row in df_subset.iterrows():

#            Ij = wAPWP_pole2I(row.Plong,row.Plat,row.Kj,row.nj)
            # Ij = wAPWP_pole2I(row.Plong,row.Plat,19.60,10)
            Ij = wAPWP_pole2I(row.plon,row.plat,19.60,10)

            IjWj = row.Wj * Ij

            meanW_IjWjSUM += IjWj
#            meanW_WjNj = meanW_WjNj + row.Wj * row.nj
            meanW_WjNj = meanW_WjNj + row.Wj * 10

        eigenValues, eigenVectors = np.linalg.eig(meanW_IjWjSUM)

        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        coordMIN = pygplates.PointOnSphere(eigenVectors[:,2]).to_lat_lon()
        coordINT = pygplates.PointOnSphere(eigenVectors[:,1]).to_lat_lon()
        coordMAX = pygplates.PointOnSphere(eigenVectors[:,0]).to_lat_lon()

        Kx = np.abs(np.sum(eigenValues[0:2])/(np.sum(eigenValues)-2*eigenValues[0]))
        Ky = np.sum(eigenValues[0:2])/(np.sum(eigenValues)-2*eigenValues[1])

        if Kx<Ky:
            e95a = 140./np.sqrt(Kx*df_subset.Wj.sum())
            e95b = 140./np.sqrt(Ky*df_subset.Wj.sum())
        else:
            e95a = 140./np.sqrt(Ky*df_subset.Wj.sum())
            e95b = 140./np.sqrt(Kx*df_subset.Wj.sum())

        # ImDm,alpha95m,kappa,N = FishM(df_subset.Plong.tolist(),
        #                               df_subset.Plat.tolist(),
        #                               df_subset.A95.tolist())
        ImDm,alpha95m,kappa,N = FishM(df_subset.plon.tolist(),
                                      df_subset.plat.tolist(),
                                      df_subset.A95.tolist()) # A95 or alpha95?

        coordMINa = get_antipode(coordMIN[0],coordMIN[1])
        coordINTa = get_antipode(coordINT[0],coordINT[1])
        coordMAXa = get_antipode(coordMAX[0],coordMAX[1])

        #For this angle comparison, finding which distance is less than 180 deg
        # should be same as finding the smaller distance (as in matlab version)
        if distance_on_unit_sphere(coordMIN[1],coordMIN[0],ImDm[0],ImDm[1]) >= distance_on_unit_sphere(coordMINa[1],coordMINa[0],ImDm[0],ImDm[1]):
            # print(counter, distance_on_unit_sphere(coordMIN[1],coordMIN[0],ImDm[0],ImDm[1]), distance_on_unit_sphere(coordMINa[1],coordMINa[0],ImDm[0],ImDm[1]), 'if') # LW debug
            omega = (coordINTa[1]-coordMINa[1])/ np.abs(coordINTa[1]-coordMINa[1])*np.degrees(np.arccos(np.sin(np.radians(coordINTa[0]))/np.cos(np.radians(coordMINa[0]))))

            # not sure how omega could ever be more than a single value (ie not an array)
            if np.isnan(omega):
                omega=np.zeros(len(omega),1)
            if not np.isreal(omega):
                omega=np.zeros(len(omega),1)

            meanW_eigDir = np.hstack(([coordMINa[1],coordMINa[0],1,coordINT[1],coordINT[0],1,coordMAX[1],coordMAX[0],1],
                                      [coordMIN[1],coordMIN[0],1,coordINTa[1],coordINTa[0],1,coordMAXa[1],coordMAXa[0],1]))

            #%----------------------------------------------------------------------

            meanW_meanW=[coordMINa[1],coordMINa[0],e95a,e95b,omega,Kx,Ky,Npts]
            # print(windowed_poles[0],meanW_meanW)
            # weightm_out = [windowed_poles[0],coordMINa[1],coordMINa[0],e95a,e95b,omega,Kx,Ky,Npts]
            #%----------------------------------------------------------------------

            if Npts==1:
                eigDirTmp = np.vstack(([coordMINa[1],coordMINa[0]],
                                       [coordINT[1],coordINT[0]],
                                       [coordMAX[1],coordMAX[0]],
                                       [coordMIN[1],coordMIN[0]],
                                       [coordINTa[1],coordINTa[0]],
                                       [coordMAXa[1],coordMAXa[0]]))

                distTMP = []
                for kk in np.arange(eigDirTmp.shape[0]):
                    distTMP.append(pygplates.GeometryOnSphere.distance(pygplates.PointOnSphere(ImDm[0],ImDm[1]),
                                                                       pygplates.PointOnSphere(eigDirTmp[kk,1],eigDirTmp[kk,0])))  #distance(ImTPM,DmTPM,eigDirTmp(kk,2),eigDirTmp(kk,1));

                indTMP = np.array(distTMP).argmin()
                meanW_meanW[0:2] = eigDirTmp[indTMP,0:2]

        ## Added by Yebo, 2019.11.10
        else:
            # print(counter, distance_on_unit_sphere(coordMIN[1],coordMIN[0],ImDm[0],ImDm[1]), distance_on_unit_sphere(coordMINa[1],coordMINa[0],ImDm[0],ImDm[1]), 'else') # LW debug
            omega = (coordINT[1]-coordMIN[1])/ np.abs(coordINT[1]-coordMIN[1])*np.degrees(np.arccos(np.sin(np.radians(coordINT[0]))/np.cos(np.radians(coordMIN[0]))))
            if np.isnan(omega):
                omega=np.zeros(len(omega),1)
            if not np.isreal(omega):
                omega=np.zeros(len(omega),1)

            meanW_eigDir = np.hstack(([coordMIN[1],coordMIN[0],1,coordINT[1],coordINT[0],1,coordMAX[1],coordMAX[0],1],
                                      [coordMINa[1],coordMINa[0],1,coordINTa[1],coordINTa[0],1,coordMAXa[1],coordMAXa[0],1]))    # LW
            meanW_meanW = [coordMIN[1],coordMIN[0],e95a,e95b,omega,Kx,Ky,Npts]

            if Npts==1:
                eigDirTmp = np.vstack(([coordMINa[1],coordMINa[0]],
                                       [coordINT[1],coordINT[0]],
                                       [coordMAX[1],coordMAX[0]],
                                       [coordMIN[1],coordMIN[0]],
                                       [coordINTa[1],coordINTa[0]],
                                       [coordMAXa[1],coordMAXa[0]]))

                distTMP = []
                for kk in np.arange(eigDirTmp.shape[0]):
                    distTMP.append(pygplates.GeometryOnSphere.distance(pygplates.PointOnSphere(ImDm[0],ImDm[1]),
                                                                       pygplates.PointOnSphere(eigDirTmp[kk,1],eigDirTmp[kk,0])))  #distance(ImTPM,DmTPM,eigDirTmp(kk,2),eigDirTmp(kk,1));

                indTMP = np.array(distTMP).argmin()
                meanW_meanW[0:2] = eigDirTmp[indTMP,0:2]

            if not np.isreal(alpha95m):
                alpha95m = 0

        # meanW_fishM = [windowed_poles[0],ImDm[0],ImDm[1],alpha95m,kappa,N]
        # APWP_fish.append(meanW_fishM)
        # APWP_weight.append(meanW_meanW)
        # df_fishm = pd.DataFrame(APWP_fish, columns=['AgeWindowMidPoint','Plat','Plong','A95','kappa','N'])
        # df_weightm = pd.DataFrame(APWP_weight, columns=['Plong','Plat','e95a','e95b','omega','kx','ky','N']) # LW

        meanW_fishM = [windowed_poles[0],ImDm[0],ImDm[1],alpha95m,kappa,N]
        meanW_meanW.insert(0, windowed_poles[0]) # LW
        APWP_fish.append(meanW_fishM)
        APWP_weight.append(meanW_meanW) # LW
        df_fishm = pd.DataFrame(APWP_fish, columns=['AgeWinM','Plat','Plong','A95','kappa','N'])
        df_weightm = pd.DataFrame(APWP_weight, columns=['AgeWinM','Plong','Plat','e95a','e95b','omega','kx','ky','N']) # LW

    return df_fishm,df_weightm


def create_ellipse(centerlon, centerlat, major_axis, minor_axis, angle, n=100):
    """
    This function enables general error ellipses

    Parameters
    -----------
    centerlon : longitude of the center of the ellipse
    centerlat : latitude of the center of the ellipse
    major_axis : Major axis of ellipse
    minor_axis : Minor axis of ellipse
    angle : angle of major axis in degrees east of north
    n : number of points with which to apporximate the ellipse

    Returns
    ---------

    """
    angle = angle * (np.pi/180)
    glon1 = centerlon
    glat1 = centerlat
    major_axis = major_axis
    minor_axis = minor_axis
    X = []
    Y = []
    for azimuth in np.linspace(-180, 180, n):
        az_rad = azimuth*(np.pi/180)
        radius = ((major_axis*minor_axis)/(((minor_axis*np.cos(az_rad-angle))
                                            ** 2 + (major_axis*np.sin(az_rad-angle))**2)**.5))

        # glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius* (180/np.pi)) # LW
        X.append(glon2) # LW
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])
    return X, Y

def shoot(lon, lat, azimuth, maxdist=None):
    """
    This function enables A95 error ellipses to be drawn around
    paleomagnetic poles in conjunction with equi
    (from: http://www.geophysique.be/2011/02/20/matplotlib-basemap-tutorial-09-drawing-circles/)
    """
    from past.utils import old_div
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    scaleLW1 = .96
    s = maxdist / scaleLW1 # LW
    faz = azimuth * np.pi / 180.

    EPS = 0.00000000005

    a = old_div(6378.13, 1.852)
    f = old_div(1, 298.257223563)
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf == 0):
        b = 0.
    else:
        b = 2. * np.arctan2(tu, cf)

    cu = old_div(1., np.sqrt(1 + tu * tu))
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (old_div(1., (r * r)) - 1.))
    x = old_div((x - 2.), x)
    c = 1. - x
    c = old_div((x * x / 4. + 1.), c)
    d = (0.375 * x * x - 1.) * x
    tu = old_div(s, (r * a * c))
    y = tu
    c = y + 1

    sy = np.sin(y)
    cy = np.cos(y)
    cz = np.cos(b + y)
    e = 2. * cz * cz - 1.
    c = y
    x = e * cy
    y = e + e - 1.
    y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
         d / 4. - cz) * sy * d + tu

    while (np.abs(y - c) > EPS):
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
             d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2 * np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2 * np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 = glon2 * 180/np.pi # LW
    glat2 = glat2 * 180/np.pi # LW
    baz = baz * 180/np.pi # LW

    return (glon2, glat2, baz)
