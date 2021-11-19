import numpy as np
import os


def get_files_in_directory(path): 
    """
    Retrieves file names from a directory \
    \n\nInput: path = directory \
    \n\nOutput: list of subdirectories
    """

    # The last conditional here is in order to ignore the /DS_store file in macs 
    return [os.path.join(path, name) for name in os.listdir(path)
            if (os.path.isfile(os.path.join(path, name)) and (not name.startswith('.')))  ]

def spherical2cartesian(theta, phi):
    """
    v[0] = theta / Latitude
    v[1] = phi   / Longitude
    """
    
    x = np.cos(theta) * np.cos(phi)  
    y = np.cos(theta) * np.sin(phi)  
    z = np.sin(theta)  
    
    return x, y, z

def GCD_cartesian(cartesian1, cartesian2):
    
    gcd =  np.arccos(np.dot(cartesian1,cartesian2))
    
    return gcd