import numpy as np

def generatePAMConstellation(M, d):
    """
    Generates a PAM constellation with M amplitude levels and distance d between adjacent levels.

    Parameters:
        M (int): Constellation size (number of amplitude levels).
        d (float): Distance between adjacent amplitude levels.

    Returns:
        np.ndarray: A 1D array containing the amplitude levels of the PAM constellation.
    """
    # Generate values from -((M-1)/2) to ((M-1)/2) with a step of 1, then scale by d.
    pam = np.arange(-((M-1)/2), ((M-1)/2) + 1, 1) * d
    pam = pam / np.sqrt(np.mean(pam**2))
    return pam