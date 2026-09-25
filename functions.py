import numpy as np
from scipy.special import k0
from scipy.spatial import distance


def G_1D(p, t=0, p_interp=None):
    '''
    Compute matrix G formed by 1D Cartesian Green's functions.
    
    Parameters
    ----------
    p : array 1D
        Vector with N coordinates.

    t : float
        Positive scalar, in the interval [0, 1[ , controlling the
        tension in spline surface.

    p_interp : array 1D
        Vector with N interpolation coordinates.

    Returns
    -------
    G : array 2D
        Matrix of Green's functions.
    '''

    # Verify p and t
    try:
        p = np.asarray(p)
    except:
        raise ValueError("p must be convertible to array")
    if p.ndim != 1:
        raise ValueError("p must be a vector")
    if p.size <= 2:
        raise ValueError("p must have more than two elements")
    try:
        t = float(t)
    except:
        raise ValueError("t must be convertible to float")
    if (t < 0) or (t >= 1):
        raise ValueError("t must be in the interval [0, 1[")

    # Verify p_interp
    if p_interp is not None:
        try:
            p_interp = np.asarray(p_interp)
        except:
            raise ValueError("p_interp must be convertible to array")
        if p_interp.ndim != 1:
            raise ValueError("p_interp must be a vector")
        if p_interp.size <= 2:
            raise ValueError("p_interp must have more than two elements")
        y = p_interp
    else:
        y = p

    tau = np.sqrt(t/(1.-t))

    # Absolute distance matrix
    R = np.abs(y[:, np.newaxis] - p)
    
    if tau == 0:
        G = R**3
    else:
        G = np.exp(-tau*R) + tau*R - 1
    return G


def G_2D(P, t=0, P_interp=None):
    '''
    Compute matrix G formed by 2D Cartesian Green's functions.
    
    Parameters
    ----------
    P : array 2D
        Matrix with N rows and 2 columns. The first and second 
        columns contain, respectively, the coordinates x and y
        of the data points.

    t : float
        Positive scalar, in the interval [0, 1[ , controlling the
        tension in spline surface.

    P_interp : array 2D
        Matrix with N rows and 2 columns. The first and second 
        columns contain, respectively, the coordinates x and y
        of the interpolating points.

    Returns
    -------
    G : array 2D
        Matrix of Green's functions.
    '''

    # Verify p and t
    try:
        P = np.asarray(P)
    except:
        raise ValueError("P must be convertible to array")
    if P.ndim != 2:
        raise ValueError("P must be a matrix")
    if P.shape[0] <= 2:
        raise ValueError("P must have more than two rows")
    if P.shape[1] != 2:
        raise ValueError("P must have two columns")
    try:
        t = float(t)
    except:
        raise ValueError("t must be convertible to float")
    if (t < 0) or (t >= 1):
        raise ValueError("t must be in the interval [0, 1[")

    # Verify P_interp
    if P_interp is not None:
        try:
            P_interp = np.asarray(P_interp)
        except:
            raise ValueError("P_interp must be convertible to array")
        if P_interp.ndim != 2:
            raise ValueError("P_interp must be a matrix")
        if P_interp.shape[0] <= 2:
            raise ValueError("P_interp must have more than two rows")
        if P_interp.shape[1] != 2:
            raise ValueError("P_interp must have two columns")
        Y = P_interp
    else:
        Y = P

    tau = np.sqrt(t/(1.0-t))

    # Euclidean distance matrix
    R = distance.cdist(Y, P, 'euclidean')
    R[R <= 1e-15] = 1e-15

    if tau == 0:
        G = R*R*(np.log(R) - 1.0)
    else:        
        G = k0(tau*R) + np.log(tau*R)
    return G


def G_3D(P, t=0, P_interp=None):
    '''
    Compute matrix G formed by 3D Cartesian Green's functions.
    
    Parameters
    ----------
    P : array 2D
        Matrix with N rows and 3 columns. The first, second and third
        columns contain, respectively, the coordinates x, y and z
        of the data points.

    t : float
        Positive scalar, in the interval [0, 1[ , controlling the
        tension in spline surface.

    P_interp : array 2D
        Matrix with N rows and 3 columns. The first, second and third
        columns contain, respectively, the coordinates x, y and z
        of the interpolating points.

    Returns
    -------
    G : array 2D
        Matrix of Green's functions.
    '''

    # Verify p and t
    try:
        P = np.asarray(P)
    except:
        raise ValueError("P must be convertible to array")
    if P.ndim != 2:
        raise ValueError("P must be a matrix")
    if P.shape[0] <= 2:
        raise ValueError("P must have more than two rows")
    if P.shape[1] != 3:
        raise ValueError("P must have 3 columns")
    try:
        t = float(t)
    except:
        raise ValueError("t must be convertible to float")
    if (t < 0) or (t >= 1):
        raise ValueError("t must be in the interval [0, 1[")

    # Verify P_interp
    if P_interp is not None:
        try:
            P_interp = np.asarray(P_interp)
        except:
            raise ValueError("P_interp must be convertible to array")
        if P_interp.ndim != 2:
            raise ValueError("P_interp must be a matrix")
        if P_interp.shape[0] <= 2:
            raise ValueError("P_interp must have more than two rows")
        if P_interp.shape[1] != 3:
            raise ValueError("P_interp must have 3 columns")
        Y = P_interp
    else:
        Y = P

    tau = np.sqrt(t/(1.0-t))

    # Euclidean distance matrix
    R = distance.cdist(Y, P, 'euclidean')
    R[R <= 1e-15] = 1e-15

    if tau == 0:
        G = R
    else:        
        G = (np.exp(-tau * R) - 1.0) / (tau * R) + 1.0
    return G