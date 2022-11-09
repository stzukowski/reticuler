import mpmath as mp
import numpy as np
from scipy.optimize import brentq
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
    
def analytical_trajectory(t_range, x0=1.5, d0=1):
    """
    Numerically solved implicit equation for the position of the tip as a 
    function of time in a semi-infinite channel of width=2.
    Reference: 
    T. Gubiec, P. Szymczak,
    "Fingered growth in channel geometry: A Loewner equation approach",
    Phys. Rev. E, 77, 041602, 2008
    
    Parameters
    -------
    t_range : float or array
        Times at which tip position will be calculated.
    x0 : float, default 1.5
        Initial position of the finger in a channel x=(0, 2).
    d0 : float, default 1.0
        Growth factor, for details check the reference.
        Constant d0 corresponds to eta=-2.


    Returns
    -------
    np.array
        A n-2 array of (x,y) coordinates of the tip at times from t_range.
    """
    try:
        iter(t_range)
    except TypeError:
        t_range = [t_range]
    x0 = x0 - 1 # original equations are for channel x=(-1,1), we use rather x=(0,2)
    prev = x0 + (t_range[0]+0.5)*1j
    trajectory = np.empty( (len(t_range), 2) )
    for i, t in enumerate(t_range):
        temp = lambda g: mp.exp(mp.pi**2/8 * t) * \
            ( 1 - mp.exp( -d0*t*mp.pi**2/4 ) * mp.sin( x0*mp.pi/2 )**2 )**0.25 + \
                (-(mp.cos(g*mp.pi/2))**0.5 / mp.sin(x0*mp.pi/2) \
                 - mp.ellipf( g*mp.pi/4, 2 ) \
                 + mp.ellipf( 0.5 * mp.asin( mp.exp(-d0*t*mp.pi**2/8) * mp.sin(x0*mp.pi/2) ), 2 ) 
                 ) * mp.sin( x0*mp.pi/2 )
        
        g0 = mp.findroot(temp, prev, solver='muller')
        prev = g0
        if prev.imag<0.5:
            prev = prev.real + 0.5j
        # to come back to channel x=(0,2) we shift real: real+1
        trajectory[i] = np.array([g0.real+1, g0.imag])
    trajectory[trajectory[:,1]<1e-6, 1] = 0
    return trajectory

def analytical_trajectory_timer(height, x0=1.5, d0=1):
    """
    Calculates time at which finger in channel reaches desired height.

    Parameters
    ----------
    height : float
        Desired height (0<height<16).
    x0 : float, optional
        Initial position of the finger in a channel x=(0, 2). The default is 1.5.
    d0 : float, optional
        Growth factor, for details check Gubiec&Szymczak, PRE, 77, 041602, 2008.
        Constant d0 corresponds to eta=-2. The default is 1.

    Returns
    -------
    float
        Time at which finger in channel reaches desired height.
    """
    temp = lambda t: height - analytical_trajectory(t, x0, d0)[0,1]
    # t_height = mp.findroot(temp, (0,3), solver='pegasus')
    t_height = brentq(temp, 0, 10)
    return float(t_height)

def analytical_trajectory_height5():
    """ 
    Imports and returns trajectory of the finger of height=5, x0=1.5, d0=1. 
    Calculated at timesteps: np.linspace(0, (t0)**(2/3), 10000)**3/2;
    Stored in a file 'analytical_finger_trajectory_height5.csv'
    
    Importing files in packages:
        https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
    
    Returns
    -------
    trajectory : np.array
        A 10000-2 array with (x,y) coordinates of the progressive tip positions.
    """
    # trajectory = np.loadtxt('analytical_finger_trajectory_height5.csv')
    trajectory_str = pkg_resources.read_text(__package__, 'analytical_finger_trajectory_height5.csv')
    trajectory = np.fromstring(trajectory_str, sep='\n').reshape(10000,2)
    return trajectory

def construct_and_compare_with_analytical(trajectory):
    error = 0
    trajectory_analytical = np.empty_like(trajectory)
    for i, pair in enumerate(trajectory):
        t = analytical_trajectory_timer(pair[1])
        trajectory_analytical[i] = analytical_trajectory(t)
        error = error + np.abs(trajectory_analytical[i, 0]-pair[0])
    return error/len(trajectory), trajectory_analytical

def compare_with_analytical(trajectory):
    error = 0
    trajectory_analytical = analytical_trajectory_height5()
    for i, pair in enumerate(trajectory):
        x = pair[0]
        y = pair[1]
        ind = np.max( (np.sum(trajectory_analytical[:,1]<=y), 1) )
        if np.abs(y-trajectory_analytical[ind-1,1])<1e-8 or np.abs(y-trajectory_analytical[ind,1])<1e-8:
            error = error + np.abs(x - trajectory_analytical[ind,0])
        else:
            x0 = trajectory_analytical[ind-1,0]
            x1 = trajectory_analytical[ind,0]
            y0 = trajectory_analytical[ind-1,1]
            y1 = trajectory_analytical[ind,1]
            #print(y0-y,y-y,y1-y)
            a = (y0-y1)/(x0-x1)
            b = y0 - a*x0
            x_cross = (y - b)/a
            error = error + np.abs(x - x_cross)
    return error/len(trajectory)