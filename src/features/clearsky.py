import warnings
import threading

import numpy as np
import pandas as pd
import numba as nb

import pvlib
from pvlib import tools
from pvlib.solarposition import _spa_python_import
from pvlib.location import Location

spa = _spa_python_import('numba')

def haurwitz(apparent_zenith):
    """Caluclate global horizontal irradiance from apparent zenith angle of sun
    using Haurwitz method.
    
    Parameters
    ----------
    apparent_zenith : array_like
        Apparent zenith angle of sun in degrees

    Returns
    -------
    array_like
        Global horizontal irradiance
        
    Notes
    -----
    Based on `pvlib.clearsky.haurwitz`
    """
    cos_zenith = tools.cosd(apparent_zenith)
    clearsky_ghi = np.zeros_like(apparent_zenith)
    cos_zen_gte_0 = cos_zenith > 0
    clearsky_ghi[cos_zen_gte_0] = (1098.0 * cos_zenith[cos_zen_gte_0] *
                                   np.exp(-0.059/cos_zenith[cos_zen_gte_0]))

    return clearsky_ghi


@nb.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:,:])', 
        nopython=True, nogil=True)
def _solar_position_loop(unixtime, lats, lons, loc_args, out):
    """Modify the array `out` array inplace to input the calculated solar 
    position at unixtime times and lats-lons locations.
    
    Notes
    -----
    Based on  `pvlib.spa.solar_position_loop` function.

    For now we assume the elevation, pressure and temp are the same at all 
    locations and times. This is just for a simple approximation.
    """
    # 
    elev = loc_args[0]
    pressure = loc_args[1]
    temp = loc_args[2]
    delta_t = loc_args[3]
    atmos_refract = loc_args[4]

    for i in range(unixtime.shape[0]):
        utime = unixtime[i]
        jd = spa.julian_day(utime)
        jde = spa.julian_ephemeris_day(jd, delta_t)
        jc = spa.julian_century(jd)
        jce = spa.julian_ephemeris_century(jde)
        jme = spa.julian_ephemeris_millennium(jce)
        R = spa.heliocentric_radius_vector(jme)

        L = spa.heliocentric_longitude(jme)
        B = spa.heliocentric_latitude(jme)
        Theta = spa.geocentric_longitude(L)
        beta = spa.geocentric_latitude(B)
        x0 = spa.mean_elongation(jce)
        x1 = spa.mean_anomaly_sun(jce)
        x2 = spa.mean_anomaly_moon(jce)
        x3 = spa.moon_argument_latitude(jce)
        x4 = spa.moon_ascending_longitude(jce)
        delta_psi = spa.longitude_nutation(jce, x0, x1, x2, x3, x4)
        delta_epsilon = spa.obliquity_nutation(jce, x0, x1, x2, x3, x4)
        epsilon0 = spa.mean_ecliptic_obliquity(jme)
        epsilon = spa.true_ecliptic_obliquity(epsilon0, delta_epsilon)
        delta_tau = spa.aberration_correction(R)
        lamd = spa.apparent_sun_longitude(Theta, delta_psi, delta_tau)
        v0 = spa.mean_sidereal_time(jd, jc)
        v = spa.apparent_sidereal_time(v0, delta_psi, epsilon)
        alpha = spa.geocentric_sun_right_ascension(lamd, epsilon, beta)
        delta = spa.geocentric_sun_declination(lamd, epsilon, beta)

        m = spa.sun_mean_longitude(jme)
        eot = spa.equation_of_time(m, alpha, delta_psi, epsilon)
        
        
        for j in range(lats.shape[0]):
            lat = lats[j]
            lon = lons[j]
            H = spa.local_hour_angle(v, lon, alpha)
            xi = spa.equatorial_horizontal_parallax(R)

            u = spa.uterm(lat)
            x = spa.xterm(u, lat, elev)
            y = spa.yterm(u, lat, elev)
            delta_alpha = spa.parallax_sun_right_ascension(x, xi, H, delta)
            delta_prime = spa.topocentric_sun_declination(delta, x, y, xi, delta_alpha, H)
            H_prime = spa.topocentric_local_hour_angle(H, delta_alpha)
            e0 = spa.topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
            delta_e = spa.atmospheric_refraction_correction(pressure, temp, e0, atmos_refract)
            e = spa.topocentric_elevation_angle(e0, delta_e)
            theta = spa.topocentric_zenith_angle(e)
            out[i, j] = theta


def _solar_position_numba(unixtime, lats, lons, elev, pressure, temp, delta_t,
                         atmos_refract, numthreads):
    """Calculate the solar position using the numba compiled functions
    and multiple threads. Very slow if functions are not numba compiled.
    
    Notes
    -----    
    Based on  `pvlib.spa.solar_position_numba` function.
    """
    # these args are the same for each thread
    loc_args = np.array([elev, pressure, temp, delta_t, atmos_refract])

    # construct dims x ulength array to put the results in
    ulength = unixtime.shape[0]
        
    results = np.zeros((ulength, lats.shape[0]), dtype=np.float64)

    if unixtime.dtype != np.float64:
        unixtime = unixtime.astype(np.float64)

    if ulength < numthreads:
        warnings.warn('The number of threads is more than the length of '
                      'the time array. Only using %s threads.'.format(ulength))
        numthreads = ulength

    if numthreads <= 1:
        _solar_position_loop(unixtime, lats, lons, loc_args, results)
        return results

    # split the input and output arrays into numthreads chunks
    time_split = np.array_split(unixtime, numthreads)
    results_split = np.array_split(results, numthreads)
    chunks = [[time_split[i], lats, lons, loc_args, results_split[i]] for i in range(numthreads)]
    # Spawn one thread per chunk
    threads = [threading.Thread(target=_solar_position_loop, args=chunk)
               for chunk in chunks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results

def spa_python(times, latitudes, longitudes,
               altitude=0, pressure=101325, temperature=12, delta_t=67.0,
               atmos_refract=None, numthreads=4, **kwargs):
    """
    Calculate the solar position using a python implementation of the
    NREL SPA algorithm [1].

    If numba is installed, the functions can be compiled to
    machine code and the function can be multithreaded.
    Without numba, the function evaluates via numpy with
    a slight performance hit.
    
    Parameters
    ----------
    time : pandas.DatetimeIndex
        Must be localized or UTC will be assumed.
    latitudes : array_like, float
        Latitudes in decimal degrees. Positive north of equator, negative
        to south.
    longitudes : array_like, float
        Longitudes in decimal degrees. Positive east of prime meridian,
        negative to west.
    altitude : float, default 0
        Distance above sea level.
    pressure : int or float, optional, default 101325
        avg. yearly air pressure in Pascals.
    temperature : int or float, optional, default 12
        avg. yearly air temperature in degrees C.
    delta_t : float, optional, default 67.0
        If delta_t is None, uses spa.calculate_deltat
        using time.year and time.month from pandas.DatetimeIndex.
        For most simulations specifing delta_t is sufficient.
        Difference between terrestrial time and UT1.
        *Note: delta_t = None will break code using nrel_numba,
        this will be fixed in a future version.*
        The USNO has historical and forecasted delta_t [3].
    atmos_refrac : None or float, optional, default None
        The approximate atmospheric refraction (in degrees)
        at sunrise and sunset.
    numthreads : int, optional, default 4
        Number of threads to use if how == 'numba'.
        
    Returns
    -------
    array_like
        Apparent zenith (degrees) with time along zeroth axis and location along 
        first axis.
    
    References
    ----------
    .. [1] I. Reda and A. Andreas, Solar position algorithm for solar
       radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
    
    Notes
    -----
    Based on  `pvlib.solarposition.spa_python` function.
    """
    
    lats = latitudes
    lons = longitudes
    elev = altitude
    pressure = pressure / 100  # pressure must be in millibars for calculation

    atmos_refract = atmos_refract or 0.5667

    if not isinstance(times, pd.DatetimeIndex):
        try:
            times = pd.DatetimeIndex(times)
        except (TypeError, ValueError):
            times = pd.DatetimeIndex([times, ])

    unixtime = np.array(times.astype(np.int64)/10**9)

    spa = _spa_python_import('numba')

    delta_t = delta_t or spa.calculate_deltat(times.year, times.month)

    app_zenith = _solar_position_numba(unixtime, lats, lons, elev, pressure, 
                                temperature, delta_t, atmos_refract, numthreads)
    return app_zenith


def compute_clearsky(times, latitudes, longitudes):
    """A slow function for calculating 3 clearsky quantities. Diffuse Horizontal 
    Irradiance, Global Horizontal Irradiance and Direct Normal Irradiance.
    
    Returns
    -------
    numpy.ndarray
    """
    clearsky = np.full(shape=(len(times), len(latitudes), 3), 
                       fill_value=np.NaN, dtype=np.float32)

    
    for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
        clearsky_for_location = (
            Location(
                latitude=lat,
                longitude=lon,
                tz='UTC'
            ).get_clearsky(times)
        )
        clearsky[:,i,:] = clearsky_for_location.values    
    
    return clearsky

if __name__=='__main__':
    
    import pandas as pd
    
    N = 40 # number of locations
    times = pd.date_range(start='2019-04-01', periods=24, freq='1h')
    latitudes = np.random.uniform(50, 59, size=N)
    longitudes = np.random.uniform(-5.9, 1, size=N)
    apparent_zenith = spa_python(times, latitudes, longitudes)
    haurwitz_ghi = haurwitz(apparent_zenith)
    pd.DataFrame(haurwitz_ghi, index=times).plot()
    print(haurwitz_ghi)