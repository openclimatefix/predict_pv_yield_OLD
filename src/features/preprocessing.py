import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin

from clearsky import spa_python, haurwitz


class RobustMinMaxScaler(MinMaxScaler):
    """Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one. 
    
    Each feature is first clipped to extremal percentages, ie clipped to the 
    1st and 99th percentile. These clipped values are then scaled to the given 
    range.

    This transformation is used to compensate for data with outlying points.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    saturation_fraction : float in range [0,1]
        Data is clipped to the saturation_fraction*100% and 
        (1-saturation_fraction)*100% values.
    
    See also
    --------
    sklearn.preprocessing.MinMaxScaler: Equivalent class without the robust 
    aspect of scaling. These classes become equivalent when saturation_fraction
    is set to 0.
    
    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    """
    
    def __init__(self, feature_range=(0, 1), saturation_fraction=0.01, 
                 copy=True):
        self.feature_range = feature_range
        self.copy = copy
        self.saturation_fraction = saturation_fraction
        
    def partial_fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the min and max
            used for later scaling along the features axis.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Transformer instance.
        """
        percentile_min = self.saturation_fraction*100
        percentile_max = (1-self.saturation_fraction)*100

        robust_data_min = np.nanpercentile(X, percentile_min, axis=0)
        robust_data_max = np.nanpercentile(X, percentile_max, axis=0)
        
        self.robust_data_min = robust_data_min
        self.robust_data_max = robust_data_max
        
        Xr = X.clip(self.robust_data_min, self.robust_data_max, axis=1)
        return super().partial_fit(Xr, y=y)
        
    def transform(self, X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)
        Xr = X.clip(self.robust_data_min, self.robust_data_max, axis=1)
        return super().transform(Xr)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class ClearskyScalar(TransformerMixin):
    """Transform PV data by normalising with respect to the Global Horizontal 
    Irradience (GHI).

    Each PV measurement is scaled by the calculated GHI at the appropriate time
    and location.
    
    Parameters
    ----------
    latitudes : array_like, float
        Latitudes in decimal degrees. Positive north of equator, negative
        to south.
    longitudes : array_like, float
        Longitudes in decimal degrees. Positive east of prime meridian,
        negative to west.
    g0 : float, optional
        A fudge factor added to each calculated GHI to avoid dividing by zero.
        Default is 0 so zero-division may present an error and these values will
        returned as NaN.
    
    See also
    --------
    sclearsky.spa_python: Function used to calculate GHI.
    
    Notes
    -----
    NaNs are treated as missing values and maintained in transform.
    """
    
    def __init__(self, latitudes, longitudes, g0=0):
        assert latitudes.shape==longitudes.shape
        self.lats = latitudes
        self.lons = longitudes
        self.g0 = g0
        
    def haurwitz_ghi(self, times):
        apparent_zenith = spa_python(times, self.lats, self.lons, numthreads=3)
        return haurwitz(apparent_zenith) + self.g0

    def transform(self, X):
        '''Divides by the GHI at each loaction and time
        
        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Input data that will be transformed. Index must be 
            pandas.DatetimeIndex type.
            
        Returns
        -------
        Xt : pandas.DataFrame of shape (n_samples, n_features)
            Transformed data.
        '''
        assert len(self.lats)==X.shape[1]
        GHI = self.haurwitz_ghi(X.index)
        GHI[GHI==0] = np.nan
        return X/GHI
    
    def inverse_transform(self, X):
        '''Multiplies by the GHI at each loaction and time.
        
        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Input data that will be transformed. Index must be 
            pandas.DatetimeIndex type.
            
        Returns
        -------
        Xt : pandas.DataFrame of shape (n_samples, n_features)
            Transformed data.
        '''        
        assert len(self.lats)==X.shape[1]
        GHI = self.haurwitz_ghi(X.index)
        return X*GHI
    
if __name__=='__main__':
    
    import pandas as pd
    
    N = 40 # number of locations
    P=24 # number of periods
    
    times = pd.date_range(start='2019-04-01', periods=P, freq='1h')
    latitudes = np.random.uniform(50, 59, size=N)
    longitudes = np.random.uniform(-5.9, 1, size=N)
    X = pd.DataFrame(np.random.normal(size=(P,N)), index=times)
    
    ghi_scalar = ClearskyScalar(latitudes,  longitudes, g0=10)
    rminmax_scalar = RobustMinMaxScaler(saturation_fraction=0.01)
    
    X_effective_area = ghi_scalar.transform(X)
    
    X_ea_scaled = rminmax_scalar.fit_transform(X_effective_area)
    
    print(X_ea_scaled)