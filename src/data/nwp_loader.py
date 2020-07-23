## TO DO
# - Fix loader so variables at different pressure levels can be used
# - look into pytorch Dataset class to improve loader
# - Easier support for signle point values instead of rectangular patch

import xarray as xr
import pandas as pd
import numpy as np

import gcsfs
from . constants import GCP_FS

NWP_ZARR_PATHS = ['solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/{}'.format(d) 
                  for d in ['2018_1-6','2018_7-12','2019_1-6','2019_7-12']]
NWP_AGG_PATH = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/aggregate'

NWP_STORES = [gcsfs.mapping.GCSMap(NWP_ZARR_PATH, gcs=GCP_FS, 
                                       check=True, create=False) 
              for NWP_ZARR_PATH in NWP_ZARR_PATHS]
NWP_AGG_STORE = gcsfs.mapping.GCSMap(NWP_AGG_PATH, gcs=GCP_FS, 
                                       check=True, create=False)

AVAILABLE_CHANNELS = pd.DataFrame(
    {'channel_name': [
        't', 'r', 'dpt', 'vis', 'si10', 'wdir10', 'prmsl', 'unknown_1', 
        'unknown_2', 'prate', 'unknown_3', 'cdcb', 'lcc', 'mcc', 'hcc', 
        'unknown_4', 'sde', 'hcct', 'dswrf', 'dlwrf', 'h', 'ws_p', 't_p', 
        'gh_p', 'r_p', 'wdir_p', 'gust10', 'gust10_m'
    ],
     'description': [
        '1.5m air temperature at surface',
        '1.5m Relative humidity',
        '1.5m dew point temperature',
        '1.5m visibility',
        '10m wind speed',
        '10m wind direction',
        'Mean sea level pressure',
        "Uncertain. This is likely '1.5m fog probability'",
        "Uncertain. This is likely 'snow fraction'",
        'Total precipitation rate',
        "Uncertain. This might be 'cloud fraction below 1000ft ASL'",
        'Cloud base',
        'Low cloud cover',
        'Medium cloud cover',
        'High cloud cover',
        "Uncertain. This might be 'wet/dry bulb freezing level height'",
        'Snow depth',
        'Height of convective cloud top',
        'Downward short-wave radiation flux',
        'Downward long-wave radiation flux',
        'Geometrical height',
        'Wind speed at multiple pressure levels',
        'Temperature at multiple pressure levels',
        'Geopotential Height at multiple pressure levels',
        'Relative humidity at multiple pressure levels',
        'Wind direction at multiple pressure levels',
        '10m wind gust',
        '10m maximum wind gust in hour (T+1 to T+36)',
     ],
    }
).set_index('channel_name', drop=True)

DEFAULT_CHANNELS = ['t', 'dswrf', 'lcc', 'mcc', 'hcc', 'r']

def xr_unique(ds):
    index = np.sort(np.unique(ds.time, return_index=True)[1])
    return ds.isel(time=index)

class NWPLoader:
    """
    Loader for UKV numerical weather prediction data. Used for easy and 
    efficient loading of NWP patches in time and space.
    
    By default this streams data from the OCF google cloud bucket. These
    NWPs are arranged in forecasts which are 3 hours apart at times 00:00, 
    03:00, 06:00, 09:00 etc every day. Each forecast covers hourly steps from 
    0 to 36 hours ahead of time it starts.
    
    You can see the full lazy loaded dataset by exploring the .dataset attribute
    

    Attributes
    ----------
    store : store (MutableMapping or str), optional
        A MutableMapping where a Zarr Group has been stored or a path to a 
        directory in file system where a Zarr DirectoryStore has been stored.
        Defaults to the standard NWP zarr in the OFC GC bucket.
    width : int, optional
        Width (east-west extent) in metres of the data patch to load. Defaults
        to 22km which gives approximately a 160 degree view on sky considering 
        average UK cloud altitude.
    height : int, optional
        Height (north-south extent) in metres of the data patch to load. See 
        width.
    time_slice : array_like (int), optional
        When selecting a time slice of data in get_rectangle(_array) methods,
        return times at these intger steps around requested time. Default [0,] 
        only returns NWP data at single given time. [-2, -1, 0, 1] would give 
        the NWP data for 2 hours before, 1 hour before, at the requested time 
        and 1 hour ahead. For all of these the most recent forecast which covers 
        these times is used. i.e. if the requested time was 06:00 then the 03:00 
        forecast would be used for steps -2 and -1, and the forecast at 06:00
        used for 0 and 1.
    channels : array_like (str), optional
        NWP channels to use when loading.
    preprocess_method : str or None, optional
        What method of preprocessing, if any, should be applied to the returned
        NWP data. Avaliable options are:
             - 'norm' : Normalise the data x -> (x-mean(x))/std(x)
             - 'minmax' :  Min-max scaling  x -> (x-min(x))/(max(x)-min(x))
             - 'log_norm' : Modified log x -> log(x-min(x)+1) and then normalise
             - 'log_minmax : Modified log and then min-max scaling
             - None
    """
    def __init__(self, 
                 store='all', 
                 width=22000,
                 height=22000,
                 time_slice=[0,],
                 channels=DEFAULT_CHANNELS,
                 preprocess_method='norm'):
        
        if preprocess_method not in [None, 'norm', 'minmax', 'log_norm', 'log_minmax']:
            raise ValueError('Selected preprocess_method not valid')
        if len(set(channels)-set(AVAILABLE_CHANNELS.index))!=0:
            raise ValueError('Selected channel list not available')
        if width%2000!=0 or height%2000!=0:
            raise ValueError("""
                Grid spacing is 2000m so height and width should 
                be multiple of this."""
            )
        if not np.all(np.array(time_slice)<36):
            raise ValueError("""
                Forecast only goes to 36 hours ahead. Max time slice must be 
                less than this"""
            )

        self.channels = channels
        drop_variables = set(AVAILABLE_CHANNELS.index) - set(channels)
        if store=='all':
            self.dataset = xr.concat([xr_unique(xr.open_zarr(store=s,  
                                    drop_variables=drop_variables,
                                    consolidated=True)) for s in NWP_STORES], dim='time')[channels]
        else:
            self.dataset = xr.open_zarr(store=store,  
                                    drop_variables=drop_variables,
                                    consolidated=True)[channels]
        # transform below gives same y-oritentation as sat
        self.dataset = self.dataset\
                        .transpose('y', 'x', 'time', 'step') \
                        .sortby('time').isel(y=slice(None, None, -1))
        
        self.width = width
        self.height = height
        self.time_slice = time_slice
        
        self.preprocess_method = preprocess_method
        if preprocess_method is not None:
            self._agg_stats = xr.open_zarr(store=NWP_AGG_STORE, 
                                           consolidated=True)[channels].load()
        self._cache = None
        self._cache_dates = [None, None]
            
    @property
    def sample_shape(self):
        """(channels, y, x, time)"""
        return (len(self.channels), self.height//2000, 
                self.width//2000, len(self.time_slice))
    
    def close(self):
        self.dataset.close()
        
    def __len__(self):
        return len(self.dataset.time)
        
    def get_rectangle(self, forecast_time, valid_time, centre_x, centre_y):
        """Get (preprocesed) section of data for the instantiated channels; 
        centred on the given spatial location; and starting at the given time.
        
        The returned rectangle patch will have spatial size defined by the 
        instantiated height and width. It will be a sequence in time of 
        NWP data with the sequence defined by the instantiated time_slice.
        
        Parameters
        ----------
        forecast_time : str, pandas.Timestamp, numpy.datetime64
            Conceptually this is the current time. We have all forecast start
            times up to including this time. The most recent forecast before or
            at this time is used.
        valid_time : str, pandas.Timestamp, numpy.datetime64
            Datetime of first NWP in sequence. i.e. when the forecast is valid 
            for.
        centre_x : float, int
            East-West coordinate of centre of image patch expressed in metres 
            east in OSGB36(EPSG:27700) system.
        centre_y : float, int
            North-South coordinate of centre of image patch expressed in metres 
            east in OSGB36(EPSG:27700) system.

        Returns
        -------
        xarray.Dataset 
            Containing one xarray.DataArray for each channel with dimensions 
            (y, x, step). `step` is defined as time since most recent forecast 
            before forecast_time. The variables here are the variables 
            instantiated in the channel parameter.
        """
        # select first forecast time before selected time
        # frecasts are 3-hourly
        valid_t0 = pd.Timestamp(valid_time).floor('60min').to_numpy()
        valid_times = np.array([valid_t0 + np.timedelta64(i, 'h') for i in self.time_slice])
        
        forecast_t0 = pd.Timestamp(forecast_time).floor('180min').to_numpy()
        forecast_times = np.array([forecast_t0 if forecast_t0<vt else pd.Timestamp(vt).floor('180min').to_numpy() for vt in valid_times])
        
        # check forecast in range of data
        if (valid_t0 >= forecast_t0+np.timedelta64(37, 'h')) or (valid_t0 < forecast_t0):
            raise ValueError(f'valid_time {valid_t0 - forecast_t0} ahead of forecast_time')
        
        # convert from km to m
        half_width = self.width / 2
        half_height = self.height / 2

        north = centre_y + half_height
        south = centre_y - half_height
        east = centre_x + half_width
        west = centre_x - half_width
        
        # cache to speed up loading on same datetime
        if [forecast_t0, valid_t0]!=self._cache_dates:
            datasets = []
            for ft in np.unique(forecast_times):
                steps = valid_times[forecast_times==ft]-ft
                ds = self.dataset.sel(time=ft, step=steps).drop_vars('time').load()
                ds['step'] = ft-forecast_t0+ds.step
                datasets.append(ds)
            self._cache = xr.concat(datasets, dim='step').assign_coords(time=forecast_t0)
            self._cache_dates = [forecast_t0, valid_t0]
            
        rectangle = self._preprocess(self._cache.sel(
                                        y=slice(south, north), 
                                        x=slice(west, east)
                                  ))
        return rectangle
    
    def get_rectangle_array(self, forecast_time, valid_time, centre_x, centre_y):
        """Get (preprocesed) array of NWP data for instantiated channels, and 
        given time and spatial location. See get_rectangle.
        
        Parameters
        ----------
        time : str, pandas.Timestamp, numpy.datetime64
            Datetime of first image in sequence.
        centre_x : float, int
            East-West coordinate of centre of image patch expressed in metres 
            east in OSGB36(EPSG:27700) system.
        centre_y : float, int
            North-South coordinate of centre of image patch expressed in metres 
            east in OSGB36(EPSG:27700) system.

        Returns
        -------
        np.ndarray 
            with dimension (variable, y, x, step).
        """     
        ds = self.get_rectangle(forecast_time, valid_time, centre_x, centre_y)
        return ds.to_array().values
    
    def _preprocess(self, x):
        if self.preprocess_method=='norm':
            x = ((x - self._agg_stats.sel(aggregate_statistic='mean')) / 
                     self._agg_stats.sel(aggregate_statistic='std')
                )
        elif self.preprocess_method=='minmax':
            x = ((x - self._agg_stats.sel(aggregate_statistic='min')) / 
                     (self._agg_stats.sel(aggregate_statistic='max') - 
                      self._agg_stats.sel(aggregate_statistic='min'))
                 )
        elif self.preprocess_method=='log_norm':
            x = np.log(x - self._agg_stats.sel(aggregate_statistic='min')+1)
            x = ((x - self._agg_stats.sel(aggregate_statistic='mean_log')) / 
                     self._agg_stats.sel(aggregate_statistic='std_log')
                )
        elif self.preprocess_method=='log_minmax':
            x = np.log(x - self._agg_stats.sel(aggregate_statistic='min')+1)
            # note that min_log is 0 using our log transform above
            x = x  / self._agg_stats.sel(aggregate_statistic='max_log')
        return x
    

if __name__=='__main__':
    nwp_loader = NWPLoader()
    nwp_loader.get_rectangle('2019-01-01 11:00', '2019-01-01 11:00', 0,0)