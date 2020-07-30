## TO DO
# - look into pytorch Dataset class to improve loader

import xarray as xr
import pandas as pd
import numpy as np

import gcsfs
from . constants import GCP_FS

SATELLITE_ZARR_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr'
SATELLITE_AGG_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/aggregate'

SATELLITE_STORE = gcsfs.mapping.GCSMap(SATELLITE_ZARR_PATH, gcs=GCP_FS, 
                                       check=True, create=False)
SATELLITE_AGG_STORE = gcsfs.mapping.GCSMap(SATELLITE_AGG_PATH, gcs=GCP_FS, 
                                       check=True, create=False)
 
# Description copied from http://eumetrain.org/data/2/204/204.pdf
AVAILABLE_CHANNELS = pd.DataFrame({
    'channel_name':[
        'VIS006',
        'VIS008',
        'IR_016',
        'IR_039',
        'WV_062',
        'WV_073',
        'IR_087',
        'IR_097',
        'IR_108',
        'IR_120',
        'IR_134',
        'HRV'],
    'description':[
        'λ_central=0.635µm, λ_min=0.56µm, λ_max=0.71µm, | Main observational purposes : Surface, clouds, wind fields',
        'λ_central=0.81µm, λ_min=0.74µm, λ_max=0.88µm, | Main observational purposes : Surface, clouds, wind fields',
        'λ_central=1.64µm, λ_min=1.50µm, λ_max=1.78µm, | Main observational purposes : Surface, cloud phase',
        'λ_central=3.90µm, λ_min=3.48µm, λ_max=4.36µm, | Main observational purposes : Surface, clouds, wind fields',
        'λ_central=6.25µm, λ_min=5.35µm, λ_max=7.15µm, | Main observational purposes : Water vapor, high level clouds, upper air analysis',
        'λ_central=7.35µm, λ_min=6.85µm, λ_max=7.85µm, | Main observational purposes : Water vapor, atmospheric instability, upper-level dynamics',
        'λ_central=8.70µm, λ_min=8.30µm, λ_max=9.1µm, | Main observational purposes : Surface, clouds, atmospheric instability',
        'λ_central=9.66µm, λ_min=9.38µm, λ_max=9.94µm, | Main observational purposes : Ozone',
        'λ_central=10.80µm, λ_min=9.80µm, λ_max=11.80µm, | Main observational purposes : Surface, clouds, wind fields, atmospheric instability',
        'λ_central=12.00µm, λ_min=11.00µm, λ_max=13.00µm, | Main observational purposes : Surface, clouds, atmospheric instability',
        'λ_central=13.40µm, λ_min=12.40µm, λ_max=14.40µm, | Main observational purposes : Cirrus cloud height, atmospheric instability',
        'Broadband (about 0.4 – 1.1 µm) | Main observational purposes : Surface, clouds']
    }).set_index('channel_name', drop=True)

DEFAULT_CHANNELS = list(AVAILABLE_CHANNELS.index)

class SatelliteLoader:
    """
    Loader for SEVIRI satellite data. Used for easy and efficient loading of
    satellite patches in time and space.
    
    By default this streams data from the OCF google cloud bucket. These
    satellite images are at 5 minute intervals at times (N*5-1) minutes. e.g.
    2019-01-01 12:04, 2019-01-01 12:09, 2019-01-01 12:14 etc.
    
    You can see the full lazy loaded dataset by exploring the .dataset attribute

    Attributes
    ----------
    store : store (MutableMapping or str), optional
        A MutableMapping where a Zarr Group has been stored or a path to a 
        directory in file system where a Zarr DirectoryStore has been stored.
        Defaults to the standard satellite zarr in the OFC GC bucket.
    width : int, optional
        Width (east-west extent) in metres of the data patch to load. Defaults
        to 22km which gives approximately a 160 degree view on sky considering 
        average UK cloud altitude.
    height : int, optional
        Height (north-south extent) in metres of the data patch to load. See 
        width.
    time_slice : array_like (int), optional
        When selecting a time slice of data in get_rectangle(_array) methods,
        return times at these intger steps before requested time. Default [0,] 
        only returns satellite data at single given time. [-2, -1, 0] would give 
        the satellite data 10 minutes before, 5 minutes before and at the time 
        of the requested time.
    channels : array_like (str), optional
        Satellite channels to use when loading.
    preprocess_method : str or None, optional
        What method of preprocessing, if any, should be applied to the returned
        satellite data. Avaliable options are:
             - 'norm' : Normalise the data x -> (x-mean(x))/std(x)
             - 'minmax' :  Min-max scaling  x -> (x-min(x))/(max(x)-min(x))
             - 'log_norm' : Modified log x -> log(x-min(x)+1) and then normalise
             - 'log_minmax : Modified log and then min-max scaling
             - None
    """
    def __init__(self, 
                 store=SATELLITE_STORE, 
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
        if not np.all(np.array(time_slice)<=0):
            raise ValueError("""
                Time slice indices must not be negative. This reflects the idea
                that the `time` requested in get_rectangle(_array) will be the 
                moment at which predictions are being made for current or future 
                power. Therefoore no future (positive) data is available."""
            )
        
        self.channels = channels
        # transforms below are needed to have same dimension order as NWPs
        self.dataset = xr.open_zarr(store=store, consolidated=True) \
                         .sel(variable=channels, y=slice(None, None, -1)) \
                         .transpose('variable', 'y', 'x', 'time') \
                         .sortby('time')
        self.width = width
        self.height = height
        self.time_slice = time_slice
        self.preprocess_method = preprocess_method
        if preprocess_method is not None:
            self._agg_stats = xr.open_zarr(store=SATELLITE_AGG_STORE, 
                                consolidated=True).sel(variable=channels).load()
            
        self._cache = None
        self._cache_date = None
            
    @property
    def sample_shape(self):
        """(channels, y, x, time)"""
        return (len(self.channels), self.height//2000, 
                self.width//2000, len(self.time_slice))
    
    def close(self):
        self.dataset.close()
        
    def __len__(self):
        return len(self.dataset.time)
        
    def get_rectangle(self, time, centre_x, centre_y):
        """Get (preprocesed) section of data for the instantiated channels; 
        centred on the given spatial location; and starting at the given time.
        
        The returned rectangle patch will have spatial size defined by the 
        instantiated height and width. It will be a sequence in time of 
        satellite images with the sequence defined by the instantiated 
        time_slice.
        
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
        xarray.Dataset 
            Containing only one xarray.DataArray named stacked_eumetsat_data 
            with dimension (variable, y, x, time). The variables here are the
            variables instantiated in the channel parameter.
        """
        t0 = np.datetime64(time)
        times = np.array([t0 + np.timedelta64(5*i, 'm') for i in self.time_slice])
        
        # convert from km to m
        half_width = self.width / 2
        half_height = self.height / 2

        north = centre_y + half_height
        south = centre_y - half_height
        east = centre_x + half_width
        west = centre_x - half_width

        # Caching will speed up next load if it is from the same datetime
        if t0!=self._cache_date:
            self._cache = self.dataset.sel(time=times).load()
            self._cache_date = t0

        rectangle = self._cache.sel(y=slice(south, north), 
                                    x=slice(west, east))
        rectangle = self._preprocess(rectangle)
        return rectangle
    
    def get_rectangle_array(self, time, centre_x, centre_y):
        """Get (preprocesed) array of satellite data for instantiated channels,
        and given time and spatial location. See get_rectangle.
        
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
            with dimension (variable, y, x, time).
        """        
        ds = self.get_rectangle(time, centre_x, centre_y)
        return ds.stacked_eumetsat_data.values
    
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
    sat_loader = SatelliteLoader(time_slice=[-2, -1])
    sat_loader.get_rectangle('2019-01-01 10:59', 0, 0)