## TO DO
# - look into pytorch Dataset class to improve loader

import xarray as xr
import pandas as pd
import numpy as np

import gcsfs

from torch.utils.data import Dataset

from . constants import GCP_FS

SATELLITE_ZARR_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/zarr'
SATELLITE_AGG_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/aggregate'

SATELLITE_STORE = gcsfs.mapping.GCSMap(SATELLITE_ZARR_PATH, gcs=GCP_FS, 
                                       check=True, create=False)
SATELLITE_AGG_STORE = gcsfs.mapping.GCSMap(SATELLITE_AGG_PATH, gcs=GCP_FS, 
                                       check=True, create=False)
 
AVAILABLE_CHANNELS = ['HRV', 'IR_016', 'IR_039','IR_087', 'IR_097', 'IR_108', 
                     'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

class SatelliteLoader(Dataset):
    """

    """
    
    def __init__(self, 
                 store=SATELLITE_STORE, 
                 width=22000,
                 height=22000,
                 channels=AVAILABLE_CHANNELS,
                 preprocess_method='norm'):
        
        if preprocess_method not in [None, 'norm', 'minmax', 'log_norm', 'log_minmax']:
            raise ValueError('Selected preprocess_method not valid')
        if len(set(channels)-set(AVAILABLE_CHANNELS))!=0:
            raise ValueError('Selected channel list not available')
        if width%2000!=0 or height%2000!=0:
            raise ValueError("Grid spacing is 2000m so height and width should be multiple of this")
        
        self.channels = channels
        self.dataset = xr.open_zarr(store=store, consolidated=True) \
                         .sel(variable=channels, y=slice(None, None, -1)) \
                         .transpose('variable', 'time', 'y', 'x')
        # Note that above the reversing of the y-coord and transposing need to
        # be reoved once sat_netcdf_to_zarr.py is run again. It was modified to
        # do these before saving.
        self.datset = self.dataset.sortby('time')
        self.width = width
        self.height = height
        self.preprocess_method = preprocess_method
        if preprocess_method is not None:
            self._agg_stats = xr.open_zarr(store=SATELLITE_AGG_STORE, 
                                consolidated=True).sel(variable=channels).load()
            
        self._cache = None
        self._cache_date = None
            
    @property
    def sample_shape(self):
        """(channels, y, x, time)"""
        return (len(self.channels), self.height//2000, self.width//2000, 1)
    
    def close(self):
        self.dataset.close()
        
    def __len__(self):
        return len(self.dataset.time)
        
    def get_rectangle(self, time, centre_x, centre_y):
        
        time = np.datetime64(time)
        # convert from km to m
        half_width = self.width / 2
        half_height = self.height / 2

        north = centre_y + half_height
        south = centre_y - half_height
        east = centre_x + half_width
        west = centre_x - half_width

        # cache to speed up loading same datetime
        if time!=self._cache_date:
            self._cache = self.dataset.sel(time=time).load()
            self._cache_date = time

        rectangle = self._cache.sel(y=slice(south, north), 
                                    x=slice(west, east))
        rectangle = self.preprocess(rectangle)
        return rectangle
    
    def get_rectangle_array(self, time, centre_x, centre_y):
        """Variables are placed in zeroth dimension"""
        ds = self.get_rectangle(time, centre_x, centre_y)
        return ds.stacked_eumetsat_data.values
    
    def preprocess(self, x):
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
    sat_loader = SatelliteLoader()
    sat_loader.get_rectangle('2019-01-01 10:59', 0, 0)