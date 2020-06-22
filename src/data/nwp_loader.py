## TO DO
# - Fix loader so variables at different pressure levels can be used
# - look into pytorch Dataset class to improve loader

import xarray as xr
import pandas as pd
import numpy as np

import gcsfs

from torch.utils.data import Dataset

from . constants import GCP_FS

NWP_ZARR_PATH = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/2019_1-6'
NWP_AGG_PATH = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/aggregate'

NWP_STORE = gcsfs.mapping.GCSMap(NWP_ZARR_PATH, gcs=GCP_FS, 
                                       check=True, create=False)
NWP_AGG_STORE = gcsfs.mapping.GCSMap(NWP_AGG_PATH, gcs=GCP_FS, 
                                       check=True, create=False)

_channels_meta_data = pd.DataFrame(
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
     'wholesale_file_number': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 
                               2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4],
     'index_after_loading': [0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 0, 1, 2, 3, 4, 5, 5, 
                             5, 5, 5, 6, 0, 0, 0, 0, 0, 0, 1],
     'wholesale_file_variable_code':[
        't', 'r', 'dpt', 'vis', 'si10', 'wdir10', 'prmsl', 'paramId_0', 
         'paramId_0', 'prate', 'paramId_0', 'cdcb', 'lcc', 'mcc', 'hcc', 
         'paramId_0', 'sde', 'hcct', 'dswrf', 'dlwrf', 'h', 'ws', 't', 'gh', 
         'r', 'wdir', 'gust', 'gust'
     ],
    }
).set_index('channel_name', drop=True)

AVAILABLE_CHANNELS = _channels_meta_data.loc[:, ['description']]
DEFAULT_CHANNELS = ['t', 'dswrf', 'lcc', 'mcc', 'hcc', 'r']


class NWPLoader(Dataset):
    """

    """
    
    def __init__(self, 
                 store=NWP_STORE, 
                 width=22000,
                 height=22000,
                 channels=DEFAULT_CHANNELS,
                 preprocess_method='norm'):
        
        if preprocess_method not in [None, 'norm', 'minmax', 'log_norm', 'log_minmax']:
            raise ValueError('Selected preprocess_method not valid')
        if len(set(channels)-set(AVAILABLE_CHANNELS.index))!=0:
            raise ValueError('Selected channel list not available')
        if width%2000!=0 or height%2000!=0:
            raise ValueError("Grid spacing is 2000m so height and width should be multiple of this")

        self.channels = channels
        drop_variables = set(_channels_meta_data.index) - set(channels)
        self.dataset = xr.open_zarr(store=store,  
                                    drop_variables=drop_variables,
                                    consolidated=True)[channels]
        self.datset = self.dataset.sortby('time')
        self.width = width
        self.height = height
        
        self.preprocess_method = preprocess_method
        if preprocess_method is not None:
            self._agg_stats = xr.open_zarr(store=NWP_AGG_STORE, 
                                           consolidated=True)[channels].load()
            
    @property
    def sample_shape(self):
        """(channels,y,x,(time,))"""
        return (len(self.channels), self.height//2000, self.width//2000, 1)
    
    def close(self):
        self.dataset.close()
        
    def __len__(self):
        return len(self.dataset.time)
        
    def get_rectangle(self, forecast_time, valid_time, centre_x, centre_y):
        # select first forecast time before selected time
        # frecasts are 3-hourly
        forecast_time = pd.Timestamp(forecast_time).floor('180min').to_pydatetime()
        valid_time = pd.Timestamp(valid_time).floor('60min').to_pydatetime()
        # check forecast in range of data
        step = valid_time - forecast_time
        if (step > pd.Timedelta('37h')) or (step < pd.Timedelta('0h')):
            raise ValueError(f'valid_time {step} ahead of forecast_time')
        
        # convert from km to m
        half_width = self.width / 2
        half_height = self.height / 2

        north = centre_y + half_height
        south = centre_y - half_height
        east = centre_x + half_width
        west = centre_x - half_width

        rectangle = self.preprocess(
                        self.dataset.sel(time=forecast_time, 
                                         step=step,
                                         x=slice(west, east), 
                                         y=slice(south, north)))
        return rectangle
    
    def get_rectangle_array(self, forecast_time, valid_time, centre_x, centre_y):
        """Variables are placed in zeroth dimension"""
        ds = self.get_rectangle(forecast_time, valid_time, centre_x, centre_y)
        return ds.to_array().values
    
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
    nwp_loader = NWPLoader()
    nwp_loader.get_rectangle('2019-01-01 11:00', '2019-01-01 11:00', 0,0)