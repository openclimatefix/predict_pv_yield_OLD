"""
Loader for UKV numerical weather forecasts

When loaded using cfgrib the following variables are available at each index 
from each file
    # wholesale 1
    # index 0 - [heightAboveGround = 1]
    't' # 1.5m air temperature at surface
    'r' # 1.5m Relative humidity
    'dpt' # 1.5m dew point temperature
    'vis' # 1.5m visibility
    
    # index 1 - [heightAboveGround = 10]
    'si10' # 10m wind speed
    'wdir10' # 10m wind direction
    
    # index 2 - [meanSea = 0]
    'prmsl' # mean sea level pressure
    
    # index 3 - [surface = 0]
    'paramId_0' # original GRIB paramId: 0 # Uncertain. This is likely "1.5m 
                                           # fog probability"
        
    # index 4 - [surface = 0]
    'paramId_0' # original GRIB paramId: 0 # Uncertain. This is likely "snow 
                                           # fraction"
    'prate' # total precipitation rate 
    
    #---------------------------------------
    # wholesale 2
    # index 0 - [atmosphere = 0]
    'paramId_0' # original GRIB paramId: 0 # Uncertain. This might be 
                                           # "cloud fraction below 1000ft ASL"
    
    # index 1 - [cloudBase = 0]
    'cdcb' # Cloud base
    
    # index 2 - [heightAboveGroundLayer = 0]
    'lcc' # Low cloud cover
    
    # index 3 - [heightAboveGroundLayer = 1524]
    'mcc' # Medium cloud cover
        
    # index 4 - [heightAboveGroundLayer = 4572]
    'hcc' # High cloud cover
    
    # index 5 [surface = 0]
    'paramId_0' # original GRIB paramId: 0 # Uncertain. This might be "wet/dry 
                                           # bulb freezing level height"
    'sde' # Snow depth
    'hcct' # Height of convective cloud top
    'dswrf' # Downward short-wave radiation flux
    'dlwrf' # Downward long-wave radiation flux
    
    # index 6 - [level = 0]
    'h' # Geometrical height
    
    #---------------------------------------
    # wholesale 3
    # index 0 - [isobaricInhPa (15,) = 1000 925 850 700 600 ... 100 70 50 30]
    'ws' # Wind speed
    't' # Temperature
    'gh' # Geopotential Height
    'r' # Relative humidity
    'wdir' # Wind direction
    
    #---------------------------------------
    # wholesale 4
    # index 0 - [heightAboveGround = 10]
    'gust' # 10m wind gust

    # index 1 - [heightAboveGround = 10]
    'gust' # 10m maximum wind gust in hour (T+1 to T+36)

"""

## TO DO
# - Fix loader so variables at different pressure levels can be used
# - look into pytorch Dataset class to improve loader
# - Do some precomputing to means and standard deviations on different variables
#     - incorporate the above into standard preprocessing

import xarray as xr
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from . constants import GCP_FS

NWP_ZARR_PATH = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/2019_1-6'
NWP_STORE = gcsfs.mapping.GCSMap(NWP_ZARR_PATH, gcs=GCP_FS, 
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
                 lazy_load=True):

        self.channels = channels
        drop_variables = set(_channels_meta_data.index) - set(channels)
        self.dataset = xr.open_zarr(store=store,  
                                    drop_variables=drop_variables,
                                    consolidated=True)[channels]
        self.datset = self.dataset.sortby('time')
        self.width = width
        self.height = height
    
    def close(self):
        self.dataset.close()
        
    def __len__(self):
        return len(self.dataset.time)
        
    def get_rectangle(self, forecast_time, valid_time, centre_x, centre_y):
        # select first forecast time before selected time
        # frecasts are 3-hourly
        forecast_time = pd.Timestamp(forecast_time).floor('180min').to_pydatetime()
        
        # check forecast in range of data
        forecast_dt = np.datetime64(t)
        valid_dt = np.datetime64(valid_time)
        step = valid_dt - forecast_dt
        if (step > np.timedelta64(36, 'h')) or (step < np.timedelta64(0, 'h')):
            raise ValueError(f'valid_time {step} ahead of forecast_time')
        
        # convert from km to m
        half_width = self.width / 2
        half_height = self.height / 2

        north = centre_y + half_height
        south = centre_y - half_height
        east = centre_x + half_width
        west = centre_x - half_width

        rectangle = self.process(
                        self.dataset.sel(time=forecast_time, 
                                         step=step,
                                         x=slice(west, east), 
                                         y=slice(south, north)))
        return rectangle
    
    def get_rectangle_array(self, forecast_time, valid_time, centre_x, centre_y):
        """Variables are placed in zeroth dimension"""
        ds = self.get_rectangle(forecast_time, valid_time, centre_x, centre_y)
        return ds.to_array().values
    
    def process(self, x):
        return x
    

if __name__=='__main__':
    nwp_loader = NWPLoader()
    nwp_loader.get_rectangle('2019-01-01 11:00', '2019-01-01 11:00', 0,0)