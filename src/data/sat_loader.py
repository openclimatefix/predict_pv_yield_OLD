## TO DO
# - look into pytorch Dataset class to improve loader
# - Do some precomputing to means and standard deviations on different variables
#     - incorporate the above into standard preprocessing

import xarray as xr
import pandas as pd
import numpy as np

import gcsfs

from torch.utils.data import Dataset

from . constants import GCP_FS

SATELLITE_ZARR_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/zarr'
SATELLITE_STORE = gcsfs.mapping.GCSMap(SATELLITE_ZARR_PATH, gcs=GCP_FS, 
                                       check=True, create=False)
 
AVAILABLE_CHANNELS = ['HRV', 'IR_016', 'IR_039','IR_087', 'IR_097', 'IR_108', 
                     'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

class SatelliteLoader(Dataset):
    """

    """
    # These values for July only data
    HRVMAX = 103.32494
    MRVMEDIAN = 12.876617
    HRVMIN = -0.2079
    HRVMEAN = 14.23
    HRVSTD = 12.876

    
    def __init__(self, 
                 store=SATELLITE_STORE, 
                 width=22000,
                 height=22000,
                 channels=AVAILABLE_CHANNELS):

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

        rectangle = self.process(self.dataset.sel(time=time, 
                                           x=slice(west, east), 
                                           y=slice(south, north)))
        return rectangle
    
    def get_rectangle_array(self, time, centre_x, centre_y):
        """Variables are placed in zeroth dimension"""
        ds = self.get_rectangle(time, centre_x, centre_y)
        return ds.stacked_eumetsat_data.values
    
    def process(self, x):
        return x
    

if __name__=='__main__':
    sat_loader = SatelliteLoader()
    sat_loader.get_rectangle('2019-01-01 10:59', 0, 0)