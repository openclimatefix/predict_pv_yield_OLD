import os
from glob import glob

import xarray as xr
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import rasterio.warp as rasteriowarp

# data locations
GCP_BUCKET = "solar-pv-nowcasting-data"
SATELLITE_DATA_PATH = 'satellite/EUMETSAT/SEVIRI_RSS/reprojected/just_UK'
PV_DATA_FILENAME = 'PV/PVOutput.org/UK_PV_timeseries_batch.nc'
PV_METADATA_FILENAME = 'PV/PVOutput.org/UK_PV_metadata.csv'
LOCAL_DATA_DIRECTORY = '/home/davidjamesfulton93/repos/predict_pv_yield/data'

# default parameters
# size of satellite patch in kilometers (ad also pixels as 1km/pix)
DEFAULT_RECTANGLE_WIDTH = DEFAULT_RECTANGLE_HEIGHT = 128 

SAT_IMAGE_MEAN = 20.444992
SAT_IMAGE_STD = 8.766013

DST_CRS = {
    'ellps': 'WGS84',
    'proj': 'tmerc',  # Transverse Mercator
    'units': 'm'  # meters
}

# Geospatial boundary in Transverse Mercator projection (meters)
SOUTH = 5513500
NORTH = 6613500
WEST =  -889500
EAST =   410500


def load_pv_metadata(filepath=None):
    if filepath is None:
        filepath = os.path.join(LOCAL_DATA_DIRECTORY, PV_METADATA_FILENAME)
        
    pv_metadata = pd.read_csv(filepath, index_col='system_id')
    pv_metadata.dropna(subset=['longitude', 'latitude'], how='any', inplace=True)

    # Convert lat lons to Transverse Mercator
    pv_metadata['x'], pv_metadata['y'] = rasteriowarp.transform(
        src_crs={'init': 'EPSG:4326'},
        dst_crs=DST_CRS,
        xs=pv_metadata['longitude'].values,
        ys=pv_metadata['latitude'].values)

    # Filter 3 PV systems which apparently aren't in the UK!
    pv_metadata = pv_metadata[
        (pv_metadata.x >= WEST) &
        (pv_metadata.x <= EAST) &
        (pv_metadata.y <= NORTH) &
        (pv_metadata.y >= SOUTH)]
    
    pv_metadata = pv_metadata.reindex(sorted(pv_metadata.index), axis=0)

    return pv_metadata

def load_pv_power(filepath=None, start='2010-12-15', end='2019-08-20'):
    if filepath is None:
        filepath = os.path.join(LOCAL_DATA_DIRECTORY, PV_DATA_FILENAME)
        
    pv_power_df = xr.open_dataset(filepath) \
                    .loc[dict(datetime=slice(start, end))] \
                    .to_dataframe() \
                    .dropna(axis='columns', how='all') \
                    .dropna(axis='rows', how='all')
    pv_power_df.columns = pv_power_df.columns.astype(np.int64)
    pv_power_df = pv_power_df.tz_localize('Europe/London').tz_convert('UTC')
    
    # Sort the columns
    pv_power_df = pv_power_df.reindex(sorted(pv_power_df.columns), axis=1)
    
    return pv_power_df


class SatelliteLoader(Dataset):
    """
    Attributes:
        paths: pd.Series which maps from UTC datetime to full filename of satellite data.
        _data_array_cache: The last lazily opened xr.DataArray that __getitem__ was asked to open.
            Useful so that we don't have to re-open the DataArray if we're asked to get
            data from the same file on several different calls.
    """
    def __init__(self, file_pattern=SATELLITE_FILE_PATTERN, 
                width=DEFAULT_RECTANGLE_WIDTH,
                height=DEFAULT_RECTANGLE_HEIGHT):
        self._load_sat_index(file_pattern)
        self._data_array_cache = None
        self._last_filename_requested = None
        self.width = width
        self.height = height
        
    def __getitem__(self, dt):
        sat_filename = self.paths[dt]
        if self._data_array_cache is None or sat_filename != self._last_filename_requested:
            self._data_array_cache = xr.open_dataarray(sat_filename).sel(time=dt)
            self._last_filename_requested = sat_filename
        return self._data_array_cache
    
    def close(self):
        if self._data_array_cache is not None:
            self._data_array_cache.close()
        
    def __len__(self):
        return len(self.index)
        
    def _load_sat_index(self, file_pattern):
        """Opens all satellite files in `file_pattern` and loads all their datetime indicies into self.index."""
        sat_filenames = glob(file_pattern)
        sat_filenames.sort()
        
        n_filenames = len(sat_filenames)
        sat_index = []
        for i_filename, sat_filename in enumerate(sat_filenames):
            if i_filename % 10 == 0 or i_filename == (n_filenames - 1):
                print('\r {:5d} of {:5d}'.format(i_filename + 1, n_filenames), end='', flush=True)
            data_array = xr.open_dataarray(sat_filename, drop_variables=['x', 'y'])
            sat_index.extend([(sat_filename, t) for t in data_array.time.values])

        sat_index = pd.DataFrame(sat_index, columns=['filename', 'datetime']).set_index('datetime').squeeze()
        self.paths = sat_index.tz_localize('UTC')
        self.index = self.paths.index
        
    def get_rectangles_for_all_data(self, centre_x, centre_y):
        """Iterate through all satellite filenames and load rectangle of imagery."""
        sat_filenames = np.sort(np.unique(self.paths.values))
        for sat_filename in sat_filenames:
            data_array = xr.open_dataarray(sat_filename)
            yield get_rectangle(data_array, time, centre_x, centre_y, self.width, self.height)
        
    def get_rectangle(self, time, centre_x, centre_y):
        data_array = self[time]
        # convert from km to m
        half_width = self.width * 1000 / 2
        half_height = self.height * 1000 / 2

        north = centre_y + half_height
        south = centre_y - half_height
        east = centre_x + half_width
        west = centre_x - half_width

        return data_array.sel(
            x=slice(west, east), 
            y=slice(north, south))


if __name__=='__main__':
    sat_loader = SatelliteLoader(os.path.join(SATELLITE_DATA_PATH, '*.nc'))


