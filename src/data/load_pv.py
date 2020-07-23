import xarray as xr
import pandas as pd
import numpy as np

import rasterio.warp as rasteriowarp

import os
from pathlib import Path

# Coordinate system of the SEVIRI OSGB36 reprojected & UKV data
from . constants import DST_CRS, NORTH, SOUTH, EAST, WEST

# Path of this submodule 
submodule_path = Path(os.path.dirname(os.path.abspath(__file__)))
# Path where setup downloads local data to
LOCAL_DATA_DIRECTORY = os.path.join(submodule_path.parent.parent, 'data')


PV_DATA_FILEPATH = 'PV/PVOutput.org/UK_PV_timeseries_batch.nc'
PV_METADATA_FILEPATH = 'PV/PVOutput.org/UK_PV_metadata.csv'


def load_pv_metadata(filepath=None):
    """Loads the PV metadata from local storage and adds columns 'x' and  'y' 
    for the system location in the coordinate system used by the NWP and 
    satellite data.
    
    Parameters
    ----------
    filepath : str, optional
        Location of the PV metadata on the local system. Defaults to file which
        should be downloaded during setup.
    
    Returns
    -------
    pandas.DataFrame of dimension (system_id, metadata_features)
    """
    if filepath is None:
        filepath = os.path.join(LOCAL_DATA_DIRECTORY, PV_METADATA_FILEPATH)
        
    pv_metadata = pd.read_csv(filepath, index_col='system_id')
    
    # drop systems without location information
    pv_metadata.dropna(subset=['longitude', 'latitude'], how='any', inplace=True)

    # Convert lat lons to Transverse Mercator
    pv_metadata['x'], pv_metadata['y'] = rasteriowarp.transform(
        src_crs={'init': 'EPSG:4326'},
        dst_crs=DST_CRS,
        xs=pv_metadata['longitude'].values,
        ys=pv_metadata['latitude'].values)

    # Filter PV systems which apparently aren't in the UK
    pv_metadata = pv_metadata[
        (pv_metadata.x >= WEST) &
        (pv_metadata.x <= EAST) &
        (pv_metadata.y <= NORTH) &
        (pv_metadata.y >= SOUTH)]
    
    pv_metadata = pv_metadata.reindex(sorted(pv_metadata.index), axis=0)

    return pv_metadata

def load_pv_power(filepath=None, start='2010-12-15', end='2019-08-20'):
    """Loads the PV power output data from local netcdf storage for a given time
    period.
    
    Parameters
    ----------
    filepath : str, optional
        Location of the PV power data on the local system. Defaults to file 
        which should be downloaded during setup.
    start : str in format 'YYYY-MM-DD', optional
        First date to load the PV data from. Defaults to first date available.
    end : str in format 'YYYY-MM-DD', optional
        Last date to load the PV data from. Defaults to last date available.
    
    Returns
    -------
    pandas.DataFrame of dimension (time, system_id)
    """
    if filepath is None:
        filepath = os.path.join(LOCAL_DATA_DIRECTORY, PV_DATA_FILEPATH)
        
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

if __name__=='__main__':
    pv_output = load_pv_power(start='2018-01-01', end='2019-12-31')
    pv_metadata = load_pv_metadata()