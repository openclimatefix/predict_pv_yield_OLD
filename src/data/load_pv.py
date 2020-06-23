import xarray as xr
import pandas as pd
import numpy as np

import rasterio.warp as rasteriowarp

import os

LOCAL_DATA_DIRECTORY = os.path.expanduser('~/OCFdata')
PV_DATA_FILEPATH = 'PV/PVOutput.org/UK_PV_timeseries_batch.nc'
PV_METADATA_FILEPATH = 'PV/PVOutput.org/UK_PV_metadata.csv'

# Coordinate system of the SEVIRI OSGB36 reprojected & UKV data
DST_CRS = 'EPSG:27700'
WEST=-239_000
SOUTH=-185_000
EAST=857_000
NORTH=1223_000

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

if __name__=='__main__':
    pv_output = load_pv_power("~/repos/predict_pv_yield/data/"+PV_DATA_FILEPATH, 
                              start='2018-01-01', 
                              end='2019-12-31')
    pv_metadata = load_pv_metadata("~/repos/predict_pv_yield/data/"+PV_METADATA_FILEPATH)
    





