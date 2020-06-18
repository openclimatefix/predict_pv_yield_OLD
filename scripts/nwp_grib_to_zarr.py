"""Simple script to convert NWP data on GCS to zarr format with some reshaping
and relabelling for easier access"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import cfgrib
import zarr
import rasterio
from datetime import datetime
import gcsfs
import subprocess
import time

################################################################################
# user defined 
################################################################################

# project
fs = gcsfs.GCSFileSystem(project='solar-pv-nowcasting', token=None)

# bucket and save path
gcssavepath = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/test'

# filepattern to load from
gcsloadpaths = ['solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV/2019/07/01']

# create new zarr store or otherwise append
create_new = True

# Channels to move to zarr. Must be list of `channel_name` defined in dataframe
# below or can be set to 'all'
channels = 'all'

# temporary directory
temp_dir = os.path.expanduser('~/staging')
os.makedirs(temp_dir, exist_ok=True)

################################################################################
# derived and predefined
################################################################################

# What variables are available, what they are called in the grib files,
# where they are in the grib files, and what we rename them to.
channels_meta_data = pd.DataFrame(
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
     'wholesale_index_variable_name':[
        't', 'r', 'dpt', 'vis', 'si10', 'wdir10', 'prmsl', 'paramId_0', 
         'paramId_0', 'prate', 'paramId_0', 'cdcb', 'lcc', 'mcc', 'hcc', 
         'paramId_0', 'sde', 'hcct', 'dswrf', 'dlwrf', 'h', 'ws', 't', 'gh', 
         'r', 'wdir', 'gust', 'gust'
     ],
    }
).set_index('channel_name', drop=True)

if channels=='all':
    channels = list(channels_meta_data.index)

# wholesale file numbers we need to load
load_filenumbers = channels_meta_data.loc[channels].wholesale_file_number.unique()

# Bounds of the UKV grid
NORTH = 1223_000
SOUTH = -185_000
EAST = 857_000
WEST = -239_000
NUM_ROWS = 704 # north-south points
NUM_COLS =  548 # east-west points
n = max(NUM_COLS, NUM_ROWS)

# calculate pixel positions
DST_TRANSFORM = rasterio.transform.from_bounds(width=NUM_COLS, height=NUM_ROWS,
    west=WEST, south=SOUTH, east=EAST, north=NORTH)
xs, ys = rasterio.transform.xy( transform=DST_TRANSFORM, 
                              rows=np.arange(n), cols=np.arange(n))
ys = ys[:NUM_ROWS][::-1]
xs = xs[:NUM_COLS]

# find the directories to load from
load_directories = []
for p in gcsloadpaths:
    out = subprocess.Popen(["gsutil", "ls", "-d", f"gs://{p}"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    load_directories.extend([x for x in stdout.decode().split('\n') if x!=''])
load_directories = sorted(list(set(load_directories)))
print('loading from : ', load_directories)

################################################################################
# function library
################################################################################
def adhoc_merge(datasets, filepattern):
    """File structures are heterogenious across time. 
    This function is to try to fix some of that to 2018-07 which is
    the file structure I was working with first
    """
    if 1 in datasets.keys():
        # identify change by cfgrib extra splitting behaviour
        # before file strucuture change [t, r, dpt, vis] are all at index 0
        if list(datasets[1][0].keys())==['t'] and list(datasets[1][1].keys())==['r', 'dpt', 'vis']:
            datasets[1][0] = cfgrib.open_dataset(filepattern.format(1), backend_kwargs=dict(filter_by_keys={'level': 1}))
            for i in range(1, len(datasets[1])-1):
                datasets[1][i] = datasets[1][i+1]
    return datasets

def load_wholesale_gribs_to_xarray(filepattern, channels):
        
    #  load in all the wholesale filenumbers needed
    cfgrib.dataset.LOG.disabled = True
    datasets = {i:cfgrib.open_datasets(filepattern.format(i)) for i in load_filenumbers}
    datasets = adhoc_merge(datasets, filepattern)
    cfgrib.dataset.LOG.disabled = False
        
    # rename channels from grib names to unique short names
    # also select t the channels we need
    dataset_list = []
        
    # for all the files we want variables from
    for whsl_num in load_filenumbers:
        file_channels_meta = channels_meta_data.loc[channels]\
                                    .query("wholesale_file_number==@whsl_num")
        # for all the chunks of this files we want variables from
        for i in file_channels_meta.index_after_loading.unique():
            # rename the variables to defined non-clashing names
            df_ = file_channels_meta.query("index_after_loading==@i")
            rename = {k:v for k,v in zip(df_.wholesale_index_variable_name, df_.index)}
            datasets[whsl_num][i] = datasets[whsl_num][i].rename(rename)[list(df_.index)]
            # gather the needed parts of the dataset
            dataset_list.append(datasets[whsl_num][i])
        
    # do the merge to a single dataset
    merged_datasets = xr.merge(dataset_list, compat='override')
    return merged_datasets


def reshape_data(ds):   
    # The UKV data is of shape <num_time_steps, num_values> and we want
    # it in shape <num_time_steps, <num_rows>, <num_columns>
    index = pd.MultiIndex.from_arrays(
        [x.flatten() for x in np.meshgrid(xs, ys)], 
        names=['x', 'y', ])
    ds['values'] = index
    reshaped = ds.unstack('values').expand_dims('time')
    all_dims = ['time', 'step', 'y', 'x', 'isobaricInhPa']
    sorted_dims = [d for d in all_dims if d in reshaped.dims]
    return reshaped.transpose(*sorted_dims)

def drop_unused_coords(ds):
    return ds.reset_coords([c for c in ds.coords if c not in ds.dims], drop=True)


def compress_and_save_to_zarr(ds, mode='a'):
    # Chunk the array thinking about how we access spatial slices
    # Here we have assumed we don't use forecast time slices or many step slices
    chunk_dict = {'time': 1, 'step': 10,  'y':-1, 'x':-1}
    if 'isobaricInhPa'in ds.dims:
        chunk_dict['isobaricInhPa'] = 5
    ds = ds.chunk(chunk_dict)
    
    if mode=='w': # if this is the first addition to the zarr file use this 
        # This encoding good compression and was as fast to load as any other 
        # set  level. It also took a reasonably short amount of time to encode 
        # compared to level 9 for only a couple of percent more stored data.
        encoding = { var_name: {
                'filters': [zarr.Delta(dtype='float32')],
                'compressor': zarr.Blosc(cname='zstd', 
                                         clevel=4, 
                                         shuffle=zarr.Blosc.AUTOSHUFFLE)}
                    for var_name in ds.keys()}
        gcsmap = gcsfs.mapping.GCSMap(gcssavepath, gcs=fs, check=False, create=True)
        ds.to_zarr(store=gcsmap, consolidated=True, encoding=encoding) 
    
    elif mode=='a': # if we are appending to an existing zarr file use this
        gcsmap = gcsfs.mapping.GCSMap(gcssavepath, gcs=fs, check=True, create=False)
        ds.to_zarr(store=gcsmap, append_dim='time', consolidated=True) 
    else:
        raise ValueError

################################################################################
# run
################################################################################

if __name__=='__main__':
    t0 = time.time()
    mode = 'w' if create_new else 'a'
    for directory in load_directories:
        year, month, day = np.array(directory.split("/")[-4:-1], dtype=int)
        
        for hour in [0,3,6,9,12,15,18,21]:
            # define forecast path root for datetime
            filename_root = f"{year}{month:02}{day:02}{hour:02}" \
                                + "00_u1096_ng_umqv_Wholesale{}.grib"
            gs_forecast_rootpath = directory + filename_root

            # check if all required wholesale files are available
            out = subprocess.Popen(["gsutil", "ls", gs_forecast_rootpath.format('[0-9]')], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT)
            stdout, stderr = out.communicate()
            available_wholesale_files = set(stdout.decode().split('\n')[:-1])
            needed_wholesale_files = {gs_forecast_rootpath.format(n) for n in load_filenumbers}
            if len(needed_wholesale_files - available_wholesale_files)>0:
                print("All wholesale files needed are not available for "\
                      + f"{year}-{month:02}-{day:02} {hour:02}:00")
            else:
                # download the data
                dl_command = f"gsutil -m cp {gs_forecast_rootpath.format('[1-4]')} {temp_dir}/."
                os.system(dl_command)
                # process and upload to zarr
                local_forecast_root = f"{temp_dir}/{filename_root}"
                ds = load_wholesale_gribs_to_xarray(local_forecast_root, channels)
                ds = reshape_data(ds)
                ds = drop_unused_coords(ds)
                compress_and_save_to_zarr(ds, mode=mode)
                # clear up and remove files
                os.system(f"rm {temp_dir}/*")
                
                # change mode to append after first file uploaded
                mode='a'
    print(time.time()-t0)
