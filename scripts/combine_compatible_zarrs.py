"""This script combines multiple different zarr stores with some very strong 
assumptions.

Note that this was developed to work alongside `nwp_grib_to_zarr.py` so that
that script could be run in parallel and uploaded to different GCS zarr files.
This is then meant to combine the resulting zarr stores.

I therefore assume that the encodings are the same; that there is chunking only
in time; that the other coordinates are identical; that the zeroth coordinate 
for all dataarrays is time. We also assume that time chunk size is 1. ie one
datetim per chunk.

This script works on google compue engine with the exact environment of the
`research-instance2` compute instance. I cannot guarentee it will work elsewhere.

At the time of writing the xarray support of zarr is quite new and may be 
subject to change. This might break this script.
"""

import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import zarr
import subprocess
import os
import gcsfs
import json
import glob
import shutil

################################################################################
# user defined
################################################################################

# project
fs = gcsfs.GCSFileSystem(project='solar-pv-nowcasting', token=None)

# source and destination zarrs
gcscombinetest_root = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/zarr_combine_test'

zarrfiles = [f"{gcscombinetest_root}/{f}" for f in ['zarr1', 'zarr2', 'zarr3']]
masterzarrfile = f"{gcscombinetest_root}/combined_zarr"

# temporary-save local folder
stagingdir = os.path.expanduser("~/staging")

# do you want to 'move' the gcs files or 'copy' them?
mode = 'copy'

################################################################################
# derived
################################################################################

assert mode in ['move', 'copy'], "mode must be 'move' or 'copy'"

mvcp = 'mv' if mode=='move' else 'cp'

zarrmaps = [gcsfs.mapping.GCSMap(f, gcs=fs, check=False, create=True) for f in zarrfiles]

# here we create some test data to check this sample script
gcsloadpath = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr'
gcsloadmap = gcsfs.mapping.GCSMap(gcsloadpath, gcs=fs, check=True, create=False)
ds = xr.open_zarr(store=gcsloadmap)
ds = ds[['t', 'lcc']]
ds.isel(x=slice(0,5), y=slice(0, 5), time=slice(5, 10)).to_zarr(store=zarrmaps[0], consolidated=True, mode='w-')
ds.isel(x=slice(0,5), y=slice(0, 5), time=slice(10, 15)).to_zarr(store=zarrmaps[1], consolidated=True, mode='w')
ds.isel(x=slice(0,5), y=slice(0, 5), time=slice(0, 5)).to_zarr(store=zarrmaps[2], consolidated=True, mode='w')
del ds, gcsloadpath, gcsloadmap

# open the zarrs to get time info
all_ds = [xr.open_zarr(store=z) for z in zarrmaps]
combined_ds = xr.merge(all_ds).sortby('time')
time_shape = combined_ds.time.shape[0]

################################################################################
# function library
################################################################################

def decode_command(command_string, split=True):
    command_list = command_string.split(' ')
    out = subprocess.Popen(command_list,
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    result = stdout.decode()
    if split:
        result = result.split('\n')
        result = [r for r in result if r!='']
    return result

def get_gsutil_json(gspath):
    return json.loads(decode_command(f"gsutil cat {gspath}", split=False))

def dump_gsutil_json(data, gspath):
    json_dump = os.path.join(stagingdir, 'json_dump')
    with open(json_dump, 'w') as f:
        json.dump(data, f, indent=4)
    decode_command(f"gsutil cp {json_dump} {gspath}", split=False)
    os.remove(json_dump)

def set_up_empty_zarr():
    
    # copy top level attributes
    os.system(f"gsutil cp gs://{zarrfiles[0]}/.z* gs://{masterzarrfile}/")
    
    # metadata needs edited
    zmeta = get_gsutil_json(f"gs://{zarrfiles[0]}/.zmetadata")
    keys = set([k.replace('/.zattrs', '').replace('/.zarray', '') 
                for k in zmeta['metadata'].keys()])-set(['.zattrs', '.zgroup'])
    for k in keys:
        if 'time' in zmeta['metadata'][f"{k}/.zattrs"]["_ARRAY_DIMENSIONS"]:
            zmeta['metadata'][f"{k}/.zarray"]['shape'][0] = time_shape
    dump_gsutil_json(zmeta, f"gs://{masterzarrfile}/.zmetadata")
    
    subdirs = [k.split('/')[-2] for k in 
               decode_command(f"gsutil ls -d gs://{zarrfiles[0]}/*") 
               if k.endswith('/') and k!=f"gs://{zarrfiles[0]}/"]
    
    # copy and update data level atributes
    for sd in subdirs:
        inpath = f"gs://{zarrfiles[0]}/{sd}"
        outpath = f"gs://{masterzarrfile}/{sd}"
        
        print(f"{inpath}")
        data = get_gsutil_json(f"{inpath}/.zarray")
        data['shape'][0] = time_shape
        dump_gsutil_json(data, f"{outpath}/.zarray")
        os.system(f"gsutil cp {inpath}/.zattrs {outpath}/.zattrs")
    
    coordzarr = os.path.join(stagingdir, 'coordszarr')
    combined_ds.coords.to_dataset().to_zarr(coordzarr, mode='w')
    os.system(f"gsutil cp -r {coordzarr}/* gs://{masterzarrfile}/")
    shutil.rmtree(coordzarr)
    
def match_move_timeindex(args):
    ti, time = args
    for i, ds in enumerate(all_ds):
        # find which zarr directory and time index the match is at
        if time in ds.time:
            file_ti = int(np.argwhere(ds.time.values.flatten()==time))
            for key in combined_ds.keys():
                paths = decode_command(f"gsutil ls gs://{zarrfiles[i]}/{key}/{file_ti}.*")
                for path in paths:
                    # new filename
                    file = '.'.join([str(ti)]+path.split('/')[-1].split('.')[1:])
                    print(f"{path} -> {masterzarrfile}/{key}/{file}")
                    os.system(f"nohup gsutil {mvcp} {path} gs://{masterzarrfile}/{key}/{file} &")
            return
        

def copy_files_to_new():
    from multiprocessing import Pool
    with Pool(4) as p:
        print(p.map(match_move_timeindex, enumerate(combined_ds.time.values)))

################################################################################
# do the work
################################################################################
                
if __name__=='__main__':
    import time
    
    t0 = time.time()
    
    print('Setting up zarr archive')
    set_up_empty_zarr()
    print(f"Setup time : {time.time()-t0}")
    print('Setup complete\n\n')
    
    t1 = time.time()
    print(f"{mode.strip('e')}ing files...\n")
    copy_files_to_new()
    print('Copied.')
    print(f"Populate time : {time.time()-t1} | full time : {time.time()-t0}")
    
    masterzarrmap = gcsfs.mapping.GCSMap(masterzarrfile, gcs=fs, create=True)
    dz = xr.open_zarr(store=masterzarrmap).load()
    
    # run check to verify all values in right place in combined zarr
    for ds in all_ds:
        x1 = ds.load()
        x2 = dz.sel(time=ds.time).load()
        print(x1.equals(x2))


