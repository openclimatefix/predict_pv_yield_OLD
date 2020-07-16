# TO DO
# many fields in NWP data have same value over huge area. Take single values 
# rather than rectangular patch.

import numpy as np
import pandas as pd
import numba as nb

import torch
import copy
from sklearn.utils import shuffle

import pvlib
from pvlib.location import Location

import threading
from multiprocessing import Value
import warnings

from . constants import DST_CRS, NORTH, SOUTH, EAST, WEST

XY_MIN = np.array([WEST, SOUTH])
XY_RANGE = np.array([EAST, NORTH])-XY_MIN

def compute_clearsky(times, latitudes, longitudes):
    clearsky = np.full(shape=(len(times), len(latitudes), 3), 
                       fill_value=np.NaN, dtype=np.float32)

    
    for i, (lat, lon) in enumerate(zip(lat, lon)):
        loc = Location(
                latitude=lat,
                longitude=lon,
                tz='UTC').get_clearsky(times)
        clearsky[:,i,:] = clearsky_for_location.values    
    
    return clearsky

def data_source_intersection(pv_output, clearsky=None, sat_loader=None, nwp_loader=None, 
                             lead_time=pd.Timedelta('0min')):
    """Return the datetimes where the pv data, the available satellite data 
    (optional), and the available numerical weather prediction data (optional)
    overlap. This is considered with a forecast lead time."""

    valid_y_times = pv_output.index.values
    
    if clearsky is not None:
        valid_y_times = np.intersect1d(clearsky.index,values, valid_y_times)
    
    if sat_loader is not None:

        for i in sat_loader.time_slice:
            # times in future we can predict y using past sat images
            available_sat_prediction_times = sat_loader.dataset.time.values + \
                                (pd.Timedelta(f'{i*5+1}min') + lead_time)
            valid_y_times = np.intersect1d(available_sat_prediction_times, valid_y_times)
        
    if nwp_loader is not None:
        nwp_times = nwp_loader.dataset.time.values
        for i in nwp_loader.time_slice:
            # past nwp forecast times required to make predictions at given
            # y times
            required_nwp_forecast_times = (valid_y_times - lead_time + pd.Timedelta(f'{i}hours'))\
                                        .floor('180min').to_array()
            in_nwp = np.in1d(required_nwp_forecast_times, nwp_times)
            valid_y_times = valid_y_times[in_nwp]
        
    return pd.DatetimeIndex(valid_y_times)

@nb.jit()
def _shuffled_indexes_for_pv(pv_output_values, consec_samples_per_datetime):
    """
    Parameters
    ----------
    pv_output_values : array_like,
        The pv-output values
    consec_samples_per_datetime: int
        number of samples per each datetime row. Value -1 means take all pv 
        systems fotr each randomly selected datetime. 
    """
    if consec_samples_per_datetime<1 and consec_samples_per_datetime!=-1:
        raise ValueError("invalid samples_per_datetime")
    rowchunks = []
    n = consec_samples_per_datetime
    i_max, j_max = pv_output_values.shape
    
    # get indices without nans
    for i in range(i_max):
        row=[]
        for j in range(j_max):
            if not np.isnan(pv_output_values[i,j]):
                row.append((i,j))
                
        if len(row)>0:
            # shuffle row
            row = [row[k] for k in np.random.permutation(np.arange(len(row)))]
            if consec_samples_per_datetime==-1: n = len(row)
            rowchunks.extend([row[m:m+n] for m in range(0, len(row), n)])
    
    # shuffle the order of the same-time-different-pv-system chunks
    rowchunks = [rowchunks[k] for k in np.random.permutation(np.arange(len(rowchunks)))]
    
    # fill with value so can be converted to numpy array
    for i, rc in enumerate(rowchunks):
        if consec_samples_per_datetime==-1: n = j_max
        rowchunks[i] = rc+[(-1,-1)]*(n-len(rc))
        
    return np.array(rowchunks, dtype=np.int32)

    
class cross_processor_batch:
    """
    A superbatch data generator object for CPU or GPU.
    Loading is done on CPU and ported to GPU if required.
    ...
    
    Parameters
    ----------
    y : pandas.DataFrame 
        PV prediction data. Index must be 
        pandas.DatetimeIndex type and columns must be unique system identifiers.
    y_meta : pandas.DataFrame
        PV metadata with columns which include the x and y coordinates of the PV 
        systems. Index must be unique system identifiers which cover all systems 
        in `y`.
    clearsky : pandas.DataFrame, optional
        Clearsky GHI data for each time and position as the data from the 
        systems in `y`.
    sat_loader : sat_loader.SatelliteLoader, optional
        Satellite image loader object which returns data with any required
        preprocessing already implemented.
    nwp_loader : nwp_loader.NWPLoader, optional
    include_tod : bool, optional
        Include time-of-day as feature.
    include_toy : bool, optional
        Include time-of-year as feature.
    include_latlon : bool, optional
        Include latitude and longitude information as features.
    batch_size : int, default 256
        Number of samples to use in batch
    batches_per_superbatch : int, default 16
        number of batches of size `batch_size` to load and store at once.
    n_superbatches : int, default 1
        Limit of number of superbatches to load in generator functionality.
    n_epochs : int, optional
        Overrides n_superbatches and we loop through the entire dataset this 
        many times.
    gpu : int, default 0
        Whether to load the data into CUDA GPU. Will initially load in CPU then
        transfer over. Value 0 means batches stay on CPU. Value 1 means each
        batch is transfered to GPU only when required. Value 2 means the full
        superbatch is transfered to GPU on loading.
    shuffle_datetime bool, default True (strongly recommended)
        Whether to shuffle the datetimes before loading so that all samples in
        batch are not in consecutive order.
    

    Attributes
    ----------
    cpu_superbatch : dict of numpy.array
    gpu_superbatch : dict of numpy.array, only if `gpu` is True.

    Methods
    -------
    next()
        Generate the next batch of data and load a new superbatch if required.
    """
    # slots for faster attribute access
    # on speed test this contributes maybe 2% speedup
    __slots__ = ['datetime', 'lead_time', 'y', 'y_meta', 'clearsky', 
              'sat_loader', 'nwp_loader', 'include_tod', 'include_toy', 
              'include_latlon', 'batch_size', 'batches_per_superbatch', 
              'superbatch_size','n_superbatches', 'n_epochs', 'gpu', 
              'batch_index','superbatch_index', 'epoch', 
              'consec_samples_per_datetime', 'indexes', 'index_number',
              'extinguished', 'reshuffle_required', 'parallel_loading_cores', 
              '_parallel_loading_cache', 'cpu_superbatch', 'gpu_superbatch',
              'gpu_batch']
    
    def __init__(self, y, y_meta, 
                 clearsky=None,
                 sat_loader=None,
                 nwp_loader=None,
                 include_tod=True,
                 include_toy=True,
                 include_latlon=False,
                 lead_time=pd.Timedelta('0min'),
                 batch_size=256, batches_per_superbatch=16, 
                 n_superbatches=1, n_epochs=None, 
                 gpu=0,
                 consec_samples_per_datetime=10,
                 parallel_loading_cores = 2,
                ):
        
        # remove UTC timezone info
        y.index = y.index.values
        
        # make sure we only keep datetimes where we have all data
        datetime = data_source_intersection(y, clearsky=clearsky,
                                            sat_loader=sat_loader, 
                                            nwp_loader=nwp_loader, 
                                            lead_time=lead_time)
        if len(datetime)==0:
            raise ValueError('Data sources do not overlap in time')
                        
        self.datetime = datetime
        self.lead_time = lead_time
        
        # store datasets
        self.y = y.reindex(datetime).astype(np.float32)
        self.y_meta = y_meta.reindex(y.columns)
        if clearsky is None:
            self.clearsky = None
        else:
            self.clearsky = clearsky.reindex(datetime).astype(np.float32)
        
        # store loaders
        self.sat_loader = sat_loader
        self.nwp_loader = nwp_loader
        
        # store metadata options
        self.include_tod = include_tod
        self.include_toy = include_toy
        self.include_latlon = include_latlon
        
        # store options
        self.batch_size = batch_size
        self.batches_per_superbatch = batches_per_superbatch
        self.superbatch_size = batch_size*batches_per_superbatch
        self.n_superbatches = n_superbatches
        self.n_epochs = n_epochs
        self.gpu = gpu

        # Initiate these indices for looping through and loading the data
        self.batch_index = -1
        self.superbatch_index = -1
        self.epoch = 0
        self.consec_samples_per_datetime = consec_samples_per_datetime
        self.indexes = _shuffled_indexes_for_pv(self.y.values, 
                                               consec_samples_per_datetime)
        self.index_number = 0
        self.extinguished = False
        self.reshuffle_required = False
        self.parallel_loading_cores = parallel_loading_cores
        # Make loader copies for parallel loading
        # This makes better use of caching for speed
        self._parallel_loading_cache = {i:{
                'nwp_loader': nwp_loader if i==0 else copy.deepcopy(nwp_loader),
                'sat_loader': sat_loader if i==0 else copy.deepcopy(sat_loader),
                'thread_current_index':-1,
                'thread_subindex':-1
            } for i in range(self.parallel_loading_cores)}
        self._instantiate_batches(0) # cpu superbatch
        if self.gpu>0: self._instantiate_batches(gpu) # gpu (super)batch
            
            
    def __next__(self):
        """Return the next batch of data and load a new superbatch if 
        required"""
        
        if self.extinguished:
            raise StopIteration
            
        self.batch_index += 1

        # load if very first batch or we have already looped though loaded data
        first_item = (self.batch_index -1) == self.superbatch_index == -1
        load_required = (first_item or (
            self.batch_index >= self.batches_per_superbatch
        ))
        
        none_greater = lambda x, y: (y is not None) and x>=y
        if load_required:
            # Stop iteration if
            # if the number of epochs has reached it's limit or
            # if the number of superbatches has reached it's limit
            if (none_greater(self.epoch, self.n_epochs) or 
                none_greater(self.superbatch_index+1, self.n_superbatches)):
                self.extinguished = True
                raise StopIteration
            
            # load more data otherwise
            
            # if we have reached the end of an epoch then reshuffle
            if self.reshuffle_required:
                self.indexes = _shuffled_indexes_for_pv(self.y.values, 
                                               self.consec_samples_per_datetime)
                self.index_number = 0
            
            self.load_next_superbatch_to_cpu()
            if self.gpu==2:
                self.transfer_superbatch_to_gpu()
            self.batch_index = 0

        batch = self.return_batch()
        
        return batch
    
    
    def __iter__(self):
            return self
    
    
    def _instantiate_batches(self, kind):
        """Instantiate space for data in memory"""
        def new_array(size):
            if kind==0:
                return np.full(shape=size, 
                                  fill_value=0., 
                                  dtype=np.float32)
            elif kind in [1,2]:
                return torch.full(size=size, 
                                  fill_value=0., 
                                  dtype=torch.float16, 
                                  device='cuda')
            else:
                raise ValueError('batch kind not valid')
        
        N = self.batch_size if kind==1 else self.superbatch_size
        
        batch = {}
        batch['y'] = new_array((N, 1))
        
        if self.include_tod:
            batch['day_fraction'] =  new_array((N, 1))
        if self.include_toy:
            batch['year_fraction'] =  new_array((N, 1))
        if self.include_latlon:
            batch['latlon'] =  new_array((N, 2, 1))
        
        if self.clearsky is not None:
            batch['clearsky'] = new_array((N, 1))
        
        if self.sat_loader is not None:
            batch['satellite'] = new_array(
                    (N,) + self.sat_loader.sample_shape
            )
        if self.nwp_loader is not None:
            batch['nwp'] = new_array( 
                                       (N,) + self.nwp_loader.sample_shape
            )
        
        if kind==0:
            self.cpu_superbatch = batch
        elif kind==1:
            self.gpu_batch = batch
        elif kind==2:
            self.gpu_superbatch = batch
        return
    
    
    def shuffle_cpu_superbatch(self):
        rand_state = np.random.get_state()
        for key in self.cpu_superbatch.keys():
                np.random.set_state(rand_state)
                np.random.shuffle(self.cpu_superbatch[key])
    
    
    def load_next_superbatch_to_cpu(self):
        
        # share this value between threads
        index_n = Value('i', self.index_number)
        # set up value for exiting thread
        thread_exit = Value('b', False)
        # keep track of if warned
        already_warned = Value('b', False)
        
        # store whether we run over end of data
        newepoch = Value('i', 0)
        
        def advance_index(thread_current_index, thread_subindex, 
                          force_new_index=False):
            # update thread indices and global next index
            thread_subindex+=1
            if (thread_subindex>=self.indexes.shape[1] 
                            or thread_current_index==-1
                            or force_new_index):
                with index_n.get_lock():
                    thread_current_index = index_n.value
                    index_n.value += 1
                    if index_n.value >= self.indexes.shape[0]:
                        index_n.value = 0
                        newepoch.value += 1
                thread_subindex = 0
            return thread_current_index, thread_subindex
        
        def day_frac(dt):
            return (dt.hour+dt.minute/60.)/24.

        def year_frac(dt):
            return dt.timetuple().tm_yday/365.
        
        def latlon_fraction(xy):
            xy_frac = xy - XY_MIN
            xy_frac = xy / XY_RANGE
            return xy_frac

        def single_thread_data_gather(n_start, n_stop, cache_dict):
            
            # unpack cache
            thread_current_index = cache_dict['thread_current_index']
            thread_subindex = cache_dict['thread_subindex']
            sat_loader = cache_dict['sat_loader']
            nwp_loader = cache_dict['nwp_loader']
            
            # check if first load
            if thread_current_index==-1:
                thread_current_index, thread_subindex = advance_index(
                    thread_current_index, thread_subindex)
            
            for n in range(n_start, n_stop):
                 
                repeats=0 # warn if repeats gets too high
                completed_new=False # loop until we find valid sample
                
                while not completed_new:
                    
                    #assume this sample will be fine
                    completed_new = True
                    
                    # allow threads to be interupted
                    if thread_exit.value: return
                    
                    # warn if repeats too high
                    repeats+=1
                    if repeats == 20 and not already_warned.value:
                        warnings.warn(
                            """
                            Warning: Number of failed loads for one datapoint
                            exceeded {}. This may imply a data issue.
                            
                            """.format(repeats))
                        already_warned.value=True
                                            
                    i, j = self.indexes[thread_current_index, thread_subindex]
                    
                    # To make array regular shaped nan values were filled with
                    # -1. Nan values always occur at end of index rows
                    if i==j==-1:
                        thread_current_index, thread_subindex = advance_index(
                                        thread_current_index, thread_subindex)
                        completed_new = False
                        continue
                        
                    self.cpu_superbatch['y'][n] = self.y.values[i, j]
                    
                    # valid time of forecast
                    dt_valid = pd.Timestamp(self.datetime[i])
                    # location of system
                    xy = self.y_meta.loc[self.y.columns[j]][['x', 'y']]
                    
                    if self.include_tod:
                        self.cpu_superbatch['day_fraction'][n] = day_frac(dt_valid)
                    if self.include_toy:
                        self.cpu_superbatch['year_fraction'][n] = year_frac(dt_valid)
                    if self.include_latlon:
                        self.cpu_superbatch['latlon'][n,...,0] = (
                                latlon_fraction(xy.values))
                    
                    if self.clearsky is not None:
                        if np.isnan(self.clearsky.values[i, j]):
                            thread_current_index, thread_subindex = (
                                advance_index(thread_current_index, 
                                              thread_subindex)
                            )
                            completed_new = False
                            continue
                        else:
                            self.cpu_superbatch['clearsky'][n] = self.clearsky.values[i, j]
                    
                    # load sat images and/or nwp from before the time prediction
                    # is made.
                    dt = dt_valid-self.lead_time
                    
                    if self.sat_loader is not None:
                        sat = sat_loader.get_rectangle_array(
                                dt.floor('5min') - pd.Timedelta('1min'), *xy
                                ).astype(np.float32)[..., np.newaxis]
                        # check for nans and shape
                        completed_new = (completed_new and 
                            not np.any(np.isnan(sat)) and 
                            self.cpu_superbatch['satellite'][n].shape==sat.shape
                        )
                        if not completed_new: 
                            thread_current_index, thread_subindex = (
                                advance_index(thread_current_index, 
                                              thread_subindex, 
                                              force_new_index=True)
                            )
                            continue
                        else:
                            self.cpu_superbatch['satellite'][n] = sat
                    
                    if self.nwp_loader is not None:
                        nwp = nwp_loader.get_rectangle_array(dt, dt_valid, 
                                *xy).astype(np.float32)[..., np.newaxis]
                        # check for nans and shape
                        completed_new = (completed_new and 
                            not np.any(np.isnan(nwp)) and 
                            self.cpu_superbatch['nwp'][n].shape==nwp.shape
                        )
                        if not completed_new: 
                            thread_current_index, thread_subindex = (
                                advance_index(thread_current_index, 
                                              thread_subindex,
                                              force_new_index=True)
                            )
                            continue
                        else:
                            self.cpu_superbatch['nwp'][n] = nwp
                            
                    
                # If it gets to here it will have got a valid sample so 
                # update the thread indexes and go for the next sample.
                thread_current_index, thread_subindex = (
                        advance_index(thread_current_index, thread_subindex)
                )
                    

                        
            # store the current loading location
            cache_dict['thread_current_index'] = thread_current_index
            cache_dict['thread_subindex'] = thread_subindex
            
            return
        
        if self.parallel_loading_cores>1:

            sample_nums = np.linspace(0, self.superbatch_size, 
                                      self.parallel_loading_cores+1).astype(int)

            chunk_args = [[sample_nums[i], sample_nums[i+1], cache_dict] 
                                  for i, cache_dict in self._parallel_loading_cache.items()]

            # Spawn one thread per chunk
            threads = [threading.Thread(target=single_thread_data_gather, args=args)
                                   for args in chunk_args]
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            finally:
                thread_exit.value = True
        
        else:
            single_thread_data_gather(0, self.superbatch_size, 
                                      self._parallel_loading_cache[0])

        # store where we loaded up to
        self.index_number = index_n.value
        self.epoch += newepoch.value
        
        # delay reshuffling in case we have hit epoch limit
        if newepoch.value >0:
            self.reshuffle_required = True
        self.superbatch_index+=1
        self.shuffle_cpu_superbatch()
        return
    

    def transfer_superbatch_to_gpu(self):
        if not self.gpu:
            raise TypeError('gpu support was set to False')
        for k, v in self.cpu_superbatch.items():
            try:
                self.gpu_superbatch[k].copy_(torch.HalfTensor(v))
            except:
                raise Exception('Problem with', k)

    
    def return_batch(self):
        
        i1 = self.batch_index*self.batch_size
        i2 = i1 + self.batch_size
        
        if self.gpu==1:
            for k, v in self.cpu_superbatch.items():
                try:
                    self.gpu_batch[k].copy_(torch.HalfTensor(v[i1:i2]))
                except:
                    raise Exception('Problem with', k)
            batch = self.gpu_batch
            
        else:
            if self.gpu==0:
                superbatch = self.cpu_superbatch
            elif self.gpu==2:
                superbatch = self.gpu_superbatch
            batch = {k:v[i1:i2] for k, v in superbatch.items()}
            
        return batch