from sklearn.utils import shuffle
import numba as nb
from numba import njit, objmode
import numpy as np
import torch

class cross_processor_batch:
    """
    A superbatch data generator object for CPU or GPU.
    Loadin gis done on CPU and ported to GPU if required.
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
    clearsky : pandas.DataFrame
        Clearsky GHI data for each time and position as the data from the 
        systems in `y`.
    satellite_loader : make_dataset.SatelliteLoader, optional
        Satellite image loader object which returns data with any required
        preprocessing already implemented.
    nwp_loader : make_dataset.NWPLoader, not yet supported
    batch_size : int, default 256
        Number of samples to use in batch
    batches_per_superbatch : int, default 16
        number of batches of size `batch_size` to load and store at once.
    n_superbatches : int, default 1
        Limit of number of superbatches to load in generator functionality.
    n_epochs : int, optional
        Overrides n_superbatches and we loop through the entire dataset this 
        many times.
    gpu : bool, default False
        Whether to load the data into CUDA GPU. Will initially load in CPU then
        transfer over.
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
    def __init__(self, y, y_meta, clearsky,
                 satellite_loader=None,
                 nwp_loader=None,
                 batch_size=256, batches_per_superbatch=16, 
                 n_superbatches=1, n_epochs=None, 
                 gpu=False,
                 shuffle_datetime=True,
                 method='fast',
                ):
        
        # make sure we only keep datetimes where we have all data
        datetime = y.index.intersection(clearsky.index)
        if satellite_loader is not None: 
            datetime = datetime.intersection(satellite_loader.index)
        if nwp_loader is not None: 
            datetime = datetime.intersection(nwp_loader.index)
        if shuffle_datetime: datetime = shuffle(datetime)
        if len(datetime)==0:
            raise ValueError('Data sources do not overlap in time')
            
        assert np.all(y_meta.index == y.columns), "metadata doesn't match y"
            
        self.shuffle_datetime = shuffle_datetime
        self.datetime = datetime
        
        # store datasets
        self.y = y.reindex(datetime).astype(np.float32)
        self.clearsky = clearsky.reindex(datetime).astype(np.float32)
        self.y_meta = y_meta
        
        # store loaders
        self.satellite_loader = satellite_loader
        self.nwp_loader = nwp_loader
        
        # store options
        self.batch_size = batch_size
        self.batches_per_superbatch = batches_per_superbatch
        self.superbatch_size = batch_size*batches_per_superbatch
        self.n_superbatches = n_superbatches
        self.n_epochs = n_epochs
        self.gpu = gpu
        self.method = method
        

        # Initiate these indices for looping through and loading the data
        self.batch_index = -1
        self.superbatch_index = -1
        self.epoch = 0
        self.datetime_index = 0
        self.y_index = 0
        self.extinguished = False

        self._instantiate_superbatch('cpu')
        if self.gpu: self._instantiate_superbatch('gpu')
            
            
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
            self.load_next_superbatch_to_cpu()
            if self.gpu:
                self.transfer_superbatch_to_gpu()
            self.batch_index = 0

        batch = self.return_batch(self.batch_index, gpu=self.gpu)
        
        return batch
    
    def __iter__(self):
            return self
    
    def _instantiate_superbatch(self, where):
        """Instantiate space for data in memory"""
        def new_array(size):
            if where=='gpu':
                return torch.full(size=size, 
                                  fill_value=0., 
                                  dtype=torch.float16, 
                                  device='cuda')
            elif where=='cpu':
                return np.full(shape=size, 
                                  fill_value=0., 
                                  dtype=np.float32)
        
        superbatch = {}
        superbatch['y'] = new_array((self.superbatch_size, 1))
        superbatch['day_fraction'] =  new_array((self.superbatch_size, 1))
        superbatch['clearsky'] = new_array((self.superbatch_size, 1))
        
        if self.satellite_loader is not None:
            superbatch['satellite'] = new_array((self.superbatch_size, 1, 
                                               self.satellite_loader.width, 
                                               self.satellite_loader.height)
                                                    )
        if self.nwp_loader is not None:
            raise NotImplementedError(
                'nwp_loader not implemented in instantiate_cpu_superbatch')
        
        if where=='cpu':
            self.cpu_superbatch = superbatch
        elif where=='gpu':
            self.gpu_superbatch = superbatch
        return
    
    def shuffle_cpu_superbatch(self):
        rand_state = np.random.get_state()
        for key in self.cpu_superbatch.keys():
                np.random.set_state(rand_state)
                np.random.shuffle(self.cpu_superbatch[key])
    
    def load_next_superbatch_to_cpu(self):
        if not hasattr(self, '_load_next_superbatch_to_cpu'):
            # generate numba function for fast loading
            self._load_next_superbatch_to_cpu = self._cpu_load_function_factory(
                        method=self.method)

        day_fraction_in = ((self.datetime.hour.values +
                            self.datetime.minute.values/60)
                           /24).astype(np.float32)
        
        # if including satellite imagery
        is_sat = np.int32(self.satellite_loader is not None)
        if is_sat:
            satellite_image_out = self.cpu_superbatch['satellite']
        else:
            satellite_image_out = np.empty((0,0,0,0), dtype=np.float32)

        args = [self.y.values, 
                self.cpu_superbatch['y'],
                self.clearsky.values, 
                self.cpu_superbatch['clearsky'], 
                day_fraction_in, 
                self.cpu_superbatch['day_fraction'],
                satellite_image_out, 
                is_sat,
                np.int32(self.superbatch_size), 
                np.int32(self.datetime_index), 
                np.int32(self.y_index)]

        # load data
        out = self._load_next_superbatch_to_cpu(*args)

        # store where we loaded up to
        self.datetime_index, self.y_index, newepoch = out
        self.epoch += newepoch
        self.superbatch_index+=1

        self.shuffle_cpu_superbatch()
        return
        
    def _cpu_load_function_factory(self, method='fast'):
        """generates numba function which loads data much faster than
        the base python."""
        assert method in ['fast', 'random'], 'method parameter not valid'
        is_sat = np.int32(self.satellite_loader is not None)
        
        shuffled_index = np.array(
                    np.meshgrid(np.arange(self.y.shape[0]), 
                    np.arange(self.y.shape[1]))
        )
        shuffled_index = shuffle(shuffled_index.reshape((2, -1)).T) \
                                               .reshape(self.y.shape+(2,))
        
        def get_sat(i,j):
            return self.satellite_loader.get_rectangle(self.datetime[i], 
                                *self.y_meta.loc[self.y.columns[j]][['x', 'y']]
                                ).values.astype(np.float32)

        @nb.jit("""int32[:](float32[:,:], float32[:,:],
                            float32[:,:], float32[:,:],
                            float32[:], float32[:,:],
                            float32[:,:,:,:], int32,
                            int32, int32, int32)""", 
                nopython=True, nogil=True)
        def fast_numba_data_gather(y_in, y_out, 
              clearsky_in, clearsky_out, 
              day_fraction_in, day_fraction_out,
              satellite_image_out, is_sat,
              superbatch_size, i, j):

            newepoch = 0
            for n in range(superbatch_size):

                completed_new=0
                while completed_new==0:

                    # loop through to find non-NaN y
                    found_new = 0
                    while found_new==0:
                        j+=1
                        if j==y_in.shape[1]:
                            j=0
                            i+=1
                        if i==y_in.shape[1]:
                            i=0
                            newepoch += 1
                        y_val = y_in[i, j]
                        found_new = 1-int(np.isnan(y_val))

                    y_out[n] = y_val
                    clearsky_out[n] = clearsky_in[i, j]
                    day_fraction_out[n] = day_fraction_in[i]

                    if is_sat:
                        with objmode(sat='float32[:,:]'):
                                sat = get_sat(i,j)

                        satellite_image_out[n] = sat

                    # sat data sometimes has NaNs. Check for this
                    completed_new = 1-int(np.any(np.isnan(sat)))

            return np.array([i,j, int(newepoch)], dtype=np.int32)
        
        
        @nb.jit("""int32[:](float32[:,:], float32[:,:],
                            float32[:,:], float32[:,:],
                            float32[:], float32[:,:],
                            float32[:,:,:,:], int32,
                            int32, int32, int32)""", 
                nopython=True, nogil=True)
        def random_numba_data_gather(y_in, y_out, 
              clearsky_in, clearsky_out, 
              day_fraction_in, day_fraction_out,
              satellite_image_out, is_sat,
              superbatch_size, i, j):

            newepoch = 0
            for n in range(superbatch_size):

                completed_new=0
                while completed_new==0:

                    # loop through to find non-NaN y
                    found_new = 0
                    while found_new==0:
                        j+=1
                        if j==y_in.shape[1]:
                            j=0
                            i+=1
                        if i==y_in.shape[1]:
                            i=0
                            newepoch += 1
                        ind0, ind1 = shuffled_index[i,j]
                        y_val = y_in[ind0, ind1]
                        found_new = 1-int(np.isnan(y_val))

                    y_out[n] = y_val
                    clearsky_out[n] = clearsky_in[ind0, ind1]
                    day_fraction_out[n] = day_fraction_in[ind0]

                    if is_sat:
                        with objmode(sat='float32[:,:]'):
                                sat = get_sat(ind0, ind1)

                        satellite_image_out[n] = sat

                    # sat data sometimes has NaNs. Check for this
                    completed_new = 1-int(np.any(np.isnan(sat)))

            return np.array([i,j, int(newepoch)], dtype=np.int32)
        
        if method=='fast':
            return fast_numba_data_gather
        elif method=='random':
            return random_numba_data_gather
            
    def transfer_superbatch_to_gpu(self):
        if not self.gpu:
            raise TypeError('gpu support was set to False')
        for k, v in self.cpu_superbatch.items():
            try:
                self.gpu_superbatch[k].copy_(torch.HalfTensor(v))
            except:
                print('Problem with', k)
                raise
    
    def return_batch(self, batch_index, gpu=False):
        if gpu:
            superbatch = self.gpu_superbatch
        else:
            superbatch = self.cpu_superbatch
        i1 = batch_index*self.batch_size
        i2 = i1 + self.batch_size
        dict_view = {k:v[i1:i2] for k, v in superbatch.items()}
        return dict_view
    
