from sklearn.utils import shuffle
import nump as np

class cross_processor_batch:
    def __init__(self, y, y_meta, clearsky,
                 satellite_loader=None,
                 nwp_loader=None,
                 batch_size=256, 
                 batches_per_superbatch=16, 
                 n_superbatches=1, 
                 n_epochs=None, 
                 gpu=False,
                 shuffle_datetime=True,
                ):

        self.y = y
        self.y_meta = y_meta
        self.clearsky = clearsky
        self.satellite_loader = satellite_loader
        self.nwp_loader = nwp_loader
        self.batch_size = batch_size
        self.batches_per_superbatch = batches_per_superbatch
        self.superbatch_size = batch_size*batches_per_superbatch
        self.n_superbatches = n_superbatches
        self.n_epochs = n_epochs
        self.gpu = gpu
        
        datetime = self.Y.index
        if satellite_loader is not None: 
            datetime = datetime.intersection(satellite_loader.index)
        if nwp_loader is not None: 
            datetime = datetime.intersection(nwp_loader.index)
        if shuffle_datetime: datetime = shuffle(shuffle_datetime)
        self.shuffle_datetime = shuffle_datetime
        self.datetime = datetime
        
        self.batch_index = 0
        self.superbatch_index = 0
        self.epoch = 0
        self.datetime_index = 0
        self.y_index = 0

        self._instantiate_superbatch('cpu')
        if self.gpu: self._instantiate_superbatch('gpu')
    
    def _instantiate_superbatch(self, where):
        def new_array(size):
            if where=='gpu':
                return torch.full(size=size, 
                                  fill_value=np.nan, 
                                  dtype=torch.float16, 
                                  device='cuda')
            elif where=='cpu':
                return np.full(size=size, 
                                  fill_value=np.nan, 
                                  dtype=np.float16)
        
        superbatch = {}
        superbatch['y'] = new_array((self.superbatch_size, 1))
        superbatch['day_fraction'] =  new_array((self.superbatch_size, 1))
        superbatch['clearsky'] = new_array((self.superbatch_size, 1))
        
        if self.satellite_loader is not None:
            superbatch['satellite'] = new_array((self.superbatch_size, 1, 
                                               satellite_loader.width, 
                                               satellite_loader.height)
                                                    )
        if self.nwp_loader is not None:
            raise ValueError(
                'nwp_loader not implemented in instantiate_cpu_superbatch'
            )
        
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
        
        # Find the next set of datetimes and y columns
        # For faster loading we load all locations at each shuffled datetime
        i = self.datetime_index
        j = self.y_index
        dts = np.empty(self.superbatch_size, dtype='datetime64[s]')
        y_inds = np.empty(self.superbatch_size, dtype=int)
        for n in range(self.superbatch_size):
            print('\r{:6d} of {:d}'.format(n+1, self.superbatch_size), 
                  end='', flush=True)
            
            completed_new=False
            while not completed_new:
                # loop through to find non-NaN y
                found_new=False
                while not found_new:
                    j+=1
                    if j==self.y.shape[1]:
                        j=0
                        i+=1
                    if i==len(self.datetime_index)
                        i=0
                        self.epoch += 1
                    y_val = self.y.loc[self.datetime[i]].iloc[:, j]
                    found_new = !np.isnan(y)

                self.cpu_superbatch['y'][n] = y
                self.cpu_superbatch['day_fraction'][n] = (self.datetime[i].hour 
                                                          + self.datetime[i]/60.)/24
                self.cpu_superbatch['clearsky'][n] = self.y.loc[self.datetime[i]].iloc[:, j]
                if self.satellite_loader is not None:
                    self.cpu_superbatch['satellite'][n] = self.satellite_loader.get_rectangle(
                        self.datetime[i], *self.y_meta.loc[self.y.columns[i]][['x', 'y']]
                    )
                if self.nwp_loader is not None:
                    raise ValueError('nwp_loader not implemented in load_next_superbatch_to_cpu')
                
                # sat data sometimes has NaNs. Check for this
                completed_new = !np.any([
                    np.any(np.isnan(self.cpu_superbatch['satellite'][n]))
                ])
                if not completed_new:
                    print('\nData had NaN values. Skipping')
                
        # store where we loaded up to
        self.datetime_index = i
        self.y_index = j
        self.superbatch_index+=1
        
        self.shuffle_cpu_superbatch()
        return
            
    def transfer_superbatch_to_gpu(self):
        if not self.gpu:
            raise ValueError('gpu support was set to False')
        for k, v in self.cpu_superbatch.items():
            try:
                self.gpu_superbatch[k].copy_(torch.HalfTensor(v))
            except:
                print('Problem with', k)
                raise
    
            
    def return_batch(self, batch_index, which):
        if which=='cpu':
            superbatch = self.cpu_superbatch
        elif which=='gpu':
            if not self.gpu:
                raise ValueError('gpu support was set to False')
            else:
                superbatch = self.cpu_superbatch
        dict_view = {k:v[batch_index*self.batch_size:(batch_index+1)*self.batch_size:] 
                     for k, v in superbatch.items()}
        return dict_view
    
    def next(self):
        load_required = self.batch_index == self.superbatch_index == 0
        load_required = load_required or (self.batch_index == self.batches_per_superbatch)
            
        if load_required:
            #Stop iteration if
            # if the number of epochs has reached it's limit or
            # if the number of superbatches has reached it's limit
            if self.epochs==self.n_epochs:
                raise StopIteration
            elif self.n_epochs is None and self.superbatch_index==self.n_superbatches:
                raise StopIteration
            
            # load more data otherwise
            self.load_next_superbatch_to_cpu()
            if self.gpu:
                self.transfer_superbatch_to_gpu()
            self.batch_index = 0

        which = 'gpu' if self.gpu else 'cpu'
        batch = self.return_batch(self.batch_index, which)

        self.batch_index += 1
        
        return batch