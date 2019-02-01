import numpy as np
from .cp_compat import numpy_or_cupy, get_array_module, has_cupy
if has_cupy:
    import cupy as cp
from . import assertion

"""
Compilation of utility functions for minibatching, memory transferer
"""


def minibatch_index(shape, minibatch, rng):
    """ Construct a minibatch index from random indexing. """
    if minibatch is None and len(shape) == 1:
        return tuple([slice(None, None, None) for s in shape])
    return tuple([rng.randint(0, s, minibatch) for s in shape])


class MinibatchEpochIndex(object):
    """ A simple class to provide a minibatch index.
    usage:
        minibatch_index = MinibatchEpochIndex(size, minibatch, rng, xp)
        # run 1 epoch
        for index in minibatch_index:
            x_minibatch = x[index]
            # do something with the minibatch
    """
    def __init__(self, size, minibatch, rng, xp):
        """
        size: int
            shape of the indexes.
        minibatch: int
            number of minibatch
        rng: np.random.RandomState or cp.random.RandomState
        """
        self._i = 0
        self._minibatch = minibatch
        self._indexes = xp.arange(size)
        self.rng = rng
        self.shuffle()

    def shuffle(self):
        self.rng.shuffle(self._indexes)

    def __len__(self):
        return int(len(self._indexes) / self._minibatch)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._i + self._minibatch > len(self._indexes):
            self._i = 0
            raise StopIteration()
        value = self._indexes[self._i: self._i + self._minibatch]
        self._i += self._minibatch
        return value


class MinibatchBase(object):
    def __init__(self, array, minibatch):
        """
        Load minibatch by slice.
        array should be already shuffled.

        array: array-like
            Array to be minibatched.
            The first dimension is considered as batch dimension.
        minibatch: int
            minibatch size
        """
        self.round = 0  # This runs 0 -> self.n_loop-1
        self.minibatch = minibatch
        self._array = array
        self.size = len(array)
        if self.size < self.minibatch:
            raise ValueError('Minibatch size should be smaller than the total '
                             'size. Given {} < {}'.format(self.size,
                                                          self.minibatch))

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, arr):
        self._array = arr

    @property
    def n_loop(self):
        """ number of loops for one epoch """
        return int(self.size / self.minibatch)

    def reset(self):
        self.round = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.round + 1 > self.n_loop:
            raise StopIteration()
        value = self._array[self.round * self.minibatch:
                            (self.round + 1) * self.minibatch]
        self.round += 1
        return value


class MinibatchData(MinibatchBase):
    """ Data for minibatching. """
    def __init__(self, array, minibatch, shuffle_index=None):
        """
        array: array-like
            Array to be minibatched.
            The first dimension is considered as batch dimension.
        minibatch: int
            minibatch size
        shuffle_index: array-like
            Index array if shuffle is necessary.
        """
        xp = get_array_module(array)
        assertion.assert_shapes('array', array, 'shuffle_index', shuffle_index,
                                axes=[0])
        if shuffle_index is not None:
            array = array[shuffle_index]
            self.restore_index = xp.arange(len(array))[shuffle_index]
        else:
            array = array
            self.restore_index = xp.arange(len(array))
        super(MinibatchData, self).__init__(array, minibatch)

    @property
    def array(self):
        index = self.restore_index.argsort()
        return self._array[index]

    def shuffle(self, shuffle_index):
        assertion.assert_shapes('array', self._array,
                                'shuffle_index', shuffle_index, axes=[0])
        self._array = self._array[shuffle_index]
        self.restore_index = self.restore_index[shuffle_index]


class _StreamedMinibatch(object):
    """ A minibatch enabling asynchronous memory transfer """
    def __init__(self, array, use_stream):
        self.array = array
        if use_stream:
            self.stream = cp.cuda.Stream(non_blocking=True)
        else:
            self.stream = None

    def send_to_gpu(self, array):
        """ array should be generated from cp.cuda.alloc_pinned_memory """
        self.array.set(array, stream=self.stream)

    def copy_from_gpu(self, array):
        if self.stream is None:
            array[...] = self.array.get()
        else:
            ptr = array.ctypes.get_as_parameter()
            self.array.data.copy_to_host_async(ptr, self.array.nbytes,
                                               self.stream)

    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()


class _StreamedMinibatchSet(MinibatchBase):
    def __init__(self, streamedMinibatches):
        self.streamedMinibatches = streamedMinibatches
        self.n_parallel = len(self.streamedMinibatches)
        self.i = 0

    def __getitem__(self, round_):
        return self.streamedMinibatches[round_ % self.n_parallel]

    def reset(self):
        self.i = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.i >= self.n_parallel:
            raise StopIteration()
        st = self.streamedMinibatches[self.i]
        self.i += 1
        return st


class AsyncMinibatchData(MinibatchBase):
    """
    Data with sequentiall memory transfer between cpu <-> gpu
    """
    def __init__(self, array, minibatch, n_parallel=3, shuffle_index=None,
                 needs_update=True, use_stream=True):
        """
        array: array-like
            Array to be minibatched.
            The first dimension is considered as batch dimension.
        minibatch: int
            minibatch size
        n_parallel: int
            How many parallel job to be used for memory transfer.
        shuffle_index: array-like
            Index array.
        needs_update: boolean
            True if array should be updated. (i.e. parameter)
        use_stream: boolean
            If False, cuda.stream is not used (just for the testing purpose.)
        """
        assertion.assert_shapes('array', array, 'shuffle_index', shuffle_index,
                                axes=[0])
        if not has_cupy:
            raise ImportError('AsyncMinibatchData needs cupy installed.')

        # Constructing np.array from pinned_memory for async Memory transfer
        pinned_array = np.frombuffer(cp.cuda.alloc_pinned_memory(
                array.nbytes), array.dtype, array.size).reshape(array.shape)

        if shuffle_index is not None:
            pinned_array[...] = array[shuffle_index]
            self.restore_index = np.arange(len(array))[shuffle_index]
        else:
            pinned_array[...] = array
            self.restore_index = np.arange(len(array))

        super(AsyncMinibatchData, self).__init__(pinned_array, minibatch)

        self.needs_update = needs_update
        self.n_parallel = np.minimum(n_parallel, self.n_loop)
        self.minibatchset = _StreamedMinibatchSet(
            [_StreamedMinibatch(
                cp.ndarray((self.minibatch, ) + self._array.shape[1:],
                           dtype=self._array.dtype), use_stream)
             for i in range(self.n_parallel)])

    @property
    def array(self):
        index = self.restore_index.argsort()
        return self._array[index]

    def shuffle(self, shuffle_index):
        # cp.ndarray index should be converted to np.ndarray
        if get_array_module(shuffle_index) is cp:
            shuffle_index = shuffle_index.get()

        assertion.assert_shapes('array', self._array,
                                'shuffle_index', shuffle_index, axes=[0])
        array = self._array.copy()
        array = array[shuffle_index]
        self._array[...] = array
        self.restore_index = self.restore_index[shuffle_index]

    def __iter__(self):
        self.reset()
        self.minibatchset.reset()
        # send the first data
        for i in range(self.n_parallel):
            self.send_to_gpu(i)
        return self

    def next(self):
        if self.round > 0:
            if self.needs_update:
                self.copy_from_gpu(self.round - 1)
            if self.round + self.n_parallel - 1 < self.n_loop:
                # send data for the next round
                self.send_to_gpu(self.round + self.n_parallel - 1)

        if self.round + 1 > self.n_loop:
            # End iteration. Wait until all the copy and send finish.
            for data in self.minibatchset:
                data.synchronize()
            raise StopIteration()

        # wait until sending is finished
        self.minibatchset[self.round].synchronize()
        value = self.minibatchset[self.round].array

        self.round += 1
        return value

    def copy_from_gpu(self, round_):
        indexes = slice(round_ * self.minibatch,
                        (round_ + 1) * self.minibatch)
        self.minibatchset[round_].copy_from_gpu(self._array[indexes])

    def send_to_gpu(self, round_):
        indexes = slice(round_ * self.minibatch,
                        (round_ + 1) * self.minibatch)
        self.minibatchset[round_].send_to_gpu(self._array[indexes])


class NoneIterator(object):
    """ Iterator just gives None """
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """ Dummy method """
        return None

    def shuffle(self, index):
        """ Dummy method """
        pass
