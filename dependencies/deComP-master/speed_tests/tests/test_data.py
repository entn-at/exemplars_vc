import pytest
import numpy as np
from decomp.utils.data import AsyncMinibatchData


@pytest.mark.parametrize("mode", ['read', 'write'])
def test_AsyncMinibatchData(benchmark, mode):
    M = 100000
    N = 50000
    n = 10
    n_parallel = 3
    semibatch = 100
    calc_time = 1.0e-4
    needs_update = False if mode == 'read' else True

    data = AsyncMinibatchData(np.ndarray((N, M), np.float32), minibatch=n,
                              n_parallel=n_parallel, needs_update=needs_update,
                              use_stream=True)
    def iteration():
        for (i, arr) in enumerate(data):
            arr *= 2
    it_list = benchmark(iteration)
