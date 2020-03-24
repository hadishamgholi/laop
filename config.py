import multiprocessing as mp

num_worker = mp.cpu_count()
batch_size = 64
use_cuda = True

use_apex = True
apex_opt_level = 'O2' # "O0", "O1", "O2", and "O3"

epochs = 10