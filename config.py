import multiprocessing as mp

num_worker = 0 #mp.cpu_count()
batch_size = 64
use_cuda = False

use_apex = False
apex_opt_level = 'O2' # "O0", "O1", "O2", and "O3"

epochs = 10