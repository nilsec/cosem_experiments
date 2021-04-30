from micron.prepare_evaluation import set_up_environment
import os

""" COSEM BLOCK 1
"""
solve_dir = "/nrs/funke/ecksteinn/micron_experiments/cosem_hela_2_block_1/04_solve"
base_dir = "/nrs/funke/ecksteinn/micron_experiments"
experiment = "cosem_hela_2_block_1"
train_number = 0
predict_number = 10
graph_number = 0
tracing_file = "/nrs/funke/ecksteinn/micron_experiments/cosem/00_data/tracings/block_1.nml"
tracing_offset = [9996, 396, 21996]
tracing_size = [2000, 2000, 2000]
""" COSEM BLOCK 2
"""
solve_dir = "/nrs/funke/ecksteinn/micron_experiments/cosem_hela_2_block_2/04_solve"
base_dir = "/nrs/funke/ecksteinn/micron_experiments"
experiment = "cosem_hela_2_block_2"
train_number = 0
predict_number = 10
graph_number = 0
tracing_file = "/nrs/funke/ecksteinn/micron_experiments/cosem/00_data/tracings/block_2.nml"
tracing_offset = [2196, 1196, 27996]
tracing_size = [2000, 2000, 2000]
""" COSEM BLOCK 3
"""
solve_dir = "/nrs/funke/ecksteinn/micron_experiments/cosem_hela_2_block_3/04_solve"
base_dir = "/nrs/funke/ecksteinn/micron_experiments"
experiment = "cosem_hela_2_block_3"
train_number = 0
predict_number = 10
graph_number = 0
tracing_file = "/nrs/funke/ecksteinn/micron_experiments/cosem/00_data/tracings/block_3.nml"
tracing_offset = [15596, 596, 22796]
tracing_size = [2000, 2000, 2000]
""" COSEM BLOCK 4
"""
solve_dir = "/nrs/funke/ecksteinn/micron_experiments/cosem_hela_2_block_4/04_solve"
base_dir = "/nrs/funke/ecksteinn/micron_experiments"
experiment = "cosem_hela_2_block_4"
train_number = 0
predict_number = 10
graph_number = 0
tracing_file = "/nrs/funke/ecksteinn/micron_experiments/cosem/00_data/tracings/block_4.nml"
tracing_offset = [10796, 496, 36996]
tracing_size = [2000, 2000, 2000]
min_solve_number = 5000
"""COSEM BLOCK 5
"""
solve_dir = "/nrs/funke/ecksteinn/micron_experiments/cosem_hela_2_block_5/04_solve"
base_dir = "/nrs/funke/ecksteinn/micron_experiments"
experiment = "cosem_hela_2_block_5"
train_number = 0
predict_number = 10
graph_number = 0
tracing_file = "/nrs/funke/ecksteinn/micron_experiments/cosem/00_data/tracings/block_5.nml"
tracing_offset = [11196, 2196, 18796]
tracing_size = [2000, 2000, 2000]
min_solve_number = 0

subsample_factor = 10
max_edges = 5
distance_threshold = 120
optimality_gap = 0.0
time_limit = 300
voxel_size = [4, 4, 4]
mount_dirs = "/nrs, /scratch, /groups, /misc"
singularity = "None"
num_cpus = 1
num_block_workers = 1
num_cache_workers = 1
queue = "normal"


solve_dir = os.path.join(os.path.join(base_dir, experiment), "04_solve")
solve_setups = [os.path.join(solve_dir, f) for f in os.listdir(solve_dir) if "setup_t{}_p{}_g{}_s".format(train_number, 
                                                                                                          predict_number,
                                                                                                          graph_number) in f]

solve_numbers = [int(f.split("_")[-1][1:]) for f in solve_setups]
solve_numbers = [n for n in solve_numbers if n >= min_solve_number]

eval_number = 0
for solve_number in solve_numbers:
    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       predict_number,
                       graph_number,
                       solve_number,
                       eval_number,
                       tracing_file,
                       tracing_offset,
                       tracing_size,
                       subsample_factor,
                       max_edges,
                       distance_threshold,
                       optimality_gap,
                       time_limit,
                       voxel_size,
                       mount_dirs,
                       singularity,
                       queue,
                       num_cpus,
                       num_block_workers,
                       num_cache_workers)
