from micron.prepare_grid import prepare_grid

grid_solve_parameters={"evidence_factor": [120, 160, 200],
                       "comb_angle_factor": [14, 16, 22, 28],
                       "start_edge_prior": [200],
                       "selection_cost": [-160, -180, -200, -220, -240]}

prepare_grid(base_dir="/nrs/funke/ecksteinn/micron_experiments",
             experiment="cosem_hela_2_block_3",
             train_number=0,
             predict_number=10,
             graph_number=0,
             grid_solve_parameters=grid_solve_parameters,
             mount_dirs="/nrs, /scratch, /groups, /misc",
             singularity="/groups/funke/home/ecksteinn/Projects/microtubules/micron/singularity/micron114.img",
             num_cpus=10,
             num_block_workers=10,
             queue="normal",
             num_cache_workers=10,
             min_solve_number=0,
             time_limit=120,
             context=400)
