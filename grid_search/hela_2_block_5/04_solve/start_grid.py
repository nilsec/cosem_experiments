import subprocess
import os

predict_number = 10
train_number = 0
graph_number = 0
script = "solve.py"
min_solve_number = 0

solve_scripts = [os.path.join(os.path.abspath("."), f) for f in os.listdir(".") ]
solve_scripts = [os.path.join(d, script) for d in solve_scripts if os.path.isdir(d) and "setup_t{}_p{}_g{}_".format(train_number, predict_number, graph_number) in d and int(d.split("_s")[-1]) >= min_solve_number]

if True:
    print(solve_scripts)
if True:
    solve_command = ""
    for f in solve_scripts:
        solve_command += "python {} & ".format(f)

solve_command = solve_command[:-2]
subprocess.run(solve_command, shell=True)
