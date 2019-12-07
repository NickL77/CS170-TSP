# project-fa19
CS 170 Fall 2019 Project

### Running Code
1. cd into `ilp/` and run `python3 ilp.py`
  * **There are a few tunable parameters:**
  * `main` has the first for-loop which iterates through files. Change the range and the `filename` variable to iterate through desired files.
  * After the first for-loop (currently commented out) there is another for-loop and the two preceeding lines which can be run to iterate through suboptimal outputs and re-run/optimize them more.
  * In both approaches in `main` the integer parameter for `solver.solve(num_iterations)` is used for a maximum number of seconds to run the solver for.
2. Outputs are separated into `outputs/optimal/` and `outputs/suboptimal/`
3. To combine outputs into `outputs/aggregate/` run `python3 aggregate.py` in `ilp/`
