# BanditPAM: Almost Linear-Time $k$-Medoids Clustering

## Dear Reviewer

We thank you for your time in reviewing our submission. 
We understand that you are performing a service to the community.
In order to ensure that review of this code is as easy as possible, we have:
- Included a one-line script to recreate all results (`repro_script.sh`)
- Heavily commented and documented the code
- Included an overview of the codebase and the files.

Thank you for reviewing our submission!

## Description of Files

The files are organized as follows:

- `src/` contains the implementation of all versions of BanditPAM with and without caching and virtual arms.
- `scripts/` contains any python scripts for experiments or tests.
- `experiments/` contains experiment scripts
  - The two main types of experiments are the scaling experiments with respect to `k` and `n`.
- `logs/` contains the logs of the experiments.

## Reproducing the results
- To reproduce all of the experimental logs and create the figures, run `chmod +x repro_script.sh && ./repro_script.sh`
- All datasets will be automatically downloaded by the script
