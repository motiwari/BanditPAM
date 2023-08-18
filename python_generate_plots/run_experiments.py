'''
This script contains scaffolding to run many experiments, listed in a config
file such as auto_exp_config.py.

This script will parse each line (= exp configuration) from the config file and
run the corresponding experiment.
'''

import importlib
import traceback
import sys
import os
import banditpam
from data_utils import *
from timeit import default_timer as timer

def remap_args(args, exp):
    '''
    Parses a config line (as a list) into an args variable (a Namespace).
    '''
    args.num_medoids = exp[1]
    args.sample_size = exp[2]
    args.seed = exp[3]
    args.dataset = exp[4]
    args.metric = exp[5]
    return args

def get_filename(exp, args):
    '''
    Create the filename suffix for an experiment, given its configuration.
    '''
    return exp[0] + \
        '-k-' + str(args.num_medoids) + \
        '-N-' + str(args.sample_size) + \
        '-s-' + str(args.seed) + \
        '-d-' + args.dataset + \
        '-m-' + args.metric

def write_medoids(prof_fname, built_medoids, swapped_medoids, num_swaps, final_loss, dist_comps):
    '''
    Write results of an experiment to the given file, including:
    medoids after BUILD step (or uniform random sampling), medoids after
    SWAP step, etc.
    '''
    with open(prof_fname, 'w+') as fout:
        fout.write("Built:" + ','.join(map(str, built_medoids)))
        fout.write("\nSwapped:" + ','.join(map(str, swapped_medoids)))
        fout.write("\nNum Swaps: " + str(num_swaps))
        fout.write("\nFinal Loss: " + str(final_loss))
        fout.write("\nDistance Computations: " + str(dist_comps))

def run_exp(args, object_name, log_fname, output_dir, time):
    '''
    Runs an experiment with the given parameters, and writes the results to the
    given logfile (and potentially timefile).
    '''
    # Load the dataset of size N
    total_images = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(len(total_images), size = args.sample_size, replace = False)]

    print("Fitting...")
    if time:
        start = timer()
        object_name.fit(imgs, args.metric)
        end = timer()
        print("Runtime:", end - start)
        t_name = os.path.join('logs', output_dir, 't' + log_fname[1:]) # Ignore the L
        with open(t_name, 'w+') as fout:
            fout.write("Runtime:" + str(end - start) + "\n")
    else:
        object_name.fit(imgs, args.metric)

    print("Done Fitting")

    built_medoids = object_name.build_medoids
    swapped_medoids = object_name.medoids
    num_swaps = object_name.steps
    final_loss = object_name.average_loss
    dist_comps = object_name.swap_distance_computations
    log_fname = os.path.join('logs', output_dir, log_fname)
    write_medoids(log_fname, built_medoids, swapped_medoids, num_swaps, final_loss, dist_comps)

def main(sys_args):
    '''
    Run all the experiments in the experiments lists specified by the -e
    argument, and write the final results to files.

    Note that PAM and FasterPAM experiments are only run for loss comparison
    (in Figure 1(a)).
    '''
    output_dir = "testing3"
    args = get_args(sys.argv[1:]) # Uses default values for now as placeholder to instantiate args

    imported_config = importlib.import_module(args.exp_config.strip('.py'))
    for exp in imported_config.experiments:
        args = remap_args(args, exp)
        log_fname = 'L-' + get_filename(exp, args)
        dir_name = os.path.join('logs', output_dir, log_fname)

        if os.path.exists(dir_name):
            # Experiments have already been conducted
            print("Warning: already have data for experiment", dir_name)
            continue
        else:
            print("Running exp:", dir_name)

        try:
            if exp[0] == 'bfp':
                kmed = banditpam.KMedoids(n_medoids=args.num_medoids, algorithm="BanditFasterPAM")
                run_exp(args, kmed, log_fname, output_dir, time=True)
            elif exp[0] == 'fp':
                kmed = banditpam.KMedoids(n_medoids=args.num_medoids, algorithm="FasterPAM")
                run_exp(args, kmed, log_fname, output_dir, time=True)
            elif exp[0] == 'naive_v1':
                kmed = banditpam.KMedoids(n_medoids=args.num_medoids, algorithm="PAM", max_iter=1)
                run_exp(args, kmed, log_fname, output_dir, time=True)
            else:
                raise Exception('Invalid algorithm specified')
        except Exception as e:
            track = traceback.format_exc()
            print(track)

    print("Finished")

if __name__ == "__main__":
    main(sys.argv)
