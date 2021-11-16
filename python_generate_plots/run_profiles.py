'''
This script contains scaffolding to run many experiments, listed in a config
file such as auto_exp_config.py.

This script will parse each line (= exp configuration) from the config file and
run the corresponding experiment. It can also run many experiments in parallel
by using the pool.apply_async calls instead of the explicit run_exp calls.

To use FP1 optimization, call:
`python run_profiles -e exp_config.py -p`
'''

import importlib
import multiprocessing as mp
import copy
import traceback

from data_utils import *
from timeit import default_timer as timer

from banditpam import KMedoids

def remap_args(args, exp):
    '''
    Parses a config line (as a list) into an args variable (a Namespace).
    Note that --fast_pam1 flag (-p) must be passed to run_profiles.py in order
    to use the FastPAM1 (FP1) optimization. E.g.
    `python run_profiles -e exp_config.py -p`
    (This is inconsistent with the manner in which other args are passed)
    '''
    args.build_ao_swap = exp[1]
    args.verbose = exp[2]
    args.num_medoids = exp[3]
    args.sample_size = exp[4]
    args.seed = exp[5]
    args.dataset = exp[6]
    args.metric = exp[7]
    args.warm_start_medoids = exp[8]
    args.cache_computed = None
    return args

def get_filename(exp, args):
    '''
    Create the filename suffix for an experiment, given its configuration.
    '''
    return exp[0] + \
        '-' + str(args.fast_pam1) + \
        '-' + args.build_ao_swap + \
        '-v-' + str(args.verbose) + \
        '-k-' + str(args.num_medoids) + \
        '-N-' + str(args.sample_size) + \
        '-s-' + str(args.seed) + \
        '-d-' + args.dataset + \
        '-m-' + args.metric + \
        '-w-' + args.warm_start_medoids


def run_exp(args, prof_fname):
    '''
    Runs an experiment with the given parameters, and writes the results to the
    files (arguments ending in _fname). We run the BUILD step and SWAP step
    separately, so as to get different profiles for each step (so we can
    measure the number of distance calls in the BUILD step and SWAP step
    individually)

    Note that the approach below to profiling is undocumented.
    See https://stackoverflow.com/questions/1584425/return-value-while-using-cprofile
    '''
    # Load the dataset of size N
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric == 'PRECOMP':
        dist_mat = np.loadtxt('tree-3630.dist')
        random_indices = np.random.choice(len(total_images), size = args.sample_size, replace = False)
        imgs = np.array([total_images[x] for x in random_indices])
        dist_mat = dist_mat[random_indices][:, random_indices]
    elif args.metric == 'TREE':
        imgs = np.random.choice(total_images, size = args.sample_size, replace = False)
    else:
        imgs = total_images[np.random.choice(len(total_images), size = args.sample_size, replace = False)]
    print(imgs.shape)

    kmed = KMedoids(n_medoids = args.num_medoids, algorithm = "BanditPAM")

    print("Fitting...")
    start = timer()
    kmed.fit(imgs, 'L2', args.num_medoids, prof_fname)
    end = timer()
    print(end - start)
    slash_idx = prof_fname.find('/')
    t_name = os.path.join('profiles', 't' + prof_fname[slash_idx + 2:]) # Ignore the /-
    with open(t_name, 'w+') as fout:
        fout.write("Runtime:" + str(end - start) + "\n")
    print("Done Fitting")


def main(sys_args):
    '''
    Run all the experiments in the experiments lists specified by the -e
    argument, and write the final results (including logstrings) to files. Can
    run multiple experiments in parallel by using the pool.apply_async calls
    below instead of the explicit run_exp calls.

    Note that clarans and em_style experiments are only run for loss comparison
    (in Figure 1(a)).
    '''
    args = get_args(sys.argv[1:]) # Uses default values for now as placeholder to instantiate args

    imported_config = importlib.import_module(args.exp_config.strip('.py'))
    pool = mp.Pool()
    for exp in imported_config.experiments:
        args = remap_args(args, exp)
        prof_fname = os.path.join('profiles', 'p-' + get_filename(exp, args))

        if os.path.exists(prof_fname) and not args.force:
            # Experiments have already been conducted
            print("Warning: already have data for experiment", prof_fname)
            continue
        else:
            print("Running exp:", prof_fname)

        '''
        WARNING: The apply_async calls below are NOT threadsafe. In particular,
        strings in python are lists, which means they are passed by reference.
        This means that if a NEW thread gets the SAME reference as the other
        threads, and updates the object, the OLD thread will write to the wrong
        file. Therefore, whenever using multiprocessing, need to copy.deepcopy()
        all the arguments. Don't need to do this for the explicit run_exp calls
        though since those references are used appropriately (executed
        sequentially)
        '''
        try:
            if exp[0] == 'naive_v1':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), naive_pam_v1.naive_build_and_swap, copy.deepcopy(medoids_fname), copy.deepcopy(B_prof_fname), copy.deepcopy(S_prof_fname))) # Copy inline to copy OTF
                run_exp(args, naive_pam_v1.naive_build_and_swap, medoids_fname, B_prof_fname, S_prof_fname)
            elif exp[0] == 'ucb':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), copy.deepcopy(prof_fname)))
                run_exp(args, prof_fname)
            elif exp[0] == 'clarans':
                # pool.apply_async(run_loss_exp, args=(copy.deepcopy(args), clarans.CLARANS_build_and_swap, copy.deepcopy(medoids_fname)))
                run_loss_exp(copy.deepcopy(args), clarans.CLARANS_build_and_swap, copy.deepcopy(medoids_fname))
            elif exp[0] == 'em_style':
                # pool.apply_async(run_loss_exp, args=(copy.deepcopy(args), em_style.EM_build_and_swap, copy.deepcopy(medoids_fname)))
                run_loss_exp(copy.deepcopy(args), em_style.EM_build_and_swap, copy.deepcopy(medoids_fname))
            else:
                raise Exception('Invalid algorithm specified')
        except Exception as e:
            track = traceback.format_exc()
            print(track)

    pool.close()
    pool.join()
    print("Finished")

if __name__ == "__main__":
    main(sys.argv)
