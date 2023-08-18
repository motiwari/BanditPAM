# '''
# This script contains scaffolding to run many experiments, listed in a config
# file such as auto_exp_config.py.
#
# This script will parse each line (= exp configuration) from the config file and
# run the corresponding experiment. It can also run many experiments in parallel
# by using the pool.apply_async calls instead of the explicit run_exp calls.
#
# To run this script, call:
# `python run_profiles -e exp_config.py`
# '''
#
# import importlib
# import multiprocessing as mp
# import copy
# import traceback
#
# import banditpam
# from data_utils import *
#
# def remap_args(args, exp):
#     '''
#     Parses a config line (as a list) into an args variable (a Namespace).
#     '''
#     args.verbose = exp[2]  # TODO(@Adarsh321123): test this for accuracy, also look at where this is used in original and see if we have those
#     args.num_medoids = exp[3]
#     args.sample_size = exp[4]
#     args.seed = exp[5]
#     args.dataset = exp[6]
#     args.metric = exp[7]
#     return args
#
# def get_filename(exp, args):
#     '''
#     Create the filename suffix for an experiment, given its configuration.
#     '''
#     return exp[0] + \
#         '-v-' + str(args.verbose) + \
#         '-k-' + str(args.num_medoids) + \
#         '-N-' + str(args.sample_size) + \
#         '-s-' + str(args.seed) + \
#         '-d-' + args.dataset + \
#         '-m-' + args.metric
#
# def write_medoids(medoids_fname, built_medoids, swapped_medoids, num_swaps, final_loss, dist_comps):
#     '''
#     Write results of an experiment to the given file, including:
#     medoids after BUILD step, medoids after SWAP step, etc.
#     '''
#     with open(medoids_fname, 'w+') as fout:
#         fout.write("Built:" + ','.join(map(str, built_medoids)))
#         fout.write("\nSwapped:" + ','.join(map(str, swapped_medoids)))
#         fout.write("\nNum Swaps: " + str(num_swaps))
#         fout.write("\nFinal Loss: " + str(final_loss))
#         fout.write("\nDistance Computations: " + str(dist_comps))
#
# def run_exp(args, object_name, medoids_fname):
#     '''
#     Runs an experiment with the given parameters, and writes the results to the
#     given logfile (and potentially timefile).
#     '''
#     # Load the dataset of size N
#     total_images, total_labels, sigma = load_data(args)
#     np.random.seed(args.seed)
#     imgs = total_images[np.random.choice(len(total_images), size = args.sample_size, replace = False)]
#
#     print("Fitting...")
#     object_name.fit(imgs, args.metric)
#     print("Done Fitting")
#
#     built_medoids = object_name.build_medoids
#     swapped_medoids = object_name.medoids
#     num_swaps = object_name.steps
#     final_loss = object_name.average_loss
#     # there are no BFP build dist comps because we use uniform random sampling
#     dist_comps = object_name.swap_distance_computations
#     write_medoids(medoids_fname, built_medoids, swapped_medoids, num_swaps, final_loss, dist_comps)
#
# def main(sys_args):
#     '''
#     Run all the experiments in the experiments lists specified by the -e
#     argument, and write the final results (including logstrings) to files. Can
#     run multiple experiments in parallel by using the pool.apply_async calls
#     below instead of the explicit run_exp calls.
#
#     Note that clarans and em_style experiments are only run for loss comparison
#     (in Figure 1(a)).
#     '''
#     # TODO: why "including logstrings" in docstring above
#     # TODO: test the parallel experiments
#     args = get_args(sys.argv[1:]) # Uses default values for now as placeholder to instantiate args
#
#     imported_config = importlib.import_module(args.exp_config.strip('.py'))
#     pool = mp.Pool()
#     for exp in imported_config.experiments:
#         args = remap_args(args, exp)
#         medoids_fname = os.path.join('profiles','Loss_plots_paper', 'L-' + get_filename(exp, args))
#
#         if os.path.exists(medoids_fname) and not args.force:
#             # Experiments have already been conducted
#             print("Warning: already have data for experiment", medoids_fname)
#             continue
#         else:
#             print("Running exp:", medoids_fname)
#
#         '''
#         WARNING: The apply_async calls below are NOT threadsafe. In particular,
#         strings in python are lists, which means they are passed by reference.
#         This means that if a NEW thread gets the SAME reference as the other
#         threads, and updates the object, the OLD thread will write to the wrong
#         file. Therefore, whenever using multiprocessing, need to copy.deepcopy()
#         all the arguments. Don't need to do this for the explicit run_exp calls
#         though since those references are used appropriately (executed
#         sequentially)
#         '''
#         try:
#             if exp[0] == 'bfp':
#                 kmed = banditpam.KMedoids(n_medoids=args.num_medoids, algorithm="BanditFasterPAM", max_iter=10000)
#                 run_exp(args, kmed, medoids_fname)
#             elif exp[0] == 'fp':
#                 kmed = banditpam.KMedoids(n_medoids=args.num_medoids, algorithm="FasterPAM", max_iter=10000)
#                 run_exp(args, kmed, medoids_fname)
#             elif exp[0] == 'naive_v1':
#                 # TODO: fix the max iter between algos and add comment explaining
#                 kmed = banditpam.KMedoids(n_medoids=args.num_medoids, algorithm="PAM", max_iter=1)
#                 run_exp(args, kmed, medoids_fname)
#             else:
#                 raise Exception('Invalid algorithm specified')
#         except Exception as e:
#             track = traceback.format_exc()
#             print(track)
#
#     pool.close()
#     pool.join()
#     print("Finished")
#
# if __name__ == "__main__":
#     main(sys.argv)
