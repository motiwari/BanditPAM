import numpy as np
import os
import banditpam
import time

from scripts.comparison_utils import print_results, store_results
from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
    FASTPAM,
)


def run_banditpam(algorithm_name, data, n_medoids, loss, cache_width):
    if algorithm_name == BANDITPAM_ORIGINAL_NO_CACHING:
        algorithm = "BanditPAM_orig"
        use_cache = False
    elif algorithm_name == BANDITPAM_ORIGINAL_CACHING:
        algorithm = "BanditPAM_orig"
        use_cache = True
    elif algorithm_name == BANDITPAM_VA_CACHING:
        algorithm = "BanditPAM"
        use_cache = True
    elif algorithm_name == BANDITPAM_VA_NO_CACHING:
        algorithm = "BanditPAM"
        use_cache = False
    elif algorithm_name == FASTPAM:
        algorithm = "FastPAM1"
        use_cache = False
    else:
        assert False, "Incorrect algorithm!"

    kmed = banditpam.KMedoids(
        n_medoids=n_medoids,
        algorithm=algorithm,
        use_cache=use_cache,
        use_perm=use_cache,
        parallelize=True,
        cache_width=cache_width,
    )
    start = time.time()
    kmed.fit(data, loss)
    end = time.time()
    runtime = end - start

    return kmed, runtime
