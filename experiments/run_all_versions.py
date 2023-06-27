import banditpam
import time

from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
    FASTPAM,
)


def run_banditpam(
    algorithm_name,
    data,
    n_medoids,
    loss,
    cache_width=2000,
    parallelize=True,
    n_swaps=100,
    build_confidence=3,
    swap_confidence=5,
):
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
        max_iter=n_swaps,
        parallelize=parallelize,
        cache_width=cache_width,
        build_confidence=build_confidence,
        swap_confidence=swap_confidence,
    )
    start = time.time()
    kmed.fit(data, loss)
    end = time.time()
    runtime = end - start

    return kmed, runtime
