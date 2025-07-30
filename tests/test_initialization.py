import unittest

from banditpam import KMedoids


class InitializationTests(unittest.TestCase):
    def test_default_initialization(self):
        kmed = KMedoids()
        assert kmed.n_medoids == 5, "n_medoids does not match"
        assert kmed.algorithm == "BanditPAM", "algorithm does not match"
        assert kmed.max_iter == 100, "max_iter does not match"
        assert kmed.build_confidence == 10, "build_confidence does not match"
        assert kmed.swap_confidence == 5, "swap_confidence does not match"
        assert kmed.use_cache, "use_cache does not match"
        assert kmed.use_perm, "use_perm does not match"
        assert kmed.cache_width == 1000, "cache_width does not match"
        assert kmed.parallelize, "parallelize does not match"

    def test_initialization_1(self):
        """
        Test that we can initialize a KMedoids object with given parameters.
        """
        kmed = KMedoids(
            n_medoids=5,
            algorithm="BanditPAM",
            max_iter=100,
            build_confidence=9,
            swap_confidence=11,
            use_cache=True,
            use_perm=True,
            cache_width=10,
            parallelize=True,
        )
        assert kmed.n_medoids == 5, "n_medoids does not match"
        assert kmed.algorithm == "BanditPAM", "algorithm does not match"
        assert kmed.max_iter == 100, "max_iter does not match"
        assert kmed.build_confidence == 9, "build_confidence does not match"
        assert kmed.swap_confidence == 11, "swap_confidence does not match"
        assert kmed.use_cache, "use_cache does not match"
        assert kmed.use_perm, "use_perm does not match"
        assert kmed.cache_width == 10, "cache_width does not match"
        assert kmed.parallelize, "parallelize does not match"

    def test_initialization_2(self):
        kmed = KMedoids(
            n_medoids=3,
            algorithm="FastPAM1",
            max_iter=1,
            build_confidence=11,
            swap_confidence=2,
            use_cache=False,
            use_perm=True,
            cache_width=60,
            parallelize=False,
        )
        assert kmed.n_medoids == 3, "n_medoids does not match"
        assert kmed.algorithm == "FastPAM1", "algorithm does not match"
        assert kmed.max_iter == 1, "max_iter does not match"
        assert kmed.build_confidence == 11, "build_confidence does not match"
        assert kmed.swap_confidence == 2, "swap_confidence does not match"
        assert not kmed.use_cache, "use_cache does not match"
        assert kmed.use_perm, "use_perm does not match"
        assert kmed.cache_width == 60, "cache_width does not match"
        assert not kmed.parallelize, "parallelize does not match"

    def test_initialization_3(self):
        kmed = KMedoids(
            n_medoids=100,
            algorithm="PAM",
            max_iter=10000,
            build_confidence=11000,
            swap_confidence=2029,
            use_cache=True,
            use_perm=False,
            cache_width=60000,
            parallelize=True,
        )
        assert kmed.n_medoids == 100, "n_medoids does not match"
        assert kmed.algorithm == "PAM", "algorithm does not match"
        assert kmed.max_iter == 10000, "max_iter does not match"
        assert (
            kmed.build_confidence == 11000
        ), "build_confidence does not match"
        assert kmed.swap_confidence == 2029, "swap_confidence does not match"
        assert kmed.use_cache, "use_cache does not match"
        assert not kmed.use_perm, "use_perm does not match"
        assert kmed.cache_width == 60000, "cache_width does not match"
        assert kmed.parallelize, "parallelize does not match"


if __name__ == "__main__":
    unittest.main()
