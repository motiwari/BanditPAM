from banditpam import KMedoids
import numpy as np


# TODO(@motiwari): change pam to alg_name everywhere
def bpam_agrees_pam(
    k: int,
    data: np.array,
    loss: str,
    test_build: bool = False,
    assert_immediately: bool = False,
    use_fp: bool = True,
):
    """
    Parameters:
        k: Number of medoids
        data: Input data to fit
        loss: Loss function to use for clustering
        test_build: Verify whether BanditPAM and PAM's medoids agree
            after the BUILD step
        assert_immediately: Fail immediately if medoids disagree.
            Set to false to test the proportion of passing tests

    Returns:
        bpam_and_pam_agree: 1 if BanditPAM and PAM agree, 0 otherwise
    """
    alg_name = "FastPAM1" if use_fp else "PAM"

    kmed_bpam = KMedoids(n_medoids=k, algorithm="BanditPAM")
    kmed_pam = KMedoids(n_medoids=k, algorithm=alg_name)
    kmed_bpam.fit(data, loss)
    kmed_pam.fit(data, loss)

    bpam_build_medoids = sorted(kmed_bpam.build_medoids.tolist())
    pam_build_medoids = sorted(kmed_pam.build_medoids.tolist())

    bpam_final_medoids = sorted(kmed_bpam.medoids.tolist())
    pam_final_medoids = sorted(kmed_pam.medoids.tolist())

    bpam_and_pam_agree = 1 if bpam_final_medoids == pam_final_medoids else 0
    if test_build:
        bpam_and_pam_agree &= bpam_build_medoids == pam_build_medoids

    if assert_immediately:
        error_message = "".join(
            map(
                str,
                [
                    "BanditPAM and {} disagree!".format(alg_name),
                    "\nBanditPAM build medoids:",
                    bpam_build_medoids,
                    "\n{} build medoids:".format(alg_name),
                    pam_build_medoids,
                    "\nBanditPAM final medoids:",
                    bpam_final_medoids,
                    "\n{} final medoids:".format(alg_name),
                    pam_final_medoids,
                ],
            )
        )
        assert bpam_and_pam_agree, error_message

    return bpam_and_pam_agree
