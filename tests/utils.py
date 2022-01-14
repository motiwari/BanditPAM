from banditpam import KMedoids
import numpy as np


# TODO(@motiwari): change pam to fp1 everywhere
def bpam_agrees_pam(
    k: int,
    data: np.array,
    loss: str,
    test_build: bool = False,
    assert_immediately: bool = False,
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
    kmed_bpam = KMedoids(n_medoids=k, algorithm="BanditPAM")
    kmed_fp1 = KMedoids(n_medoids=k, algorithm="FastPAM1")
    kmed_bpam.fit(data, loss)
    kmed_fp1.fit(data, loss)

    bpam_build_medoids = sorted(kmed_bpam.build_medoids.tolist())
    pam_build_medoids = sorted(kmed_fp1.build_medoids.tolist())

    bpam_final_medoids = sorted(kmed_bpam.medoids.tolist())
    pam_final_medoids = sorted(kmed_fp1.medoids.tolist())

    bpam_and_pam_agree = 1 if bpam_final_medoids == pam_final_medoids else 0
    if test_build:
        bpam_and_pam_agree &= (bpam_build_medoids == pam_build_medoids)

    if assert_immediately:
        error_message = ''.join(map(str, [
                                        "BanditPAM and FastPAM1 disagree!",
                                        "\nBanditPAM build medoids:",
                                        bpam_build_medoids,
                                        "\nFastPAM1 build medoids:",
                                        pam_build_medoids,
                                        "\nBanditPAM final medoids:",
                                        bpam_final_medoids,
                                        "\nFastPAM1 final medoids",
                                        pam_final_medoids,
                                        ]
                                    )
                                )
        assert bpam_and_pam_agree, error_message

    return bpam_and_pam_agree
