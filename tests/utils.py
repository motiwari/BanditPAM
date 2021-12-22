from banditpam import KMedoids


def bpam_agrees_pam(k, data, loss, test_build=False, assert_immediately=False):
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
    kmed_naive = KMedoids(n_medoids=k, algorithm="naive")
    kmed_bpam.fit(data, loss)
    kmed_naive.fit(data, loss)

    bpam_build_medoids = sorted(kmed_bpam.build_medoids.tolist())
    pam_build_medoids = sorted(kmed_naive.build_medoids.tolist())

    bpam_final_medoids = sorted(kmed_bpam.medoids.tolist())
    pam_final_medoids = sorted(kmed_naive.medoids.tolist())

    bpam_and_pam_agree = 1 if bpam_final_medoids == pam_final_medoids else 0
    if test_build:
        bpam_and_pam_agree &= (bpam_build_medoids == pam_build_medoids)

    if assert_immediately:
        error_message = ''.join(map(str, [
                                        "BanditPAM and PAM disagree!",
                                        "\nBanditPAM build medoids:",
                                        bpam_build_medoids,
                                        "\nPAM build medoids:",
                                        pam_build_medoids,
                                        "\nBanditPAM final medoids:",
                                        bpam_final_medoids,
                                        "\nPAM final medoids",
                                        pam_final_medoids,
                                        ]
                                    )
                                )
        assert bpam_and_pam_agree, error_message

    return bpam_and_pam_agree
