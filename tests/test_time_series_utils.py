from pycaret.utils.time_series import remove_harmonics_from_sp


def test_harmonic_removal():
    """Tests the removal of harmonics"""

    # 1.0 No harmonics removed ----
    results = remove_harmonics_from_sp([2, 51, 5])
    assert results == [2, 51, 5]

    # 2.0 One base frequency removed ----
    results = remove_harmonics_from_sp([2, 52, 3])
    assert results == [52, 3]

    # 3.0 Remove more than one base period ----
    results = remove_harmonics_from_sp([50, 3, 11, 100, 39])
    assert results == [11, 100, 39]

    # 4.0 Order of replacement ----

    # 4.1 Only one removed
    # 4.1A Ordered by raw strength
    results = remove_harmonics_from_sp([2, 3, 4, 50])
    assert results == [3, 4, 50]
    # 4.1B Ordered by harmonic max
    results = remove_harmonics_from_sp(
        [2, 3, 4, 50], harmonic_order_method="harmonic_max"
    )
    assert results == [50, 3, 4]
    # 4.1C Ordered by harmonic strength
    results = remove_harmonics_from_sp(
        [2, 3, 4, 50], harmonic_order_method="harmonic_strength"
    )
    assert results == [4, 3, 50]

    # 4.2 More than one removed
    # 4.2A Ordered by raw strength
    results = remove_harmonics_from_sp([3, 2, 6, 50])
    assert results == [6, 50]
    # 4.2B Ordered by harmonic max
    results = remove_harmonics_from_sp(
        [3, 2, 6, 50], harmonic_order_method="harmonic_max"
    )
    assert results == [6, 50]
    results = remove_harmonics_from_sp(
        [2, 3, 6, 50], harmonic_order_method="harmonic_max"
    )
    assert results == [50, 6]
    # 4.2C Ordered by harmonic strength
    results = remove_harmonics_from_sp(
        [3, 2, 6, 50], harmonic_order_method="harmonic_strength"
    )
    assert results == [6, 50]
    results = remove_harmonics_from_sp(
        [2, 3, 6, 50], harmonic_order_method="harmonic_strength"
    )
    assert results == [6, 50]

    # 4.2D Other variants
    results = remove_harmonics_from_sp(
        [10, 20, 30, 40, 50, 60], harmonic_order_method="harmonic_strength"
    )
    assert results == [20, 40, 60, 50]
    results = remove_harmonics_from_sp(
        [10, 20, 30, 40, 50, 60], harmonic_order_method="harmonic_max"
    )
    assert results == [60, 40, 50]

    # 5.0 These were giving precision issues earlier. Now fixed by rounding internally. ----

    # 5.1
    results = remove_harmonics_from_sp([50, 100, 150, 49, 200, 51, 23, 27, 10, 250])
    assert results == [150, 49, 200, 51, 23, 27, 250]

    # 5.2
    results = remove_harmonics_from_sp([49, 98, 18])
    assert results == [98, 18]

    # 5.3
    results = remove_harmonics_from_sp([50, 16, 15, 17, 34, 2, 33, 49, 18, 100, 32])
    assert results == [15, 34, 33, 49, 18, 100, 32]
