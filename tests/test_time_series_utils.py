from pycaret.utils.time_series import remove_harmonics_from_sp


def test_harmonic_removal():
    """Tests the removal of harmonics"""

    # 1.0 No harmonics removed ----
    results = remove_harmonics_from_sp([2, 51, 5])
    assert results == [2, 51, 5]

    # 2.0 1 base frequency removed ----
    results = remove_harmonics_from_sp([2, 52, 3])
    assert results == [52, 3]

    # 3.0 Remove more than 1 base period ----
    results = remove_harmonics_from_sp([50, 3, 11, 100, 39])
    assert results == [11, 100, 39]

    # 4.0 Order of replacement ----
    # TODO: Should this return [3, 52] or [52, 3]
    # Add an option later to have the user select this.
    results = remove_harmonics_from_sp([2, 3, 52])
    assert results == [3, 52]

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
