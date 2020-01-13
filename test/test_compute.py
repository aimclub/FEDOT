from operators.compute import sqr


def test_sqr_correct():
    values = [3, 4, 5]
    expected_result = [9, 16, 25]
    actual_result = sqr(values=values)

    assert expected_result == actual_result
