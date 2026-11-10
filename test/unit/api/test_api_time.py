from datetime import timedelta

from fedot.api.time import ApiTime


def test_tuning_resources_reserve_final_cv_check_and_refit():
    timer = ApiTime(time_for_automl=1, with_tuning=True)
    timer.assumption_fit_spend_time_single_fold = timedelta(seconds=5)
    timer.assumption_fit_spend_time = timedelta(seconds=15)
    timer.composing_spend_time = timedelta(seconds=10)

    assert timer.determine_resources_for_tuning() == 25


def test_tuning_is_skipped_if_initial_cv_cannot_leave_time_for_search():
    timer = ApiTime(time_for_automl=1, with_tuning=True)
    timer.assumption_fit_spend_time_single_fold = timedelta(seconds=10)
    timer.assumption_fit_spend_time = timedelta(seconds=30)

    assert timer.determine_resources_for_tuning() == 10
    assert not timer.have_time_for_tuning()


def test_expired_tuning_budget_stays_negative():
    timer = ApiTime(time_for_automl=0.75, with_tuning=True)
    timer.assumption_fit_spend_time_single_fold = timedelta(seconds=12)
    timer.assumption_fit_spend_time = timedelta(seconds=36)

    assert timer.determine_resources_for_tuning() == -15
    assert not timer.have_time_for_tuning()


def test_tuning_is_allowed_when_cv_and_minimum_search_time_fit_budget():
    timer = ApiTime(time_for_automl=2, with_tuning=True)
    timer.assumption_fit_spend_time_single_fold = timedelta(seconds=5)
    timer.assumption_fit_spend_time = timedelta(seconds=15)

    assert timer.determine_resources_for_tuning() == 95
    assert timer.have_time_for_tuning()


def test_tuning_is_allowed_without_automl_timeout():
    timer = ApiTime(time_for_automl=None, with_tuning=True)

    assert timer.have_time_for_tuning()
