import datetime
import time

from fedot.core.pipelines.tuning.timer import TunerTimer


def test_tuner_timer():
    iter_number = 100
    time_limit = datetime.timedelta(minutes=0.01)
    start = datetime.datetime.now()
    reached = False
    with TunerTimer(timeout=time_limit) as timer:
        for _ in range(iter_number):
            time.sleep(1)
            if timer.is_time_limit_reached():
                reached = True
                break

    spent_time = (datetime.datetime.now() - start).seconds
    assert reached and spent_time == 1
