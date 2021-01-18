from fedot.core.composer.timer import CompositionTimer, TunerTimer
import time
import datetime


def test_composition_timer():
    generation_num = 100
    reached = False
    start = datetime.datetime.now()
    with CompositionTimer(verbose=True) as timer:
        for generation in range(generation_num):
            time.sleep(1)
            if timer.is_time_limit_reached(max_lead_time=datetime.timedelta(minutes=0.01),
                                           generation_num=generation):
                reached = True
                break

    spent_time = (datetime.datetime.now() - start).seconds
    assert reached and spent_time == 1


def test_tuner_timer():
    iter_number = 100
    time_limit = datetime.timedelta(minutes=0.01)
    start = datetime.datetime.now()
    reached = False
    with TunerTimer(verbose=True) as timer:
        for iteration in range(iter_number):
            time.sleep(1)
            if timer.is_time_limit_reached(limit=time_limit):
                reached = True
                break

    spent_time = (datetime.datetime.now() - start).seconds
    assert reached and spent_time == 1
