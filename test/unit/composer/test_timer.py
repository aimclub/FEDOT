import datetime
import random
import time

import numpy as np

from fedot.core.composer.timer import CompositionTimer, TunerTimer


def test_composition_timer():
    generation_num = 100
    reached = False
    start = datetime.datetime.now()
    with CompositionTimer(max_lead_time=datetime.timedelta(minutes=0.01)) as timer:
        for generation in range(generation_num):
            time.sleep(1)
            if timer.is_time_limit_reached(generation_num=generation):
                reached = True
                break

    spent_time = (datetime.datetime.now() - start).seconds
    assert reached and spent_time == 1


def test_tuner_timer():
    iter_number = 100
    time_limit = datetime.timedelta(minutes=0.01)
    start = datetime.datetime.now()
    reached = False
    with TunerTimer(max_lead_time=time_limit) as timer:
        for _ in range(iter_number):
            time.sleep(1)
            if timer.is_time_limit_reached():
                reached = True
                break

    spent_time = (datetime.datetime.now() - start).seconds
    assert reached and spent_time == 1
