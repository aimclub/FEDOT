import logging
import multiprocessing
import pathlib
import time
from random import random

from joblib import Parallel, delayed

from fedot.core.log import Log, default_log


def write_to_log(message: str, shared_q: multiprocessing.Queue = None, log_lvl: int = None):
    Log().reset_logging_level(log_lvl)
    with Log.using_mp_worker(shared_q):
        time.sleep(random())
        default_log().info(message)


# def preinit(q: multiprocessing.Queue):  # Needed for the alternate solution with globals...too dirty
#     global shared_q
#     shared_q = q


if __name__ == '__main__':
    messages = [f'PRINT {i}' for i in range(20)]
    n_jobs = 4
    log = Log(output_logging_level=logging.INFO)

    # with Log.using_mp_listener() as shared_q:
    #     with closing(Pool(4)) as p:
    #         list(p.imap_unordered(partial(write_to_log, shared_q=shared_q, log_lvl=log.logger.level), messages))
    parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch="2*n_jobs")
    with Log.using_mp_listener() as shared_q:
        eval_inds = parallel(delayed(write_to_log)(message=msg, shared_q=shared_q, log_lvl=log.logger.level)
                             for msg in messages)

    # check if every message was written to the log
    log_path = Log().log_file
    content = pathlib.Path(log_path).read_text().splitlines()
    for msg in messages:
        exist = False
        for line in content:
            if line and line.endswith(msg):
                exist = True
                break
        if not exist:
            print(f'{msg} NOT IN CONTENT')
