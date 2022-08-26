import multiprocessing
import pathlib
import time
from contextlib import closing
from functools import partial
from logging.handlers import QueueHandler
from multiprocessing import Pool
from random import random

from fedot.core.log import Log, default_log


def worker_configurer(shared_q: multiprocessing.Queue, name: str):
    h = QueueHandler(shared_q)
    default_log(prefix=name).logger.addHandler(h)


def write_to_log(message: str, shared_q: multiprocessing.Queue):
    name = multiprocessing.current_process().name
    worker_configurer(shared_q, name)
    time.sleep(random())
    default_log(prefix=name).info(message)


# def preinit(q: multiprocessing.Queue):  # Needed for the alternate solution with globals...too dirty
#     global shared_q
#     shared_q = q


if __name__ == '__main__':
    messages = [f'PRINT {i}' for i in range(100)]

    processes = []
    with Log.using_mp() as shared_q:
        with closing(Pool(4)) as p:
            list(p.imap_unordered(partial(write_to_log, shared_q=shared_q), messages))
        # with closing(Pool(4, preinit, [queue])) as p:
        #     list(p.imap_unordered(write_to_log, messages))
    # with closing(Pool(4)) as p:
    #     list(p.imap_unordered(write_to_log, messages))

    # check if every message was written to the log
    log_path = Log('random').log_file
    content = pathlib.Path(log_path).read_text()
    all_msgs = True
    for mes in messages:
        if mes not in content:
            all_msgs = False
            print(f'{mes} NOT IN CONTENT')
    if all_msgs:
        print('OK')
    else:
        print('NOT OK')
