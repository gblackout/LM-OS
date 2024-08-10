import signal
from typing import Callable
import os
from os.path import join as joinpath
import logging
import sys
import datetime
import json
import torch


PRED_L, PRED_R = '\033[91m', '\033[00m'
PGREEN_L, PGREEN_R = '\033[92m', '\033[00m'
PYELLOW_L, PYELLOW_R = '\033[93m', '\033[00m'
PLIGHTPURPLE_L, PLIGHTPURPLE_R = '\033[94m', '\033[00m'
PPURPLE_L, PPURPLE_R = '\033[95m', '\033[00m'
PCYAN_L, PCYAN_R = '\033[96m', '\033[00m'


def jlload(fp):
    res = []
    with open(fp, 'r') as f:
        for line in f:
            res.append(json.loads(line))
    return res


def jldump(data, fp):
    with open(fp, 'w') as f:
        for e in data:
            f.write(json.dumps(e) + '\n')


def jload(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def jdump(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def all_exists(*args):
    return all(e is not None for e in args)


def any_exists(*args):
    return any(e is not None for e in args)


class FuncTimeOutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise FuncTimeOutError("Function timed out")


def has_same_obj_in_list(obj, ls):
    return any(obj is e for e in ls)


def wrap_function_with_timeout(func: Callable, timeout: int):
    def wrapped_function(*args, **kwargs):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Reset the alarm
        except FuncTimeOutError:
            return None
        return result

    return wrapped_function


def make_parent_dirs(fp: str):
    parts = fp.split('/')
    if len(parts) == 1:
        return

    parent_dir = joinpath(*parts[:-1])
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)


def split_string_with_separators(full_str, sep_ls):
    res = {}

    uid_ls = [sep + str(ind) for ind, sep in enumerate(sep_ls)]
    uid2sep = dict([(sep+str(ind), sep) for ind, sep in enumerate(sep_ls)])

    # first we iteratively find the start ind of the first occurrence of the sep
    start = 0
    for sep_ind, sep_uid in enumerate(uid_ls):
        first_sep_start = full_str.find(uid2sep[sep_uid], start)

        if first_sep_start != -1:
            sub_str_start = first_sep_start + len(uid2sep[sep_uid])
            res[sep_uid] = sub_str_start
            start = sub_str_start
            # print(res, sub_str_start, start)
        else:
            res[sep_uid] = None
    # print(res)
    # once we find all the substring starts, we figure out the substring ends
    for sep_ind, sep_uid in enumerate(uid_ls):
        sub_str_start = res[sep_uid]
        if sub_str_start is None:
            res[sep_uid] = [None, None, None]
            continue

        # the sub_str_end is essentially the sub_str_start of the next nonempty sep minus the sep length
        sub_str_end = len(full_str)
        for next_sep_ind, next_sep_uid in enumerate(uid_ls[sep_ind + 1:]):
            next_sub_str_start = res[next_sep_uid]
            if next_sub_str_start is not None:
                sub_str_end = next_sub_str_start - len(uid2sep[next_sep_uid])
                break

        res[sep_uid] = [sub_str_start, sub_str_end, full_str[sub_str_start:sub_str_end]]

    return [[uid2sep[sep_uid], *res[sep_uid]] for sep_uid in uid_ls]


def prep_logging(log_path):
    logging.basicConfig(
        handlers=[
            logging.FileHandler(joinpath(log_path, 'run_log.txt')),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def make_output_dir(output_dir: str):
    assert all_exists(output_dir)
    datetime_str = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    output_dir += datetime_str
    os.makedirs(output_dir)
    return output_dir


def mem_stats():
    a, ma, r, mr = \
        torch.cuda.memory_allocated() / 1024 / 1024, \
        torch.cuda.max_memory_allocated() / 1024 / 1024, \
        torch.cuda.memory_reserved() / 1024 / 1024, \
        torch.cuda.max_memory_reserved() / 1024 / 1024

    log_str = \
        f'Allocated {a:.0f} MB   ' \
        f'Max Allocated {ma:.0f} MB\n' \
        f'Reserved {r:.0f} MB   ' \
        f'Max Reserved {mr:.0f} MB\n'\

    print(log_str)


def readline_stripped(fp: str):
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            yield line