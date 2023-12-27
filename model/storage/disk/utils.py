import datetime
import os
import shutil
import sys
from model.data import ModelId


def get_local_miners_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "models")


def get_local_miner_dir(base_dir: str, hotkey: str) -> str:
    return os.path.join(get_local_miners_dir(base_dir), hotkey)


def get_local_model_dir(base_dir: str, hotkey: str, model_id: ModelId) -> str:
    return os.path.join(
        get_local_miner_dir(base_dir, hotkey),
        model_id.namespace + "_" + model_id.name + "_" + model_id.commit,
    )


def get_newest_datetime_under_path(path: str) -> datetime.datetime:
    newest_filetime = sys.maxsize

    # Check to see if any file at any level was modified more recently than the current one.
    for cur_path, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(cur_path, filename)
            mod_time = os.stat(path).st_mtime
            if mod_time < newest_filetime:
                newest_filetime = mod_time

    if newest_filetime == sys.maxsize:
        return datetime.datetime.max

    return datetime.datetime.fromtimestamp(newest_filetime)


def remove_dir_out_of_grace(path: str, grace_period_seconds: int):
    last_modified = get_newest_datetime_under_path(path)
    grace = datetime.timedelta(seconds=grace_period_seconds)

    if last_modified < datetime.datetime.now() - grace:
        shutil.rmtree(path=path, ignore_errors=True)
