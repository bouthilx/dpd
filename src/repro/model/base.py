import os
from pathlib import Path

try:
    import mahler.client as mahler
except ImportError:
    mahler = None

import torch

from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.model', __file__)


CHECKPOINT_DIR = os.environ.get('REPRO_CHECKPOINT_DIR', str(Path.home()))
CHECKPOINT_FILE_TEMPLATE = os.path.join(CHECKPOINT_DIR, '{task_id}')
TMP_CHECKPOINT_FILE_TEMPLATE = "{file_path}.tmp"


def get_checkpoint_file_path():
    task_id = None
    if mahler is not None:
        task_id = mahler.get_current_task_id()
    if task_id is None:
        print("Not running with mahler, no ID to create model file path for checkpointing.")
        return None

    return CHECKPOINT_FILE_TEMPLATE.format(task_id=str(task_id))


def build_model(name=None, **kwargs):
    return factories[name](**kwargs)


def save_checkpoint(file_path, model, optimizer, lr_scheduler, **metadata):

    if not file_path:
        return

    state_dict = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        lr_scheduler=lr_scheduler.state_dict() if lr_scheduler else None,
        metadata=metadata)

    tmp_file_path = TMP_CHECKPOINT_FILE_TEMPLATE.format(file_path=file_path)

    if not os.path.isdir(os.path.dirname(tmp_file_path)):
        os.makedirs(os.path.dirname(tmp_file_path))

    with open(tmp_file_path, 'wb') as f:
        state_dict = torch.save(state_dict, f)

    os.rename(tmp_file_path, file_path)


def load_checkpoint(file_path, model, optimizer, lr_scheduler):

    if not file_path or not os.path.exists(file_path):
        return

    with open(file_path, 'rb') as f:
        state_dict = torch.load(f)

    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    if lr_scheduler:
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    return state_dict['metadata']


def clear_checkpoint(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
