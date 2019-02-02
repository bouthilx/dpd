import os

import mahler.client as mahler

import torch

from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.model', __file__)


CHECKPOINT_FILE_TEMPLATE = os.path.join(os.environ['REPRO_CHECKPOINT_DIR'], '{task_id}')
TMP_CHECKPOINT_FILE_TEMPLATE = "{file_path}.tmp"


def get_checkpoint_file_path():
    task_id = mahler.get_current_task_id()
    if task_id is None:
        print("Not running with mahler, no ID to create model file path.")
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
