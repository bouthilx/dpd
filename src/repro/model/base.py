import os

import torch

from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.model', __file__)


CHECKPOINT_FILE_TEMPLATE = os.path.join(os.environ['REPRO_CHECKPOINT_DIR'], '{task_id}')
TMP_CHECKPOINT_FILE_TEMPLATE = CHECKPOINT_FILE_TEMPLATE + '.tmp'


def build_model(name=None, **kwargs):
    return factories[name](**kwargs)


def save_checkpoint(mahler_client, model, optimizer, lr_scheduler, **metadata):
    task = mahler_client.get_current_task()
    if task is None:
        print("Not running with mahler, no ID to identify artifacts")
        return None

    state_dict = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        lr_scheduler=lr_scheduler.state_dict() if lr_scheduler else None,
        metadata=metadata)

    tmp_file_path = TMP_CHECKPOINT_FILE_TEMPLATE.format(task_id=str(task.id))
    file_path = CHECKPOINT_FILE_TEMPLATE.format(task_id=str(task.id))

    if not os.path.isdir(os.path.dirname(tmp_file_path)):
        os.makedirs(os.path.dirname(tmp_file_path))

    with open(tmp_file_path, 'wb') as f:
        state_dict = torch.save(state_dict, f)

    os.rename(tmp_file_path, file_path)


def load_checkpoint(mahler_client, model, optimizer, lr_scheduler):
    task = mahler_client.get_current_task()
    if task is None:
        print("Not running with mahler, no ID to identify artifacts")
        return None

    file_path = CHECKPOINT_FILE_TEMPLATE.format(task_id=str(task.id))

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'rb') as f:
        state_dict = torch.load(f)

    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    if lr_scheduler:
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    return state_dict['metadata']


def clear_checkpoint(mahler_client):
    task = mahler_client.get_current_task()
    if task is None:
        print("Not running with mahler, no ID to identify artifacts")
        return None

    file_path = CHECKPOINT_FILE_TEMPLATE.format(task_id=str(task.id))

    if os.path.exists(file_path):
        os.remove(file_path)
