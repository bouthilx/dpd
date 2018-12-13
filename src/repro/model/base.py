import io

from kleio.client.logger import kleio_logger

import torch

from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.model', __file__)


def build_model(name=None, **kwargs):
    return factories[name](**kwargs)


def save_checkpoint(model, optimizer, filename, **metadata):
    file_like_object = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), file_like_object)
    file_like_object.seek(0)
    kleio_logger.log_artifact(filename, file_like_object, **metadata)


def load_checkpoint(model, optimizer, filename, query={}, logger=None):
    if logger is None:
        print("No logger to load artifacts")
        logger = kleio_logger
    artifacts = logger.load_artifacts(filename, dict(query))

    artifact = None
    for artifact in artifacts:
        continue

    if artifact is None:
        print("No artifacts found")
        return None

    file_like_object, metadata = artifact
    state_dict = torch.load(file_like_object.download())
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return metadata
