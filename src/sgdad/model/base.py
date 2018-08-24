from kleio.client.logger import kleio_logger

import torch

from sgdad.utils.factory import fetch_factories


factories = fetch_factories('sgdad.model', __file__)


def build_model(name, **kwargs):
    return factories[name](**kwargs)
    

def save_model(model, filename, **metadata):
    file_like_object = io.BytesIO()
    torch.save(model.state_dict(), file_like_object)
    file_like_object.seek(0)
    kleio_logger.log_artifact(filename, file_like_object, **metadata)


def load_model(model, filename):
    artifacts = kleio_logger.load_artifacts('weights', {})
    if not artifacts:
        return None
    file_like_object, metadata = artifacts[-1]
    state_dict = torch.load(file_like_object.download())
    model.load_state_dict(state_dict)
    return metadata


