from torch.utils.data.sampler import Sampler

from .sphericalcow import SphericalCow


class InfiniteRandomSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        sampled = set()
        i = 0
        while True:
            for j, sample in enumerate(self.sampler):
                i += 1
                sample_tuple = tuple(sorted(sample))
                if sample_tuple not in sampled:
                    sampled.add(sample_tuple)
                    yield sample
                else:
                    print(i, j, "duplicate")


def extend(loader):
    if isinstance(loader, SphericalCow):
        data_loader = loader.dataloader
    else:
        data_loader = loader

    infinite_loader = data_loader.__class__(
        data_loader.dataset, batch_size=1, shuffle=False, sampler=None,
        batch_sampler=InfiniteRandomSampler(data_loader.batch_sampler), num_workers=0,
        collate_fn=data_loader.collate_fn, pin_memory=data_loader.pin_memory,
        drop_last=data_loader.drop_last, timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn)

    if isinstance(loader, SphericalCow):
        loader.dataloader = infinite_loader
        infinite_loader = loader

    return infinite_loader
    

def build(data):
    wrapped_data = {}
    wrapped_data.update(data)
    for set_name in ['train', 'valid', 'test']:
        wrapped_data[set_name] = extend(data[set_name])

    return wrapped_data
