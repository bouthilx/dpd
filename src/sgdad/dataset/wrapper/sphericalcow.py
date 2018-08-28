class SphericalCowIterator(object):
    def __init__(self, iterator):
        self.iterator = iterator

    def __next__(self):
        items = next(self.iterator)

        spherical_cows = []

        for item in items:
            if item.type() == "torch.FloatTensor":
                mean = item.mean(0)
                std = item.std(0)
                spherical_cow = torch.randn_like(item) * std + mean
            elif item.type() == "torch.LongTensor":
                lower_bound = item.min()
                upper_bound = item.max()
                if lower_bound.item() == upper_bound.item() == 0:
                    spherical_cow = torch.zeros_like(item)
                else:
                    spherical_cow = torch.randint_like(item, lower_bound, upper_bound)
            else:
                raise TypeError("Don't know spherical cow species for '{}'".format(item.type()))

            spherical_cows.append(spherical_cow)

        return spherical_cows


class SphericalCow(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __getattr__(self, name):
        if hasattr(self.dataloader, name):
            return getattr(self.dataloader, name)

    def __iter__(self):
        return SphericalCowIterator(iter(self.dataloader))


def build(data, target, level):
    wrapped_data = {}
    wrapped_data.update(data)
    
    for set_name in ['train', 'valid', 'test']:
        wrapped_data[set_name] = SphericalCow(data[set_name])

    return wrapped_data
