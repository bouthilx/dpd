from multiprocessing import Process
import uuid


def build(workers, resources={}, **kwargs):
    return ResourceManager(workers, resources)


class ResourceManager(Process):
    def __init__(self, workers, resources):
        super(ResourceManager, self).__init__()
        self.id = uuid.uuid4().hex
        self.workers = workers
        self.resources = resources

    def run(self):
        pass

    def allocate(self):
        pass

    def release(self):
        pass
