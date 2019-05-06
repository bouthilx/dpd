import json
import time
import os

from repro.utils.monitor import MonitorComponent


class CheckPointer(MonitorComponent):

    """ CheckPoint is a class that monitors a set of Resumable Objects and save their state regularly. """

    def __init__(self, every=1, archive_folder='/tmp/checkpoints/repro/'):
        self.every = every
        self.running = False
        self.tracked_objects = []
        self.archive_folder = archive_folder
        self.last_save = 0
        os.makedirs(self.archive_folder, exist_ok=True)

    def checkpoint_this(self, obj, name=None) -> bool:
        if name is None:
            name = type(obj).__name__

        self.tracked_objects.append((name, obj))
        return True

    def _save_obj(self, name, obj):
        from repro.utils.resumable import state

        file_name = f'{self.archive_folder}/{name}.json'
        with open(file_name, 'w') as f:
            json.dump(state(obj), f, indent=2)

    def _save_tracked_objects(self):
        for name, obj in self.tracked_objects:
            self._save_obj(name, obj)

    def save_now(self):
        self._save_tracked_objects()
        self.last_save = time.time()

    def run(self):
        if self.every is None:
            self.save_now()
        else:
            sleep_time = self.every - (time.time() - self.last_save)
            if sleep_time < 0:
                self.save_now()


def load_checkpoint(name):
    return json.load(open(name, 'r'))


def resume_from_checkpoint(obj, name):
    from repro.utils.resumable import resume
    return resume(obj, load_checkpoint(name))
