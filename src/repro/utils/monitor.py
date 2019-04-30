import time

from ctypes import c_bool
from typing import Optional, List
from multiprocessing import Process, Value


class MonitorComponent:
    def run(self):
        raise NotImplementedError()

    def shutdown(self):
        pass


class Monitor:
    """ Generic Monitor that runs in parallel"""

    def __init__(self, loop_delay=0.01):
        self.running = Value(c_bool, False)
        self.components: List[MonitorComponent] = []
        self.loop_delay = loop_delay
        self.process: Optional[Process] = None

    def add_component(self, obj: MonitorComponent) -> MonitorComponent:
        if self.running.value:
            raise RuntimeError('Cannot insert a MonitorComponent while the monitor is running!')

        self.components.append(obj)
        return obj

    def run(self) -> None:
        self.process = Process(target=self._run, args=())
        self.running.value = True
        self.process.start()

    def _run(self):
        while self.running.value:
            start = time.time()

            for comp in self.components:
                comp.run()

            # Do not run too fast to not use 100% of the cpu doing nothing
            elapsed = time.time() - start
            if elapsed < self.loop_delay:
                time.sleep(self.loop_delay - elapsed)

    def stop(self, kill=False):
        self.running.value = False
        if self.process is not None:
            if not kill:
                self.process.join()
            else:
                self.process.kill()

            for comp in self.components:
                comp.shutdown()


if __name__ == '__main__':

    m = Monitor()

    m.run()

    m.stop()
