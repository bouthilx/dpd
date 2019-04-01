import pprint

try:
    import mahler.client as mahler
except ImportError:
    mahler = None


class Logger:
    def __init__(self):
        self.task_id = mahler.get_current_task_id() if mahler else None
        self.connect()

    def connect(self):
        if self.task_id is None:
            # Either mahler is not installed, or the task is not executed by a mahler worker.
            print("Not running with mahler, cannot save metrics.")
            self.mahler_client = None
            self.task = None
        else:
            self.mahler_client = mahler.Client()
            self.task = self.mahler_client.find(id=self.task_id)
    
    def close(self):
        if self.mahler_client:
            self.mahler_client.close()
    
    def add_metric(self, stats):
        if self.mahler_client:
            self.mahler_client.add_metric(self.task_id, stats)
        else:
            pprint.pprint(stats)
