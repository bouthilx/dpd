import getpass
import logging
import os
import pprint
import signal
import time
import uuid
from multiprocessing import Process

try:
    from fabric import Connection
except:
    Connection = None

try:
    import mahler.core.resources
except ImportError as e:
    print(e)
    mahler = None


from repro.hpo.resource.builtin import ResourceManager


logger = logging.getLogger(__name__)


FLOW_OPTIONS_TEMPLATE = "{array}time=2:59:00;job-name={job_name}"

FLOW_TEMPLATE = "flow-submit {container} --root {root_dir} --prolog '{prolog}' --options '{options}'"

COMMAND_TEMPLATE = "mahler execute{container}{tags}{options}"

SUBMIT_COMMANDLINE_TEMPLATE = "{flow} launch {command}"


def append_signal(sig, f):

    old = None
    if callable(signal.getsignal(sig)):
        old = signal.getsignal(sig)

    def helper(*args, **kwargs):
        if old is not None:
            old(*args, **kwargs)
        f(*args, **kwargs)

    signal.signal(sig, helper)


# NOTE: When submitting sections with mahler, need to take into account how many workers each of
#       them has.
#       TODO: Evaluate number of workers per sections based on resource[usage]
# TODO: Support local for debugging and multi-cluster for deployment.
class MahlerResourceManager(ResourceManager):

    def __init__(self, workers, resources, container, tags, workers_per_job=10, monitoring_interval=300):
        super(MahlerResourceManager, self).__init__(workers, resources)
        self.id = uuid.uuid4().hex
        self.container = container
        self.tags = tags
        self.workers_per_job = workers_per_job
        self.monitoring_interval = monitoring_interval
        self.schedulers = []
        assert Connection is not None, 'Fabric must be installed'

        mahler.core.resources.load_config(type='remoteflow')
        config = mahler.core.config.scheduler.remoteflow
        self.hosts = config.hosts
        for host in self.hosts.values():
            host.setdefault('max_workers', config.max_workers)
            host.setdefault('submission_root', config.submission_root)

        self.user = getpass.getuser()

        # TODO: Support cpu job only. Force gpu resource until then.
        self.resources['gpu'] = 1

        append_signal(signal.SIGTERM, self.stop_gracefully)

    def run(self):
        self.stop = False

        for host_name, host in self.hosts.items():
            scheduler = Scheduler(self.id, self.user, host_name, self.workers, self.workers_per_job,
                                  self.resources,
                                  host['max_workers'],
                                  host['prolog'], host['submission_root'],
                                  container=self.container,
                                  monitoring_interval=self.monitoring_interval)
            scheduler.start()
            self.schedulers.append(scheduler)

        while not self.stop:
            time.sleep(1)

    def stop_gracefully(self, signum=None, frame=None):
        for scheduler in self.schedulers:
            scheduler.terminate()

        self.stop = True

            # Duplicate workers on all clusters
            # Count co-workers and dedicated workers

            # lauch in parallel a resources manager for each cluster. They will duplicate
            # everything.


class Scheduler(Process):
    def __init__(self, id, user, host, workers, workers_per_job, resources, max_workers, prolog,
                 submission_root, container, monitoring_interval=300):
        super(Scheduler, self).__init__()
        self.id = id
        self.user = user
        self.host = host
        self.workers = workers
        self.workers_per_job = workers_per_job
        self.resources = resources
        self.max_workers = max_workers
        self.prolog = prolog
        self.submission_root = submission_root
        self.container = container
        self.monitoring_interval = monitoring_interval
        self.tags = ['worker']
        append_signal(signal.SIGTERM, self.stop_gracefully)

    def init(self):
        """
        """
        self.stop = False
        logger.info(f'{self.user}@{self.host} Initiating ssh connection')
        self.connection = Connection(self.host, user=self.user)
        logger.info(f'{self.user}@{self.host} Pulling {self.container} on logging node')
        out = self.connection.run('sregistry pull {}'.format(self.container), hide=True, warn=True)
        logger.info(out.stdout)
        logger.debug(out.stderr)
        logger.info(f'{self.user}@{self.host} Pulling completed')

        if self.submission_root is None:
            raise ValueError(f"submission_root is not defined for host {self.host} nor globally.")
        submission_dir = os.path.join(self.submission_root, self.container)
        logger.info(f'{self.user}@{self.host} Creating `{submission_dir}`')
        self.connection.run('mkdir -p {}'.format(submission_dir))

    def status(self):
        """
        """
        command = 'squeue -r -o "%t %j" -u {user}'.format(user=getpass.getuser())
        workers = {}
        logger.debug(f'squeue on {self.host}')
        out = self.connection.run(command, hide=True, warn=True)
        out = out.stdout
        states = dict()
        for line in out.split("\n")[1:]:  # ignore `ST` header
            line = line.strip()
            if not line:
                continue

            state, name = line.split(' ')

            if state == 'CG':
                continue

            if state not in states:
                states[state] = 0

            states[state] += 1

            if name == self.id and state not in workers:
                workers[state] = 0

            if name == self.id:
                workers[state] += 1

        return workers, states

    def run(self):
        self.init()

        while True:
            workers, resources = self.status()

            logger.info(f'{self.user}@{self.host} status:')
            logger.info(f'{self.user}@{self.host}    workers:')
            logger.info(pprint.pformat(workers))
            logger.info(f'{self.user}@{self.host}    resources:')
            logger.info(pprint.pformat(resources))

            n_tasks = max(self.workers - sum(workers.values()), 0)
            available = max(self.max_workers - sum(resources.values()), 0)
            n_tasks = min(n_tasks, available)

            if n_tasks:
                logger.info(f'{self.user}@{self.host} Submitting {n_tasks}')
                self.submit(n_tasks)
            else:
                logger.info(f'{self.user}@{self.host} No task to submit')

            logger.debug(f'{self.user}@{self.host} Waiting {self.monitoring_interval} seconds')
            start = time.time()
            while time.time() - start < self.monitoring_interval and not self.stop:
                time.sleep(1)

            if self.stop:
                logger.info(f'{self.user}@{self.host} Leaving')
                break

    def stop_gracefully(self, signum=None, frame=None):
        # scancel the dedicated workers?
        # maybe not, they can keep running other tasks
        # But, the manager should make sure to stop all trials prior leaving.
        self.stop = True

    def submit(self, n_workers):

        array_option = 'array=1-{};'.format(n_workers)
        flow_options = FLOW_OPTIONS_TEMPLATE.format(
            array=array_option, job_name=self.id)

        resources = []
        for name, value in self.resources.items():
            if name == 'cpu':
                resources.append('cpus-per-task={}'.format(value))
            elif name == 'gpu':
                resources.append('gres=gpu:{}'.format(value))
            elif name == 'mem':
                resources.append('mem={}'.format(value))
            # elif name not in IGNORE_RESOURCES:
            #     raise ValueError('Unknown option: {}'.format(name))

        flow_options += ";" + ";".join(resources)

        submission_dir = os.path.join(self.submission_root, self.container)

        flow_command = FLOW_TEMPLATE.format(
            container=self.container, root_dir=submission_dir, prolog=self.prolog,
            options=flow_options)

        options = {}
        if self.workers_per_job:
            options['num-workers'] = self.workers_per_job

        if options:
            options = ' ' + ' '.join('--{}={}'.format(k, v) for k, v in options.items())
        else:
            options = ''

        command = COMMAND_TEMPLATE.format(
            container=" --container " + self.container,
            tags=" --tags " + " ".join(self.tags),
            options=options)

        submit_command = SUBMIT_COMMANDLINE_TEMPLATE.format(flow=flow_command, command=command)

        logger.info(f"{self.user}@{self.host} Executing '{submit_command}'")
        out = self.connection.run(submit_command, hide=True, warn=True)
        logger.info("Command output:")
        logger.info(out.stdout)
        logger.info(out.stderr)


if mahler:
    def build(workers, operator=None, resources={}, container=None, tags=tuple()):
        full_resource_spec = {}

        if operator is not None:
            full_resource_spec.update(operator.resources)

        full_resource_spec.update(resources)

        return MahlerResourceManager(workers, full_resource_spec, container=container, tags=tags)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    manager = MahlerResourceManager(10, {'cpu': 4, 'gpu': 1, 'mem': '10GB'}, container='test',
                                    tags=['nada'])
