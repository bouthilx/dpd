from multiprocessing import Process

try:
    import mahler
except ImportError:
    mahler = None

from repro.hpo.resource.builtin import ResourceManager


# NOTE: When submitting sections with mahler, need to take into account how many workers each of
#       them has.
#       TODO: Evaluate number of workers per sections based on resource[usage]
# TODO: Support local for debugging and multi-cluster for deployment.
class MahlerResourceManager(ResourceManager):

    def __init__(self, workers, resources, container, tags):
        super(MahlerResourceManager, self).__init__(workers, resources)
        self.container = container
        self.tags = tags

    def start(self):
        print(f'mahler -v execute --tags {" ".join(self.tags)} --num-workers {self.workers}')
       
        for host in self.hosts:
            scheduler = Scheduler(host)
            scheduler.start()
            self.schedulers.append(schedulers)

    def stop(self):
        for scheduler in self.schedulers:
            scheduler.stop()

            # Duplicate workers on all clusters
            # Count co-workers and dedicated workers
            
            # lauch in parallel a resources manager for each cluster. They will duplicate
            # everything.


class Scheduler(Process):
    def __init__(self, user, host, workers, max_workers, prolog, 
                 submission_root,
                 monitoring_interval=300):
        self.user = user
        self.host = host
        self.workers = workers
        self.max_workers = max_workers
        self.submission_root = submission_root
        self.monitoring_interval = monitoring_interval

    def init(self):
        """
        """
        logger.info(f'Initiating ssh connection for {self.user}@{self.host}')
        self.connection = Connection(self.host, user=self.user)
        logger.info(f'Pulling {self.container} on logging node of {self.host}')
        out = self.connection.run('sregistry pull {}'.format(self.container), hide=True, warn=True)
        logger.info(out.stdout)
        logger.debug(out.stderr)
        logger.info(f'Pulling finished on {self.host}')

    def status(self):
        """
        """
        command = 'squeue -r -o "%t %j" -u {user}'.format(user=getpass.getuser())
        workers = 
        jobs = {}
        max_workers = 0
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
                states[line] = 0

            states[line] += 1

            if name == self.name and state not in workers:
                workers[state] = 0

            if name == self.name:
                workers[state] += 1

        return workers, resources

    def submit(self, n_jobs):
        tags = ['worker']
        pass

    def run(self):
        while True:
            workers, resources = self.status()
            
            logger.info(f'Host {self.host} status:')
            logger.info(f'    workers:')
            logger.info(pprint.pformat(workers))
            logger.info(f'    resources:')
            logger.info(pprint.pformat(resources))

            n_tasks = max(self.workers - sum(workers.values()), 0)
            available = max(sum(resources.values()) - self.max_workers, 0)
            n_tasks = min(n_tasks, available)

            if n_tasks:
                logger.info(f'Submitting {n_tasks} on {self.host}')
                self.submit(n_tasks)

            logger.debug(f'Waiting {self.monitoring_interval} seconds')
            time.sleep(self.monitoring_interval)

    def stop_gracefully(self):
        # scancel the dedicated workers?
        # maybe not, they can keep running other tasks
        # But, the manager should make sure to stop all trials prior leaving.
        pass

    def submit_single_host(self, n_workers):

        array_option = 'array=1-{};'.format(n_workers)
        flow_options = FLOW_OPTIONS_TEMPLATE.format(
            array=array_option, job_name=self.name)

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

        if self.submission_root is None:
            raise ValueError(
                "submission_root is not defined for host {} nor globally.".format(host_name))
        submission_dir = os.path.join(self.submission_root, container)
        # TODO: Run mkdirs -p with connection.run instead of python's `os`.
        #       this folder should be created in _ensure_remote_setup.
        if not os.path.isdir(submission_dir):
            connection.run('mkdir -p {}'.format(submission_dir))

        flow_command = FLOW_TEMPLATE.format(
            container=container, root_dir=submission_dir, prolog=self.prolog, options=flow_options)

        options = {}
        if self.num_workers:
            options['num-workers'] = self.num_workers

        if options:
            options = ' ' + ' '.join('--{}={}'.format(k, v) for k, v in options.items())
        else:
            options = ''

        command = COMMAND_TEMPLATE.format(
            container=" --container " + self.container,
            tags=" --tags " + " ".join(self.tags),
            options=options)

        submit_command = SUBMIT_COMMANDLINE_TEMPLATE.format(flow=flow_command, command=command)

        logger.info(f"Executing on {self.host}:")
        logger.info(submit_command)
        out = self.connection.run(submit_command, hide=True, warn=True)
        logger.info("Command output:")
        logger.info(out.stdout)
        logger.info(out.stderr)


if mahler:
    def build(workers, resources={}, container=None, tags=tuple()):
        return MahlerResourceManager(workers, resources, container=container, tags=tags)
