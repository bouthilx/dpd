import asyncio
import argparse
import os
import pprint
import hashlib

from kleio.core.io.trial_builder import TrialBuilder
from kleio.core.trial import status
from kleio.core.utils import sorteddict, flatten, unflatten

import yaml

from sgdad.utils.commandline import execute


# add job-name to submissions, something like {command}{experiment}{dataset}{model}{version}

####
# Save execution
###

# Looks like main difference is the config path of script template and its options.

KLEIO_TEMPLATE = """\
kleio save --branch-original --config /config/kleio.core/kleio_config.yaml \
--tags {tags}"""

EXECUTION_SCRIPT_TEMPLATE = "python3.6 /repos/sgd-space/src/sgdad/train.py --config={file_path}"

REGISTER_COMMANDLINE_TEMPLATE = "{kleio} {script}"

####
# Save analysis
####

# Note: When computing analysis, config passed in configs/some/path/config.yaml
#       Prepend /repos/sgd-space/ before running the save script.

ANALYSIS_SCRIPT_TEMPLATE = (
    "python3.6 /repos/sgd-space/src/sgdad/analyze.py --config={file_path} --trial-id {trial_id}")

####
# Submit identical for execution and analysis, only differs for tags, array and job-name
####

FLOW_OPTIONS_TEMPLATE = "{array}mem=30000M;time=2:59:00;job-name={job_name}"

FLOW_TEMPLATE = "flow-submit {container} --config {file_path} --options {options}{optionals}"

SUBMIT_KLEIO_TEMPLATE = """\
kleio run --allow-host-change \
--config /config/kleio.core/kleio_config.yaml \
--tags {tags}\
"""

SUBMIT_COMMANDLINE_TEMPLATE = "{flow} launch {kleio}"

CONTAINER_KLEIO_CONFIG = '/config/kleio.core/kleio_config.yaml'

SUBMISSION_FILE_TEMPLATE = "{model}.{version}.sh"


def assert_env(name):
    if name not in os.environ:
        raise RuntimeError("Environement variable ${} is not set.".format(name))


assert_env('SGD_SPACE_SUBMISSION_DIR')
assert_env('SGD_SPACE_HASH_DIR')

SUBMISSION_ROOT = os.environ['SGD_SPACE_SUBMISSION_DIR']
HASH_DIR = os.environ['SGD_SPACE_HASH_DIR']
CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, 'configs')


def build_database():
    if os.path.exists(CONTAINER_KLEIO_CONFIG):
        with open(CONTAINER_KLEIO_CONFIG, 'r') as f:
            return TrialBuilder().build_database({'config': f})

    # Rely on local config
    return TrialBuilder().build_database({})


database = build_database()


def build_parser():
    parser = argparse.ArgumentParser(description='Script to train a model')

    subparsers = parser.add_subparsers(title='subcommands', dest='command')

    build_registration_parser(subparsers)
    build_scheduling_parser(subparsers)
    return parser


def build_registration_parser(subparsers):
    registration_subparser = subparsers.add_parser('register')

    add_common_arguments(registration_subparser)

    registration_subparser.set_defaults(func=register)


def build_scheduling_parser(subparsers):
    scheduling_subparser = subparsers.add_parser('schedule')

    add_common_arguments(scheduling_subparser)

    scheduling_subparser.add_argument(
        '--container', required=True,
        help='Container to use for the execution')

    scheduling_subparser.set_defaults(func=schedule)


def add_common_arguments(parser):
    parser.add_argument(
        'type', choices=['experiments', 'analyses'],
        help='Type of the execution.')

    parser.add_argument(
        '--config',
        help='Configuration file path for the analyses')

    parser.add_argument(
        '--execution-version',
        help='Version of the execution to analyze')

    parser.add_argument(
        '--experiment', choices=fetch_experiments(),
        required=True, help='Name of the experiment')

    parser.add_argument(
        '--datasets', choices=fetch_all_datasets(),
        nargs="*", help='Datasets to use. Defaults to all available.')

    parser.add_argument(
        '--models', choices=fetch_all_models(),
        nargs="*", help='Models to use. Defaults to all available.')

    parser.add_argument('--version', required=True,
                        help='Version of the trial (execution or analysis)')

    parser.add_argument('--tags', nargs="*", default=[], help='Additional tags for the trials')

    parser.add_argument('--values', type=str, default="",
                        help='Values to overwrite in configurations')

    # TODO remove print_only, and turn it into a test for kleio, if not using
    # kleio to register this execution, then print-only
    parser.add_argument('--print-only', action='store_true',
                        help='Print executions but do not execute.')

    parser.add_argument('--generate-only', action='store_true',
                        help='Generate sbatch scripts but do not submit.')


def get_dataset_dir(experiment, dataset, root=CONFIG_ROOT):
    return os.path.join(root, experiment, 'zoo', dataset)


def get_config_file_path(experiment, dataset, model, root=CONFIG_ROOT):
    return os.path.join(root, experiment, 'zoo', dataset, model + ".yaml")


def get_submission_script_path(experiment, dataset, model, version):
    submission_dir = get_dataset_dir(experiment, dataset, root=SUBMISSION_ROOT)
    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)
    submission_file_name = SUBMISSION_FILE_TEMPLATE.format(model=model, version=version)
    return os.path.join(submission_dir, submission_file_name)


def fetch_experiments():
    return os.listdir(CONFIG_ROOT)


def fetch_all_datasets():
    datasets = set()
    for experiment in fetch_experiments():
        for dataset in fetch_dataset_dirs(experiment):
            datasets.add(dataset)

    return list(sorted(datasets))


def fetch_all_models():
    models = set()
    for experiment in fetch_experiments():
        for dataset in fetch_dataset_dirs(experiment):
            for model in fetch_models(experiment, dataset, []):
                models.add(model)

    return list(sorted(models))


def fetch_dataset_dirs(experiment=None):
    return os.listdir(os.path.join(CONFIG_ROOT, experiment, 'zoo'))


def fetch_models(experiment, dataset, models):
    for model in os.listdir(get_dataset_dir(experiment, dataset)):
        if model.split(".")[-1] != "yaml":
            continue

        model = model[:-5]

        if models and model not in models:
            continue

        yield model


def fetch_datasets(experiment, datasets):
    for dataset in fetch_dataset_dirs(experiment):
        if not os.path.isdir(get_dataset_dir(experiment, dataset)):
            continue

        if datasets and dataset not in datasets:
            continue

        yield dataset


def get_submissions_scripts(datasets, models, experiment, version):
    # Note: It only creates a blank file
    for dataset, model, _ in get_configs(datasets, models, experiment):

        file_path = get_submission_script_path(experiment, dataset, model, version)

        yield dataset, model, file_path


def get_configs(datasets, models, experiment):

    for dataset in fetch_datasets(experiment, datasets):

        for model in fetch_models(experiment, dataset, models):

            file_path = get_config_file_path(experiment, dataset, model)

            yield dataset, model, file_path


def execute_experiments(args):
    # Save executions and then deploy them
    execute_commandlines(register_executions(args), args)
    execute_commandlines(submit(args), args)


def analyze_experiments(args):
    # Save analyses and then deploy them
    execute_commandlines(register_analyses(args))
    execute_commandlines(submit(args))


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


def register(args):
    if args.type == 'experiments':
        execute_commandlines(register_executions(args), args)
    elif args.type == 'analyses':
        execute_commandlines(register_analyses(args), args)
    else:
        raise SystemExit("Job type not supported: {}".format(args.type))


def schedule(args):
    if args.type in ['experiments', 'analyses']:
        execute_commandlines(submit(args), args)
    else:
        raise SystemExit("Job type not supported: {}".format(args.type))


def execute_commandlines(commandline_builder, args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    futures = []
    for commandline in commandline_builder:
        futures.append(execute(commandline, print_only=args.print_only))
        if len(futures) > 15:
            loop.run_until_complete(asyncio.gather(*futures))
            futures = []

    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()


def verify_configurations(configurations):
    are_lists = [isinstance(value, (list, tuple)) for value in configurations.values()]
    are_all_lists = are_lists and all(are_lists)

    if are_all_lists:
        lengths = [len(value) for value in configurations.values()]
        length = lengths[0]
        have_diff_lengths = all(i_length != length for i_length in lengths)
    else:
        length = None
        have_diff_lengths = False

    if any(are_lists) and (not all(are_lists) or have_diff_lengths):
        raise ValueError("Values must either all be lists of the same length or all "
                         "items.\n{}".format(pprint.pformat(configurations)))

    return are_all_lists, length


def fetch_configurations(values):
    configurations = dict()
    for value in values.split(";"):
        key = value.split("=")[0]
        value = "=".join(value.split("=")[1:])
        try:
            configurations[key] = eval(value)
        except SyntaxError as e:
            print(values)
            raise SyntaxError("Cannot parse '{}'".format(value)) from e

    are_lists, length = verify_configurations(configurations)

    if not are_lists:
        yield configurations

    else:
        for i in range(length):
            yield {key: value[i] for key, value in configurations.items()}


def hash_dict(content):
    string = str(sorteddict(content)).encode('utf-8')
    return hashlib.sha256(string).hexdigest()[:32]


FILE_EXISTS_DIFF_CONFIG_ERROR = """\
File {file_path} already exists with a different content.

File content:
{file_content}

Hash content:
{hash_content}

File content lead to hash code {file_hash_code}.
Hash content lead to hash code {hash_code}.
"""


def convert_path_to_hash_dir(file_path):

    if file_path.startswith(CONFIG_ROOT):
        return file_path.replace(CONFIG_ROOT, os.path.join(HASH_DIR, 'configs'))
    elif file_path.startswith(SUBMISSION_ROOT):
        return file_path.replace(SUBMISSION_ROOT, os.path.join(HASH_DIR, 'submit'))

    raise ValueError("Cannot convert path if not in $CONFIG_ROOT or $SUBMISSION_ROOT")


def create_hashed_file_path(file_path, hash_content):

    file_path = convert_path_to_hash_dir(file_path)

    hash_content = sorteddict(hash_content)

    hash_code = hash_dict(hash_content)

    basefilename, fileext = os.path.splitext(os.path.abspath(file_path))

    hashed_file_path = "{file_path}.{hash_code}{ext}".format(
        file_path=basefilename, hash_code=hash_code, ext=fileext)

    if os.path.exists(hashed_file_path):
        with open(hashed_file_path, 'r') as f:
            content = sorteddict(yaml.load(f))

        if str(content) != str(hash_content):

            raise RuntimeError(FILE_EXISTS_DIFF_CONFIG_ERROR.format(
                file_path=hashed_file_path,
                file_content=pprint.pformat(content),
                hash_content=pprint.pformat(hash_content),
                file_hash_code=hash_dict(content),
                hash_code=hash_code))
    else:
        dir_path = os.path.dirname(hashed_file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        with open(hashed_file_path, 'w') as f:
            yaml.dump(hash_content.raw(), f)

    return hashed_file_path


def create_configuration_file(base_file_path, configuration):
    with open(base_file_path, 'r') as f:
        base_configuration = yaml.load(f)

    flattened_full_configuration = flatten(base_configuration)
    flattened_full_configuration.update(flatten(configuration))
    full_configuration = unflatten(flattened_full_configuration)

    file_path = create_hashed_file_path(base_file_path, sorteddict(full_configuration))

    return file_path


def register_executions(args):

    for dataset, model, base_file_path in get_configs(args.datasets, args.models, args.experiment):

        assert_new_version(args, dataset, model)

        tags = [args.version, args.experiment, dataset, model, 'execution'] + args.tags

        for configuration in fetch_configurations(args.values):
            configuration_file_path = create_configuration_file(base_file_path, configuration)
            param_tags = create_param_tags(configuration)
            kleio = KLEIO_TEMPLATE.format(tags=";".join(tags + param_tags))
            script = EXECUTION_SCRIPT_TEMPLATE.format(file_path=configuration_file_path)
            commandline = REGISTER_COMMANDLINE_TEMPLATE.format(kleio=kleio, script=script)
            yield commandline


def register_analyses(args):

    if args.config is None or args.execution_version is None:
        raise SystemExit(
            "deploy.py register analysis: error: the following arguments are required: "
            "--config, --execution-version")

    for dataset, model, base_file_path in get_configs(args.datasets, args.models, args.experiment):

        assert_new_version(args, dataset, model)

        tags = [args.version, args.experiment, dataset, model, 'analysis'] + args.tags

        for trial_id in fetch_completed_trial_ids(args, dataset, model):
            for configuration in fetch_configurations(args.values):
                configuration_file_path = create_configuration_file(args.config, configuration)
                param_tags = create_param_tags(configuration)
                kleio = KLEIO_TEMPLATE.format(tags=";".join(tags + param_tags))
                script = ANALYSIS_SCRIPT_TEMPLATE.format(
                    file_path=configuration_file_path, trial_id=trial_id)
                commandline = REGISTER_COMMANDLINE_TEMPLATE.format(kleio=kleio, script=script)
                yield commandline


def assert_new_version(args, dataset, model):
    tags = [args.version, args.type, args.experiment, dataset, model] + args.tags
    query = {'tags': {'$all': tags}}
    total = database.count('trials.reports', query)
    if total > 0:
        raise ValueError(
            "There is already {count} trials with version {version} for tags {tags}".format(
                count=total, version=args.version, tags=tags[1:]))


def create_param_tags(configuration):
    return ["{}={}".format(key, value) for key, value in flatten(configuration).items()]


def fetch_completed_trial_ids(args, dataset, model):
    tags = [args.execution_version, args.experiment, dataset, model, 'execution']
    query = {
        'tags': {'$all': tags},
        'registry.status': {'$eq': 'completed'}}

    for trial_doc in database.read('trials.reports', query):
        yield trial_doc['_id']


def submit(args):

    iterator = get_submissions_scripts(args.datasets, args.models, args.experiment, args.version)
    for dataset, model, script_path in iterator:
        tags = [args.version, args.type, args.experiment, dataset, model] + args.tags
        query = {
            'tags': {'$all': tags},
            'registry.status': {'$in': status.RESERVABLE}}

        total = database.count('trials.reports', {'tags': {'$all': tags}})
        runnable = database.count('trials.reports', query)

        print()
        print("{:>40} Total:    {:>5}".format(";".join(tags), total))
        print("{:>40} Runnable: {:>5}".format("", runnable))

        if not runnable:
            continue

        array_option = "array=1-{}".format(min(runnable, 10))
        jobname_option = "{type}.{experiment}.{dataset}.{model}.{version}".format(
            type=args.type, experiment=args.experiment, dataset=dataset, model=model,
            version=args.version)

        # To add tags like analysis type (l2-norm, fisher-rao-norm, pr, etc)
        if args.tags:
            jobname_option += "." + ".".join(args.tags)
        options = FLOW_OPTIONS_TEMPLATE.format(array=array_option, job_name=jobname_option)

        # Note: the script_path is a blank file and will be filled by `flow-submit`
        flow = FLOW_TEMPLATE.format(
            file_path=script_path, container=args.container, options=options,
            optionals=" --generate-only" if args.generate_only else "")
        kleio = KLEIO_TEMPLATE.format(
            experiment=args.experiment, dataset=dataset, model=model,
            version=args.version)

        yield SUBMIT_COMMANDLINE_TEMPLATE.format(flow=flow, kleio=kleio)


if __name__ == "__main__":
    main()
