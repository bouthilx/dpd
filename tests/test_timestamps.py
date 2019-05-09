from multiprocessing import Process
from repro.cli.hpop import main
from tempfile import mkdtemp
from repro.utils.tests.compare import compare_dict
import json
import math
import time
import os

tmp_dir = mkdtemp()


def make_arguments(experience, workers=1, max_trials=10, more_args=None, version='0',
                   container='none', backend='builtin'):
    if more_args is None:
        more_args = []

    return [
        '-vv',
        'execute',
        '--save-out'       , f'{tmp_dir}/{experience}.json'] + more_args + [
        'curves',
        '--configurators'   , 'random_search',      # 'bayesopt',
        '--dispatcher'      , 'asha',
        '--backend'         , 'builtin',
        '--config-dir-path' , 'configs/hpop',
        '--max-trials'      , str(max_trials),
        '--workers'         , str(workers),
        '--seeds'           , '1',
        '--version'         , version,
        '--backend'         , backend,
        '--container'       , container
    ]


def get_results(file_name):
    return json.load(open(file_name, 'r'))


def resume_test():
    import shutil
    resume_test_arguments = make_arguments(
        'resume_test',
        workers=2,
        max_trials=10,
        more_args=[
            '--checkpoint', f'{tmp_dir}'
        ]
    )
    print('Running main experiment')
    start = time.time()
    main(resume_test_arguments)

    full_runtime = time.time() - start
    print('Main experiment finished reading results')
    target_results = get_results(f'{tmp_dir}/resume_test.json')

    import pdb
    pdb.set_trace()


def resume_test_mahler(container):
    import shutil
    resume_test_arguments = make_arguments(
        'resume_test',
        workers=2,
        max_trials=10,
        backend='mahler',
        version='test-resume-mahler-2',
        container=container,
        more_args=[
            '--checkpoint', f'{tmp_dir}',
        ]
    )
    print(resume_test_arguments)
    print('Running main experiment')
    start = time.time()
    main(resume_test_arguments)

    full_runtime = time.time() - start
    print('Main experiment finished reading results')
    target_results = get_results(f'{tmp_dir}/resume_test.json')

    import pprint
    pprint.pprint(target_results)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    import sys
    # simple_repro_test_all
    # resume_test()
    resume_test_mahler(sys.argv[1])
