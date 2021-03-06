from multiprocessing import Process
from repro.cli.hpop import main
from repro.utils.tests.compare import compare_dict
from tempfile import mkdtemp
import json
import math
import time
import os

tmp_dir = mkdtemp()


def make_arguments(experience, workers=1, max_trials=10, more_args=None):
    if more_args is None:
        more_args = []

    return [
        '-v',
        'execute',
        '--save-out'       , f'{tmp_dir}/{experience}.json'] + more_args + [
        'coco',
        '--problem-ids'     , '1',
        '--dimensions'      , '2',
        '--instances'       , '1',
        '--dispatchers'     , 'stub',
        '--configurators'   , 'random_search',      # 'bayesopt',
        '--config-dir-path' , 'configs/hpot/hpo/',
        '--max-trials'      , str(max_trials),
        '--workers'         , str(workers),
        '--seeds'           , '1', '2'
    ]


def get_results(file_name):
    return json.load(open(file_name, 'r'))


def simple_repro_test(workers=1):
    objectives = []
    simple_repro_test_arguments = make_arguments(
        'simple_repro_test',
        workers=workers
    )

    for i in range(0, 10):
        try:
            run = Process(target=main, args=(simple_repro_test_arguments,))
            run.start()
            run.join()

            results = get_results(f'{tmp_dir}/simple_repro_test.json')

            #                               I am not lost are you ?
            trials = results['coco,f001,d002,i01']['stub']['random_search']['2']
            mx = 0
            last_objective = ''
            for k, result in trials.items():
                if mx < int(k):
                    mx = int(k)
                    last_objective = result

            last_objective = last_objective[-1]['objective']
            objectives.append(last_objective)
        except Exception as e:
            print(json.dumps(results, indent=2))
            raise e

    target = objectives[0]
    average = (sum(objectives) / len(objectives))

    print('-' * 80)
    print(f'{target} =? {average}')
    assert math.fabs(average - target) < 1e06


def simple_repro_test_all():
    simple_repro_test(workers=1)
    simple_repro_test(workers=2)
    simple_repro_test(workers=4)


def resume_test():
    import shutil
    resume_test_arguments = make_arguments(
        'resume_test',
        workers=8,
        max_trials=100,
        more_args=[
            '--checkpoint', f'{tmp_dir}'
        ]
    )
    print('Running main experiment')
    start = time.time()
    run = Process(target=main, args=(resume_test_arguments,))
    run.start()
    run.join()

    full_runtime = time.time() - start
    print('Main experiment finished reading results')
    target_results = get_results(f'{tmp_dir}/resume_test.json')

    # next should not resume
    shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print('Run Killable experiment')
    print('-' * 80)
    # Start a Process that we are going to kill
    kill_s = time.time()
    run_kill = Process(target=main, args=(resume_test_arguments,))
    run_kill.start()
    time.sleep(full_runtime / 2)
    run_kill.terminate()
    kill_runtime = time.time() - kill_s

    print('Resume experiment')
    print('-' * 80)
    # Resume previous job
    resume_s = time.time()
    run_resumed = Process(target=main, args=(resume_test_arguments,))
    run_resumed.start()
    run_resumed.join()
    resume_runtime = time.time() - resume_s

    resumed_results = get_results(f'{tmp_dir}/resume_test.json')

    print('Compare')
    print('-' * 80)
    #print(target_results)

    experiements = target_results['index'].keys()
    results1 = target_results['results']
    results2 = resumed_results['results']

    for exp in experiements:
        for v1, v2 in zip(results1[exp], results2[exp]):
            for step1, step2 in zip(v1['results'], v2['results']):
                compare_dict(step1, step2)

    print(f'One Shot {full_runtime}, Kill {kill_runtime}, Resume: {resume_runtime}')


if __name__ == '__main__':
    # simple_repro_test_all
    resume_test()
