from multiprocessing import Process
from repro.cli.hpop import main
from tempfile import mkdtemp
import json
import math
import time

tmp_dir = mkdtemp()


def make_arguments(experience, workers=1, max_trials=10):
    return [
        'execute',
        '--json-file'       , f'{tmp_dir}/{experience}.json',
        'coco',
        '--problem-ids'     , '1',
        '--dimensions'      , '2',
        '--instances'       , '1',
        '--dispatchers'     , 'stub',
        '--configurators'   , 'random_search', #'bayesopt',
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
    resume_test_arguments = make_arguments(
        'resume_test',
        workers=1,
        max_trials=50
    )

    start = time.time()
    run = Process(target=main, args=(resume_test_arguments,))
    run.start()
    run.join()

    full_runtime = time.time() - start
    target_results = get_results(f'{tmp_dir}/resume_test.json')

    # Start a Process that we are going to kill
    run_kill = Process(target=main, args=(resume_test_arguments,))
    run.start()

    time.sleep(full_runtime / 2)
    run_kill.kill()

    # Resume previous job
    run_resumed = Process(target=main, args=(resume_test_arguments,))
    run_resumed.start()
    run_resumed.join()

    resumed_results = get_results(f'{tmp_dir}/resume_test.json')

    for (k, v), (k2, v2) in zip(target_results.items(), resumed_results.items()):
        print(k, k2, v, v2)


if __name__ == '__main__':
    # simple_repro_test_all
    resume_test()
