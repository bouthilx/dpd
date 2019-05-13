import json
import numpy as np

import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as go

from dataclasses import dataclass
import datetime
from typing import List
from collections import OrderedDict
from collections import defaultdict


def merge_reports(filenames: List[str]):
    pass


def read_report(filename):
    data = json.load(open(filename, 'r'))

    return data


# output: {index, results}
# index: {problem_id: List[Tags]}
# results: {problem_id: List[Trials]}
# Trials: {results: List[Steps], timestamps, params, id}


class WorkSegment:
    """ Work Segment represent the time that a worker spent working on a trial before being suspended """
    def __init__(self, trial_id, start, end, observations=None, trial_idhex=None):
        self.trial_id = trial_id
        self.start = start
        self.end = end
        self.observations = observations
        self.trial_idhex = trial_idhex

    def runtime(self):
        return self.end - self.start

    def min_objective(self):
        return min([obs['objective'] for time, obs in self.observations])


class WorkerAllocationHistory:
    """ WorkerAllocationHistory store all the tasks a worker spent its time on """
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.trials = []
        self.prev = 0

    def runtime(self):
        return sum([trial.runtime() for trial in self.trials])

    def _generate_gantt_entry(self, trial):
        entry = dict(
            Task=f'Worker-{self.worker_id}',
            Start=self.prev,
            Finish=self.prev + trial.runtime()
            #Complete=trial.min_objective() * 100
        )

        self.prev = self.prev + trial.runtime()
        return entry

    def generate_gantt(self):
        return [self._generate_gantt_entry(trial) for trial in self.trials]

    def min_objective_ts(self):
        # Trial are already sorted in the correct order
        min_objective = float('inf')
        ts = []
        duration = 0

        for trial in self.trials:
            last_time = trial.start

            for time, obs in trial.observations:
                objective = obs['objective']

                if objective < min_objective:
                    min_objective = objective

                #print(time, last_time, trial.start, trial.trial_idhex)
                assert time > last_time
                duration += time - last_time

                ts.append((duration, min_objective))
                last_time = time

        return ts


class WorkSegmentMinObjectiveIterator:
    def __init__(self, allocations: List[WorkerAllocationHistory]):
        self.objective_ts = [[0, float('inf'), alloc.min_objective_ts()] for alloc in allocations]
        self.time = float('-inf')
        self.index = 0
        self.length = sum([len(obs[2]) for obs in self.objective_ts])

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.index == self.length:
            raise StopIteration

        min_time = float('+inf')
        min_alloc = None

        # find the next observation
        for alloc_index, (index, min_objective, ts) in enumerate(self.objective_ts):
            try:
                ts_time, current_objective = ts[index]

                if ts_time < min_time:
                    min_time = ts_time
                    min_alloc = alloc_index
            except IndexError:
                pass

        assert self.time < min_time
        self.time = min_time

        index, min_objective, ts = self.objective_ts[min_alloc]

        # we are using that obs
        self.objective_ts[min_alloc][0] += 1

        # Check if the new observation is a new min for its alloc
        ts_time, new_objective = ts[index]
        if min_objective > new_objective:
            self.objective_ts[min_alloc][1] = new_objective

        # Check if the new observation is a new min overall
        min_objective = float('+inf')
        for alloc_index, (index, alloc_min_objective, ts) in enumerate(self.objective_ts):
            if alloc_min_objective < min_objective:
                min_objective = alloc_min_objective

        self.index += 1
        return self.time, min_objective


def generate_timeline(trials):
    timeline = OrderedDict()

    for trial in trials:
        timestamps = trial['timestamps']
        # observations
        results = trial['results']

        obs_index = 0

        for (ttype, timestamp) in timestamps:
            # 2019-05-08 20:58:51.984617
            date = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
            time = date.timestamp()

            if time in timeline:
                print('Observation will be overridden')
            else:
                timeline[time] = []

            observation = None
            if ttype == 'observe':
                observation = results[obs_index]
                obs_index += 1

            timeline[time].append((ttype, trial, date, observation))
    return timeline


def generate_work_segments(trials):
    """ Generate a WorkSegment for each Start - Stop pair"""
    work_segments = OrderedDict()

    for tindex, trial in enumerate(trials):
        trial_id = trial['id']
        timestamps = trial['timestamps']
        convert = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        timestamps = list(map(lambda x: (x[0], convert(x[1])), timestamps))
        timestamps.sort(key=lambda x: x[1])

        # observations
        results = trial['results']

        obs_index = 0
        segment_start = 0
        segment_observation = []

        for (ttype, date) in timestamps:
            # 2019-05-08 20:58:51.984617
            time = date.timestamp()

            if ttype == 'start':
                segment_start = time
                segment_observation = []

            elif ttype == 'observe':
                observation = results[obs_index]
                obs_index += 1
                segment_observation.append((time, observation))

            elif ttype == 'stop':
                work_segments[segment_start] = WorkSegment(
                    trial_id=tindex,
                    start=segment_start,
                    end=time,
                    observations=segment_observation,
                    trial_idhex=trial_id
                )
                segment_start = 0
                segment_observation = []

    return work_segments


def allocate_work_segments(work_segments, worker_count):
    workers_allocs = [WorkerAllocationHistory(i) for i in range(worker_count)]

    for time, segment in work_segments.items():
        # find the worker with the least amount of work
        min_time = float('inf')
        min_worker = None

        # ---
        for worker in workers_allocs:
            wtime = worker.runtime()
            if min_time > wtime:
                min_time = wtime
                min_worker = worker
        # ---

        min_worker.trials.append(segment)

    return workers_allocs


def make_gantt(workers_allocs):
    gantt_entries = []
    for worker in workers_allocs:
        gantt_entries.extend(worker.generate_gantt())

    # index_col='Complete',
    fig = ff.create_gantt(gantt_entries, group_tasks=True, show_colorbar=True)
    py.offline.plot(fig, filename='gantt-group-tasks-together')


@dataclass
class AggregatedCurve:
    curves: List[any]
    min_x: float
    max_x: float

    def __call__(self, x, reduce=np.mean):
        return reduce([curve(x) for curve in self.curves])


def aggregate_objective_curves(data, problem_ids: List[str]):
    """ for a set of problem ids generate a AggregatedCurve"""
    from scipy.interpolate import interp1d

    runs_objective = []
    min_x = float('+inf')
    max_x = float('-inf')

    # Generate the data for each problem
    for problem_id in problem_ids:
        tags = data['index'][problem_id]
        trials = data['results'][problem_id]
        worker_tag = None

        for tag in tags:
            if tag.startswith('w-'):
                worker_tag = tag

        worker_count = int(worker_tag[2:])

        work_segments = generate_work_segments(trials)
        workers_allocs = allocate_work_segments(work_segments, worker_count)

        iterator = WorkSegmentMinObjectiveIterator(workers_allocs)
        cost_evol = list(iterator)

        # We extract each dim
        x = [xy[0] for xy in cost_evol]
        y = [xy[1] for xy in cost_evol]

        # We interpolate so we can compare each run even when timings do not match
        cs = interp1d(x, y, bounds_error=False)
        runs_objective.append(cs)

        min_x = min(min(x), min_x)
        max_x = max(max(x), max_x)

    return AggregatedCurve(
        curves=runs_objective,
        min_x=min_x,
        max_x=max_x
    )


def aggregation_selector(data, select=None, diff_by=None):
    """ Select how problem should be aggregated together by looking at their tags """
    problem_ids = data['index'].keys()

    # index: {
    #   "ef90ce916efb051": [
    #       "42",
    #       "d-asha",
    #       "c-random_search",
    #       "s-1",
    #       "w-8",
    #       "b-curves"
    #   ],
    #   "14a0ddb1ff41630": [
    #       "42",
    #       "d-asha",
    #       "c-random_search",
    #       "s-2",
    #       "w-8",
    #       "b-curves"
    #   ]
    # }
    if select is None:
        select = {
            # select only random_search
            'c-random_search'
        }
    if diff_by is None:
        # generate a curve for each tag
        diff_by = {
            'w'
        }

    # Select the problems that we are interested in
    selected_ids = []

    for problem_id in problem_ids:
        tags = data['index'][problem_id]
        if not select.difference(tags):
            selected_ids.append(problem_id)

    # Generate different groups for each diff tags
    groups = defaultdict(list)

    def find_tag(tag_list, target_tag):
        for tag in tag_list:
            if tag.startswith(target_tag):
                return tag
        return None

    for diff_tag in diff_by:
        for problem_id in problem_ids:
            tags = data['index'][problem_id]
            tag_value = find_tag(tags, diff_tag)

            if tag_value is None:
                print(f'Warning: Problem without (diff_tag: {diff_tag})')
                continue

            groups[tag_value].append(problem_id)

    return groups


def visualize_objective_with_error(data, problem_ids: List[str], points=20):
    """ Aggregate the the problem in `problem_ids` """

    aggregated_curve = aggregate_objective_curves(data, problem_ids)

    # Generate the graph
    x_space = np.linspace(
        start=aggregated_curve.min_x,
        stop=aggregated_curve.max_x,
        num=points,
        dtype=np.float,
        endpoint=True
    )
    y_raw = [aggregated_curve(x, reduce=lambda x: x) for x in x_space]

    y_min  = [np.min(obs)  for obs in y_raw]
    y_mean = [np.mean(obs) for obs in y_raw]
    y_max  = [np.max(obs)  for obs in y_raw]
    y_sd   = [np.std(obs)  for obs in y_raw]

    upper_bound = go.Scatter(
        name='upper',
        x=x_space,
        y=y_max,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
    mean = go.Scatter(
        x=x_space,
        y=y_mean,
        line=dict(color='rgb(0,176,246)'),
        mode='lines',
        name='Objective',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )
    lower_bound = go.Scatter(
        name='lower',
        x=x_space,
        y=y_min,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        showlegend=False
    )

    fig = go.Figure(data=[lower_bound, mean, upper_bound])
    py.plot(fig, filename='shaded_lines')


def visualize_problem(problem_id, data):
    tags = data['index'][problem_id]
    trials = data['results'][problem_id]

    worker_tag = None

    for tag in tags:
        if tag.startswith('w-'):
            worker_tag = tag

    worker_count = int(worker_tag[2:])
    print(worker_count)

    work_segments = generate_work_segments(trials)
    workers_allocs = allocate_work_segments(work_segments, worker_count)
    #
    # ts = workers_allocs[0].min_objective_ts()
    # x = [xy[0] for xy in ts]
    # y = [xy[1] for xy in ts]
    #
    # g = go.Scatter(x=x, y=y, mode='lines', name='objective')
    # py.plot([g], filename='objectives')

    iterator = WorkSegmentMinObjectiveIterator(workers_allocs)
    length = len(iterator)
    print(length)

    cost_evol = list(iterator)

    x = [xy[0] for xy in cost_evol]
    y = [xy[1] for xy in cost_evol]

    g = go.Scatter(x=x, y=y, mode='lines', name='objective')
    py.plot([g], filename='objectives')


if __name__ == '__main__':
    data = read_report('tests/output_sample.json')

    groups = aggregation_selector(
        data,
        {'c-random_search'},  # Graph only Random search
        {'w'}                 # Do a graph per worker count
    )

    # for each group do a graph: in this case each worker config
    for key, problem_ids in groups.items():
        visualize_objective_with_error(data, problem_ids, 300)
