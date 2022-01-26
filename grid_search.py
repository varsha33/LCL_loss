import _thread as thread
import json
import logging
import multiprocessing
import os
import random
import sys
import threading
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable

import optuna
from optuna.samplers import GridSampler
from optuna.trial import Trial
from tqdm.auto import tqdm, trange


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


logger = logging.getLogger(__name__)


def quit_function(fn_name):
    print(f'[TIMEOUT] {fn_name} take too long!', file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if function takes longer than s seconds
    '''

    def outer(fn):
        def inner(*args, **kwargs):
            if s > 0:
                timer = threading.Timer(s, quit_function, args=[fn.__name__])
                timer.start()
                try:
                    result = fn(*args, **kwargs)
                finally:
                    timer.cancel()
                return result
            else:
                return fn(*args, **kwargs)

        return inner

    return outer


class StopWhenNotImproved:
    def __init__(self, patience: int, min_trials: int):
        self.patience = patience
        self.min_trials = min_trials
        self.no_improve_cnt = 0
        self.trial_cnt = 0
        self.best_value = None

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.trial_cnt += 1
            current_value = trial.value
            if self.best_value is None:
                self.best_value = current_value
            else:
                if current_value > self.best_value:
                    self.best_value = current_value
                    self.no_improve_cnt = 0
                else:
                    if self.trial_cnt > self.min_trials:
                        self.no_improve_cnt += 1
                        if self.no_improve_cnt >= self.patience:
                            study.stop()


class RecordCallback:
    def __init__(self, metric: str, save_home: str, prev_record: dict = None):
        self.metric = metric
        self.save_home = Path(save_home)
        self.save_home.mkdir(parents=True, exist_ok=True)
        self.record = {}
        if prev_record is not None:
            self.record = deepcopy(prev_record)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_cnt = len(self.record)
            self.record[trial_cnt] = trial.params
            self.record[trial_cnt][self.metric] = trial.value
            json.dump(self.record, open(self.save_home / 'search_record.json', 'w'), indent=4)


class RandomGridSampler(GridSampler):
    def __init__(self, search_space, filter_fn: Optional[Callable] = None) -> None:
        super().__init__(search_space=search_space)
        if filter_fn is not None:
            self._all_grids = filter_fn(self._all_grids, self._param_names)
            self._n_min_trials = len(self._all_grids)
        random.shuffle(self._all_grids)


def fetch_hyperparas_suggestions(search_space: Dict, trial: Trial):
    return {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}


def single_process(item, data_loaders, log, train_fn, metric, verbose):
    suggestions, i = item
    log = deepcopy(log)
    log.param.SEED = log.param.SEED[i]
    log.param.update(suggestions)

    if verbose:
        train_fn(log, data_loaders, test_flag=False)
    else:
        with HiddenPrints():
            train_fn(log, data_loaders, test_flag=False)
    return log[metric]


def grid_search(
        log,
        save_home,
        search_space: Dict,
        data_loaders,

        train_fn: Callable,
        filter_fn: Optional[Callable] = None,
        metric: Optional[Union[str, Callable]] = 'f1_macro',
        direction: Optional[str] = 'maximize',

        prev_results: Optional[dict] = None,

        verbose: Optional[bool] = False,
        n_repeats: Optional[int] = 1,
        n_trials: Optional[int] = 100,
        n_jobs: Optional[int] = 1,
        min_trials: Optional[Union[int, float]] = -1,
        study_patience: Optional[Union[int, float]] = -1,
        prune_threshold: Optional[float] = -1,
        study_timeout: Optional[int] = None,
        parallel: Optional[bool] = False,
        study_name: Optional[str] = None,
        **kwargs: Any):
    worker = partial(
        single_process,
        data_loaders=data_loaders,
        log=log,
        train_fn=train_fn,
        metric=metric,
        verbose=verbose
    )
    study = optuna.create_study(
        study_name=study_name,
        sampler=RandomGridSampler(search_space, filter_fn=filter_fn),
        direction=direction,
    )

    if prev_results is not None:
        distributions = {k: optuna.distributions.CategoricalDistribution(v) for k, v in search_space.items()}
        for i, (k, v) in enumerate(prev_results.items()):
            v = deepcopy(v)
            metric_v = v.pop(metric)
            grid = tuple([v[k] for k in study.sampler._param_names])
            grid_id = study.sampler._all_grids.index(grid)
            trial = optuna.trial.create_trial(
                params=v,
                distributions=distributions,
                value=metric_v,
                system_attrs={
                    'grid_id'     : grid_id,
                    'search_space': study.sampler._search_space,
                }
            )
            study.add_trial(trial)

    n_grids = len(study.sampler._all_grids)
    if isinstance(min_trials, float):
        min_trials = int(min_trials * n_grids)
    if isinstance(study_patience, float):
        study_patience = int(study_patience * n_grids)

    callbacks = [RecordCallback(metric=metric, save_home=save_home, prev_record=prev_results)]
    if study_patience > 0:
        callbacks.append(StopWhenNotImproved(patience=study_patience, min_trials=min_trials))

    if parallel:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(n_repeats)

        def parallel_objective(trial: Trial):
            suggestions = fetch_hyperparas_suggestions(search_space, trial)
            metric_value = 0
            for val in tqdm(pool.imap_unordered(worker, [(suggestions, i) for i in range(n_repeats)]), total=n_repeats):
                metric_value += val
            value = metric_value / n_repeats
            return value

        study.optimize(parallel_objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,), callbacks=callbacks, timeout=study_timeout)

    else:

        def objective(trial: Trial):
            suggestions = fetch_hyperparas_suggestions(search_space, trial)
            metric_value = 0
            for i in trange(n_repeats):
                val = worker((suggestions, i))
                metric_value += val
                if prune_threshold > 0 and trial._trial_id > 0:
                    if (trial.study.best_value - val) > (prune_threshold * trial.study.best_value):
                        return metric_value / (i + 1)
            value = metric_value / n_repeats
            return value

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,), callbacks=callbacks, timeout=study_timeout)

    logger.info(f'[END: BEST VAL / PARAMS] Best value: {study.best_value}, Best paras: {study.best_params}')
    return study
