from typing import Callable, Dict, List       

from repro.hpo.dispatcher.dispatcher import HPODispatcher


class Stub(HPODispatcher):
    def should_suspend(self, trial_id: str) -> bool:
        return False

    def should_resume(self, trial_id) -> bool:
        return True

    
def build(space, configurator_config, max_trials, seed):
    return Stub(space, configurator_config, max_trials, seed)
