from collections import OrderedDict
from mirl.analyzers.base_analyzer import Analyzer
from mirl.collectors.step_collector import SimpleCollector
from mirl.utils.logger import logger

class GeneralizationAnalyzer(Analyzer):
    def __init__(
        self,
        agent,
        num_eval_steps,
        max_path_length,
        step_kwargs={"":{}},
        **envs,
    ):
        self.num_eval_steps = num_eval_steps
        self.max_path_length = max_path_length
        memory_size = num_eval_steps // max_path_length
        self.collectors = {}
        for k, env in envs.items():
            if hasattr(env, "action_repeat"):
                action_repeat = env.action_repeat
            else:
                action_repeat = 1
            for step_name, kwargs in step_kwargs.items():
                col_name = k+step_name
                self.collectors[col_name] = SimpleCollector(
                    env,
                    agent,
                    action_repeat=action_repeat,
                    memory_size=memory_size,
                    agent_step_kwargs=kwargs
                )

    def analyze(self, epoch): 
        for name, coll in self.collectors.items():
            print("generalization analyze: %s"%name)
            coll.collect_new_steps(
                self.num_eval_steps,
                self.max_path_length,
                step_mode='exploit'
            )
            ret = coll.get_diagnostics()['Return Mean']
            logger.tb_add_scalar("return/%s"%name, ret, epoch)

    def get_diagnostics(self):
        info = OrderedDict()
        for name, coll in self.collectors.items():
            coll_info = coll.get_diagnostics()
            for k in coll_info:
                info[name+'/'+k] = coll_info[k]
        return info