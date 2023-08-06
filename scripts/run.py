
from mirl.external_libs.robosuite_wrapper import robosuite_container
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
from mirl.utils.launch_utils import parse_cmd, run_experiments
run_experiments(*parse_cmd())  
