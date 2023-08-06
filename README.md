# MIRL
A repo for quickly reproducing RL algorithms and test your idea. We have implementd many fundamental elements in advanced model-based and visual RL algorithms.

## Run

You can run experiments by the following instruction: 

* python scripts/run.py configs/rsp_uai/learned_v2

or 

* python scripts/run.py configs/rsp_uai/learned_v2/rst_cartpole_01.json

When given a directory, run experiments with all the JSON files in the given directory in sequence.



#### Command 

For example, we can change the by directly given the corresponding key and value. 

```
python scripts/run.py sac.json --gpu true
```

If we want to change the initial kwargs of instances, you need to specify the name.

```
python scripts/run.py sac.json --agent.lr 0.001
```

>>>>>>> master
