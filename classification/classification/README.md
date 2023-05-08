Run Weights ans Biases Sweep:
1. Initialize sweeps
``wandb sweep --project classification_example sweep_config.yaml``

2. Run agent
``wandb agent projector-team/classification_example/6thou4el``

3. You can set max number of runs 
NUM=10
SWEEPID="6thou4el"
wandb agent --count $NUM $SWEEPID