#!/bin/sh

ENV=RockProblem
SOLVER=SARSA
SEED=123
MAX_STEPS=200
N_RUNS=10000
TEST=1000
EPSILON_START=0.5

args="--env $ENV --solver $SOLVER --seed $SEED --max_steps $MAX_STEPS --n_runs $N_RUNS --test_run $TEST --epsilon_start $EPSILON_START --preferred_actions"
echo $args

python main.py $args
