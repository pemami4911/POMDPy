#!/bin/sh

ENV=RockProblem
SOLVER=SARSA
SEED=123
MAX_STEPS=200
N_RUNS=10000
TEST=1000
EPSILON_START=0.4
EPSILON_END=0.05
EPSILON_DECAY=0.01

args="--env $ENV --solver $SOLVER --seed $SEED --max_steps $MAX_STEPS --n_runs $N_RUNS \
    --test_run $TEST --epsilon_start $EPSILON_START --epsilon_decay $EPSILON_DECAY --epsilon_end $EPSILON_END --preferred_actions"
echo $args

python main.py $args
