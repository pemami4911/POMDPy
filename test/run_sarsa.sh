#!/bin/sh

ENV=RockSample
SOLVER=SARSA
SEED=123
MAX_STEPS=200
N_EPOCHS=50
N_SIMS=50
TEST=10
EPSILON_START=0.4
EPSILON_MINIMUM=0.05
EPSILON_DECAY=0.99

args="--env $ENV --solver $SOLVER --seed $SEED --max_steps $MAX_STEPS --n_epochs $N_EPOCHS \
    --test $TEST --epsilon_start $EPSILON_START --epsilon_decay $EPSILON_DECAY --epsilon_end $EPSILON_MINIMUM --preferred_actions"
echo $args

python main.py $args
