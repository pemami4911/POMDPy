#!/bin/sh

ENV=RockSample
SOLVER=POMCP
SEED=123
MAX_STEPS=200
N_EPOCHS=5
N_SIMS=1000
N_START_STATES=1000
MAX_PARTICLE_COUNT=2500
UCB_COEFFICIENT=5.0
EPSILON_START=0.4
EPSILON_DECAY=0.99

args="--env $ENV --solver $SOLVER --seed $SEED --max_steps $MAX_STEPS --n_epochs $N_EPOCHS --epsilon_start $EPSILON_START \
    --epsilon_decay $EPSILON_DECAY --n_sims $N_SIMS --n_start_states $N_START_STATES \
    --max_particle_count $MAX_PARTICLE_COUNT --ucb_coefficient $UCB_COEFFICIENT --preferred_actions"
echo $args

python pomcp.py $args
