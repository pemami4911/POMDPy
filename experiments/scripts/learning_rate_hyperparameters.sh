#!/bin/bash

ENV=Tiger
SOLVER=LinearAlphaNet
EPSILON_DECAY=0.99
EPSILON_DECAY_STEP=75
EPSILON_MINIMUM=0.02
EPSILON_START=0.2
LEARNING_RATE=0.01
LEARNING_RATE_DECAY=0.996
LEARNING_RATE_DECAY_STEP=50
LEARNING_RATE_MINIMUM=0.00025
MAX_STEPS=50
SEED=111
N_EPOCHS=1000
TEST=10

echo "learning rate hyperparameter search"
echo "==================================="

learning_rates=(0.01 0.02 0.05)

for i in ${learning_rates[@]}; do
	LEARNING_RATE=${i}
	args="--env $ENV --solver $SOLVER --seed $SEED --max_steps $MAX_STEPS --n_epochs $N_EPOCHS --epsilon_start $EPSILON_START \
    	     --epsilon_decay $EPSILON_DECAY --epsilon_decay_step $EPSILON_DECAY_STEP --epsilon_minimum $EPSILON_MINIMUM  \
   	     --learning_rate $LEARNING_RATE --learning_rate_decay $LEARNING_RATE_DECAY --learning_rate_decay_step $LEARNING_RATE_DECAY_STEP \
    	     --learning_rate_minimum $LEARNING_RATE_MINIMUM --test $TEST --use_tf"
	echo $args

	python vi.py $args > "experiments/results/hyperparameters/learning_rate_${i}.txt" &
done
echo "done"
exit 0

