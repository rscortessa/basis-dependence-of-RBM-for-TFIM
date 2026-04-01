#!bin/bash

angles=(0 7.5 15 22.5 30 37.5 45 52.5 60 67.5 75 82.5 90)
g_s=(0.5 1.0 1.5)
L=10
NR=1000
model="QIM"
NN=1.0
reports=5
trials=100

for angle in "${angles[@]}"; do
    for g in "${g_s[@]}"; do
	python study.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --NN "$NN" --trials "$trials" &
    done
    wait
done
