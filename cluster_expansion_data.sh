#!bin/bash

angles=(7.5 30 52.5 75)
g_s=(1.0 1.5)
L=10
NR=1000
model="QIM"
NN=(0.5 1.0 2.0 3.0)
reports=5
trials=100

for nn in "${NN[@]}$"; do
    for g in "${g_s[@]}"; do
	for angle in "${angles[@]}"; do
	    python study.py --L "$L" --g "$g" --angle "$angle" --architecture "$architecture" --NN "$nn" --trials "$trials" &
	done
	wait
    done
done
