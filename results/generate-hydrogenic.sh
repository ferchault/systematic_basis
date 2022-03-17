#!/bin/bash
cd ..
for Z in 1 2 3 4 5 6
do
	for nprimitives in 1 2 3 4 5 6 7;
	do
		echo $Z $nprimitives
		python prototyping/hydrogenic-scan.py $Z $nprimitives &> results/hydrogenic/run-$Z-$nprimitives.log
	done
done
