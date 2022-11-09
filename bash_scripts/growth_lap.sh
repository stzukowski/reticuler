#!/bin/bash

function run_single_exp {
	eta=`printf '%.1f' "$(echo $1/10 | bc)"`
	
	# results will be saved in 'outputFile' + '.json'
	# final plot in 'outputFile' + '.pdf'
	outputFile="eta`printf '%02d' $1`"
	cd ~/$experimentDirectory
	( export PYTHONUNBUFFERED="1" # otherwise stdout/stderr are buffered in memory and printed to log file only when process terminates
	reticulate -out "$outputFile" \
				--growth_params "{\"growth_thresh_type\":$growthThreshType,\"growth_thresh\":$growthThresh}" \
				-ic $initialCondition --kwargs_box "{\"seeds_x\":$seedsX,\"height\":$boxHeight}" \
				--pde_solver_params "{\"equation\":$equation}" \
				--extender_params "{\"eta\":$eta,\"ds\":$ds,\"bifurcation_type\":$bifurcationType}" \
				--final_plot \
				> $outputFile.log 2>&1 &)
	echo "Experiment $experimentName, eta=$1/10 started."
}

function wait_till {
	sleep 60
	num=$(ps -C MATLAB | wc -l)
	echo "`date +"%c"`  Number of currently running processes: $num"
	while [ $num -gt $1 ] # 5?
	do
		sleep 5m; 
		num=$(ps -C MATLAB | wc -l);
		echo "`date +"%c"`   Number of currently running processes: $num.";
	done
}

################ GROWTH SETTINGS ################
# equation can have 2 values:
# 0 - Laplace equation
# 1 - Poisson equation
equation=0

# initial_condition:
# 0 - constant flux on top
# 1 - reflective top
initialCondition=0
boxHeight=50
seedsX=[1.5]

ds=0.01 # spatial step

# growth limit (when to stop the simulation) 
growthThreshType=1 # 0: number of steps, 1: height, 2: network length
growthThresh=10 # Poisson - for number of steps: around 2/ds for short tree and 3.5/ds for long tree, for network length: 12.832

# bifurcationType:
# 0 - no bifurcation
# 1 - bifurcation when a1>bifurcation_treshold
# 2 - bifurcation when a3/a1<bifurcation_treshold
bifurcationType=1

echo "Starting growth."
for subexp in test_laplace
do
	experimentName="${subexp}"
	experimentDirectory="reticuler/archive/growth/$experimentName"
	mkdir -p ~/$experimentDirectory
	
	for iEta in {0..18..3}
	do
		run_single_exp $iEta
	done
done


: <<'COMMENTSECTION'
sleep 60
num=$(ps -C MATLAB | wc -l)
echo "`date +"%c"`  Amount of currently running processes: $num"
while [ $num -gt 5 ]
do
	sleep 5m; 
	num=$(ps -C MATLAB | wc -l);
	echo "`date +"%c"`   Amount of currently running processes: $num.";
done

for counter in {1..9}
do
	experimentName="precision_calibration/dt005/dt005_$counter"
	
	experimentDirectory="matlab/growth/experiments/$experimentName"
	outsDirectory="matlab/growth/outs/$experimentName"

	for i in 5 10 15 
	# {1..1}
	do
		cd ~/$experimentDirectory
		mkdir -p Laplace_eta$i;
		cp -a files/. Laplace_eta$i/
		cd Laplace_eta$i
		echo "$i" > eta.txt
		echo "0.005" > dt.txt
		echo "$experimentName" > experimentName.txt
		(matlab -nodesktop -nosplash -r -nodisplay mainHandler > ~/$outsDirectory/out_$i.txt 2>&1 &)
		echo "Growth for eta=$i/10 has started."
	done
done
: <<'COMMENTSECTION'
( ./b_growth.sh > ~/matlab/growth/outs/precision_calibration/out_bash.txt 2>&1 & )
COMMENTSECTION
