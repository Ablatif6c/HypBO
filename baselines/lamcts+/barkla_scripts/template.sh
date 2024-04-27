#!/bin/bash -l
#SBATCH -J $job_prefix$_d$dim$_$seed_start$_$seed_count$
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o  $func_name$_d$dim$_$seed_start$_$seed_count$_std.out
#SBATCH -p cooper
#SBATCH -N 1 --exclude=node[001]
#SBATCH -n $n_cores$
#SBATCH --mem-per-cpu=9000M
#SBATCH -t 3-00:00:00

echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

echo =========================================================   
echo Job output begins                                           
echo -----------------                                           
echo

module load apps/anaconda3/2019.10-pytorch
conda activate hbo
export OMP_NUM_THREADS=1
python main.py --func_name $func_name$ --dim $dim$  --seed_start $seed_start$ --seed_count $seed_count$ --n_init $n_init$ --budget $budget$
 
echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
