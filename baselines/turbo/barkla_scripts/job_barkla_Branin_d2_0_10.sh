#!/bin/bash -l
#SBATCH -J B_d2_0_10
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o  Branin_d2_0_10_std.out
#SBATCH -p cooper
#SBATCH -N 1 --exclude=node[001]
#SBATCH -n 6
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
python main.py --func_name Branin --dim 2  --seed_start 0 --seed_count 10 --n_init 5 --budget 100
 
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
