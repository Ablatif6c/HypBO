#SBATCH -o main_std.out
sbatch job_barkla_Branin.sh
echo 'job_barkla_Branin.sh' started...

sbatch job_barkla_Sphere.sh
echo 'job_barkla_Sphere.sh' started...

sbatch job_barkla_Levy.sh
echo 'job_barkla_Levy.sh' started...

