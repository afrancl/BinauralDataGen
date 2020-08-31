#!/bin/sh

#SBATCH --time=0-5:00:00  # -- first number is days requested, second number is hours.  After this time the job is cancelled. 
#SBATCH -p normal # Partition to submit to
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=francl@mit.edu # -- use this to send an automated email when:
#SBATCH -o /home/francl/spatialize_stim/convolve_stim_%A_%a.stdout
#SBATCH -e /home/francl/spatialize_stim/convolve_stim_%A_%a.stder
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem-per-cpu=2000
#SBATCH --array=0-1
module add openmind/singularity/2.5.1

#total = 1746
offset=0
trainingID=$(($SLURM_ARRAY_TASK_ID + $offset))

singularity exec --nv -B /om -B /nobackup -B /scratch /om/user/francl/SoundLocalization/tf1.4.simg python -u convolve_stim_vary_env_no_hanning.py $trainingID
