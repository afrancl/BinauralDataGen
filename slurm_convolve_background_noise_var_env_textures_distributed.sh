#!/bin/sh

#SBATCH --time=2-6:30:00  # -- first number is days requested, second number is hours.  After this time the job is cancelled. 
#SBATCH -p mcdermott 
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=francl@mit.edu # -- use this to send an automated email when:
#SBATCH -o /home/francl/convolve_bkgd.stdout
#SBATCH -e /home/francl/convolve_bkgd.stder
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem-per-cpu=50000
#SBATCH --array=0-4
module add openmind/singularity/2.5.1

#offset should go to 1000 and array from 1 to 638
#total = 3276
offset=0
trainingID=$(($SLURM_ARRAY_TASK_ID + $offset))

#singularity exec --nv -B /om -B /nobackup tf1.4.simg python -u tf_record_gen_training_no_resample.py $SLURM_ARRAY_TASK_ID

singularity exec --nv -B /om -B /nobackup /om/user/francl/plottingFn.simg python3 -u convolve_background_noise_var_env_textures_distributed.py $trainingID
