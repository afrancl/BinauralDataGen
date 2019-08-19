#!/bin/sh

#SBATCH --time=0-5:00:00  # -- first number is days requested, second number is hours.  After this time the job is cancelled. 
#SBATCH --qos=use-everything
#SBATCH -p om_all_nodes # Partition to submit to
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=francl@mit.edu # -- use this to send an automated email when:
#SBATCH -o /home/francl/5deg_error_files/tf_gen_%A_%a.stdout
#SBATCH -e /home/francl/5deg_error_files/tf_gen_%A_%a.stder
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1500
#SBATCH --array=0-3%250
module add openmind/singularity/2.5.1

#total = 1157
offset=0
trainingID=$(($SLURM_ARRAY_TASK_ID + $offset))

#singularity exec --nv -B /om -B /nobackup tf1.4.simg python -u tf_record_gen_training_no_resample.py $SLURM_ARRAY_TASK_ID

singularity exec --nv -B /om -B /nobackup /om/user/francl/tfv1.13_nnresample.simg python -u tf_record_gen_training_vary_env_elev_label_no_hanning.py $trainingID
