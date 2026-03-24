#!/bin/bash
#PBS -l select=1:ncpus=16:mem=150gb:ngpus=1
#PBS -l walltime=47:59:59
#PBS -N gfed_model_all
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -V

module load Python

source /rds/general/user/aac115/home/venvs/pytorch/bin/activate

LOGDIR="$HOME/pbs_logs"
LOGFILE="$LOGDIR/gfed_model_all.log"

cd /rds/general/user/aac115/home/fireveg/

python -u xgb_gfed_model_all.py >> "$LOGFILE" 2>&1



:'
Ok so some of this is obvious, some of it isn't, so I'll run through everything in order.
This is a very very basic script.

You always want select=1, then choose the number of CPUs, the membory needed and the number of GPUs
The wall time of the job in HH:MM:SS - check the RCS website for the info on queue sizes and time limits: https://icl-rcs-user-guide.readthedocs.io/en/latest/
The name of the job to appear on the HPC
Directory of the output file, then directory of the error file. I have these set to NaN because I use logs instead (-V).

Load Python on the HPC

Load your environment

For me this is where I keep my log files for each job - the location then the name.

Change directory to the relevant folder

Run the relevant script with -u. The last part (from >> onwards) tells the HPC where to write the logfile and to combine the output and error into one.

Then on the HPC submit the job with "qsub submit_job.sh". Status of jobs can be checked by qstat -u <username>. Jobs can be deleted by qdel <number>.pbs-7
'