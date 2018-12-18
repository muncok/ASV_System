#! /bin/sh
echo "n_enr, trial_type, thres_type, out_dir"

n_enr=$1
trial_type=$2
thres=$3
out_dir=$4

#python run_trial.py -n_enr $n_enr -type $trial_type -out_dir "$out_dir/base" -sv_mode base -n_process 80

for i in 1
do
    python run_trial.py -n_enr $n_enr -type $trial_type -thresh_type $thres -out_dir "$out_dir/inc/$i" -sv_mode inc -n_process 80
    python run_trial.py -n_enr $n_enr -type $trial_type -thresh_type $thres -out_dir "$out_dir/inc/$i" -sv_mode inc -n_process 80 -incl_init
    python run_trial.py -n_enr $n_enr -type $trial_type -thresh_type $thres -out_dir "$out_dir/inc/$i" -sv_mode inc -n_process 80 -update
    python run_trial.py -n_enr $n_enr -type $trial_type -thresh_type $thres -out_dir "$out_dir/inc/$i" -sv_mode inc -n_process 80 -incl_init -update
done
