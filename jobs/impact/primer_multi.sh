dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
mem=$6
time=$7
partition=$8

method_list=('bacon')
setting_list=('dynamic')
n_test_list=(100)
start_pred_list=(-1)

for setting in ${setting_list[@]}; do
    for n_test in ${n_test_list[@]}; do
        for start_pred in ${start_pred_list[@]}; do
            for method in ${method_list[@]}; do
                for rs in {1..20}; do
                    job_name="I_${dataset}_${model}_${method}_${setting}_${start_pred}_${n_test}"

                    sbatch --mem=${mem}G \
                           --time=$time \
                           --partition=$partition \
                           --job-name=$job_name \
                           --output=jobs/logs/impact/$job_name \
                           --error=jobs/errors/impact/$job_name \
                           jobs/impact/runner.sh $dataset $model $preprocessing \
                           $n_estimators $max_depth $method $start_pred $setting $n_test $rs
                done
            done
        done
    done
done