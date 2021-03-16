dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
method=$6
check_pct=$7
train_frac=$8
mem=$9
time=${10}
partition=${11}

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    job_name="C_${dataset}_${model}_${method}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/cleaning/$job_name \
           --error=jobs/errors/cleaning/$job_name \
           jobs/cleaning/runner.sh $dataset $model $preprocessing \
           $n_estimators $max_depth $method 'tree_output' 1.0 'leaf_path' 61 \
           $check_pct $train_frac $rs
done