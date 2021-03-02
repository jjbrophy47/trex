dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
train_frac_to_remove=$6
n_checkpoints=$7
mem=$8
time=$9
partition=${10}

method_list=('random' 'klr-leaf_output' 'knn-leaf_output' 'maple')

for method in ${method_list[@]}; do
    for rs in {1..40}; do
        job_name="R_${dataset}_${model}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/roar/$job_name \
               --error=jobs/errors/roar/$job_name \
               jobs/roar/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $method $train_frac_to_remove $n_checkpoints $rs
    done
done