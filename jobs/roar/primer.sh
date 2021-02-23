dataset=$1
model=$2
n_estimators=$3
max_depth=$4
train_frac_to_remove=$5
n_checkpoints=$6
mem=$7
time=$8
partition=$9

method_list=('random' 'klr-leaf_output' 'maple' 'knn-leaf_output')

for rs in {0..19}; do
    for method in ${method_list[@]}; do
        job_name="R_${dataset}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/roar/$job_name \
               --error=jobs/errors/roar/$job_name \
               jobs/roar/runner.sh $dataset $model $n_estimators \
               $max_depth $method $train_frac_to_remove $n_checkpoints $rs
    done
done