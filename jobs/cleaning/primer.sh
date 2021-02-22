dataset=$1
model=$2
n_estimators=$3
max_depth=$4
check_pct=$5
train_frac=$6
mem=$7
time=$8
partition=$9

rs_list=(1 2 3 4 5)
method_list=('random' \
             'klr|leaf_output' 'svm|leaf_output' \
             'klr_loss|leaf_output' 'svm_loss|leaf_output' \
             'tree_loss' 'leaf_influence' 'maple' \
             'teknn|leaf_output' 'teknn_loss|leaf_output' 'tree_prototype')

for method in ${method_list[@]}; do
    for rs in ${rs_list[@]}; do
        job_name="C_${dataset}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/cleaning/$job_name \
               --error=jobs/errors/cleaning/$job_name \
               jobs/cleaning/runner.sh $dataset $model $n_estimators \
               $max_depth $method $check_pct $train_frac $rs
    done
done