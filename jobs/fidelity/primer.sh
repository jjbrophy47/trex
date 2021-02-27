dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
mem=$6
time=$7
partition=$8

surrogate_list=('klr' 'svm' 'knn')
tree_kernel_list=('leaf_output' 'tree_output' 'leaf_path' 'feature_path')
metric_list=('spearman' 'mse')
rs_list=(1 2 3 4 5)

for surrogate in ${surrogate_list[@]}; do
    for tree_kernel in ${tree_kernel_list[@]}; do
        for metric in ${metric_list[@]}; do
            for rs in ${rs_list[@]}; do
                job_name="F_${dataset}_${model}_${surrogate}_${tree_kernel}"

                sbatch --mem=${mem}G \
                       --time=$time \
                       --partition=$partition \
                       --job-name=$job_name \
                       --output=jobs/logs/fidelity/$job_name \
                       --error=jobs/errors/fidelity/$job_name \
                       jobs/fidelity/runner.sh $dataset $model $preprocessing \
                       $n_estimators $max_depth $surrogate $tree_kernel $metric $rs
            done
        done
    done
done
