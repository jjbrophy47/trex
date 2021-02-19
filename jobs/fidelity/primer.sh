dataset=$1
model=$2
n_estimators=$3
max_depth=$4
mem=$5
time=$6
partition=$7

surrogate_list=('klr' 'svm' 'knn')
tree_kernel_list=('leaf_output' 'tree_output' 'leaf_path')
rs_list=(1 2 3 4 5)

for surrogate in ${surrogate_list[@]}; do
    for tree_kernel in ${tree_kernel_list[@]}; do
        for rs in ${rs_list[@]}; do
            job_name="F_${dataset}_${surrogate}_${tree_kernel}"

            sbatch --mem=${mem}G \
                   --time=$time \
                   --partition=$partition \
                   --job-name=$job_name \
                   --output=jobs/logs/fidelity/$job_name \
                   --error=jobs/errors/fidelity/$job_name \
                   jobs/fidelity/runner.sh $dataset $model \
                   $n_estimators $max_depth $surrogate $tree_kernel $rs
        done
    done
done
