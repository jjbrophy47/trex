dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
klr_tree_kernel=$6
klr_C=$7
knn_tree_kernel=$8
knn_n_neighbors=$9
mem=${10}
time=${11}
partition=${12}

rs_list=(1 2 3 4 5 6 7 8 9 10)
method_list=('klr-leaf_output' 'svm-leaf_output' \
             'leaf_influence' 'maple' 'knn-leaf_output')
method_list=('klr' 'knn')

for method in ${method_list[@]}; do
    for rs in ${rs_list[@]}; do
        job_name="RT_${dataset}_${model}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/runtime/$job_name \
               --error=jobs/errors/runtime/$job_name \
               jobs/runtime/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $method \
               $klr_tree_kernel $klr_C $knn_tree_kernel $knn_n_neighbors $rs
    done
done