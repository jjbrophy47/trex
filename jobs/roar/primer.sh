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
rs_start=${13}

# method_list=('random' 'klr' 'knn' 'maple' 'maple+')
method_list=('random_pos' 'random_neg' 'random_pred' 'maple+')

for method in ${method_list[@]}; do
    for (( rs = $rs_start; rs < $rs_start + 20; rs++ )); do
        job_name="R_${dataset}_${model}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/roar/$job_name \
               --error=jobs/errors/roar/$job_name \
               jobs/roar/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $method $klr_tree_kernel $klr_C \
               $knn_tree_kernel $knn_n_neighbors $rs
    done
done