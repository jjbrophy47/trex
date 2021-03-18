dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
desired_pred=$6
n_test=$7
klr_C=$8
knn_tree_kernel=$9
knn_n_neighbors=${10}
mem=${11}
time=${12}
partition=${13}

# method_list=('random' 'klr' 'knn' 'maple' 'maple+')
method_list=('klr_tree_output' 'klr_tree_output_sim' 'klr_leaf_path' 'klr_leaf_path_sim' 'maple')

for method in ${method_list[@]}; do
    for rs in {1..20}; do
        job_name="I_${dataset}_${model}_${method}_${desired_pred}_${n_test}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/impact/$job_name \
               --error=jobs/errors/impact/$job_name \
               jobs/impact/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $method $desired_pred $n_test \
               $klr_C $knn_tree_kernel $knn_n_neighbors $rs
    done
done