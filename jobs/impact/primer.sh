dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
desired_pred=$6
n_test=$7
klr_tree_kernel=$8
klr_C=$9
knn_tree_kernel=${10}
knn_n_neighbors=${11}
mem=${12}
time=${13}
partition=${14}

# method_list=('random' 'klr' 'knn' 'maple' 'maple+')
method_list=('random_pred' 'klr')

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
               $klr_tree_kernel $klr_C $knn_tree_kernel $knn_n_neighbors $rs
    done
done