dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
check_pct=$6
train_frac=$7
klr_tree_kernel=$8
klr_C=$9
knn_tree_kernel=${10}
knn_n_neighbors=${11}
mem=${12}
time=${13}
partition=${14}

rs_list=(1 2 3 4 5)
method_list=('random' \
             'klr-leaf_output' 'svm-leaf_output' \
             'klr_loss-leaf_output' 'svm_loss-leaf_output' \
             'tree_loss' 'leaf_influence' 'maple' \
             'knn-leaf_output' 'knn_loss-leaf_output' 'tree_prototype')

method_list=('knn')

for method in ${method_list[@]}; do
    for rs in ${rs_list[@]}; do
        job_name="C_${dataset}_${model}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/cleaning/$job_name \
               --error=jobs/errors/cleaning/$job_name \
               jobs/cleaning/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $method $klr_tree_kernel $klr_C \
               $knn_tree_kernel $knn_n_neighbors $check_pct $train_frac $rs
    done
done
