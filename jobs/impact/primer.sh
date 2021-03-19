dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
start_pred=$6
klr_C=$7
knn_tree_kernel=$8
knn_n_neighbors=$9
mem=${10}
time=${11}
partition=${12}

method_list=('random' 'klr_tree_output' 'klr_tree_output_sim' 'klr_leaf_path' 'klr_leaf_path_sim' 'maple')
setting_list=('static' 'dynamic')
n_test_list=(1)

for method in ${method_list[@]}; do
    for setting in ${setting_list[@]}; do
        for n_test in ${n_test_list[@]}; do
            for rs in {1..20}; do
                job_name="I_${dataset}_${model}_${method}_${setting}_${n_test}"

                sbatch --mem=${mem}G \
                       --time=$time \
                       --partition=$partition \
                       --job-name=$job_name \
                       --output=jobs/logs/impact/$job_name \
                       --error=jobs/errors/impact/$job_name \
                       jobs/impact/runner.sh $dataset $model $preprocessing \
                       $n_estimators $max_depth $method $start_pred $setting $n_test \
                       $klr_C $knn_tree_kernel $knn_n_neighbors $rs
            done
        done
    done
done