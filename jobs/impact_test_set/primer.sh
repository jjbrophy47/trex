dataset=$1
model=$2
n_estimators=$3
max_depth=$4
mem=$5
time=$6
partition=$7

method_list=(
             'klr_og_tree_output_alpha'
             'klr_og_tree_output_sim'
             'klr_og_tree_output_sim_inf'
             'klr_og_tree_output'
             'klr_og_leaf_path_alpha'
             'klr_og_leaf_path_sim'
             'klr_og_leaf_path_sim_inf'
             'klr_og_leaf_path'
             'klr_og_leaf_output_alpha'
             'klr_og_leaf_output_sim'
             'klr_og_leaf_output_sim_inf'
             'klr_og_leaf_output'
             'klr_og_weighted_leaf_path_alpha'
             'klr_og_weighted_leaf_path_sim'
             'klr_og_weighted_leaf_path_sim_inf'
             'klr_og_weighted_leaf_path'
             'random'
             'fast_leaf_influence'
)
setting_list=('static')
frac_remove_list=(0.1 0.25 0.5)

for setting in ${setting_list[@]}; do
    for method in ${method_list[@]}; do
        for frac_remove in ${frac_remove_list[@]}; do
            job_name="ITS_${dataset}_${model}_${method}_${setting}_${frac_remove}"
            sbatch --mem=${mem}G \
                   --time=$time \
                   --partition=$partition \
                   --job-name=$job_name \
                   --output=jobs/logs/impact_test_set/$job_name \
                   --error=jobs/errors/impact_test_set/$job_name \
                   jobs/impact_test_set/runner.sh $dataset $model \
                   $n_estimators $max_depth $setting $method $frac_remove
        done
    done
done
