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
             'klr_og_tree_output'
             'klr_og_leaf_path_alpha'
             'klr_og_leaf_path_sim'
             'klr_og_leaf_path'
             'klr_og_leaf_output_alpha'
             'klr_og_leaf_output_sim'
             'klr_og_leaf_output'
             'klr_og_weighted_leaf_path_alpha'
             'klr_og_weighted_leaf_path_sim'
             'klr_og_weighted_leaf_path'
             'random'
             'fast_leaf_influence'
)

for method in ${method_list[@]}; do
    job_name="IHL_${dataset}_${model}_${method}"
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/impact_high_loss/$job_name \
           --error=jobs/errors/impact_high_loss/$job_name \
           jobs/impact_high_loss/runner.sh $dataset $model \
           $n_estimators $max_depth $method
done