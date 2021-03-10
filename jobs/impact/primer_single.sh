dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
method=$6
mem=$7
time=$8
partition=$9

for rs in {1..20}; do
    job_name="I_${dataset}_${model}_${method}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/impact/$job_name \
           --error=jobs/errors/impact/$job_name \
           jobs/impact/runner.sh $dataset $model $preprocessing \
           $n_estimators $max_depth $method \
           'tree_output' 1.0 'tree_output' 61 $rs
done