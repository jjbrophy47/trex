dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
surrogate=$6
tree_kernel=$7
metric=$8
mem=$9
time=${10}
partition=${11}

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    job_name="F_${dataset}_${model}_${surrogate}_${tree_kernel}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/fidelity/$job_name \
           --error=jobs/errors/fidelity/$job_name \
           jobs/fidelity/runner.sh $dataset $model $preprocessing \
           $n_estimators $max_depth $surrogate $tree_kernel $metric $rs
done
