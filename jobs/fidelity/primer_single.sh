dataset=$1
model=$2
n_estimators=$3
max_depth=$4
surrogate=$5
tree_kernel=$6
metric=$7
mem=$8
time=$9
partition=${10}

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    job_name="F_${dataset}_${surrogate}_${tree_kernel}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/fidelity/$job_name \
           --error=jobs/errors/fidelity/$job_name \
           jobs/fidelity/runner.sh $dataset $model \
           $n_estimators $max_depth $surrogate $tree_kernel $metric $rs
done
