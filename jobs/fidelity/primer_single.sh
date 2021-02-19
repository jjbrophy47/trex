dataset=$1
model=$2
n_estimators=$3
max_depth=$4
surrogate=$5
tree_kernel=$6
mem=$7
time=$8
partition=$9

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    job_name="F_${dataset}_${surrogate}_${tree_kernel}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=F_TREX_$dataset \
           --output=jobs/logs/fidelity/$dataset \
           --error=jobs/errors/fidelity/$dataset \
           jobs/fidelity/trex_runner.sh $dataset $model \
           $n_estimators $max_depth $surrogate $tree_kernel $rs
done
