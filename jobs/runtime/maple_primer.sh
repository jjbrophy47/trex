dataset=$1
n_estimators=$2
max_depth=$3
mem=$4
time=$5
partition=$6

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=RT_MAPLE_$dataset \
           --output=jobs/logs/runtime/$dataset \
           --error=jobs/errors/runtime/$dataset \
           jobs/runtime/maple_runner.sh $dataset $n_estimators \
           $max_depth $rs
done
