dataset=$1
model=$2
processing=$3
tune_frac=$4
mem=$5
time=$6
partition=$7

rs_list=(1 2 3 4 5 6 7 8 9 10)

for rs in ${rs_list[@]}; do
    job_name="P_${dataset}_${model}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/performance/$job_name \
           --error=jobs/errors/performance/$job_name \
           jobs/performance/runner.sh $dataset $model $rs $processing $tune_frac
done
