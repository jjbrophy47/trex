dataset=$1
model=$2
preprocessing=$3
scoring=$4
tune_frac=$5
mem=$6
time=$7
partition=$8

for rs in {1..40}; do
    job_name="P_${dataset}_${model}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/performance/$job_name \
           --error=jobs/errors/performance/$job_name \
           jobs/performance/runner.sh $dataset $model $rs $preprocessing $scoring $tune_frac
done
