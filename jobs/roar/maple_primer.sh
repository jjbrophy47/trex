dataset=$1
n_estimators=$2
max_depth=$3
mem=$4
time=$5
partition=$6

for rs in {0..19}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=R_MAPLE_$dataset \
           --output=jobs/logs/roar/$dataset \
           --error=jobs/errors/roar/$dataset \
           jobs/roar/maple_runner.sh $dataset $n_estimators \
           $max_depth $rs
done
