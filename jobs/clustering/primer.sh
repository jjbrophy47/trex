dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
mem=$5
time=$6
partition=$7

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=CLS_$dataset \
       --output=jobs/logs/clustering/$dataset \
       --error=jobs/errors/clustering/$dataset \
       jobs/clustering/runner.sh $dataset $n_estimators $max_depth
