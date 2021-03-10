# CatBoost: all methods, experiments are executed in chunks of 20 runs at a time
./jobs/roar/primer.sh 'churn' 'cb' 'standard' 100 3 'leaf_path' 0.01 'leaf_path' 61 2 300 'short' 1
./jobs/roar/primer.sh 'surgical' 'cb' 'standard' 250 5 'tree_output' 1.0 'leaf_path' 15 2 300 'short' 1
./jobs/roar/primer.sh 'vaccine' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 2 300 'short' 1
./jobs/roar/primer.sh 'amazon' 'cb' 'standard' 250 7 'tree_output' 1.0 'feature_path' 7 15 1440 'short' 1
./jobs/roar/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 2 300 'short' 1
./jobs/roar/primer.sh 'adult' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 2 300 'short' 1
./jobs/roar/primer.sh 'census' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 6 300 'short' 1

# RF: all methods, experiments are executed in chunks of 20 runs at a time
./jobs/roar/primer.sh 'churn' 'rf' 'standard' 100 3 'leaf_output' 0.01 'leaf_path' 61 2 300 'short' 1
./jobs/roar/primer.sh 'surgical' 'rf' 'standard' 250 5 'leaf_path' 0.001 'feature_output' 61 6 300 'short' 1
./jobs/roar/primer.sh 'vaccine' 'rf' 'standard' 250 5 'leaf_output' 0.001 'feature_output' 61 15 300 'short' 1
./jobs/roar/primer.sh 'amazon' 'rf' 'standard' 250 7 'tree_output' 0.01 'tree_output' 61 2 300 'short' 1
./jobs/roar/primer.sh 'bank_marketing' 'rf' 'standard' 250 5 'leaf_output' 0.001 'leaf_path' 61 20 300 'short' 1
./jobs/roar/primer.sh 'adult' 'rf' 'standard' 250 5 'leaf_output' 0.001 'feature_output' 61 17 300 'short' 1
./jobs/roar/primer.sh 'census' 'rf' 'standard' 250 5 'tree_output' 0.001 'leaf_path' 61 15 300 'short' 1
