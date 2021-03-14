# CatBoost: currently all methods
# ./jobs/roar/primer.sh 'churn' 'cb' 'standard' 100 3 'leaf_path' 0.01 'leaf_path' 61 2 300 'short' 1
# ./jobs/roar/primer.sh 'surgical' 'cb' 'standard' 250 5 'tree_output' 1.0 'leaf_path' 15 2 300 'short' 1
# ./jobs/roar/primer.sh 'vaccine' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 2 300 'short' 1
# ./jobs/roar/primer.sh 'amazon' 'cb' 'standard' 250 7 'tree_output' 1.0 'feature_path' 7 15 1440 'short' 1
# ./jobs/roar/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 2 300 'short' 1
# ./jobs/roar/primer.sh 'adult' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 2 300 'short' 1
# ./jobs/roar/primer.sh 'census' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 6 300 'short' 1

# RF: currently all methods
# ./jobs/roar/primer.sh 'churn' 'rf' 'standard' 100 3 'leaf_output' 0.01 'leaf_path' 61 2 300 'short' 1
# ./jobs/roar/primer.sh 'surgical' 'rf' 'standard' 250 7 'leaf_path' 0.001 'feature_output' 61 10 300 'short' 1
# ./jobs/roar/primer.sh 'vaccine' 'rf' 'standard' 250 7 'leaf_output' 0.001 'feature_output' 61 15 300 'short' 1
./jobs/roar/primer.sh 'amazon' 'rf' 'categorical' 10 3 'tree_output' 0.01 'tree_output' 61 20 300 'short' 1
# ./jobs/roar/primer.sh 'bank_marketing' 'rf' 'standard' 250 7 'leaf_output' 0.001 'leaf_path' 61 20 300 'short' 1
# ./jobs/roar/primer.sh 'adult' 'rf' 'standard' 250 7 'leaf_output' 0.001 'feature_output' 61 17 300 'short' 1

# CatBoost: leaf influence, experiments are executed in chunks of 20 runs at a time
# ./jobs/roar/primer_single.sh 'churn' 'cb' 'standard' 100 3 'fast_leaf_influence' 2 5040 'long' 1
# ./jobs/roar/primer_single.sh 'surgical' 'cb' 'standard' 250 5 'fast_leaf_influence' 2 5040 'long' 1
# ./jobs/roar/primer_single.sh 'vaccine' 'cb' 'standard' 250 5 'fast_leaf_influence' 2 5040 'long' 1
# ./jobs/roar/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'fast_leaf_influence' 15 1440 'long' 1
# ./jobs/roar/primer_single.sh 'bank_marketing' 'cb' 'standard' 250 5 'fast_leaf_influence' 10 5040 'long' 1
# ./jobs/roar/primer_single.sh 'adult' 'cb' 'standard' 250 5 'fast_leaf_influence' 10 5040 'long' 1
# ./jobs/roar/primer_single.sh 'census' 'cb' 'standard' 250 5 'fast_leaf_influence' 20 5040 'long' 1
