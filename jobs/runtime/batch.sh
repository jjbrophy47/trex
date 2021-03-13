# CatBoost: all methods
# NOTE: currently only running leaf_influence and fast_leaf_influence
./jobs/runtime/primer.sh 'churn' 'cb' 'standard' 100 3 'leaf_path' 0.01 'leaf_path' 61 2 5040 'long'
./jobs/runtime/primer.sh 'surgical' 'cb' 'standard' 250 5 'tree_output' 1.0 'leaf_path' 15 2 5040 'long'
./jobs/runtime/primer.sh 'vaccine' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 2 5040 'long'
./jobs/runtime/primer.sh 'amazon' 'cb' 'standard' 250 7 'tree_output' 1.0 'feature_path' 7 15 5040 'long'
./jobs/runtime/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 5 5040 'long'
./jobs/runtime/primer.sh 'adult' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 5 5040 'long'
./jobs/runtime/primer.sh 'census' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 20 5040 'long'

# RF: all methods
# NOTE: currently only running leaf_influence and fast_leaf_influence
# ./jobs/runtime/primer.sh 'churn' 'rf' 'standard' 100 3 'leaf_output' 0.01 'leaf_path' 61 2 300 'short'
# ./jobs/runtime/primer.sh 'surgical' 'rf' 'standard' 250 5 'leaf_path' 0.001 'feature_output' 61 6 300 'short'
# ./jobs/runtime/primer.sh 'vaccine' 'rf' 'standard' 250 5 'leaf_output' 0.001 'feature_output' 61 15 300 'short'
# ./jobs/runtime/primer.sh 'amazon' 'rf' 'categorical' 250 7 'tree_output' 0.01 'tree_output' 61 2 300 'short'
# ./jobs/runtime/primer.sh 'bank_marketing' 'rf' 'standard' 250 5 'leaf_output' 0.001 'leaf_path' 61 20 300 'short'
# ./jobs/runtime/primer.sh 'adult' 'rf' 'standard' 250 5 'leaf_output' 0.001 'feature_output' 61 17 300 'short'
# ./jobs/runtime/primer.sh 'census' 'rf' 'standard' 250 5 'tree_output' 0.001 'leaf_path' 61 15 300 'short'
