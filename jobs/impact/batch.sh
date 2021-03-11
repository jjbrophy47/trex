# CB: all methods except Leaf Influence
./jobs/impact/primer.sh 'churn' 'cb' 'standard' 100 3 'leaf_path' 0.01 'leaf_path' 61 2 300 'short'
./jobs/impact/primer.sh 'surgical' 'cb' 'standard' 250 5 'tree_output' 1.0 'leaf_path' 15 2 300 'short'
./jobs/impact/primer.sh 'vaccine' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 2 300 'short'
./jobs/impact/primer.sh 'amazon' 'cb' 'standard' 250 7 'tree_output' 1.0 'feature_path' 7 15 1440 'short'
./jobs/impact/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 2 300 'short'
./jobs/impact/primer.sh 'adult' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 61 2 300 'short'
./jobs/impact/primer.sh 'census' 'cb' 'standard' 250 5 'tree_output' 1.0 'tree_output' 31 6 300 'short'

# RF: all methods except Leaf Influence
./jobs/impact/primer.sh 'churn' 'rf' 'standard' 100 3 'leaf_output' 0.01 'leaf_path' 61 2 300 'short'
./jobs/impact/primer.sh 'surgical' 'rf' 'standard' 250 7 'leaf_path' 0.001 'feature_output' 61 10 300 'short'
./jobs/impact/primer.sh 'vaccine' 'rf' 'standard' 250 7 'leaf_output' 0.001 'feature_output' 61 15 300 'short'
./jobs/impact/primer.sh 'amazon' 'rf' 'standard' 10 3 'tree_output' 0.01 'tree_output' 61 25 300 'short'
./jobs/impact/primer.sh 'bank_marketing' 'rf' 'standard' 250 7 'leaf_output' 0.001 'leaf_path' 61 20 300 'short'
./jobs/impact/primer.sh 'adult' 'rf' 'standard' 250 7 'leaf_output' 0.001 'feature_output' 61 17 300 'short'
./jobs/impact/primer.sh 'census' 'rf' 'standard' 250 7 'tree_output' 0.001 'leaf_path' 61 30 300 'short'

# CB: Leaf Influence, needs much more time than the others
./jobs/impact/primer_single.sh 'churn' 'cb' 'standard' 100 3 'leaf_influence' 2 4320 'long'
./jobs/impact/primer_single.sh 'surgical' 'cb' 'standard' 250 5 'leaf_influence' 2 4320 'long'
./jobs/impact/primer_single.sh 'vaccine' 'cb' 'standard' 250 5 'leaf_influence' 2 4320 'long'
./jobs/impact/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'leaf_influence' 25 1440 'short'
./jobs/impact/primer_single.sh 'bank_marketing' 'cb' 'standard' 250 5 'leaf_influence' 15 4320 'long'
./jobs/impact/primer_single.sh 'adult' 'cb' 'standard' 250 5 'leaf_influence' 15 4320 'long'
# ./jobs/impact/primer_single.sh 'census' 'cb' 'standard' 250 5 'leaf_influence' 6 4320 'long'