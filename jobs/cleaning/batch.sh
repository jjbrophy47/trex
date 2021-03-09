# all methods
# ./jobs/cleaning/primer.sh 'churn' 'cb' 'standard' 100 3 0.3 1.0 2 1440 'short'
# ./jobs/cleaning/primer.sh 'surgical' 'cb' 'standard' 250 5 0.12 1.0 2 1440 'short'
# ./jobs/cleaning/primer.sh 'vaccine' 'cb' 'standard' 250 5 0.18 1.0 10 1440 'short'
# ./jobs/cleaning/primer.sh 'amazon' 'cb' 'standard' 250 7 0.10 1.0 10 1440 'short'
# ./jobs/cleaning/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 0.18 1.0 10 1440 'short'
# ./jobs/cleaning/primer.sh 'adult' 'cb' 'standard' 250 5 0.15 1.0 10 1440 'short'
# ./jobs/cleaning/primer.sh 'census' 'cb' 'standard' 250 5 0.10 1.0 25 1440 'short'

# experiments that will not finish
# scancel --name=C_census_cb_leaf_influence
# scancel --name=C_census_knn-leaf_output
# scancel --name=C_census_tree_prototype

# experiments that need more time
# scancel --name=C_vaccine_cb_leaf_influence
# scancel --name=C_amazon_cb_tree_prototype
# scancel --name=C_amazon_cb_knn-leaf_output
# scancel --name=C_amazon_cb_maple
# scancel --name=C_amazon_cb_leaf_influence
# scancel --name=C_bank_marketing_cb_leaf_influence
# scancel --name=C_adult_cb_leaf_influence
# ./jobs/cleaning/primer_single.sh 'vaccine' 'cb' 'standard' 250 5 'leaf_influence' 0.18 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'tree_prototype' 0.10 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'knn-leaf_output' 0.10 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'maple' 0.10 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'leaf_influence' 0.10 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'bank_marketing' 'cb' 'standard' 250 5 'leaf_influence' 0.18 1.0 10 4320 'long'
# ./jobs/cleaning/primer_single.sh 'adult' 'cb' 'standard' 250 5 'leaf_influence' 0.15 1.0 10 4320 'long'

# susbet of methods
./jobs/cleaning/primer.sh 'churn' 'cb' 'standard' 100 3 0.3 1.0 'leaf_path' 0.01 'leaf_path' 61 2 300 'short'
./jobs/cleaning/primer.sh 'surgical' 'cb' 'standard' 250 5 0.12 1.0 'tree_output' 1.0 'leaf_path' 15 2 300 'short'
./jobs/cleaning/primer.sh 'vaccine' 'cb' 'standard' 250 5 0.18 1.0 'tree_output' 1.0 'tree_output' 61 2 300 'short'
./jobs/cleaning/primer.sh 'amazon' 'cb' 'standard' 250 7 0.1 1.0 'tree_output' 1.0 'feature_path' 7 15 1440 'short'
./jobs/cleaning/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 0.18 1.0 'tree_output' 1.0 'tree_output' 31 2 300 'short'
./jobs/cleaning/primer.sh 'adult' 'cb' 'standard' 250 5 0.15 1.0 'tree_output' 1.0 'tree_output' 61 2 300 'short'
./jobs/cleaning/primer.sh 'census' 'cb' 'standard' 250 5 0.1 1.0 'tree_output' 1.0 'tree_output' 31 6 300 'short'