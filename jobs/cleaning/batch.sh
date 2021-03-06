# all methods
# NOTE: currently just KLR_OG and KLR_LOSS_OG, both REVERSED!
# ./jobs/cleaning/primer.sh 'churn' 'cb' 'standard' 100 3 0.3 1.0 'leaf_path' 0.01 'leaf_path' 61 2 900 'short'
# ./jobs/cleaning/primer.sh 'surgical' 'cb' 'standard' 250 5 0.12 1.0 'tree_output' 1.0 'leaf_path' 15 10 900 'short'
# ./jobs/cleaning/primer.sh 'vaccine' 'cb' 'standard' 250 5 0.18 1.0 'tree_output' 1.0 'tree_output' 61 10 900 'short'
# ./jobs/cleaning/primer.sh 'amazon' 'cb' 'standard' 250 7 0.1 1.0 'tree_output' 0.1 'leaf_path' 61 35 1440 'short'
# ./jobs/cleaning/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 0.18 1.0 'tree_output' 1.0 'tree_output' 31 10 900 'short'
# ./jobs/cleaning/primer.sh 'adult' 'cb' 'standard' 250 5 0.15 1.0 'tree_output' 1.0 'tree_output' 61 10 900 'short'
# ./jobs/cleaning/primer.sh 'census' 'cb' 'standard' 250 5 0.1 1.0 'tree_output' 1.0 'tree_output' 31 30 900 'short'

# experiments that will not finish
# scancel --name=C_census_cb_leaf_influence
# scancel --name=C_census_tree_prototype

# experiments that need more time
# scancel --name=C_vaccine_cb_leaf_influence
# scancel --name=C_amazon_cb_knn
# scancel --name=C_amazon_cb_tree_prototype
# scancel --name=C_amazon_cb_maple
# scancel --name=C_amazon_cb_leaf_influence
# scancel --name=C_bank_marketing_cb_leaf_influence
# scancel --name=C_adult_cb_leaf_influence
# NOTE: primer_single uses tree_output, 1.0, and leaf_path, 61 for klr and knn, respectively
# ./jobs/cleaning/primer_single.sh 'vaccine' 'cb' 'standard' 250 5 'leaf_influence' 0.18 1.0 10 2880 'long'
./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'knn' 0.10 1.0 25 7200 'long'
# ./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'tree_prototype' 0.10 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'maple' 0.10 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'leaf_influence' 0.10 1.0 10 2880 'long'
# ./jobs/cleaning/primer_single.sh 'bank_marketing' 'cb' 'standard' 250 5 'leaf_influence' 0.18 1.0 10 4320 'long'
# ./jobs/cleaning/primer_single.sh 'adult' 'cb' 'standard' 250 5 'leaf_influence' 0.15 1.0 10 4320 'long'

# CB: fast_leaf_influence, needs more time
./jobs/cleaning/primer_single.sh 'churn' 'cb' 'standard' 100 3 'fast_leaf_influence' 0.3 1.0 2 1440 'short'
./jobs/cleaning/primer_single.sh 'surgical' 'cb' 'standard' 250 5 'fast_leaf_influence' 0.12 1.0 10 1440 'short'
./jobs/cleaning/primer_single.sh 'vaccine' 'cb' 'standard' 250 5 'fast_leaf_influence' 0.18 1.0 10 1440 'short'
./jobs/cleaning/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'fast_leaf_influence' 0.1 1.0 35 1440 'short'
./jobs/cleaning/primer_single.sh 'bank_marketing' 'cb' 'standard' 250 5 'fast_leaf_influence' 0.18 1.0 10 1440 'short'
./jobs/cleaning/primer_single.sh 'adult' 'cb' 'standard' 250 5 'fast_leaf_influence' 0.15 1.0 10 1440 'short'
./jobs/cleaning/primer_single.sh 'census' 'cb' 'standard' 250 5 'fast_leaf_influence' 0.1 1.0 30 1440 'short'