# TESTING ONE TEST INSTANCE AT A TIME

# CB: currently random_minority and random_majority
./jobs/impact/primer.sh 'churn'          'cb' 'standard' 100 3 2  300  'short'
./jobs/impact/primer.sh 'surgical'       'cb' 'standard' 250 5 2  300  'short'
./jobs/impact/primer.sh 'vaccine'        'cb' 'standard' 250 5 15 300  'short'
./jobs/impact/primer.sh 'amazon'         'cb' 'standard' 250 7 25 2880 'long'
./jobs/impact/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 15 300  'short'
./jobs/impact/primer.sh 'adult'          'cb' 'standard' 250 5 15 300  'short'
./jobs/impact/primer.sh 'census'         'cb' 'standard' 250 5 20 300  'short'

# RF: currently random_minority and random_majority
./jobs/impact/primer.sh 'churn'          'rf' 'standard'    100 3 2  300 'short'
./jobs/impact/primer.sh 'surgical'       'rf' 'standard'    250 7 10 300 'short'
./jobs/impact/primer.sh 'vaccine'        'rf' 'standard'    250 7 15 300 'short'
./jobs/impact/primer.sh 'amazon'         'rf' 'categorical' 10  3 25 300 'short'
./jobs/impact/primer.sh 'bank_marketing' 'rf' 'standard'    250 7 20 300 'short'
./jobs/impact/primer.sh 'adult'          'rf' 'standard'    250 7 17 300 'short'
./jobs/impact/primer.sh 'census'         'rf' 'standard'    250 7 30 300 'short'

# CB: methods that need more time and / or memory
./jobs/impact/primer_single.sh 'churn'          'cb' 'standard' 100 3 'fast_leaf_influence' 2  600  'short'
./jobs/impact/primer_single.sh 'surgical'       'cb' 'standard' 250 5 'fast_leaf_influence' 2  600  'short'
./jobs/impact/primer_single.sh 'vaccine'        'cb' 'standard' 250 5 'fast_leaf_influence' 15  600  'short'
./jobs/impact/primer_single.sh 'amazon'         'cb' 'standard' 250 7 'fast_leaf_influence' 25 1440 'short'
./jobs/impact/primer_single.sh 'bank_marketing' 'cb' 'standard' 250 5 'fast_leaf_influence' 15 600  'short'
./jobs/impact/primer_single.sh 'adult'          'cb' 'standard' 250 5 'fast_leaf_influence' 15 600  'short'
./jobs/impact/primer_single.sh 'census'         'cb' 'standard' 250 5 'fast_leaf_influence' 20 600  'short'

./jobs/impact/primer_single.sh 'churn' 'cb' 'standard' 100 3 'fast_leaf_influence' 2 4320 'long'
./jobs/impact/primer_single.sh 'surgical' 'cb' 'standard' 250 5 'fast_leaf_influence' 2 4320 'long'
./jobs/impact/primer_single.sh 'vaccine' 'cb' 'standard' 250 5 'fast_leaf_influence' 2 4320 'long'
./jobs/impact/primer_single.sh 'amazon' 'cb' 'standard' 250 7 'fast_leaf_influence' 25 1440 'short'
./jobs/impact/primer_single.sh 'bank_marketing' 'cb' 'standard' 250 5 'fast_leaf_influence' 15 4320 'long'
./jobs/impact/primer_single.sh 'adult' 'cb' 'standard' 250 5 'fast_leaf_influence' 15 4320 'long'
./jobs/impact/primer_single.sh 'census' 'cb' 'standard' 250 5 'fast_leaf_influence' 20 4320 'long'

# TESTING MULTIPLE TEST INSTANCES AT THE SAME TIME

# CB: currently random_minority and random_majority
./jobs/impact/primer_multi.sh 'churn'          'cb' 'standard' 100 3 2  300  'short'
./jobs/impact/primer_multi.sh 'surgical'       'cb' 'standard' 250 5 2  300  'short'
./jobs/impact/primer_multi.sh 'vaccine'        'cb' 'standard' 250 5 2  300  'short'
./jobs/impact/primer_multi.sh 'amazon'         'cb' 'standard' 250 7 15 1440 'short'
./jobs/impact/primer_multi.sh 'bank_marketing' 'cb' 'standard' 250 5 2  300  'short'
./jobs/impact/primer_multi.sh 'adult'          'cb' 'standard' 250 5 2  300  'short'
./jobs/impact/primer_multi.sh 'census'         'cb' 'standard' 250 5 20 300  'short'

# RF: currently random_minority and random_majority
./jobs/impact/primer_multi.sh 'churn'          'rf' 'standard'    100 3 2  300 'short'
./jobs/impact/primer_multi.sh 'surgical'       'rf' 'standard'    250 7 10 300 'short'
./jobs/impact/primer_multi.sh 'vaccine'        'rf' 'standard'    250 7 15 300 'short'
./jobs/impact/primer_multi.sh 'amazon'         'rf' 'categorical' 10  3 25 300 'short'
./jobs/impact/primer_multi.sh 'bank_marketing' 'rf' 'standard'    250 7 20 300 'short'
./jobs/impact/primer_multi.sh 'adult'          'rf' 'standard'    250 7 17 300 'short'
./jobs/impact/primer_multi.sh 'census'         'rf' 'standard'    250 7 30 300 'short'
