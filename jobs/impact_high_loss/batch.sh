# TESTING ONE TEST INSTANCE AT A TIME

# CB
./jobs/impact/primer.sh 'churn'          'cb' 'standard' 100 3 2  300  'short'
./jobs/impact/primer.sh 'surgical'       'cb' 'standard' 250 5 2  300  'short'
./jobs/impact/primer.sh 'vaccine'        'cb' 'standard' 250 5 15 300  'short'
./jobs/impact/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 15 300  'short'
./jobs/impact/primer.sh 'adult'          'cb' 'standard' 250 5 15 300  'short'

# RF
./jobs/impact/primer.sh 'churn'          'rf' 'standard'    100 3 5  300 'short'
./jobs/impact/primer.sh 'surgical'       'rf' 'standard'    250 7 20 300 'short'
./jobs/impact/primer.sh 'vaccine'        'rf' 'standard'    250 7 20 300 'short'
./jobs/impact/primer.sh 'bank_marketing' 'rf' 'standard'    250 7 20 300 'short'
./jobs/impact/primer.sh 'adult'          'rf' 'standard'    250 7 30 300 'short'

# TESTING MULTIPLE TEST INSTANCES AT THE SAME TIME

# CB
./jobs/impact/primer_multi.sh 'churn'          'cb' 'standard' 100 3 2  300  'short'
./jobs/impact/primer_multi.sh 'surgical'       'cb' 'standard' 250 5 2  300  'short'
./jobs/impact/primer_multi.sh 'vaccine'        'cb' 'standard' 250 5 15 300  'short'
./jobs/impact/primer_multi.sh 'bank_marketing' 'cb' 'standard' 250 5 15 300  'short'
./jobs/impact/primer_multi.sh 'adult'          'cb' 'standard' 250 5 15 300  'short'

# RF
./jobs/impact/primer_multi.sh 'churn'          'rf' 'standard'    100 3 2  300 'short'
./jobs/impact/primer_multi.sh 'surgical'       'rf' 'standard'    250 7 10 300 'short'
./jobs/impact/primer_multi.sh 'vaccine'        'rf' 'standard'    250 7 15 300 'short'
./jobs/impact/primer_multi.sh 'bank_marketing' 'rf' 'standard'    250 7 20 300 'short'
./jobs/impact/primer_multi.sh 'adult'          'rf' 'standard'    250 7 17 300 'short'
