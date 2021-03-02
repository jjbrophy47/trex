# all methods
./jobs/impact/primer.sh 'churn' 'cb' 'standard' 100 3 2 1440 'short'
./jobs/impact/primer.sh 'surgical' 'cb' 'standard' 250 5 2 1440 'short'
./jobs/impact/primer.sh 'vaccine' 'cb' 'standard' 250 5 10 1440 'short'
./jobs/impact/primer.sh 'amazon' 'cb' 'standard' 250 7 15 1440 'short'
./jobs/impact/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 10 1440 'short'
./jobs/impact/primer.sh 'adult' 'cb' 'standard' 250 5 10 1440 'short'
./jobs/impact/primer.sh 'census' 'cb' 'standard' 250 5 20 1440 'short'

# methods that will not finish in less than 3 days
--scancel --name=I_census_cb_leaf_influence
