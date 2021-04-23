# TESTING ONE TEST INSTANCE AT A TIME

# CB
./jobs/impact_high_loss/primer.sh 'churn'          'cb' 'standard' 100 3 2  300  'short'
./jobs/impact_high_loss/primer.sh 'surgical'       'cb' 'standard' 250 7 2  300  'short'
./jobs/impact_high_loss/primer.sh 'vaccine'        'cb' 'standard' 250 7 15 300  'short'
./jobs/impact_high_loss/primer.sh 'bank_marketing' 'cb' 'standard' 250 7 15 300  'short'
./jobs/impact_high_loss/primer.sh 'adult'          'cb' 'standard' 250 7 15 300  'short'
./jobs/impact_high_loss/primer.sh 'synthetic'      'cb' 'standard' 250 7 15 300  'short'

# RF
./jobs/impact_high_loss/primer.sh 'churn'          'rf' 'standard'    100 3 5  300 'short'
./jobs/impact_high_loss/primer.sh 'surgical'       'rf' 'standard'    250 7 20 300 'short'
./jobs/impact_high_loss/primer.sh 'vaccine'        'rf' 'standard'    250 7 20 300 'short'
./jobs/impact_high_loss/primer.sh 'bank_marketing' 'rf' 'standard'    250 7 20 300 'short'
./jobs/impact_high_loss/primer.sh 'adult'          'rf' 'standard'    250 7 30 300 'short'
