import os

txt = 'dataset,metrics,fitted_operation,approach,iterations,experiment_number,score_train,score_test,time_spent,memory_spent,pipeline'
for f in os.listdir('comparison_results'):
    txt += '\n' + '\t'.join(f.split('_')) + '\t' + open('comparison_results/' + f, 'r').read()

open('long_comparison_.csv', 'w').write(txt)
