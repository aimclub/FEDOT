import os

txt = 'dataset\tmetrics\tfitted_operation\tapproach\titerations\texperiment_number\tscore_train\tscore_test\ttime_spent\tmemory_spent\tpipeline'
for f in os.listdir('comparison_results'):
    txt += '\n' + '\t'.join(f.split('_')) + '\t' + open('comparison_results/' + f, 'r').read()

open('long_comparison_.csv', 'w').write(txt)
