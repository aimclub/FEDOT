import bnlearn as bn
# from random import randint, seed
# seed(42)

# data_list=['titanic', 'sprinkler', 'alarm', 'andes', 'asia', 'sachs', 'water', 'random', 'stormofswords']
d = ['sprinkler', 'alarm', 'andes', 'asia', 'sachs']
# экспорт dataset
for i in d:
    df = bn.import_example(data=i, n=1000)
    df.to_csv('examples/data/'+i+'_bnln'+'.csv')

# экспорт dag
for i in d:
    file = i
    textfile = open('examples/data/'+file+'_bnln'+".txt", "w")
    model = bn.import_DAG(file)
    df_=(model)['adjmat']
    nodes = df_.index.values
    for j in nodes:
        children = nodes[df_.loc[j]]
        if len(children)!=0:
            for i in children:
                wr = j+' '+i
                textfile.write(wr)
                textfile.write('\n')
    textfile.close()     


