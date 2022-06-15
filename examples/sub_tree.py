import networkx as nx



number_nodes = 8
pop_size=10
DAG_list=[]
while len(DAG_list)<pop_size:
    is_all = True
    DAG = []
    while is_all == True:
        G=nx.gnp_random_graph(number_nodes,0.5,directed=True)
        DAG = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
        if len(DAG.nodes) == number_nodes:
            is_all =False
            DAG_list.append(DAG)

li=list(map(lambda x: x.edges(), DAG_list))
{0: 'asia', 1: 'tub', 2: 'smoke', 3: 'lung', 4: 'bronc', 5: 'either', 6: 'xray', 7: 'dysp'}