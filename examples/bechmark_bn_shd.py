
def child_dict(net: list):
    res_dict = dict()
    for e0, e1 in net:
        if e1 in res_dict:
            res_dict[e1].append(e0)
        else:
            res_dict[e1] = [e0]
    return res_dict

def precision_recall(pred_net: list, true_net: list, decimal = 2):
    pred_dict = child_dict(pred_net)
    true_dict = child_dict(true_net)
    corr_undir = 0
    corr_dir = 0
    for e0, e1 in pred_net:
        flag = True
        if e1 in true_dict:
            if e0 in true_dict[e1]:
                corr_undir += 1
                corr_dir += 1
                flag = False
        if (e0 in true_dict) and flag:
            if e1 in true_dict[e0]:
                corr_undir += 1
    pred_len = len(pred_net)
    true_len = len(true_net)
    shd = pred_len + true_len - corr_undir - corr_dir
    return {'AP': round(corr_undir/pred_len, decimal), 
    'AR': round(corr_undir/true_len, decimal), 
    'AHP': round(corr_dir/pred_len, decimal), 
    'AHR': round(corr_dir/true_len, decimal), 
    'SHD': shd}


with open('examples/data/asia.txt') as f:
    lines = f.readlines()
true_net = []
for l in lines:
    e0 = l.split()[0]
    e1 = l.split()[1].split('\n')[0]
    true_net.append((e0, e1))


bamt_net = [('asia', 'tub'), ('asia', 'dysp'), ('tub', 'either'), ('tub', 'dysp'), 
('tub', 'lung'), ('lung', 'smoke'), ('lung', 'dysp'), ('bronc', 'smoke'), ('either', 'xray'),
 ('either', 'bronc'), ('either', 'smoke'), ('either', 'lung'), ('dysp', 'bronc')]

pred_net_K2 = [('asia', 'tub'), ('smoke', 'asia'), ('smoke', 'tub'), ('lung', 'asia'), ('bronc', 'asia'),
('bronc', 'tub'), ('bronc', 'lung'), ('bronc', 'either'), ('bronc', 'xray'), ('either', 'asia'),
('either', 'tub'), ('xray', 'asia'), ('xray', 'tub'), ('dysp', 'asia'), ('dysp', 'tub')]



pred_net_BDeu = [('asia', 'bronc'), ('tub', 'bronc'), ('smoke', 'asia'), ('smoke', 'tub'), ('lung', 'bronc'),
('either', 'bronc'), ('xray', 'bronc'), ('dysp', 'asia'), ('dysp', 'tub')]

print('bamt ->', precision_recall(bamt_net, true_net))
print('GA_BDeu ->', precision_recall(pred_net_BDeu, true_net))
print('GA_K2 ->', precision_recall(pred_net_K2, true_net))

