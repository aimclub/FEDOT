
['andes_bnln', 'asia_bnln', 'sachs_bnln', 'alarm_bnln', 'sprinkler_bnln']
file = 'alarm_bnln'
with open('examples/data/'+file+'.txt') as f:
    lines = f.readlines()
true_net = []
for l in lines:
    e0 = l.split()[0]
    e1 = l.split()[1].split('\n')[0]
    true_net.append((e0, e1))

print(true_net)