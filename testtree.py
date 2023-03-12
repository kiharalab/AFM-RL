import os
os.chdir('/net/kihara/home/taderinw/rice_home/')
import pickle
from node import Node
with open ('/net/kihara/home/taderinw/rice_scratch/2AZE/test_pathlzrd/2AZE_0.8_model_tree', 'rb') as fp:
    current_tree = pickle.load(fp)
print(current_tree)
node = current_tree
path_string = ''
while len(node.children) != 0:
    bestscore= float("-inf")
    best_child = []
    for c in node.children:
        print(c)
        exploit=c.reward/c.visits
        if exploit >= bestscore:
            best_child.append(c)
            bestscore=exploit

        bc = np.random.choice(best_child)

        path_string += str(bc.name) + '-->'
        node = bc
print(path_string)
