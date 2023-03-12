import numpy as np
import pandas as pd
import Bio.PDB as bpdb

def get_interractions(chains):
	interraction_list = []
	chain_to_int = {}
	for i in range(0,len(chains)):
		for j in range(i,len(chains)):
			if i==j:continue
			interraction_list.append([i,j])
			print (i,j)
	print (interraction_list)
	return interraction_list



def transform_chain(chain_obj,rot,tran,invert):
    rotation = rot
    translation = tran
    chain_copy = chain_obj.copy()
    	
    if invert:
    	invt_trans = -rotation.dot(translation.T) 
    	rot = np.eye(3)
    	chain_copy.transform(rot,(-translation.T))
    	chain_copy.transform(rotation,np.zeros(3))
    else:
    	chain_copy.transform(rotation.T,translation.T)
    return chain_copy



chain_list = ['A','B','C']

inter_list = get_interractions(chain_list)
print (np.random.choice([x for x in range(0,len(inter_list))]))

model_list = []
model_list.append(inter_list[0])
model_list.append(inter_list[2])
#model_list.append(inter_list[3])
    
print()
print (model_list)
print()
first_set = set(map(tuple, inter_list))
secnd_set = set(map(tuple, model_list))
diff_ = first_set.symmetric_difference(secnd_set)
print (diff_)
print (list(diff_))
a = [list(x) for x in diff_]
print (a)

pairwisedata = {}
for x in inter_list:
	url = 'data/2AZE/' + str(chain_list[x[0]]) + '-' + str(chain_list[x[1]]) + '.out'
	fh = pd.read_csv(url) #file.open(url)
	print (fh.head())
	#pairwisedata[str(x)] = data



pdbparser = bpdb.PDBParser(PERMISSIVE=1)
url_a = 'data/2AZE/A-2AZE.pdb'
url_b = 'data/2AZE/B-2AZE.pdb'
chain_a = pdbparser.get_structure(0,url_a)[0]['A']
chain_b = pdbparser.get_structure(0,url_b)[0]['B']

print()
print()
print()
print(chain_a)
print(chain_b)


rot = np.array([-0.162,  -0.900,  0.405,   0.899,   -0.304,  -0.316,  0.407,   0.313,   0.858]).reshape(3,3)
trans = np.array([60.352,  2.002,   -3.399]).reshape(3)

new_a = transform_chain(chain_a,rot,trans,True)
new_b = transform_chain(chain_b,rot,trans,False)

structure = bpdb.StructureBuilder.Model(0) # create empty structure
structure.add(chain_a)
structure.add(new_b)
io = bpdb.PDBIO()
io.set_structure(structure)
output_file = url_a = 'data/2AZE/sample_original.pdb'
io.save(output_file)



structure2 = bpdb.StructureBuilder.Model(0) # create empty structure
structure2.add(new_a)
structure2.add(chain_b)
io = bpdb.PDBIO()
io.set_structure(structure2)
output_file = url_a = 'data/2AZE/sample_invert.pdb'
io.save(output_file)