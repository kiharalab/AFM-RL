#!/usr/bin/env python3
import sys
import Bio.PDB as bpdb
import numpy as np
import itertools
import re

pdbparser = bpdb.PDBParser(PERMISSIVE=1,QUIET = True)

def make_complex(topology,action,chain_list):
	print(topology,action)
	base_index = topology[0]
	base_struct_path = chain_list[base_index[0]] + '-' + chain_list[base_index[1]] + '-'+ str(action[0]) + '.pdb'
	base_structure = pdbparser.get_structure('',base_struct_path)[0]
	assembled_chains = [x.get_id() for x in base_structure.get_chains()]
	print(assembled_chains)

	for i in range(1,len(topology)):
		print(list(base_structure.get_chains()))
		next_topology = topology[i]
		next_action = actions[i]

		next_struct_path = chain_list[next_topology[0]] + '-' + chain_list[next_topology[1]] + '-'+ str(next_action) + '.pdb'
		next_structure = pdbparser.get_structure('',next_struct_path)[0]
		next_ids = [x.get_id() for x in next_structure.get_chains()]
		print(next_ids)
		# which id is being super-imposed and which id is being added to the assemble
		if next_ids[0] in assembled_chains:
			super_imp_id = next_ids[0]
			new_chain_id = next_ids[1]
		else:
			super_imp_id = next_ids[1]
			new_chain_id = next_ids[0]

		current_atoms = list(base_structure[super_imp_id].get_atoms())
		super_imp_atoms = list(next_structure[super_imp_id].get_atoms())

		sup = bpdb.Superimposer()
		sup.set_atoms(current_atoms, super_imp_atoms)
		sup.apply(super_imp_atoms)
		rmsd = sup.rms
		print(round(rmsd,3))
		rotation = sup.rotran[0]
		translation = sup.rotran[1]

		# make a copy, transform and add to the current assemble
		new_chain = next_structure[new_chain_id].copy()
		new_chain.transform(rotation,translation)
		base_structure.add(new_chain)

		assembled_chains.append(new_chain_id)

	return base_structure


topology = [[1, 2], [1, 3], [0, 3]]
actions = [0,0,0]
chain_list = sys.argv[1].split(',')
print(chain_list)
assembled_model = make_complex(topology,actions,chain_list)
print(list(assembled_model.get_chains()))
io = bpdb.PDBIO()
io.set_structure(assembled_model)
output_file = 'assembled_complex.pdb'
io.save(output_file)
exit()



a_structure = pdbparser.get_structure('',sys.argv[1])[0]
b_structure = pdbparser.get_structure('',sys.argv[2])[0]
print(list(a_structure.get_chains()))
print(list(b_structure.get_chains()))
native_atoms = list(a_structure['B'].get_atoms())
decoy_atoms = list(b_structure['B'].get_atoms())
sup = bpdb.Superimposer()
sup.set_atoms(native_atoms, decoy_atoms)
sup.apply(decoy_atoms)
rmsd = sup.rms
print(round(rmsd,3))
rotation = sup.rotran[0]
translation = sup.rotran[1]
new_p = b_structure['P'].copy()
new_p.transform(rotation,translation)
a_structure.add(new_p)
io = bpdb.PDBIO()
io.set_structure(a_structure)
output_file = '1a0r_complex.pdb'
io.save(output_file)

