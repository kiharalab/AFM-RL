#!/usr/bin/env python3
import sys
import Bio.PDB as bpdb
import numpy as np
import itertools
from scipy.spatial import cKDTree

pdb_path = sys.argv[1]
outname = pdb_path.replace(".pdb","")

pdbparser = bpdb.PDBParser(PERMISSIVE=1,QUIET = True)
pdb_file = pdbparser.get_structure('',pdb_path)[0]

print('# of atom : ', len(list(pdb_file.get_atoms())))
print('# of chains : ',len(list(pdb_file.get_chains())))
print(list(pdb_file.get_chains()))
chain_ids = list(pdb_file.get_chains())
#structure = bpdb.StructureBuilder.Model(0) # create empty structure
#io = bpdb.PDBIO()
for chain_a in pdb_file.get_chains():
    for chain_b in pdb_file.get_chains():
        if chain_a.get_id() == chain_b.get_id():
            continue
        print(chain_a.get_id(),chain_b.get_id())
        structure = bpdb.StructureBuilder.Model(0) # create empty structure
        io = bpdb.PDBIO()
        structure.add(chain_a)
        structure.add(chain_b)
        io.set_structure(structure)
        io.save(outname + "_extracted_pair_" + chain_a.get_id() + "_"+ chain_b.get_id() + ".pdb")
    break

