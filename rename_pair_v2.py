#!/usr/bin/env python3
import sys
import os
import Bio.PDB as bpdb
import numpy as np
import itertools
from scipy.spatial import cKDTree
from Bio import SeqIO,pairwise2
from Bio.PDB.Polypeptide import three_to_one


pdb_file = sys.argv[1]
pair_a = sys.argv[2]
pair_b = sys.argv[3]

pdbparser = bpdb.PDBParser(PERMISSIVE=1,QUIET = True)
structure = pdbparser.get_structure('',pdb_file)[0]

renames =[pair_a,pair_b] # I only need A and B# {"A": pair_a,"B": pair_b}

idx =0
for chain in structure:
    old_name = chain.get_id()
    new_name = renames[idx] #.get(old_name)
    chain.parent = None
    if new_name:
        print(f"renaming chain {old_name} to {new_name}")
        chain.id = new_name
    else:
        print(f"keeping chain name {old_name}")
    idx += 1

io = bpdb.PDBIO()
io.set_structure(structure)
io.save(pdb_file.replace('.pdb','_rename.pdb'))
