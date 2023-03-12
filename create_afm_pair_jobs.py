#!/usr/bin/env python3
import sys
import os
import Bio.PDB as bpdb
import numpy as np
import itertools
from scipy.spatial import cKDTree
from Bio import SeqIO,pairwise2
from Bio.PDB.Polypeptide import three_to_one

def parse_structure_fasta(fasta_path):
        chains_fasta = {}
        decoy_fasta = SeqIO.parse(fasta_path, "fasta")
        for record in decoy_fasta:
                chain_id = record.id.split('|')[0].split(':')[1]
                chains_fasta[chain_id] = str(record.seq)
        return chains_fasta

fasta_file = sys.argv[1]
pdb_name = fasta_file.split('.')[0]
#print(pdb_name)
#exit()
fasta_details = parse_structure_fasta(fasta_file)
print(fasta_details.keys())
str_ = ''
identical_pairs = {}
for x,y in itertools.combinations(fasta_details.keys(),2):
        print(x,y)
        pair = x+y
        total_len = 0
        #os.mkdir(pair)
        current_seq = ''
        pair_fasta = ''
        pair_fasta += '>' + x + '\n'
        pair_fasta += fasta_details[x] + '\n'
        current_seq += fasta_details[x]
        total_len += len(fasta_details[x])
        pair_fasta += '>' + y + '\n'
        pair_fasta += fasta_details[y] + '\n'
        current_seq += fasta_details[y]
        total_len += len(fasta_details[y])
        count = 2
        redun = 1
        while(0):
            if total_len < 3000 and redun <3:
                pair_fasta += '>' + y + str(count) + '\n'
                pair_fasta += fasta_details[y] + '\n'
                current_seq += fasta_details[y]
                total_len += len(fasta_details[y])
                count += 1
                redun += 1
            else:
                break
        str_ += pair + '.fasta,'
        with open(pair + '.fasta','w') as fh:
                fh.write(pair_fasta)
        if current_seq not in identical_pairs:
            identical_pairs[current_seq] = pair
            cmd = 'pname='+pdb_name+' pdb=' +pair+' afm_pair_msa_only.sh'
            os.system(cmd)
            #print(cmd)
        else:
            with open(pair + '_' + identical_pairs[current_seq] + '.is.same','w') as f:
                f.write(pair + '_' + identical_pairs[current_seq])
