#!/usr/bin/env python3
import sys
import os
import Bio.PDB as bpdb
import numpy as np
import itertools
from scipy.spatial import cKDTree
from Bio import SeqIO,pairwise2
from Bio.PDB.Polypeptide import three_to_one
import glob

def parse_structure_fasta(fasta_path):
        chains_fasta = {}
        decoy_fasta = SeqIO.parse(fasta_path, "fasta")
        for record in decoy_fasta:
                chain_id = record.id.split('|')[0].split(':')[1]
                chains_fasta[chain_id] = str(record.seq)
        return chains_fasta

fasta_file = sys.argv[1]
path = sys.argv[2]
data_path = sys.argv[3]
#print(fasta_file)
pdb = fasta_file.split('/')[-1].split('.')[0].upper()
print(pdb)
fasta_details = parse_structure_fasta(fasta_file)
print(fasta_details.keys())
print('Creating and coping the redundant pairs')
for same_file in glob.glob(path +'*.same'):
    similar = same_file.split('/')[-1].split('.')[0].split('_')
    print(similar)
    os.system('mkdir '+ path + similar[0])
    os.system('mkdir '+ path + similar[0]+ '/' +similar[0])
    os.system('cp ' +   path + similar[1]+ '/' +similar[1] + '/ranked_*.pdb ' + path + similar[0]+ '/' +similar[0] + '/')
    #print(similar)
    #break

for x,y in itertools.combinations(fasta_details.keys(),2):
        print(x,y)
        pair = x+y
        os.chdir(path + '/' + pair + '/' + pair)
        ranksum_str = '0,ranksum\n'
        model_num = 0
        r_num = 0
        for decoy in range(0,25):
                print('working on : ',pair, decoy)
                os.system("extract_pairs.py ranked_"+ str(decoy) +".pdb")
                for pair_file in glob.glob('ranked_'+ str(decoy) +'_extracted_pair_*.pdb'):
                    cmd = 'rename_pair_v2.py ' +pair_file + ' '+ x.upper() + ' ' + y.upper()
                    os.system(cmd)
                    cmd = "cat "+pair_file.replace(".pdb", "")+"_rename.pdb |sed 's/CYX/CYS/g; s/HSE\|HSX\|HSP\|HSD\|HIE/HIS/g' | nicenPDB --erase-altloc > "+pair_file+"_nicen.pdb"
                    os.system(cmd)
                    cmd = "RENUM_PRINT_ALN=1 RENUM_REASSIGN_CHAINS=1 RENUM_SET_OCCUPANCY_1=1 renumber_rechain_afm "+data_path+"/"+pdb+"/"+x.upper()+"-"+y.upper()+".pdb "+pair_file+"_nicen.pdb"
                    os.system(cmd)
                    cmd = 'cp '+pair_file+'_nicen.pdb.myrenum '+ pair_file.replace(".pdb", "") + '_' + x.upper() + '-' + y.upper() + '-' + str(decoy) + '.pdb'
                    #ranksum_str += pair_file.replace(".pdb", "") + '_' + x.upper() + '-' + y.upper() + '-' + str(decoy) + '.pdb' + ',' + str(decoy) + '\n'
                    ranksum_str += x.upper()+'-'+y.upper()+ '-'+str(r_num)+ '.pdb,' + str(r_num) + '\n'
                    r_num +=1
                    os.system(cmd)
                for final_file in sorted(glob.glob('ranked_'+str(decoy)+'_*'+x.upper()+'-'+y.upper()+'-*.pdb')):
                    os.system('mv ' + final_file+ ' '+x.upper()+'-'+y.upper()+ '-'+str(model_num) + '.pdb')
                    model_num +=1
        os.system('RENUM_PRINT_ALN=1 RENUM_REASSIGN_CHAINS=1 RENUM_SET_OCCUPANCY_1=1 renumber_rechain_afm '+data_path+'/'+pdb+'/'+x.upper()+"-"+y.upper()+'.pdb ' + x.upper() + '-' + y.upper() + '-*.pdb')
        os.system('rename -f s/pdb.myrenum/pdb/ *.pdb.myrenum')
        cmd = 'mv ' + x.upper() + '-' + y.upper() + '-*.pdb '+ path + '/ready'
        os.system(cmd)
        
        with open(path + '/ready/' + x.upper() + '-' + y.upper() + '.ranksum','w') as fh:
                fh.write(ranksum_str)

