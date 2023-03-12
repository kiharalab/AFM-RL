#Bug: using dictionary to keep rmsd and energy values is bad as it will only keep distinct key
#TODO: change it from dictionary to list or another data structure
import sys
import os
import itertools
import io
import Bio.PDB as bpdb
from protein import Protein
import string
import glob
import re
import copy

def transform_topology(top_list):
	#['2-3', '0-2', '1-3']
	interaction_list = []
	for order in top_list:
		d_list = []
		data = list(order.split(','))
		for item in data:
			d_list.append(int(item))
		interaction_list.append(d_list)

	return interaction_list

if len(sys.argv) < 4:
    print('Usage Path, chains, lr... e.g ')
    exit()

path = sys.argv[1]
chains = sys.argv[2]
lr = sys.argv[3]
target = path.split('/')[0]
out_path = path + 'output2/'
out_type = sys.argv[4] #'filtered'
unbound = sys.argv[5]
myrank = ''#os.environ['SLURM_PROCID']

#fil = target + '_' + lr + '_mcrl.out'
myfile = 'decoy_subsample' if myrank == '0' else 'decoy_subsample' + myrank
files = [myfile]
count = 0
protein = Protein(len(chains),chains,target,300,out_type,unbound)


rmsd_outfile = path + 'output/' + myrank + '.rmsd.out.v2'
penalty = 10

decoy_information_list = []

try:
    for f in files:
        fh = open(path + f)
        count += 1
        #lr = str(f.split('_')[1])
        decoy_pref = 'Decoy_'
        #print('Starting complex generation for ',count,' out of ',len(files))
        for line in fh.readlines():
            if 'Decoy' in line:
                decoy_number = int(line.split(',')[1].split(' ')[-1])
                decoy_energy = float(line.split(',')[-1])
                val = line.split('[[')[-1].split(']]')
                topology = val[0].replace(']','').replace('[','')
                topology = re.sub('(,[^,]*),', r'\1|', topology).split('|')
                topology = transform_topology(topology)
                actions = val[1].split(']')[0].replace('[','')
                actions = actions.split(',')[1:] # re.sub('(,[^,]*),', r'\1|', actions).strip(',').split('|')
                actions = [int(x) for x in actions]
                #print(actions)
                #protein = Protein(len(chains),chains,target,300,out_type)
                transformed_protein = protein.get_transformed_atoms(topology,actions)
                prot_rmsd = copy.deepcopy(transformed_protein)
                #protein.write_complex(transformed_protein,decoy_number,decoy_pref,out_path)
                #decoy_rmsd = protein.calc_rmsd(prot_rmsd,protein.native)
                #decoy_energy_new = protein.get_physics_score_with_manual_weights_full(prot_rmsd,penalty)
                decoy_energy_new = protein.get_physics_score_plus_weighted_misc(prot_rmsd)
                decoy_rmsd = protein.get_all_combination_rmsd(prot_rmsd,protein.native)
                decoy_information_list.append([lr + '_' + str(decoy_number),decoy_energy_new,decoy_rmsd])
                #print(str(lr +'_' + str(decoy_number)),decoy_energy,decoy_rmsd)

        fh.close()
    with open(rmsd_outfile, 'w') as f:
        for lines in decoy_information_list:
            f.write('%s\t%s\t%s\n' % (lines[0], lines[1], lines[2]))
except Exception as e:
	print(e)#.stackTrace())


