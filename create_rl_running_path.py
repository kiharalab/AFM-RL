import os
import sys

pdb = sys.argv[1]

os.system('mkdir example_pdb')
os.system('cp AF/RESULT_220/sampled_set/' + pdb + '/ready/* example_pdb/')        
os.system('cp AF/RESULT_220/sampled_set/' + pdb + '.* example_pdb')
os.system('mv '+pdb+'.fasta '+pdb+'.fasta.txt')
