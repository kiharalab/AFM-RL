import Bio.PDB as bpdb
#from node import Node
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import itertools
from subprocess import run, PIPE
from io import StringIO
import math
import traceback
import sys
import pickle

class Protein:
    def __init__(self,chain_length,chain_list,protein_name,threshold,pairthreshold,metr_thres,log,path,lr,eps,outpath,term_thres,dsize,unbound='no',f_cla='lr',f_o=0):
        self.no_chains = chain_length
        self.chain_list = chain_list
        self.chain_pdb = [0] * chain_length
        self.unbound = '/unbound' if unbound == 'yes' else ''
        self.complex_name = protein_name
        self.datapath = path + '/data/mcrl_v2/' + self.complex_name + self.unbound + '/'
        self.outpath = outpath
        self.dsize = dsize
        self.pdbparser = bpdb.PDBParser(PERMISSIVE=1)
        #self.read_chains()
        self.interactions = self.build_interactions(self.no_chains)
        self.pairwisedata = {}
        #self.read_pairwise_data()
        self.energydata = {}
        self.read_pairwise_energy()
        self.current_best_score = float("inf")
        self.clash_threshold = threshold
        self.pair_clash_threshold = pairthreshold
        self.logger = log
        self.energy_tragetory = []
        self.lr = str(lr)
        self.eps = str(eps) # no using it again
        self.tabu_list = []
        self.metro_thres = metr_thres
        self.clash_rejection_count = 0
        self.accepted_via_metro_count = 0
        self.goal_states = {}
        self.restart = False
        self.terminal_state_thres = term_thres
        self.setup_cvdata()
        self.fit_classifier = f_cla# lr or sgd
        self.fitting_option = f_o


    def setup_cvdata(self):
        self.pdb_fold_key = {'2IAA': 0,'1IOD': 0,'1A6A': 0,'1W88': 0,'1NNU': 0,'1VPN': 0,
                             '1QGW': 1,'1W85': 1,'4IHH': 1,'1IZB': 1,'1RHM': 1,'6RLX': 1,
                             '1B9X': 2,'1CT1': 2,'4RT4': 2,'1CYD': 2,'3LL8': 2,'1DJR': 2,
                             '1A0R': 3,'1NVV': 3,'4FTG': 3,'1D1I': 3,'1CN3': 3,'1VCB': 3,
                             '1BCP': 4,'1E6V': 4,'1E6Y': 4,'1XU1': 4,'2A6Q': 4,'2AFH': 4,
			     '2GA4': 4,'3AB0': 4,'3DHG': 4,'4F7U': 4,'4GWP': 4,'4M75': 4,
			     '4MBE': 4,'4Y5R': 4,'5OF3': 4,'5WHB': 4,'5X3T': 4,'5YQG': 4,
			     '6GFO': 4,'6M8R': 4,'6SSP': 4,'6VVO': 4}
        
        self.pdb_folds = [['2IAA', '1IOD', '1A6A', '1W88', '1NNU', '1VPN'],
                          ['1QGW', '1W85', '4IHH', '1IZB', '1RHM', '6RLX'],
                          ['1B9X', '1CT1', '4RT4', '1CYD', '3LL8', '1DJR'],
                          ['1A0R', '1NVV', '4FTG', '1D1I', '1CN3', '1VCB']]
        
        self.fold_mean = [[59814.9139, 39576.2861, 26.8545343, 1182.23269],
                          [74566.7756, 44486.7951, 28.0968089, 1527.28827],
                          [75434.0507, 55616.232, 31.0517583, 1602.41573],
                          [67268.4053, 50562.747, 30.0913716, 1470.42086],
			  [107394.6,74619.17,38.25,1913.36]]
        
        self.fold_std = [[85129.3555, 29148.116, 8.82279123, 1038.77187],
                         [99815.7165, 30754.9865, 8.62248953, 1288.64961],
                         [106655.886, 34150.0143, 9.77033796, 1386.82762],
                         [93222.1031, 30737.5179, 9.08688233, 1219.5189],
			 [175334.13,44421.75,12.32,1962.64]]


        
        self.norm_mean = [[32800.0, 28200.0, 28.4, 477.0, 467000.0, 0.333]]
        self.norm_std  = [[42500.0, 16200.0, 9.77, 572.0, 379000.0, 0.115]]
        self.lr_new_coefs = [[-2.55, -0.63, -7.41, -1.9,   4.89,  4.67]]
        self.lr_new_intercept = [[-6.38]]
        self.xgb_regressor = pickle.load(open('/home/taderinw/project/mcrl_dock_v2/XGBRegressor.pkl', "rb"))
        self.xgb_classify  = pickle.load(open('/home/taderinw/project/mcrl_dock_v2/XGBClassify.pkl', "rb"))

        self.lr_new_coefs_novmqa = [[-1.6,  -5.42, -1.5,  -5.89,  8.47]]
        self.lr_new_intercept_novmqa = [[-3.02]]
        self.xgb_regressor_novmqa = pickle.load(open('/home/taderinw/project/mcrl_dock_v2/XGBRegressor_NoVMQA.pkl', "rb"))
        self.xgb_classify_novmqa  = pickle.load(open('/home/taderinw/project/mcrl_dock_v2/XGBClassify_NoVMQA.pkl', "rb"))

        self.lr_fold_coefs = [[-13.92584679,  -3.89752457,   1.59714855,   3.8197474],
                              [-16.83184463, -10.14854125,   3.68073296,   3.44914561],
                              [-13.08768276,  -6.02503142,   2.71877151,   2.22073669],
                              [-10.17703192,  -5.90930274,   2.6883329,    1.98685209],
			      [-2.98, -1.2 , -0.3 , -1.64]]
                 
        self.lr_fold_intercept = [-8.97249568,-16.04887972,-11.1165083,-8.98509789,-4.32]
        
        self.sgd_fold_coefs = [[-13.00625316,  -3.28622772,   1.09738135,   3.77927834],
                               [-17.12387764, -10.90165737,   3.31910653,   3.38697912],
                               [-12.58308174,  -5.62206348,   3.10180099,   1.88807282],
                               [-9.28318724, -5.60035135,  2.37184435,  1.65089818]]
        
        self.sgd_fold_intercept = [-8.71728076,-16.05284388,-11.09491896,-8.68515076]
    
    def get_fitted_model_score(self,protein,episode):
        try:
            # structure = bpdb.StructureBuilder.Model(0) # create empty structure
            # for x in range(0,len(protein)): # loop through rest of chain
            #     structure.add(protein[x])

            if self.fitting_option == 4:
                return self.get_fitted_model_score_new_lr_model(protein,episode)
            elif self.fitting_option == 5:
                return self.get_fitted_model_score_xgb_classify(protein,episode)
            elif self.fitting_option == 6:
                return self.get_fitted_model_score_xgb_regressor(protein,episode)
            elif self.fitting_option == 7:
                return self.get_fitted_model_score_new_lr_model_reverse(protein,episode)
            elif self.fitting_option == 8:
                return self.get_fitted_model_score_new_lr_model_novmqa(protein,episode)
            elif self.fitting_option == 9:
                return self.get_fitted_model_score_xgb_classify_novmqa(protein,episode)
            elif self.fitting_option == 10:
                return self.get_fitted_model_score_xgb_regressor_novmqa(protein,episode)
            elif self.fitting_option == 11:
                return self.get_fitted_model_score_new_lr_model_reverse_novmqa(protein,episode)


            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)

            #print('complete_score: ',complete_score)

            f_idx = self.fitting_option #self.pdb_fold_key[self.complex_name.split('_')[0]] # protein name is being split because of _ added to the variations
            
            mean = self.fold_mean[f_idx] 
            std  = self.fold_std[f_idx]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            
            if self.fit_classifier == "lr":
                coefs = self.lr_fold_coefs[f_idx]
                intercept = self.lr_fold_intercept[f_idx]
            else:
                coefs = self.sgd_fold_coefs[f_idx]
                intercept = self.sgd_fold_intercept[f_idx]

            weighted_score = np.dot(score_norm, coefs) + intercept

            #flip the size so we can keep minimizing
            weighted_score = weighted_score * -1.0

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score
                             
    def get_fitted_model_score_new_lr_model(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            complete_score.append(voromqa)

            mean = self.norm_mean[0] 
            std  = self.norm_std[0]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)

            coefs = self.lr_new_coefs[0]
            intercept = self.lr_new_intercept[0]
            #print(complete_score,coefs,intercept,score_norm)
            weighted_score = np.dot(score_norm, coefs) + intercept
            #print(weighted_score)
            #flip the size so we can keep minimizing
            weighted_score = weighted_score[0] #* -1.0
            

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score

    def get_fitted_model_score_new_lr_model_novmqa(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            #voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            #complete_score.append(voromqa)

            mean = self.norm_mean[0][:-1]
            std  = self.norm_std[0][:-1]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)

            coefs = self.lr_new_coefs_novmqa[0]
            intercept = self.lr_new_intercept_novmqa[0]
            #print(complete_score,coefs,intercept,score_norm)
            weighted_score = np.dot(score_norm, coefs) + intercept
            #print(weighted_score)
            #flip the size so we can keep minimizing
            weighted_score = weighted_score[0] #* -1.0


        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")

        return weighted_score

    def get_fitted_model_score_new_lr_model_reverse(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            complete_score.append(voromqa)

            mean = self.norm_mean[0] 
            std  = self.norm_std[0]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)

            coefs = self.lr_new_coefs[0]
            intercept = self.lr_new_intercept[0]
            #print(complete_score,coefs,intercept,score_norm)
            weighted_score = np.dot(score_norm, coefs) + intercept
            #print(weighted_score)
            #flip the size so we can keep minimizing
            weighted_score = weighted_score[0] * -1.0
            

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score

    def get_fitted_model_score_new_lr_model_reverse_novmqa(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            #voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            #complete_score.append(voromqa)

            mean = self.norm_mean_novmqa[0]
            std  = self.norm_std_novmqa[0]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)

            coefs = self.lr_new_coefs_novmqa[0]
            intercept = self.lr_new_intercept_novmqa[0]
            #print(complete_score,coefs,intercept,score_norm)
            weighted_score = np.dot(score_norm, coefs) + intercept
            #print(weighted_score)
            #flip the size so we can keep minimizing
            weighted_score = weighted_score[0] * -1.0

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")

        return weighted_score

    def get_fitted_model_score_xgb_classify(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            complete_score.append(voromqa)
            
            print(complete_score)
            mean = self.norm_mean[0] 
            std  = self.norm_std[0]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            score_norm = np.array(score_norm).reshape(1,-1)

            weighted_score = self.xgb_classify.predict_proba(score_norm)[0][0] # get the prediction that this model belong to class 0 (i.e irmsd > 4.0)

            weighted_score = weighted_score #* -1.0 no need to flip sign, just minimize the probability. 

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score

    def get_fitted_model_score_xgb_classify_novmqa(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            #voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            #complete_score.append(voromqa)

            #print(complete_score)
            mean = self.norm_mean[0][:-1]
            std  = self.norm_std[0][:-1]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            score_norm = np.array(score_norm).reshape(1,-1)

            weighted_score = self.xgb_classify_novmqa.predict_proba(score_norm)[0][0] # get the prediction that this model belong to class 0 (i.e irmsd > 4.0)

            weighted_score = weighted_score #* -1.0 no need to flip sign, just minimize the probability. 

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")

        return weighted_score

    def get_fitted_model_score_xgb_regressor(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            complete_score.append(voromqa)

            mean = self.norm_mean[0] 
            std  = self.norm_std[0]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            score_norm = np.array(score_norm).reshape(1,-1)

            weighted_score = self.xgb_regressor.predict(score_norm)[0]

            weighted_score = weighted_score

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score

    def get_fitted_model_score_xgb_regressor_novmqa(self,protein,episode):
        try:
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            itscore = self.get_it_score(protein,episode)
            #voromqa = self.get_voromqa_score(protein,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)
            complete_score.append(itscore)
            #complete_score.append(voromqa)

            mean = self.norm_mean[0][:-1]
            std  = self.norm_std[0][:-1]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            score_norm = np.array(score_norm).reshape(1,-1)

            weighted_score = self.xgb_regressor_novmqa.predict(score_norm)[0]

            weighted_score = weighted_score

        except Exception as e:
            weighted_score = float("inf")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            exit()
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")

        return weighted_score

    def build_interactions(self,no_chains):
        interaction_list = []
        for i in range(0,no_chains):
            for j in range(i,no_chains):
                if i==j:continue
                interaction_list.append([i,j])
        return interaction_list

    def get_transformed_atoms(self,chain_list,prediction):
        result  = [0] * self.no_chains # to hold all result
        base_index = chain_list[0][0] # get base protein chain index
        base_protein = self.chain_pdb[base_index] # get base protein chain which contains atoms
        result[base_index] = base_protein # add first chain to it's position unaltered

        rotations = self.get_rotational_paths(chain_list)
        # print('rotations from rotation_path : ', rotations)
        invert = False
        for rotation_index in range(0,len(rotations)):
            rotation_path = rotations[rotation_index]
            protein_index = rotation_path[-1]
            cumulative_transformation = self.chain_pdb[protein_index] # get the protein chain to transform
            after_current_transformation = [] # not needed
            while len(rotation_path) > 1:
                after_current_transformation = [] # not really needed
                lig = rotation_path[len(rotation_path) - 1]
                rec = rotation_path[len(rotation_path) - 2]

                found,edge,pos = self.find_edge(chain_list,rec,lig)
                if found == False:
                    found,edge,pos = self.find_edge(chain_list,lig,rec)
                    invert = True
                after_current_transformation = self.transform_chain(cumulative_transformation,str(edge),prediction[pos],invert)
                invert = False # reseting invert flag
                cumulative_transformation = after_current_transformation
                rotation_path = rotation_path[:-1]
            result[protein_index] = cumulative_transformation
        # print('resulting complex : ',result)
        return result


    def get_transformed_atoms_afm(self,chain_list,prediction):
        base_index = chain_list[0]#[0]
        #print('chain list and prediciton',chain_list,prediction)
        #print('base_index[0]',base_index[0])
        #print('self.chain_list[base_index[0]]',self.chain_list[base_index[0]])
        base_struct_path = self.datapath + self.chain_list[base_index[0]] + '-' + self.chain_list[base_index[1]] + '-'+ str(prediction[0]) + '.pdb'
        #print('base_struct_path',base_struct_path)
        base_structure = self.pdbparser.get_structure('',base_struct_path)[0]
        assembled_chains = [x.get_id() for x in base_structure.get_chains()]
        #print('assembled_chains',assembled_chains)

        for i in range(1,len(chain_list)):
            #print(list(base_structure.get_chains()))
            next_topology = chain_list[i]
            next_action = prediction[i]

            next_struct_path = self.datapath + self.chain_list[next_topology[0]] + '-' + self.chain_list[next_topology[1]] + '-'+ str(next_action) + '.pdb'
            next_structure = self.pdbparser.get_structure('',next_struct_path)[0]
            next_ids = [x.get_id() for x in next_structure.get_chains()]
            #print(next_ids)
            # which id is being super-imposed and which id is being added to the assemble
            if next_ids[0] in assembled_chains:
                super_imp_id = next_ids[0]
                new_chain_id = next_ids[1]
            else:
                super_imp_id = next_ids[1]
                new_chain_id = next_ids[0]

            current_atoms = [atom for atom in base_structure[super_imp_id].get_atoms() if atom.name =="CA"]
            super_imp_atoms = [atom for atom in next_structure[super_imp_id].get_atoms() if atom.name =="CA"]
            #print(len(current_atoms),len(super_imp_atoms))
            #print(current_atoms,super_imp_atoms)

            sup = bpdb.Superimposer()
            sup.set_atoms(current_atoms, super_imp_atoms)
            sup.apply(super_imp_atoms)
            rmsd = sup.rms
            
            rotation = sup.rotran[0]
            translation = sup.rotran[1]

            # make a copy, transform and add to the current assemble
            new_chain = next_structure[new_chain_id].copy()
            new_chain.transform(rotation,translation)
            base_structure.add(new_chain)

            assembled_chains.append(new_chain_id)

        #print('final_structure chains : ', list(base_structure.get_chains()))
        return base_structure


    def get_rotational_paths(self,all_edges):
        these_edges = all_edges.copy()
        result = []
        candidates_queue = []
        root_index = all_edges[0][0] # Doesn't neccesary starts at 0
        initial_base = []
        initial_base.append(root_index) 
        candidates_queue.append(initial_base)
        while len(candidates_queue) > 0:
            current_base = candidates_queue[-1]
            candidates_queue = candidates_queue[:-1] # remove because it's not considered again
            current_vertex = current_base[-1]
            for i in reversed(these_edges):
                current_edge = i
                if current_edge[0] == current_vertex or current_edge[1] == current_vertex:
                    other_edge = current_edge[1] if current_edge[0] == current_vertex else current_edge[0]
                    correct_path = current_base.copy()
                    correct_path.append(other_edge)
                    result.append(correct_path)

                    candidates_queue.append(correct_path)
                    candidates_queue.append(current_base)

                    these_edges.remove(i)
        return result


    def find_edge(self,all_edges,rec,lig):
        result = []
        found = False
        pos = None
        for x in range(0,len(all_edges)):
            current = all_edges[x]
            if current[0] == rec and current[1] == lig:
                result = current
                found = True
                pos = x
                break
        return [found,result,pos]


    def make_complex(self,chain_list,decoy_index):
    	#self.make_spanning_tree(chain_list)
    	chain_transformations = []
    	for x in range(0,self.no_chains):
    		chain_name = self.chain_list[x]
    		chain_parent = self.spanning_tree.get_parents(chain_name) # get this in reverse
    		current_transformation = self.chain_pdb[x]
    		if len(chain_parent) == 0:#root node
    			chain_transformations.append(current_transformation) #no transformation needed
    		else:
    			for x in chain_parent: #TODO add pair to tree creation
    				current_transformation = self.transform_chain(current_transformation,x.pair,decoy_index)
    			chain_transformations.append(current_transformation)

    	return chain_transformations

    def tabu_complex_list(self,chain_list,decoy_index):
        if str(chain_list + decoy_index) in self.tabu_list:
            return True
        else:
            return False

    def print_metro_stats(self):
        self.logger.info("Total number of Rejected models due to clashes : " + str(self.clash_rejection_count))
        self.logger.info("Total number of Accepted models via metropolic : " + str(self.accepted_via_metro_count))

    def update_tabu_complex_list(self,chain_list,decoy_index):
        self.tabu_list.append(str(chain_list + decoy_index))


    def update_goal_state_stats(self,chain_list,decoy_index):
        new_state = str(chain_list) + str(decoy_index)
        if new_state in self.goal_states:
            self.goal_states[new_state] += 1
            if self.goal_states[new_state] == self.terminal_state_thres:
                self.restart = True
                self.goal_states[new_state] = 0
        else:
            self.goal_states[new_state] = 1
        

    def make_and_score_complex_new(self,chain_list,decoy_index,episode):
        reward = 0
        #If complex is already in list of infeasible protein list then no need to check again
        if self.tabu_complex_list(chain_list,decoy_index):
            reward -2
            self.clash_rejection_count += 1
            return reward

        transformed_protein = self.get_transformed_atoms(chain_list,decoy_index)

        model_score = self.get_physics_score(transformed_protein,episode)
        #model_score = self.get_it_score(transformed_protein,episode)

        if model_score == float("inf"):
                reward = 0
        elif model_score <= self.current_best_score:
            self.current_best_score = model_score
            self.logger.info("Decoy " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(model_score))
            self.energy_tragetory.append(model_score) #kepping track of found energies
            reward = 100
        else:
            atom_clash = 0
            #atom_clash = self.calculate_atom_clashes(transformed_protein,3,episode)
            if atom_clash >= self.clash_threshold:
                self.update_tabu_complex_list(chain_list,decoy_index)
                self.clash_rejection_count += 1
                reward = -2 # too many clash
            else:
                #implement metropolic criterion
                energy_diff = model_score - self.current_best_score
                temp = (-energy_diff)/np.log(self.metro_thres)
                prob = np.exp(-(energy_diff) / temp)
                if np.random.uniform() < prob:
                    self.logger.info("Decoy " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(model_score))
                    #self.write_complex(transformed_protein,episode)
                    reward = prob * 100
                    self.accepted_via_metro_count += 1
                else:
                    self.logger.info("Decoy " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(model_score))
                    reward = 10
        
        return reward

    def intermidiate_thres_v1(self,current_length):
        if current_length == 2:
            return 100
        elif current_length == 3:
            return 200
        elif current_length == 4:
            return 300
        elif current_length == 5:
            return 400
        elif current_length == 6:
            return 500
        elif current_length == 7:
            return 600
        elif current_length == 8:
            return 700
        elif current_length == 9:
            return 800
        elif current_length == 10:
            return 900

    def intermidiate_thres(self,current_length):
        if current_length == 2:
            return 200
        elif current_length == 3:
            return 300
        elif current_length == 4:
            return 400
        elif current_length == 5:
            return 500
        elif current_length == 6:
            return 650
        elif current_length == 7:
            return 750
        elif current_length == 8:
            return 850
        elif current_length == 9:
            return 950
        elif current_length == 10:
            return 1000
        elif current_length == 11:
            return 1050
        elif current_length == 12:
            return 1100
        elif current_length == 13:
            return 1150
        elif current_length == 14:
            return 1200
        elif current_length == 15:
            return 1250
            

    def validate_progress(self,chain_list,decoy_index):
        if len(chain_list) == 0:
            return False
        #print('*****',chain_list,decoy_index)
        transformed_protein = self.get_transformed_atoms_afm(chain_list,decoy_index)
        atom_clash = 0
        atom_clash = self.calculate_atom_clashes(transformed_protein,3,1)
        #print('atom_clash : ',atom_clash)
        chain_length = len(decoy_index) + 1 
        if atom_clash >= self.intermidiate_thres(chain_length):
            return True
        else:
            return False

    def make_and_score_complex(self,chain_list,decoy_index,episode):
        reward = 0

        # if episode == self.terminal_state_thres:
        #     self.restart = True
            
        #If complex is already in list of infeasible protein list then no need to check again
        if self.tabu_complex_list(chain_list,decoy_index):
            reward -5
            self.clash_rejection_count += 1
            self.logger.info("NOTACCEPTED " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(10000000))
            return reward

        transformed_protein = self.get_transformed_atoms_afm(chain_list,decoy_index)
        atom_clash = 0
        atom_clash = self.calculate_atom_clashes(transformed_protein,3,episode)
        #print('atom_clash : ',atom_clash)
        #print('self.clash_threshold :',self.clash_threshold)
        if atom_clash >= self.clash_threshold:
    		# print('Too many clashes, penalising... no scoring needed')
            self.update_tabu_complex_list(chain_list,decoy_index)
            self.clash_rejection_count += 1
            self.logger.info("NOTACCEPTED " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(10000000))
            #self.write_complex_v2(transformed_protein,episode)
            reward = -5 # too many clash
        else:
            #model_score = self.get_physics_score_tunned(transformed_protein,episode)
            #model_score = self.get_physics_score_with_manual_weights(transformed_protein,episode,10)
            model_score = self.get_fitted_model_score(transformed_protein,episode)
            #print('model_score :',model_score)
            #model_score = self.get_voromqa_score(transformed_protein,episode) #self.get_physics_score_plus_weighted_misc(transformed_protein,episode)

            #Error Occurred while calculating physics score, just ignore episode and moveon
            if model_score == float("inf"):
                reward = 0
            elif model_score <= self.current_best_score: #original
            #elif model_score >= self.current_best_score:
                self.current_best_score = model_score
                self.logger.info("Decoy " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(model_score))
                self.energy_tragetory.append(model_score) #kepping track of found energies

                #track visit count for this goal state
                self.update_goal_state_stats(chain_list,decoy_index)

    			#write good complex to file
                #self.write_complex(transformed_protein,episode)
                #self.write_complex_v2(transformed_protein,episode)
                reward = 100
            else:
                #implement metropolic criterion
                energy_diff = model_score - self.current_best_score
                temp = (-energy_diff)/np.log(self.metro_thres)
                prob = np.exp(-(energy_diff) / temp)
                if np.random.uniform() < prob:
                    self.logger.info("Decoy " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(model_score))
                    #self.write_complex(transformed_protein,episode)
                    reward = prob * 100
                    self.accepted_via_metro_count += 1
                else:
                    self.logger.info("Decoy " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", " + str(model_score))
                    reward = -1
                # reward = -1
    	#if episode%100 == 0:
    	#	self.write_complex(transformed_protein,episode)
        return reward

    def get_physics_score(self,protein,episode):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(protein)): # loop through rest of chain
                structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)
            output = StringIO()
            io.save(output)
            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            score = float(p.stdout)
        except Exception as e:
            score = float("inf")
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return score

    def get_physics_score_with_manual_weights(self,protein,episode,penalty):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(protein)): # loop through rest of chain
                structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)
            output = StringIO()
            io.save(output)
            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            score = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)

            score += sasa * penalty
            score += rg * penalty
            score += clashes * penalty


        except Exception as e:
            score = float("inf")
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return score

    def get_physics_score_subset(self,protein,episode):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(protein)): # loop through rest of chain
                structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)
            output = StringIO()
            io.save(output)
            p = run(['./score_subset', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            score = float(p.stdout)
        except Exception as e:
            score = float("inf")
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return score


    def get_physics_score_tunned(self,protein,episode):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(protein)): # loop through rest of chain
                structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)
            output = StringIO()
            io.save(output)
            p = run(['./score_olddata_2_5cutoff_norm', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            score = float(p.stdout)
        except Exception as e:
            score = float("inf")
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return score

    def get_voromqa_score(self,protein,episode):
        try:
            #structure = bpdb.StructureBuilder.Model(0) # create empty structure
            #for x in range(0,len(protein)): # loop through rest of chain
            #    structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)

            p = run(['/home/taderinw/bin/voronota/voronota-voromqa'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout.strip('-').split()[0])

            weighted_score = physcore #* -1.0

        except Exception as e:
            weighted_score = float("inf")
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score

    def get_physics_score_plus_weighted_misc(self,protein,episode):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(protein)): # loop through rest of chain
                structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)
            output = StringIO()
            io.save(output)

            p = run(['./score', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            physcore = float(p.stdout)

            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)

            complete_score = [physcore]
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)

            mean = [107394.6,74619.17,38.25,1913.36]
            std  = [175334.13,44421.75,12.32,1962.64]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            
            coefs = [-2.98, -1.2 , -0.3 , -1.64]
            intercept = -4.32

            weighted_score = np.dot(score_norm, coefs) + intercept

            #flip the size so we can keep minimizing
            weighted_score = weighted_score * -1.0

        except Exception as e:
            weighted_score = float("inf")
            self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score

    def get_physics_score_tunned_with_rg_clash(self,protein,episode):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(protein)): # loop through rest of chain
                structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)
            output = StringIO()
            io.save(output)
            p = run(['./score_full', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            score = p.stdout.strip('\n')
            phy_score = score.split(',')[1::]
            phy_score = [float(x) for x in phy_score]
            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            
            complete_score = phy_score
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)

            mean = [4391.63,-2058.54,450907.81,22755.02,-22074.52,-6462.22,24778.38,6510.02,-205.49,-2300.68,-362.6,-381.29,65856.18,35.31,1547.85]
            std = [4504.81,1955.77,464942.98,21142.89,22126.09,7146.71,24520.86,7241.01,204.71,3108.29,256.33,486.32,41827.64,10.84,1313.28]
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            
            coefs = [55.3,-159.0,-79.3,7.44,462.0,47.2,362.0,32.1,0.221,0.944,-4.7,-20.5,-9.12,-5.64,-110.0]
            intercept = -81.85

            weighted_score = np.dot(score_norm, coefs) + intercept
            #weighted_score = abs(weighted_score)

        except Exception as e:
            weighted_score = float("inf")
            #self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score


    def get_physics_score_tunned_with_rg_clash_v2(self,protein,episode):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(protein)): # loop through rest of chain
                structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)
            output = StringIO()
            io.save(output)
            p = run(['./score_full', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            score = p.stdout.strip('\n')
            phy_score = score.split(',')[1::]
            phy_score = [float(x) for x in phy_score]
            sasa = self.get_surface_area(protein)
            rg = self.radius_of_gyration(protein)
            clashes = self.calculate_atom_clashes(protein,3,episode)
            
            complete_score = phy_score
            complete_score.append(sasa)
            complete_score.append(rg)
            complete_score.append(clashes)

            mean = [4403.6,-2063.82,452132.05,22811.37,-22128.24,-6472.72,24838.11,6519.94,-205.97,-2300.43,-362.81,-382.44,65977.68,35.35,1551.77]
            std = [4508.7,1957.98,465358.65,21187.36,22135.14,7150.98,24533.45,7245.24,204.82,3120.4,255.74,485.54,41809.81,10.84,1315.22]
                  
            score_norm = (np.array(complete_score) - np.array(mean)) / np.array(std)
            
            coefs = [1.16,15.11,-6.94,12.41,18.88,7.46,31.15,6.31,-0.05,-3.32,-2.32,-10.2,-14.,1.19,-14.65]
            intercept = -13.82

            weighted_score = np.dot(score_norm, coefs) + intercept
            #weighted_score = abs(weighted_score)

        except Exception as e:
            weighted_score = float("inf")
            #self.logger.info("Exception occured while trying to calculate physics score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        
        return weighted_score


    def get_it_score(self,protein,episode):
        try:
            #structure = bpdb.StructureBuilder.Model(0) # create empty structure
            #for x in range(0,len(protein)): # loop through rest of chain
            #    structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)
            moving_chains = np.random.choice(self.chain_list)
            p = run(['./itscore_mcrl', ' ',''+moving_chains+''], encoding='ascii',stdout=PIPE,input=output.getvalue())
            score = float(p.stdout.split()[-1])
        except Exception as e:
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            score = float("inf")
            self.logger.info("Exception occured while trying to calculate it_score at Episode : " + str(episode) + ", Error details : " + str(e) + " ")
        return score


    def get_surface_area(self,protein):
        try:
            # structure = bpdb.StructureBuilder.Model(0) # create empty structure
            # for x in range(0,len(protein)): # loop through rest of chain
            #     structure.add(protein[x])
            io = bpdb.PDBIO()
            io.set_structure(protein)
            output = StringIO()
            io.save(output)
            p = run(['./score_area', '--weight','all'],input=output.getvalue(), encoding='ascii',stdout=PIPE)
            score = float(p.stdout)
        except Exception as e:
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            #exit()
            score = float("inf")
        
        return score



    def radius_of_gyration(self,decoy):
        coord = list()
        mass = list()
        structure = bpdb.StructureBuilder.Model(0) # create empty structure
        for x in range(0,len(decoy)):
            if decoy[x] != 0:
                structure.add(decoy[x])

        for atom in structure.get_atoms():
            coord.append(list(atom.get_coord()))
            mass.append(atom.mass)

        # below calculation taken from pymol implemenatation
        xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord, mass)]
        tmass = sum(mass)
        rr = sum(mi*i + mj*j + mk*k for (i, j, k), (mi, mj, mk) in zip(coord, xm))
        mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
        rg = math.sqrt(rr / tmass-mm)
        return(round(rg, 3))


    def read_chains(self):
    	for x in range(0,self.no_chains):
    		url = self.datapath + self.chain_list[x] + '-' + self.complex_name + '.pdb'
    		model = self.pdbparser.get_structure(x,url)[0]
    		chain_name = self.chain_list[x]
    		self.chain_pdb[x] = model[str(chain_name)] # self.pdbparser.get_structure(x,url)[0][str(self.chain_list[x])]
    	# print('self.chain_pdb',self.chain_pdb)

    def read_pairwise_data(self):
    	for x in self.interactions:
    		url = self.datapath + str(self.chain_list[x[0]]) + '-' + str(self.chain_list[x[1]]) + '.out.filtered.pure'
    		self.pairwisedata[str(x)] = pd.read_csv(url,delimiter=' ',header=None) # use delimiter='\s+' for regular .out file


    def read_pairwise_energy(self):
        for x in self.interactions:
            url = self.datapath + str(self.chain_list[x[0]]) + '-' + str(self.chain_list[x[1]]) + '.ranksum'
            #energy = pd.read_csv(url,delimiter=',')
            #min_index = energy[1].nsmallest().index#energy[1].idxmin()
            min_index = [0,1,2,3,4]#energy['ranksum'][0:self.dsize].nsmallest().index
            self.energydata[str(x)] = min_index


    def transform_chain(self,chain_obj,chain_pair,data_loc,invert):
    	# print('Currently transforming chain : ',chain_obj.get_id(),' using chain_pair : ',chain_pair, ' and position', data_loc, ' invert is :', invert)
    	rotation = np.array(self.pairwisedata[str(chain_pair)].loc[data_loc,:8]).reshape(3,3)
    	translation = np.array(self.pairwisedata[str(chain_pair)].loc[data_loc,9:11]).reshape(3)
    	chain_copy = chain_obj.copy()
    	
    	if invert:
    		rot = np.eye(3)
    		chain_copy.transform(rot,(-translation.T))
    		chain_copy.transform(rotation,np.zeros(3))
    	else:
    		chain_copy.transform(rotation.T,translation.T)

    	return chain_copy



    def write_complex_v2(self,complex_structure,episode):
        try:
            # structure = bpdb.StructureBuilder.Model(0) # create empty structure
            # for x in range(0,len(complex_structure)): # loop through rest of chain
            #     structure.add(complex_structure[x])
            io = bpdb.PDBIO()
            io.set_structure(complex_structure)            
            output_file = self.outpath + 'output/' + 'decoy_' + str(episode) + '_' + self.lr + '_.pdb'
            #print(output_file)
            io.save(output_file)
        except Exception as e:
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            self.logger.info("Exception occured while trying to write complex to file at Episode :  " + str(episode) + ", Error details : " + str(e) + " ")

    def write_complex(self,complex_structure,episode):
        try:
            structure = bpdb.StructureBuilder.Model(0) # create empty structure
            for x in range(0,len(complex_structure)): # loop through rest of chain
                structure.add(complex_structure[x])
            io = bpdb.PDBIO()
            io.set_structure(structure)            
            output_file = self.outpath + 'output/' + 'decoy_' + str(episode) + '_' + self.lr + '_.pdb'
            print(output_file)
            io.save(output_file)
        except Exception as e:
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            self.logger.info("Exception occured while trying to write complex to file at Episode :  " + str(episode) + ", Error details : " + str(e) + " ")

    def radius_of_gyration(self,decoy):
        coord = list()
        mass = list()
        # structure = bpdb.StructureBuilder.Model(0) # create empty structure
        # for x in range(0,len(decoy)):
        #     if decoy[x] != 0:
        #         structure.add(decoy[x])
                
        for atom in decoy.get_atoms():
            coord.append(list(atom.get_coord()))
            mass.append(atom.mass)

        # below calculation taken from pymol implemenatation
        xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord, mass)]
        tmass = sum(mass)
        rr = sum(mi*i + mj*j + mk*k for (i, j, k), (mi, mj, mk) in zip(coord, xm))
        mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
        rg = math.sqrt(rr / tmass-mm)
        return(round(rg, 3))



    def calculate_atom_clashes(self,structure,radius,episode):
        structure_kdtree = []
        result = []
        atomlen = 0
        #radius = 1.5 # Todo remove this after testing
        #print(structure)
        for chains in structure:
            chain_cordinates = []
            if chains == 0:
                continue
            for atom in chains.get_atoms():
                chain_cordinates.append(list(atom.coord))
                atomlen += 1
            structure_kdtree.append(cKDTree(np.array(chain_cordinates)))


        for x,y in itertools.combinations(structure_kdtree,2):
            a = x.query_ball_tree(y, radius)
            chain_count = 0
            for element in a:
                if len(element) > 0:
                    chain_count += 1
            if chain_count > self.pair_clash_threshold:
                #print('pair_clash_threshold violated')
                return 1000000

            b = x.query_ball_tree(y, 1.5)
            small_count = 0
            for item in b:
                if len(item)> 0:
                    small_count += 1
            if small_count > 100:
                #print('small_count region violated')
                return 1000000

            result += a

        count = 0
        for i in result:
            if len(i) > 0:
                count += 1

        return count


    def get_interface_residue(self,structure):
        structure_kdtree = []
        chains_residue = {}
        interface_residue = {}
        result = []
        c_id = 0
        chains_atom_record = {}
        chain_list = []
        #append all atom coordinate in chain to kdtree
        for chains in structure:
            chain_cordinates = []
            chains_residue[c_id] = []
            chains_atom_record[c_id] = []
            interface_residue[c_id] = []
            chain_list.append(chains.get_id())
            for atom in chains.get_atoms():
                #if atom.get_name() == 'O':
                #    continue
                chain_cordinates.append(list(atom.coord))
                chains_atom_record[c_id].append(atom)
            structure_kdtree.append(cKDTree(np.array(chain_cordinates)))
            c_id += 1
        
        # get all atom that is in the interface of other chains
        radius = 10
        for x,y in itertools.combinations(structure_kdtree,2):
            a = x.query_ball_tree(y, radius)
            b = y.query_ball_tree(x, radius)
            chains_residue[structure_kdtree.index(y)] += list(itertools.chain(*a))
            chains_residue[structure_kdtree.index(x)] += list(itertools.chain(*b))
        
        # for each interface atom, get residue id
        for i in chains_residue:
            chains_residue[i] = list(set(chains_residue[i]))
            for atoms in chains_residue[i]:
                residue = chains_atom_record[i][atoms].get_parent().get_id()
                if residue[0] == 'W':
                    continue
                interface_residue[i].append(residue[1])
            interface_residue[i] = list(set(interface_residue[i]))

        return interface_residue,chain_list

    def calculate_irmsd(self,native,decoy):
        native_intf_residue,chain_list = self.get_interface_residue(native)
        native_intf_atoms = []
        decoy_intf_atoms = []
        backbone_atom = ["CA","C","N","O"]
    
        for chain in native_intf_residue:
            for resi in native_intf_residue[chain]:
                for atom in native[chain_list[chain]][resi]:
                    if atom.get_name() in backbone_atom:
                        native_intf_atoms.append(atom)
                        
                        if decoy[chain][resi][atom.get_name()]:
                            decoy_intf_atoms.append(decoy[chain][resi][atom.get_name()])

        sup = bpdb.Superimposer()
        sup.set_atoms(native_intf_atoms, decoy_intf_atoms)
        sup.apply(decoy_intf_atoms)
        return round(sup.rms,5)


    def calculate_fnat(self,structure,decoy):
        structure_kdtree = []
        c_id = 0
        chains_atom_record = {}
        chain_list = []
        #append all atom coordinate in chain to kdtree
        for chains in structure:
            chain_cordinates = []
            chains_atom_record[c_id] = []
            chain_list.append(chains.get_id())
            for atom in chains.get_atoms():
                chain_cordinates.append(list(atom.coord))
                chains_atom_record[c_id].append(atom)
            structure_kdtree.append(cKDTree(np.array(chain_cordinates)))
            c_id += 1
        
        # get all atom that is in the interface of other chains
        radius = 5
        contacting_residue_pair = []
        native_contact = 0
        decoy_contact = 0
        for x,y in itertools.combinations(structure_kdtree,2):
            a = x.query_ball_tree(y, radius)            
            for idx in range(0,len(a)):
                if len(a[idx]) > 0:
                    x_idx = structure_kdtree.index(x)
                    y_idx = structure_kdtree.index(y)
                    chain_pair = str(x_idx) + ',' + str(y_idx)
                    x_resi = chains_atom_record[x_idx][idx].get_parent().get_id()[1]
                    decoy_x_residue = list(decoy[x_idx].get_atoms())[idx].get_parent()
                    for y_atom in a[idx]:
                        y_resi = chains_atom_record[y_idx][y_atom].get_parent().get_id()[1]
                        decoy_y_residue = list(decoy[y_idx].get_atoms())[y_atom].get_parent()
                        data = chain_pair + str(x_resi) + ',' + str(y_resi)
                        
                        if data not in contacting_residue_pair:
                            contacting_residue_pair.append(data)
                            native_contact += 1
                            if self.check_contact_between_residue(decoy_x_residue,decoy_y_residue):
                                decoy_contact += 1
                        
        fnat = decoy_contact/native_contact
        return fnat

    def check_contact_between_residue(self,a_residue,b_residue):
        a_atoms = [atom for atom in a_residue]
        b_atoms = [atom for atom in b_residue]
    
        for a_atms in a_atoms:
            for b_atms in b_atoms:
                if np.linalg.norm(a_atms.coord - b_atms.coord) < 5.0:
                    return True
        return False

    def log_decoy_info(self,chain_list,decoy_index,episode):
        self.make_and_score_complex(chain_list,decoy_index,episode)
        #self.logger.info("Decoy " + str(episode) + ", " + str(chain_list) + ", " + str(decoy_index) + ", 999999")


    def validate_progress_with_contrainst(self,chain_list,decoy_index,interacting_pair,non_interacting_pair):
        if len(chain_list) == 0:
            return False
        transformed_protein = self.get_transformed_atoms(chain_list,decoy_index)
        atom_clash = 0
        atom_clash = self.calculate_atom_clashes(transformed_protein,3,1)
        chain_length = len(decoy_index) + 1 
        if atom_clash >= self.intermidiate_thres(chain_length):
            return True
        else:
            if not interacting_pair:
                return False
            if len(non_interacting_pair) == 0:
                return False
            structure_kdtree = []
            atomlen = 0
            radius = 5.0
            for chains in transformed_protein:
                chain_cordinates = []
                if chains == 0:
                    continue
                for atom in chains.get_atoms():
                    chain_cordinates.append(list(atom.coord))
                    atomlen += 1
                structure_kdtree.append(cKDTree(np.array(chain_cordinates)))


            for x,y in itertools.combinations(structure_kdtree,2):
                a = x.query_ball_tree(y, radius)
                chain_count = 0
                for element in a:
                    if len(element) > 0:
                        x_index = structure_kdtree.index(x)
                        y_index = structure_kdtree.index(y) 
                        if y_index not in interacting_pair[x_index]:
                            #print('returning true because x and y should not interact',x_index,y_index)
                            return True
                        
                        if list([x_index,y_index]) in non_interacting_pair or list([y_index,x_index]) in non_interacting_pair:
                            #print('retruing true because x and y are for sure not interacting',x_index,y_index)
                            return True
            return False
