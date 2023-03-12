AFM-RL
## Citation:
```
@article{,
  title={AFM-RL: Large Protein Complex Docking Using AlphaFold-Multimer and Reinforcement Learning.},
  author={Tunde Aderinwale, Rashidedin Jahandideh, and Daisuke Kihara},
  journal={}
}
```

## What is AFM-RL?

**AFM-RL is a general RL framework for assembling large protein complex.**
AFM-RL is an extension of our RL-MLZerD framework for multiple docking. AFM-RL leverages the RL framework alongside AlphaFold-Multimer (AFM). 
AFM is used to generate  a set of pairwise model for all possible combination of pairs in the complex. The set of pairwise models are the set of actions available to the RL agent. The set of different combinations of chains as they are being assembled is defined as the state for the agent. The assembled models are evaluated, and rewards are assigned based on the model quality. 


## Framework Flowchart
**There are 4 major steps performed by the AFM-RL agent**   
- Generate AFM models: This is the first step where redundant pairwise models are generated.
- onvert AFM models to RL Framework input: The second step is a post-processing step done on AFM output models to convert to RL framwork input. Theses correspond to the pairwise poses available for assemble (In this work we set all possible actions to be 75 - [i.e 75 different pairwise poses])
- Run RL-Search Framework: The third step is the RL assembly stage. The RL agent explore plausible combinations of the pairs and the exploration path with most reward. The RL agent is assigned a reward based on the model quality. Which serves as a signal back to it as it explore and exploit the available state/action environment. 
- Generate output model: Finally, the resulting assembly of the complex is generated.  


## Dependencies

- scipy: pip/conda install scipy==0.12
- BioPython: pip/conda install biopython==3.7.0
- Numpy: pip/conda install numpy

## Usage

Step 1: 
```
python3 create_afm_pair_jobs.py input.fasta.
then ./submit_af220.sh input.fasta
```

Step 2:
```
python3 prep_af_pair_for_docking.py input.fasta.
then  python3 reate_rl_running_path.py
```

Step 3:

```
python3 run_qagent.py: 
  --protein name of target for docking. i.e 1AVO 
  --nofchains number of chains. i.e 14 
  --chains name of each chains. i.e ABCDEFGHIJKLMN
  --path this is the path to the directory containing data directory, i.e ./  
  --episodes number of episodes to simulate. i.e 50000
  --pool_size size of parwise pool, corresponding to number of available action. i.e 75
  --out_dir output directory for results. i.e 1AVO_docking
  --classifier which classifier to use, i.e sgd
  --clash_threshold  threshold for allowed clashes in a model. Defaults to 300. Advisable to increase it in proportional to # of chains

  e.g python3 run_qagent.py --protein 1AVO --nofchains 14 --chains ABCDEFGHIJKLMN --path ./ --clash_threshold 1500 --pair_clash_threshold 20 --lr 1.0 --episodes 10000 --pool_size 75 --out_dir example_1AVO_assembly --metro_threshold 100 --not_int_pair ""  --int_pair ""  --terminal_thres 100 --unbound "" --classifier 1 --fit_score 1

```

Step 4:
After simulation, to generate the pdb files for all the assembled and accepted structure run:

```
python3 generate_mcrl_decoy_parallel_afm.py example_1AVO_assembly ABCDEFGHIJKLMN 1.0 out no
```

## License
**MIT License**
Copyright (c) 2021 Tunde Aderinwale, Rashidedin Jahandideh, Daisuke Kihara, and Purdue University 

