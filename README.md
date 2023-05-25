# InduCE: Inductive Counterfactual Explanations for Graph Neural Networks

![InduCE-Pipeline](https://github.com/idea-iitd/InduCE/blob/main/induce_diagram.png)
Fig: Pipeline of the policy learning algorithm in INDUCE. δ indicates the maximum number of allowed perturbations.


### Environment Details
We use Python 3.9.12 to run each file. We use Pytorch '1.11.0' with cuda '10.2.0'. Use ENV.yml file to create the environment required to run the source code. 
```
conda env create --file=ENV.yml
```
### Directory Related Instructions
```
cd code_supplementary
```
<!-- Make the following directories to store results:
```
mkdir logs/syn1
mkdir logs/syn4
mkdir logs/syn5
mkdir results/syn1
mkdir results/syn4
mkdir results/syn5
mkdir saved_model/syn1
mkdir saved_model/syn4
mkdir saved_model/syn5
```
-->

### Execution Instructions

#### INDUCTIVE VERSION
Train:
```
python ./src/train.py --dataset <syn4/syn1/syn5> --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 15 --maxepisodes 500 --seed 42 --train_on_correct_only --k 4 --train_on_non_zero --verbose --save_prefix inductive_non0_correct_only
```
Test:
```
python ./src/test.py --dataset <syn4/syn1/syn5> --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 15 --seed 42 --k 4  --verbose --save_prefix inductive_non0_correct_only 

```
#### TRANSDUCTIVE VERSION
```
python ./src/train_transductive.py --dataset <syn4/syn1/syn5> --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 15 --maxepisodes 500 --seed 37 --train_on_non_zero  --train_on_correct_only --k 4 --verbose --save_prefix transductive_non0_correct_only 
```
*NOTE: 
1. For deletion version: use flag --del_only
2. For checking usage of arguments refer to cmd_args.py

<!-- #### UPDATED RUNNING TIME
We optimized our evaluation time further and achieved a speed-up of 3.67x and 4.1x over InduCE-inductive and InduCE-transductive respectively (i.e., over values reported in the paper). Thus, Table 4 of the paper has been updated as follows:

| Method  | Tree-Cycles | Tree-Grid  | BA-Shapes |
| ------------- | ------------- | ------------- | ------------- |
| GEM  | 0.16  | 0.73  | 8.64  |
| CFGNNExplainer  | 1295.66  | 2382.51  | 3964.36  |
| CF^2(&alpha; = 0.6) | 304.13  | 154.88 | 2627.09 |
| CF^2(&alpha; = 0)  | 165.56  | 249.92  | 2565.87  |
| InduCE-inductive | 4.36  | 17.64  | 68.33  |
| InduCE-transductive  | 66.08  | 331.58  |  6546.48 | 

Table 4: Running times (in seconds) of each algorithm.

According to the above values, we see that InduCE-inductive provides **79x** average speed-up over the transductive baselines as opposed to **28x** (reported in the paper).-->
