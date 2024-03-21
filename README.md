![](new_architecture.png)

# Physics-Informed Graph Neural Networks for Water Distribution Systems

Official Code for the paper "Physics-Informed Graph Neural Networks for Water Distribution Systems". \
All system and package requirements are listed in the document 'REQUIREMENTS.txt'.

## Simulating scenarios to generate data

Data on heads, pressures, demands and flows for WDS can be generated for longer periods of time (Vrachimis et al. https://github.com/KIOS-Research/BattLeDIM) using: 

``` python
python dataset_generator.py
```

A number of arguments can be passed to dataset generation parameters:

``` python
'--wdn'             "Specify the WDS for which you want to simulate the scenarios; default is l_town. Choices are ['hanoi', 'fossolo', 'pescara', 'l_town', 'zhijiang']" 
'--start_scenario'  "Specify the start scenario name, must be an integer; default is 1"
'--end_scenario'    "Specify the end scenario name, must be an integer; default is 50"
'--sigma_dem'       "Specify the standard deviation of the noise to be added to the demand patterns; default is 0.1."   
'--sigma_dia'       "Specify the standard deviation of the noise to be added to the diameters; default is 1/30"   
'--_seed'           "Specify the random seed for noise; default is None, where it will be set to the scenario name for every scenario."   
'--start_time'      "Specify the start time of the simulation; default is 2018-01-01 00:00, the simulation will be done every 30 minutes starting from this time."   
'--end_time'        "Specify the end time of the simulation; default is 2018-01-14 23:30."   
```

The simulation will produce an xlsx and multiple csv files in the folder 'Results-Clean' in the respective directories. The xlsx file will be used for training the models. 

## Training and Evaluation


Models can be trained using  
```python 
python run.py
```
A number of arguments can be passed to specify model types and hyperparameters:

``` python
'--wdn'             "Specify the WDS for which you want to simulate the scenarios; default is l_town. Choices are ['hanoi', 'fossolo', 'pescara', 'l_town', 'zhijiang']" 
'--mode'            "train_test i.e. train and test a new model, or evaluate i.e. evaluate on an already trained model; default is train_test. "
'--warm_start'      "Specify True if you want to further train a partially trained model. model_path must also be specified; default is False."
'--model_path'      "Specify model path in case of re-training or evaluation; default is None."
'--start_scenario'  "Specify the start scenario name, must be an integer; default is 1"
'--end_scenario'    "Specify the end scenario name, must be an integer; default is 20"
'--n_days'          "Specify the number of days of data to be used for training; default is 2 days."
'--batch_size'      "Specify the mini-batch size; default is 48."
'--n_epochs'        "Specify the number of epochs of training; default is 3000."    
'--lr'              "Specify the learning rate; default is 1e-4."
'--decay_step'      "Specify the step size of the lr scheduler; default is 300."
'--decay_rate'      "Specify the decay rate of the lr scheduler; default is 0.75."
'--I'               "Specify the number of GCN layers; default is 5."
'--n_iter'          "Specify the minimum number of iterations; default is 10."
'--r_iter'          "Specify the maximum number of additional (random iterations; default is 5."
'--n_mlp'           "Specify the number of layers in the MLP; default is 2."
'--M_l'             "Specify the latent dimension; default is 128."

```

Trained models can be used for evaluation using run.py by specifying the 'evaluate' mode and 'model_path'.


## Important Information

Every WDS is specified by an '.inp' file. We have included those files for all WDS (except Hanoi):

- Except for 'hanoi', we modify the '.inp' file to generate different scenarios.
- The '.inp' files for Hanoi WDS can be downloaded from: https://github.com/KIOS-Research/LeakDB (Vrachimis et al. 2018).
