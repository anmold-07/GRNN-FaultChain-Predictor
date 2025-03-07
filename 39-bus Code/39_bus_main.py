import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to file path to access common files

from evaluation import EvalutionClassThreeComparitive
from evaluation import EvalutionClassThreeComparitive_time
from evaluation import EvaluationClassIterationMCPlot
from evaluation import EvaluationClassIterationMCPlot_time

from grqnSolver import load_config

from grqnSolver import DGRQNSolverClass
from qLearningSovler import QLearningSolverClass
from faultChainSolver import FaultChainSovlerClass

import torch
import torch.nn as nn
import pickle


config = load_config()

TOTAL_EPISODES = 100
MC = 1               # Monte Carlo Iterations
iteration_track = 0  # 0 for fixed time base and 1 for fixed number of iterations S based 
M_episodes_qlearning = TOTAL_EPISODES
M_episodes_drqn = TOTAL_EPISODES

if iteration_track==0:  # run
    time_taken = 60     # set fixed time (in seconds) - run for 5 minutes


# set the working directory
print(f"Current working directory: {os.getcwd()}\n")
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)                                    # Change the working directory to parent directory
print(f"New working directory: {os.getcwd()}\n")

# select case
selectCase = "39"        # "14, "30", "39", "118"
LOADING_FACTOR = 0.55
FCsovler = FaultChainSovlerClass(LOADING_FACTOR, selectCase) # instantiated object!

if selectCase=="39":
    NUM_ACTIONS = 46
    if LOADING_FACTOR == 0.6:
        dataSetString = "loading06_39bus.h5" 
        seqBuffer = "39bus_06_OfflineSequentialBuffer250" 

    elif LOADING_FACTOR == 0.55:
        dataSetString = "loading055_39bus.h5"  
        seqBuffer = "39bus_055_OfflineSequentialBuffer250" 

    buffer_path  = os.path.join("39-bus Code", seqBuffer)

dataset_path = os.path.join('Datasets', dataSetString) # ground truth datasets located in the Datasets/ 

dataSet, t = FCsovler.load_dataset(dataset_path) # finding risky FC dictionary for evaluation of results!
FCsovler.print_datatset_information(dataSet, t)
M = 5                 # 5% threhsold for risky FC determination in percentage
riskyFaultChainDict = FCsovler.find_risky_fault_chains(dataSet, M)


# common parameters across diff. algorithms
eps_end = 0.01
eps_start = 1.00
GAMMA = config["GAMMA"]

# GRQN Parameters
grqn_params = {
    "dimInputSignals": config["dimInputSignals"],
    "dimOutputSignals": config["dimOutputSignals"],
    "dimHiddenSignals": config["dimHiddenSignals"],
    "nFilterTaps": config["nFilterTaps"],
    "bias": config["bias"],
    "nonlinearityHidden": torch.tanh,
    "nonlinearityOutput": torch.tanh,
    "nonlinearityReadout": nn.Tanh,
    "dimReadout": [2],
    "dimEdgeFeatures": config["dimEdgeFeatures"],
}

training_params = {
    "numNodes": int(selectCase),
    "EXPLORE" : config["EXPLORE"],
    "replay_buffer_size": config["replay_buffer_size"],
    "sample_length": config["sample_length"],
    "batch_size": config["batch_size"],
    "learning_rate": config["learning_rate"],
}

# collect results
riskyGRQN_kappa_3 = []
numRiskyGRQN_kappa_3 = []
totalRiskGRQN_kappa_3 = []
totalNumRiskyGRQN_kappa_3 = []

riskyGRQN_kappa_2 = []
numRiskyGRQN_kappa_2 = []
totalRiskGRQN_kappa_2 = []
totalNumRiskyGRQN_kappa_2 = []

riskyGRQN_kappa_1 = []
numRiskyGRQN_kappa_1 = []
totalRiskGRQN_kappa_1 = []
totalNumRiskyGRQN_kappa_1 = []

riskyQLearning = []
numRiskyQLearning = []
totalRiskQLearning = []
totalNumRiskyQLearning = []

risky_TE = []
numRisky_TE = []
totalRisk_TE = []
totalNumRisky_TE = []


# run a for loop for MC times:
for _ in range(MC):

    ProposedAlgorithm_kappa_3 = DGRQNSolverClass(TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, eps_end, GAMMA, eps_start, 3, **training_params, **grqn_params)
    # making evaluation script efficient
    ProposedAlgorithm_kappa_3.riskyFaultChainDict = riskyFaultChainDict
    with open(buffer_path, "rb") as fp:
        a, b = pickle.load(fp)
    ProposedAlgorithm_kappa_3.replay_buffer.counter = a 
    ProposedAlgorithm_kappa_3.replay_buffer.storage = b
    if iteration_track:
        ProposedAlgorithm_kappa_3.train_grqn_already_experience()
    else:
        ProposedAlgorithm_kappa_3.train_grqn_already_experience_time(time_taken)


    ProposedAlgorithm_kappa_2 = DGRQNSolverClass(TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, eps_end, GAMMA, eps_start, 2, **training_params, **grqn_params)
    # making evaluation script efficient
    ProposedAlgorithm_kappa_2.riskyFaultChainDict = riskyFaultChainDict
    with open(buffer_path, "rb") as fp:
        a, b = pickle.load(fp)
    ProposedAlgorithm_kappa_2.replay_buffer.counter = a 
    ProposedAlgorithm_kappa_2.replay_buffer.storage = b
    if iteration_track:
        ProposedAlgorithm_kappa_2.train_grqn_already_experience()
    else:
        ProposedAlgorithm_kappa_2.train_grqn_already_experience_time(time_taken)


    ProposedAlgorithm_kappa_1 = DGRQNSolverClass(TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, eps_end, GAMMA, eps_start, 1, **training_params, **grqn_params)
    # making evaluation script efficient
    ProposedAlgorithm_kappa_1.riskyFaultChainDict = riskyFaultChainDict
    with open(buffer_path, "rb") as fp:
        a, b = pickle.load(fp)
    ProposedAlgorithm_kappa_1.replay_buffer.counter = a 
    ProposedAlgorithm_kappa_1.replay_buffer.storage = b
    if iteration_track:
        ProposedAlgorithm_kappa_1.train_grqn_already_experience()
    else:
        ProposedAlgorithm_kappa_1.train_grqn_already_experience_time(time_taken)



    KaiSunAlgorithm = QLearningSolverClass(M_episodes_qlearning, NUM_ACTIONS, LOADING_FACTOR, selectCase, eps_end, GAMMA, dataSetString, M)
    # making evaluation script efficient
    KaiSunAlgorithm.riskyFaultChainDict = riskyFaultChainDict
    if iteration_track:
        KaiSunAlgorithm.run_epoch()
    else:
        KaiSunAlgorithm.run_epoch_time(time_taken)



    KaiSunAlgorithm_TE = QLearningSolverClass(M_episodes_qlearning, NUM_ACTIONS, LOADING_FACTOR, selectCase, eps_end, GAMMA, dataSetString, M)
    EPSILON_20 = 1.3
    # making evaluation script efficient
    KaiSunAlgorithm_TE.riskyFaultChainDict = riskyFaultChainDict
    if iteration_track:
        KaiSunAlgorithm_TE.transition_extension_run_epoch(EPSILON_20)
    else:
        KaiSunAlgorithm_TE.transition_extension_run_epoch_time(EPSILON_20, time_taken)

    # for 2 algorithm comparison (non-MC):
    #evaluationKaiSun = EvalutionClassComparitive(LOADING_FACTOR, selectCase, M, dataSetString, M_episodes_qlearning, 0, KaiSunAlgorithm.answer, KaiSunAlgorithm.repeatedSequences, KaiSunAlgorithm.numRiskyFaultChains, KaiSunAlgorithm.allFalseCount, M_episodes_drqn, EXPLORE, ProposedAlgorithm.answer, ProposedAlgorithm.repeatedSequences, ProposedAlgorithm.numRiskyFaultChains, ProposedAlgorithm.allFalseCount, ProposedAlgorithm.loss)
    #evaluationKaiSun.riskyFaultChainDict = riskyFaultChainDict
    #evaluationKaiSun.display_results()

    # for 3 algorithm comparison (non-MC):
    if iteration_track:
        evaluationAll = EvalutionClassThreeComparitive(riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, M_episodes_qlearning, 0, KaiSunAlgorithm.answer, KaiSunAlgorithm.repeatedSequences, KaiSunAlgorithm.numRiskyFaultChains, KaiSunAlgorithm.allFalseCount, M_episodes_drqn, config["EXPLORE"], ProposedAlgorithm_kappa_3.answer, ProposedAlgorithm_kappa_3.repeatedSequences, ProposedAlgorithm_kappa_3.numRiskyFaultChains, ProposedAlgorithm_kappa_3.allFalseCount, ProposedAlgorithm_kappa_3.loss, M_episodes_qlearning, 0, KaiSunAlgorithm_TE.answer, KaiSunAlgorithm_TE.repeatedSequences, KaiSunAlgorithm_TE.numRiskyFaultChains, KaiSunAlgorithm_TE.allFalseCount, ProposedAlgorithm_kappa_2.answer, ProposedAlgorithm_kappa_2.repeatedSequences, ProposedAlgorithm_kappa_2.numRiskyFaultChains, ProposedAlgorithm_kappa_2.allFalseCount, ProposedAlgorithm_kappa_2.loss, ProposedAlgorithm_kappa_1.answer, ProposedAlgorithm_kappa_1.repeatedSequences, ProposedAlgorithm_kappa_1.numRiskyFaultChains, ProposedAlgorithm_kappa_1.allFalseCount, ProposedAlgorithm_kappa_1.loss)
    else:
        evaluationAll = EvalutionClassThreeComparitive_time(riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, M_episodes_qlearning, 0, KaiSunAlgorithm.answer_time, KaiSunAlgorithm.repeatedSequences, KaiSunAlgorithm.numRiskyFaultChains_time, KaiSunAlgorithm.allFalseCount, M_episodes_drqn, config["EXPLORE"], ProposedAlgorithm_kappa_3.answer_time, ProposedAlgorithm_kappa_3.repeatedSequences, ProposedAlgorithm_kappa_3.numRiskyFaultChains_time, ProposedAlgorithm_kappa_3.allFalseCount, ProposedAlgorithm_kappa_3.loss, M_episodes_qlearning, 0, KaiSunAlgorithm_TE.answer_time, KaiSunAlgorithm_TE.repeatedSequences, KaiSunAlgorithm_TE.numRiskyFaultChains_time, KaiSunAlgorithm_TE.allFalseCount, ProposedAlgorithm_kappa_2.answer_time, ProposedAlgorithm_kappa_2.repeatedSequences, ProposedAlgorithm_kappa_2.numRiskyFaultChains_time, ProposedAlgorithm_kappa_2.allFalseCount, ProposedAlgorithm_kappa_2.loss, ProposedAlgorithm_kappa_1.answer_time, ProposedAlgorithm_kappa_1.repeatedSequences, ProposedAlgorithm_kappa_1.numRiskyFaultChains_time, ProposedAlgorithm_kappa_1.allFalseCount, ProposedAlgorithm_kappa_1.loss)
    

    if iteration_track:
        #evaluationAll.display_results()
        numRiskyGRQN_kappa_3.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_3))
        totalNumRiskyGRQN_kappa_3.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_3[-1])
        riskyGRQN_kappa_3.append(list(evaluationAll.cumRewardsGRQN_kappa_3))
        totalRiskGRQN_kappa_3.append(evaluationAll.cumRewardsGRQN_kappa_3[-1])

        numRiskyGRQN_kappa_2.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_2))
        totalNumRiskyGRQN_kappa_2.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_2[-1])
        riskyGRQN_kappa_2.append(list(evaluationAll.cumRewardsGRQN_kappa_2))
        totalRiskGRQN_kappa_2.append(evaluationAll.cumRewardsGRQN_kappa_2[-1])

        numRiskyGRQN_kappa_1.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_1))
        totalNumRiskyGRQN_kappa_1.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_1[-1])
        riskyGRQN_kappa_1.append(list(evaluationAll.cumRewardsGRQN_kappa_1))
        totalRiskGRQN_kappa_1.append(evaluationAll.cumRewardsGRQN_kappa_1[-1])

        numRiskyQLearning.append(list(evaluationAll.cumNumRiskyFaultChainsQLearning))
        totalNumRiskyQLearning.append(evaluationAll.cumNumRiskyFaultChainsQLearning[-1])
        riskyQLearning.append(list(evaluationAll.cumRewardsQLearning))
        totalRiskQLearning.append(evaluationAll.cumRewardsQLearning[-1])

        numRisky_TE.append(list(evaluationAll.cumNumRiskyFaultChains_TE))
        totalNumRisky_TE.append(evaluationAll.cumNumRiskyFaultChains_TE[-1])
        risky_TE.append(list(evaluationAll.cumRewards_TE))
        totalRisk_TE.append(evaluationAll.cumRewards_TE[-1])
    else:
        #evaluationAll.display_results()
        numRiskyGRQN_kappa_3.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_3))
        totalNumRiskyGRQN_kappa_3.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_3[-1][0])
        riskyGRQN_kappa_3.append(list(evaluationAll.cumRewardsGRQN_kappa_3))
        totalRiskGRQN_kappa_3.append(evaluationAll.cumRewardsGRQN_kappa_3[-1][0])

        numRiskyGRQN_kappa_2.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_2))
        totalNumRiskyGRQN_kappa_2.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_2[-1][0])
        riskyGRQN_kappa_2.append(list(evaluationAll.cumRewardsGRQN_kappa_2))
        totalRiskGRQN_kappa_2.append(evaluationAll.cumRewardsGRQN_kappa_2[-1][0])

        numRiskyGRQN_kappa_1.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_1))
        totalNumRiskyGRQN_kappa_1.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_1[-1][0])
        riskyGRQN_kappa_1.append(list(evaluationAll.cumRewardsGRQN_kappa_1))
        totalRiskGRQN_kappa_1.append(evaluationAll.cumRewardsGRQN_kappa_1[-1][0])

        numRiskyQLearning.append(list(evaluationAll.cumNumRiskyFaultChainsQLearning))
        totalNumRiskyQLearning.append(evaluationAll.cumNumRiskyFaultChainsQLearning[-1][0])
        riskyQLearning.append(list(evaluationAll.cumRewardsQLearning))
        totalRiskQLearning.append(evaluationAll.cumRewardsQLearning[-1][0])

        numRisky_TE.append(list(evaluationAll.cumNumRiskyFaultChains_TE))
        totalNumRisky_TE.append(evaluationAll.cumNumRiskyFaultChains_TE[-1][0])
        risky_TE.append(list(evaluationAll.cumRewards_TE))
        totalRisk_TE.append(evaluationAll.cumRewards_TE[-1][0])


 

""" 
# data for reporting results directly
print(totalNumRiskyGRQN_kappa_3, totalRiskGRQN_kappa_3)
print(totalNumRiskyGRQN_kappa_2, totalRiskGRQN_kappa_2)
print(totalNumRiskyQLearning, totalRiskQLearning)
print(totalNumRisky_TE, totalRisk_TE)

# data for plotting average and variance
print(numRiskyGRQN_kappa_3, riskyGRQN_kappa_3)
print(numRiskyGRQN_kappa_2, riskyGRQN_kappa_2)
print(numRiskyQLearning, riskyQLearning)
print(numRisky_TE, risky_TE)
"""

# final result analysis
print("")

if iteration_track:
    finalPlot = EvaluationClassIterationMCPlot(totalNumRiskyGRQN_kappa_3, totalRiskGRQN_kappa_3, totalNumRiskyQLearning, totalRiskQLearning, totalNumRisky_TE, totalRisk_TE, totalNumRiskyGRQN_kappa_2, totalRiskGRQN_kappa_2, totalNumRiskyGRQN_kappa_1, totalRiskGRQN_kappa_1)
    finalPlot.display_statistics()
else:
    finalPlot = EvaluationClassIterationMCPlot_time(totalNumRiskyGRQN_kappa_3, totalRiskGRQN_kappa_3, totalNumRiskyQLearning, totalRiskQLearning, totalNumRisky_TE, totalRisk_TE, totalNumRiskyGRQN_kappa_2, totalRiskGRQN_kappa_2, totalNumRiskyGRQN_kappa_1, totalRiskGRQN_kappa_1)
    finalPlot.display_statistics()

print("")

risky_data_list = [riskyGRQN_kappa_3, riskyGRQN_kappa_2, riskyGRQN_kappa_1, riskyQLearning, risky_TE]

if iteration_track:
    with open( str(M_episodes_qlearning) + "_risky_data_list" , "wb") as fp:
        pickle.dump( (risky_data_list, M_episodes_qlearning), fp)    
else:
    with open("time_risky_data_list", "wb") as fp:
        pickle.dump( (risky_data_list, M_episodes_qlearning), fp)    

finalPlot.plot_fc_risk(risky_data_list, M_episodes_qlearning)


print("")

num_data_list = [numRiskyGRQN_kappa_3, numRiskyGRQN_kappa_2, numRiskyGRQN_kappa_1, numRiskyQLearning, numRisky_TE]

if iteration_track:
    with open( str(M_episodes_qlearning) + "_num_data_list" , "wb") as fp:
        pickle.dump( (num_data_list, M_episodes_qlearning), fp)    
else:
    with open("time_num_data_list", "wb") as fp:
        pickle.dump( (num_data_list, M_episodes_qlearning), fp)

finalPlot.plot_num_risky_fcs(num_data_list, M_episodes_qlearning)