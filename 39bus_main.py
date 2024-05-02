from Evaluation import EvalutionClassComparitive 
from Evaluation import EvalutionClassThreeComparitive
from Evaluation import EvalutionClassThreeComparitive_time
from Evaluation import EvaluationClassIterationMCPlot
from Evaluation import EvaluationClassIterationMCPlot_time


from grqnSolver import DGRQNSolverClass
from QLearningSovler import QLearningSolverClass
from FaultChainSolver import FaultChainSovlerClass


import torch
import torch.nn as nn
import pickle


# common parameters across differnet algorithms
NUM_ACTIONS = 46
selectCase = "39"     # "14, "30", "39", "118"
LOADING_FACTOR = 0.55
dataSetString = "loading055_39bus.h5"
M = 5                 # 5% threhsold for risky FC determination in percentage
EPSILON_1_m = 0.01
GAMMA = 0.99

M_episodes_qlearning = 1200
M_episodes_drqn = M_episodes_qlearning

iteration_track = 0   # 0 for fixed time base and 1 for fixed number of iterations S based 
time_taken = 60*5     # only relevant if you're running time based solution 


# finding risky FC dictionary for evaluation of results!
FCsovler = FaultChainSovlerClass(LOADING_FACTOR, selectCase) # instantiated object!
dataSet = FCsovler.load_dataset(dataSetString)
FCsovler.print_datatset_information(dataSet)
riskyFaultChainDict = FCsovler.find_risky_fault_chains(dataSet, M)


# parameters for GRQN
numNodes = int(selectCase)
dimInputSignals = 1
dimOutputSignals = 12
dimHiddenSignals = 12
nFilterTaps = [3, 3]
bias = True
nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearityReadout = nn.Tanh
dimReadout = [2]
dimEdgeFeatures = 1


eps = 1.00
replay_buffer_size = 5000
sample_length = 3
batch_size = 32
learning_rate = 0.005    # make sure this changes over time!
kappa = 3
EXPLORE = 250            # sort of irrelevant now since I have the buffer separate now!


MC = 20


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

    ProposedAlgorithm_kappa_3 = DGRQNSolverClass(M_episodes_drqn, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, EPSILON_1_m, GAMMA, numNodes, EXPLORE, replay_buffer_size, sample_length, batch_size, learning_rate, eps, dimInputSignals, dimOutputSignals, dimHiddenSignals, nFilterTaps, bias, nonlinearityHidden, nonlinearityOutput, nonlinearityReadout, dimReadout, dimEdgeFeatures, 3)
    # making evaluation script efficient
    ProposedAlgorithm_kappa_3.riskyFaultChainDict = riskyFaultChainDict
    with open("055_OfflineSequentialBuffer250", "rb") as fp:
        a, b = pickle.load(fp)
    ProposedAlgorithm_kappa_3.replay_buffer.counter = a 
    ProposedAlgorithm_kappa_3.replay_buffer.storage = b
    if iteration_track:
        ProposedAlgorithm_kappa_3.train_grqn_already_experience()
    else:
        ProposedAlgorithm_kappa_3.train_grqn_already_experience_time(time_taken)


    ProposedAlgorithm_kappa_2 = DGRQNSolverClass(M_episodes_drqn, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, EPSILON_1_m, GAMMA, numNodes, EXPLORE, replay_buffer_size, sample_length, batch_size, learning_rate, eps, dimInputSignals, dimOutputSignals, dimHiddenSignals, nFilterTaps, bias, nonlinearityHidden, nonlinearityOutput, nonlinearityReadout, dimReadout, dimEdgeFeatures, 2)
    # making evaluation script efficient
    ProposedAlgorithm_kappa_2.riskyFaultChainDict = riskyFaultChainDict
    with open("055_OfflineSequentialBuffer250", "rb") as fp:
        a, b = pickle.load(fp)
    ProposedAlgorithm_kappa_2.replay_buffer.counter = a 
    ProposedAlgorithm_kappa_2.replay_buffer.storage = b
    if iteration_track:
        ProposedAlgorithm_kappa_2.train_grqn_already_experience()
    else:
        ProposedAlgorithm_kappa_2.train_grqn_already_experience_time(time_taken)


    ProposedAlgorithm_kappa_1 = DGRQNSolverClass(M_episodes_drqn, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, EPSILON_1_m, GAMMA, numNodes, EXPLORE, replay_buffer_size, sample_length, batch_size, learning_rate, eps, dimInputSignals, dimOutputSignals, dimHiddenSignals, nFilterTaps, bias, nonlinearityHidden, nonlinearityOutput, nonlinearityReadout, dimReadout, dimEdgeFeatures, 1)
    # making evaluation script efficient
    ProposedAlgorithm_kappa_1.riskyFaultChainDict = riskyFaultChainDict
    with open("055_OfflineSequentialBuffer250", "rb") as fp:
        a, b = pickle.load(fp)
    ProposedAlgorithm_kappa_1.replay_buffer.counter = a 
    ProposedAlgorithm_kappa_1.replay_buffer.storage = b
    if iteration_track:
        ProposedAlgorithm_kappa_1.train_grqn_already_experience()
    else:
        ProposedAlgorithm_kappa_1.train_grqn_already_experience_time(time_taken)



    KaiSunAlgorithm = QLearningSolverClass(M_episodes_qlearning, NUM_ACTIONS, LOADING_FACTOR, selectCase, EPSILON_1_m, GAMMA, dataSetString, M)
    # making evaluation script efficient
    KaiSunAlgorithm.riskyFaultChainDict = riskyFaultChainDict
    if iteration_track:
        KaiSunAlgorithm.run_epoch()
    else:
        KaiSunAlgorithm.run_epoch_time(time_taken)



    KaiSunAlgorithm_TE = QLearningSolverClass(M_episodes_qlearning, NUM_ACTIONS, LOADING_FACTOR, selectCase, EPSILON_1_m, GAMMA, dataSetString, M)
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
        evaluationAll = EvalutionClassThreeComparitive(riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, M_episodes_qlearning, 0, KaiSunAlgorithm.answer, KaiSunAlgorithm.repeatedSequences, KaiSunAlgorithm.numRiskyFaultChains, KaiSunAlgorithm.allFalseCount, M_episodes_drqn, EXPLORE, ProposedAlgorithm_kappa_3.answer, ProposedAlgorithm_kappa_3.repeatedSequences, ProposedAlgorithm_kappa_3.numRiskyFaultChains, ProposedAlgorithm_kappa_3.allFalseCount, ProposedAlgorithm_kappa_3.loss, M_episodes_qlearning, 0, KaiSunAlgorithm_TE.answer, KaiSunAlgorithm_TE.repeatedSequences, KaiSunAlgorithm_TE.numRiskyFaultChains, KaiSunAlgorithm_TE.allFalseCount, ProposedAlgorithm_kappa_2.answer, ProposedAlgorithm_kappa_2.repeatedSequences, ProposedAlgorithm_kappa_2.numRiskyFaultChains, ProposedAlgorithm_kappa_2.allFalseCount, ProposedAlgorithm_kappa_2.loss, ProposedAlgorithm_kappa_1.answer, ProposedAlgorithm_kappa_1.repeatedSequences, ProposedAlgorithm_kappa_1.numRiskyFaultChains, ProposedAlgorithm_kappa_1.allFalseCount, ProposedAlgorithm_kappa_1.loss)
    else:
        evaluationAll = EvalutionClassThreeComparitive_time(riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, M_episodes_qlearning, 0, KaiSunAlgorithm.answer_time, KaiSunAlgorithm.repeatedSequences, KaiSunAlgorithm.numRiskyFaultChains_time, KaiSunAlgorithm.allFalseCount, M_episodes_drqn, EXPLORE, ProposedAlgorithm_kappa_3.answer_time, ProposedAlgorithm_kappa_3.repeatedSequences, ProposedAlgorithm_kappa_3.numRiskyFaultChains_time, ProposedAlgorithm_kappa_3.allFalseCount, ProposedAlgorithm_kappa_3.loss, M_episodes_qlearning, 0, KaiSunAlgorithm_TE.answer_time, KaiSunAlgorithm_TE.repeatedSequences, KaiSunAlgorithm_TE.numRiskyFaultChains_time, KaiSunAlgorithm_TE.allFalseCount, ProposedAlgorithm_kappa_2.answer_time, ProposedAlgorithm_kappa_2.repeatedSequences, ProposedAlgorithm_kappa_2.numRiskyFaultChains_time, ProposedAlgorithm_kappa_2.allFalseCount, ProposedAlgorithm_kappa_2.loss, ProposedAlgorithm_kappa_1.answer_time, ProposedAlgorithm_kappa_1.repeatedSequences, ProposedAlgorithm_kappa_1.numRiskyFaultChains_time, ProposedAlgorithm_kappa_1.allFalseCount, ProposedAlgorithm_kappa_1.loss)
    

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

print("")

finalPlot = EvaluationClassIterationMCPlot_time(totalNumRiskyGRQN_kappa_3, totalRiskGRQN_kappa_3, totalNumRiskyQLearning, totalRiskQLearning, totalNumRisky_TE, totalRisk_TE, totalNumRiskyGRQN_kappa_2, totalRiskGRQN_kappa_2, totalNumRiskyGRQN_kappa_1, totalRiskGRQN_kappa_1)
finalPlot.display_statistics()

print("")

risky_data_list = [riskyGRQN_kappa_3, riskyGRQN_kappa_2, riskyGRQN_kappa_1, riskyQLearning, risky_TE]
with open("time_risky_data_list", "wb") as fp:
    pickle.dump((risky_data_list, M_episodes_qlearning), fp)
finalPlot.plot_fc_risk(risky_data_list, M_episodes_qlearning)

print("")

num_data_list = [numRiskyGRQN_kappa_3, numRiskyGRQN_kappa_2, numRiskyGRQN_kappa_1, numRiskyQLearning, numRisky_TE]
with open("time_num_data_list", "wb") as fp:
    pickle.dump((num_data_list, M_episodes_qlearning), fp)
finalPlot.plot_num_risky_fcs(num_data_list, M_episodes_qlearning)