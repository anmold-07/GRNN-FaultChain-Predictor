import sys
import os
import torch
import pickle
import argparse
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to file path to access common files

from grqnSolver import load_config

from evaluation import (
    EvalutionClassThreeComparitive,
    EvalutionClassThreeComparitive_time,
    EvaluationClassIterationMCPlot,
    EvaluationClassIterationMCPlot_time,
)

from grqnSolver import DGRQNSolverClass
from qLearningSovler import QLearningSolverClass
from faultChainSolver import FaultChainSovlerClass


def initialize_and_train_algorithm(algorithm_class, kappa, riskyFaultChainDict, replay_buffer_data, iteration_track, time_taken, eps_end, GAMMA, eps_start, TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, training_params, grqn_params):
    # Initialize the algorithm
    algorithm = algorithm_class(TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, eps_end, GAMMA, eps_start, kappa, **training_params, **grqn_params)
    
    algorithm.riskyFaultChainDict = riskyFaultChainDict
    
    # Use pre-loaded replay buffer data
    a, b = replay_buffer_data
    algorithm.replay_buffer.counter = a
    algorithm.replay_buffer.storage = b
    
    if iteration_track:
        algorithm.train_grqn_already_experience()
    else:
        algorithm.train_grqn_already_experience_time(time_taken)

    return algorithm


def main(config, args):

    selectCase = args.case
    LOADING_FACTOR = args.load
    M = args.threshold                
    MC = args.MC             
    iteration_track = args.if_iteration
    TOTAL_EPISODES = args.num_episodes
    time_taken = args.time_taken

    if selectCase == "39":
        NUM_ACTIONS = 46
        dataset_map = {
            0.6: ("loading06_39bus.h5", "39bus_06_OfflineSequentialBuffer250"),
            0.55: ("loading055_39bus.h5", "39bus_055_OfflineSequentialBuffer250"),
        }
        dataSetString, seqBuffer = dataset_map.get(LOADING_FACTOR, (None, None))
        buffer_path = os.path.join("39-bus Code", seqBuffer)

    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)

    # Initialize Fault Chain Solver
    FCsovler = FaultChainSovlerClass(LOADING_FACTOR, selectCase)
    dataset_path = os.path.join('Datasets', dataSetString)
    dataSet, t = FCsovler.load_dataset(dataset_path)
    FCsovler.print_datatset_information(dataSet, t)
    riskyFaultChainDict = FCsovler.find_risky_fault_chains(dataSet, M)

    # Common parameters
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

    # Load replay buffer once
    with open(buffer_path, "rb") as fp:
        replay_buffer_data = pickle.load(fp)


    experiment_names = ["GRQN_3", "QLearning", "TE", "GRQN_2", "GRQN_1", ]
    results = {name: {"numRisky": [], "totalNumRisky": [], "risky": [], "totalRisk": []} for name in experiment_names}


    # run a for loop for MC times:
    for _ in range(MC):

        algorithms = {}

        # run GRNN approaches
        for name in ["GRQN_3", "GRQN_2", "GRQN_1"]:
            kappa = int(name.split("_")[1])  # Extract kappa
            algorithms[name] = initialize_and_train_algorithm(DGRQNSolverClass, kappa, riskyFaultChainDict, replay_buffer_data, iteration_track, time_taken, eps_end, GAMMA, eps_start, TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, training_params, grqn_params)

        # run baseline approahces
        baseline_algo = QLearningSolverClass(TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, eps_end, GAMMA, dataSetString, M)
        baseline_algo.riskyFaultChainDict = riskyFaultChainDict

        baseline_algo_TE = QLearningSolverClass(TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, eps_end, GAMMA, dataSetString, M)
        EPSILON_20 = 1.3
        baseline_algo_TE.riskyFaultChainDict = riskyFaultChainDict

        if iteration_track:
            baseline_algo.run_epoch()
            baseline_algo_TE.transition_extension_run_epoch(EPSILON_20)
        else:
            baseline_algo.run_epoch_time(time_taken)
            baseline_algo_TE.transition_extension_run_epoch_time(EPSILON_20, time_taken)

        # Store the results for each algorithm in the results dictionary
        if iteration_track:
            evaluation = EvalutionClassThreeComparitive(riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, 
                                                        TOTAL_EPISODES, 0, baseline_algo.answer, baseline_algo.repeatedSequences, baseline_algo.numRiskyFaultChains, baseline_algo.allFalseCount, 
                                                        TOTAL_EPISODES, config["EXPLORE"], algorithms["GRQN_3"].answer, algorithms["GRQN_3"].repeatedSequences, algorithms["GRQN_3"].numRiskyFaultChains, algorithms["GRQN_3"].allFalseCount, algorithms["GRQN_3"].loss, 
                                                        TOTAL_EPISODES, 0, baseline_algo_TE.answer, baseline_algo_TE.repeatedSequences, baseline_algo_TE.numRiskyFaultChains, baseline_algo_TE.allFalseCount,
                                                        algorithms["GRQN_2"].answer, algorithms["GRQN_2"].repeatedSequences, algorithms["GRQN_2"].numRiskyFaultChains, algorithms["GRQN_2"].allFalseCount, algorithms["GRQN_2"].loss, 
                                                        algorithms["GRQN_1"].answer, algorithms["GRQN_1"].repeatedSequences, algorithms["GRQN_1"].numRiskyFaultChains, algorithms["GRQN_1"].allFalseCount, algorithms["GRQN_1"].loss)
            
            for name in ["GRQN_3", "GRQN_2", "GRQN_1"]:
                # Collect results for each experiment
                results[name]["numRisky"].append(list(evaluation.cumNumRiskyFaultChainsGRQN_kappa_3))
                results[name]["totalNumRisky"].append(evaluation.cumNumRiskyFaultChainsGRQN_kappa_3[-1] if name == "GRQN_3" else evaluation.cumNumRiskyFaultChainsGRQN_kappa_2[-1] if name == "GRQN_2" else evaluation.cumNumRiskyFaultChainsGRQN_kappa_1[-1])
                results[name]["risky"].append(list(evaluation.cumRewardsGRQN_kappa_3))
                results[name]["totalRisk"].append(evaluation.cumRewardsGRQN_kappa_3[-1] if name == "GRQN_3" else evaluation.cumRewardsGRQN_kappa_2[-1] if name == "GRQN_2" else evaluation.cumRewardsGRQN_kappa_1[-1])

            # Collect results for baseline (QLearning) and TE
            results["QLearning"]["numRisky"].append(list(evaluation.cumNumRiskyFaultChainsQLearning))
            results["QLearning"]["totalNumRisky"].append(evaluation.cumNumRiskyFaultChainsQLearning[-1])
            results["QLearning"]["risky"].append(list(evaluation.cumRewardsQLearning))
            results["QLearning"]["totalRisk"].append(evaluation.cumRewardsQLearning[-1])

            results["TE"]["numRisky"].append(list(evaluation.cumNumRiskyFaultChains_TE))
            results["TE"]["totalNumRisky"].append(evaluation.cumNumRiskyFaultChains_TE[-1])
            results["TE"]["risky"].append(list(evaluation.cumRewards_TE))
            results["TE"]["totalRisk"].append(evaluation.cumRewards_TE[-1])

        else:
            evaluation = EvalutionClassThreeComparitive_time(riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, 
                                                            TOTAL_EPISODES, 0, baseline_algo.answer_time, baseline_algo.repeatedSequences, baseline_algo.numRiskyFaultChains_time, baseline_algo.allFalseCount, 
                                                            TOTAL_EPISODES, config["EXPLORE"], algorithms["GRQN_3"].answer_time, algorithms["GRQN_3"].repeatedSequences, algorithms["GRQN_3"].numRiskyFaultChains_time, algorithms["GRQN_3"].allFalseCount, algorithms["GRQN_3"].loss, 
                                                            TOTAL_EPISODES, 0, baseline_algo_TE.answer_time, baseline_algo_TE.repeatedSequences, baseline_algo_TE.numRiskyFaultChains_time, baseline_algo_TE.allFalseCount,
                                                            algorithms["GRQN_2"].answer, algorithms["GRQN_2"].repeatedSequences, algorithms["GRQN_2"].numRiskyFaultChains, algorithms["GRQN_2"].allFalseCount, algorithms["GRQN_2"].loss, 
                                                            algorithms["GRQN_1"].answer, algorithms["GRQN_1"].repeatedSequences, algorithms["GRQN_1"].numRiskyFaultChains, algorithms["GRQN_1"].allFalseCount, algorithms["GRQN_1"].loss)
            
            for name in ["GRQN_3", "GRQN_2", "GRQN_1"]:
                # Collect results for each experiment
                results[name]["numRisky"].append(list(evaluation.cumNumRiskyFaultChainsGRQN_kappa_3))
                results[name]["totalNumRisky"].append(evaluation.cumNumRiskyFaultChainsGRQN_kappa_3[-1][0] if name == "GRQN_3" else evaluation.cumNumRiskyFaultChainsGRQN_kappa_2[-1][0] if name == "GRQN_2" else evaluation.cumNumRiskyFaultChainsGRQN_kappa_1[-1][0])
                results[name]["risky"].append(list(evaluation.cumRewardsGRQN_kappa_3))
                results[name]["totalRisk"].append(evaluation.cumRewardsGRQN_kappa_3[-1][0] if name == "GRQN_3" else evaluation.cumRewardsGRQN_kappa_2[-1][0] if name == "GRQN_2" else evaluation.cumRewardsGRQN_kappa_1[-1])[0]

            # Collect results for baseline (QLearning) and TE
            results["QLearning"]["numRisky"].append(list(evaluation.cumNumRiskyFaultChainsQLearning))
            results["QLearning"]["totalNumRisky"].append(evaluation.cumNumRiskyFaultChainsQLearning[-1][0])
            results["QLearning"]["risky"].append(list(evaluation.cumRewardsQLearning))
            results["QLearning"]["totalRisk"].append(evaluation.cumRewardsQLearning[-1][0])

            results["TE"]["numRisky"].append(list(evaluation.cumNumRiskyFaultChains_TE))
            results["TE"]["totalNumRisky"].append(evaluation.cumNumRiskyFaultChains_TE[-1][0])
            results["TE"]["risky"].append(list(evaluation.cumRewards_TE))
            results["TE"]["totalRisk"].append(evaluation.cumRewards_TE[-1][0])


        #     #evaluationAll.display_results()
        #     numRiskyGRQN_kappa_3.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_3))
        #     totalNumRiskyGRQN_kappa_3.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_3[-1][0])
        #     riskyGRQN_kappa_3.append(list(evaluationAll.cumRewardsGRQN_kappa_3))
        #     totalRiskGRQN_kappa_3.append(evaluationAll.cumRewardsGRQN_kappa_3[-1][0])

        #     numRiskyGRQN_kappa_2.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_2))
        #     totalNumRiskyGRQN_kappa_2.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_2[-1][0])
        #     riskyGRQN_kappa_2.append(list(evaluationAll.cumRewardsGRQN_kappa_2))
        #     totalRiskGRQN_kappa_2.append(evaluationAll.cumRewardsGRQN_kappa_2[-1][0])

        #     numRiskyGRQN_kappa_1.append(list(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_1))
        #     totalNumRiskyGRQN_kappa_1.append(evaluationAll.cumNumRiskyFaultChainsGRQN_kappa_1[-1][0])
        #     riskyGRQN_kappa_1.append(list(evaluationAll.cumRewardsGRQN_kappa_1))
        #     totalRiskGRQN_kappa_1.append(evaluationAll.cumRewardsGRQN_kappa_1[-1][0])

        #     numRiskyQLearning.append(list(evaluationAll.cumNumRiskyFaultChainsQLearning))
        #     totalNumRiskyQLearning.append(evaluationAll.cumNumRiskyFaultChainsQLearning[-1][0])
        #     riskyQLearning.append(list(evaluationAll.cumRewardsQLearning))
        #     totalRiskQLearning.append(evaluationAll.cumRewardsQLearning[-1][0])

        #     numRisky_TE.append(list(evaluationAll.cumNumRiskyFaultChains_TE))
        #     totalNumRisky_TE.append(evaluationAll.cumNumRiskyFaultChains_TE[-1][0])
        #     risky_TE.append(list(evaluationAll.cumRewards_TE))
        #     totalRisk_TE.append(evaluationAll.cumRewards_TE[-1][0])

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
        finalPlot = EvaluationClassIterationMCPlot( results["GRQN_3"]["totalNumRisky"], results["GRQN_3"]["totalRisk"], 
                                                    results["QLearning"]["totalNumRisky"], results["QLearning"]["totalRisk"], 
                                                    results["TE"]["totalNumRisky"], results["TE"]["totalRisk"], 
                                                    results["GRQN_2"]["totalNumRisky"], results["GRQN_2"]["totalRisk"], 
                                                    results["GRQN_1"]["totalNumRisky"], results["GRQN_1"]["totalRisk"] )
        finalPlot.display_statistics()
    else:
        finalPlot = EvaluationClassIterationMCPlot_time( results["GRQN_3"]["totalNumRisky"], results["GRQN_3"]["totalRisk"], 
                                                        results["QLearning"]["totalNumRisky"], results["QLearning"]["totalRisk"], 
                                                        results["TE"]["totalNumRisky"], results["TE"]["totalRisk"], 
                                                        results["GRQN_2"]["totalNumRisky"], results["GRQN_2"]["totalRisk"], 
                                                        results["GRQN_1"]["totalNumRisky"], results["GRQN_1"]["totalRisk"] )
        finalPlot.display_statistics()

    print("")

    risky_data_list = [results["GRQN_3"]["risky"], results["GRQN_2"]["risky"], results["GRQN_2"]["risky"], results["QLearning"]["risky"], results["TE"]["risky"]]

    if iteration_track:
        with open( str(TOTAL_EPISODES) + "_risky_data_list" , "wb") as fp:
            pickle.dump( (risky_data_list, TOTAL_EPISODES), fp)    
    else:
        with open("time_risky_data_list", "wb") as fp:
            pickle.dump( (risky_data_list, TOTAL_EPISODES), fp)    

    finalPlot.plot_fc_risk(risky_data_list, TOTAL_EPISODES)


    print("")

    num_data_list = [results["GRQN_3"]["numRisky"], results["GRQN_2"]["numRisky"], results["GRQN_2"]["numRisky"], results["QLearning"]["numRisky"], results["TE"]["numRisky"]]

    if iteration_track:
        with open( str(TOTAL_EPISODES) + "_num_data_list" , "wb") as fp:
            pickle.dump( (num_data_list, TOTAL_EPISODES), fp)    
    else:
        with open("time_num_data_list", "wb") as fp:
            pickle.dump( (num_data_list, TOTAL_EPISODES), fp)

    finalPlot.plot_num_risky_fcs(num_data_list, TOTAL_EPISODES)



if __name__ == '__main__':

    # python "39-bus Code/39bus_main.py" --case 39 --load 0.55 --threshold 5 --MC 1 --if_iteration 1 --num_episodes 50
    # python "39-bus Code/39bus_main.py" --case 39 --load 0.55 --threshold 5 --MC 1 --if_iteration 0 --time_taken 60

    parser = argparse.ArgumentParser(description="Run the GRNN Fault Chain Solver.")
    parser.add_argument('--case', type=str, choices=['39', '118'], required=True, help="Select the case: '39', or '118'.")
    parser.add_argument('--load', type=float, choices=[0.55, 0.6, 1.0], required=True, help="Loading factor: 0.55, 0.6 for 39-bus and 0.6, 1.0 for 118-bus.")
    parser.add_argument('--threshold', type=int, required=True, help="Risky FC threshold in percentage.")
    parser.add_argument('--MC', type=int, required=True, help="number of Monte Carlo iterations.")
    parser.add_argument('--if_iteration', type=int, choices=[0, 1], required=True, help="Fixed iterations (1) or fixed time (0).")

    parser.add_argument('--num_episodes', type=int, default=50, help="Number of episodes for Q-learning. Only relevant when iterations = 1.")
    parser.add_argument('--time_taken', type=int, default=60, help="Time duration for fixed time execution when iterations are 0.")

    args = parser.parse_args()

    config = load_config()
    main(config, args)