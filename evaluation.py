from matplotlib import markers
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from faultChainSolver import FaultChainSovlerClass

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class EvalutionClassThreeComparitive(object):
    def __init__(self, riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, 
                 M_episodes_QLearning, EXPLOREQLearning, answerQLearning, repeatedSequencesQLearning, numRiskyFaultChainsQLearning, allFalseCountQLearning, 
                 M_episodes_GRQN, EXPLOREGRQN, answerGRQN_kappa_3, repeatedSequencesGRQN_kappa_3, numRiskyFaultChainsGRQN_kappa_3, allFalseCountGRQN_kappa_3, lossGRQN_kappa_3, 
                 M_episodes_TE, EXPLORE_TE, answer_TE, repeatedSequences_TE, numRiskyFaultChains_TE, allFalseCount_TE, 
                 answerGRQN_kappa_2, repeatedSequencesGRQN_kappa_2, numRiskyFaultChainsGRQN_kappa_2, allFalseCountGRQN_kappa_2, lossGRQN_kappa_2, 
                 answerGRQN_kappa_1, repeatedSequencesGRQN_kappa_1, numRiskyFaultChainsGRQN_kappa_1, allFalseCountGRQN_kappa_1, lossGRQN_kappa_1):
        
        self.M = M

        self.M_episodes_QLearning = M_episodes_QLearning
        self.EXPLOREQLearning = EXPLOREQLearning
        self.answerQLearning = answerQLearning
        self.repeatedSequencesQLearning = repeatedSequencesQLearning
        self.numRiskyFaultChainsQLearning = numRiskyFaultChainsQLearning
        self.allFalseCountQLearning = allFalseCountQLearning

        self.LOADING_FACTOR = LOADING_FACTOR
        self.selectCase = selectCase
        self.FCsovler = FaultChainSovlerClass(self.LOADING_FACTOR, self.selectCase) # instantiated object!
        
        self.M_episodes_GRQN = M_episodes_GRQN
        self.EXPLOREGRQN = EXPLOREGRQN

        self.answerGRQN_kappa_3 = answerGRQN_kappa_3
        self.repeatedSequencesGRQN_kappa_3 = repeatedSequencesGRQN_kappa_3
        self.numRiskyFaultChainsGRQN_kappa_3 = numRiskyFaultChainsGRQN_kappa_3
        self.allFalseCountGRQN_kappa_3 = allFalseCountGRQN_kappa_3
        self.lossGRQN_kappa_3 = lossGRQN_kappa_3

        self.answerGRQN_kappa_2 = answerGRQN_kappa_2
        self.repeatedSequencesGRQN_kappa_2 = repeatedSequencesGRQN_kappa_2
        self.numRiskyFaultChainsGRQN_kappa_2 = numRiskyFaultChainsGRQN_kappa_2
        self.allFalseCountGRQN_kappa_2 = allFalseCountGRQN_kappa_2
        self.lossGRQN_kappa_2 = lossGRQN_kappa_2

        self.answerGRQN_kappa_1 = answerGRQN_kappa_1
        self.repeatedSequencesGRQN_kappa_1 = repeatedSequencesGRQN_kappa_1
        self.numRiskyFaultChainsGRQN_kappa_1 = numRiskyFaultChainsGRQN_kappa_1
        self.allFalseCountGRQN_kappa_1 = allFalseCountGRQN_kappa_1
        self.lossGRQN_kappa_1 = lossGRQN_kappa_1

        self.M_episodes_TE = M_episodes_TE
        self.EXPLORE_TE = EXPLORE_TE
        self.answer_TE = answer_TE
        self.repeatedSequences_TE = repeatedSequences_TE
        self.numRiskyFaultChains_TE = numRiskyFaultChains_TE
        self.allFalseCount_TE = allFalseCount_TE
        self.riskyFaultChainDict = riskyFaultChainDict
        
        # evaluate all the relevant results!
        self.eval_rewards()
        self.eval_num_risky_fcs()
        self.eval_fc_risk()
        self.eval_epsilon_decay()
        self.riskyCount = self.eval_how_many_risky_fcs()
        
    def display_results(self):
        self.plot_loss()
        self.plot_rewards()
        self.plot_num_risky_fcs()
        self.plot_fc_risk()        
        self.plot_epsilon_decay()

        print(" ")
        print("----Results: Q-Learning-----")
        print("Q-Learning: Found ",  self.riskyCount[0], " Risky FCs in ", self.M_episodes_QLearning, " search trials")
        print("Q-Learning: The cumulative risk is ", sum(self.rewardsQLearning))
        print("Q-Learning: Number of ALL False conditions: ", self.allFalseCountQLearning)
        print(" ")
        print("----Results: GRQN (kappa = 3)-----")
        print("GRQN: Found ",  self.riskyCount[1], " Risky FCs in ", self.M_episodes_GRQN, " search trials")
        print("GRQN: The cumulative risk is ", sum(self.rewardsGRQN_kappa_3))
        print("GRQN: Number of ALL False conditions: ", self.allFalseCountGRQN_kappa_3)
        print(" ")
        print("----Results: GRQN (kappa = 2)-----")
        print("GRQN: Found ",  self.riskyCount[2], " Risky FCs in ", self.M_episodes_GRQN, " search trials")
        print("GRQN: The cumulative risk is ", sum(self.rewardsGRQN_kappa_2))
        print("GRQN: Number of ALL False conditions: ", self.allFalseCountGRQN_kappa_2)
        print(" ")
        print("----Results: GRQN (kappa = 1)-----")
        print("GRQN: Found ",  self.riskyCount[3], " Risky FCs in ", self.M_episodes_GRQN, " search trials")
        print("GRQN: The cumulative risk is ", sum(self.rewardsGRQN_kappa_1))
        print("GRQN: Number of ALL False conditions: ", self.allFalseCountGRQN_kappa_1)
        print(" ")
        print("----Results: Trans Extension-----")
        print("TE: Found ",  self.riskyCount[4], " Risky FCs in ", self.M_episodes_TE, " search trials")
        print("TE: The cumulative risk is ", sum(self.rewards_TE))
        print("TE: Number of ALL False conditions: ", self.allFalseCount_TE)

    def eval_how_many_risky_fcs(self):
        # how many of the risky fault chains are discovered!
        countQLearning = 0
        for episode in self.answerQLearning:
            if tuple(episode[2]) in self.riskyFaultChainDict:
                countQLearning += 1

        countGRQN_kappa_3 = 0
        for episode in self.answerGRQN_kappa_3:
            if tuple(episode[2]) in self.riskyFaultChainDict:
                countGRQN_kappa_3 += 1

        countGRQN_kappa_2 = 0
        for episode in self.answerGRQN_kappa_2:
            if tuple(episode[2]) in self.riskyFaultChainDict:
                countGRQN_kappa_2 += 1

        countGRQN_kappa_1 = 0
        for episode in self.answerGRQN_kappa_1:
            if tuple(episode[2]) in self.riskyFaultChainDict:
                countGRQN_kappa_1 += 1 

        count_TE= 0
        for episode in self.answer_TE:
            if tuple(episode[2]) in self.riskyFaultChainDict:
                count_TE += 1

        return (countQLearning, countGRQN_kappa_3, countGRQN_kappa_2, countGRQN_kappa_1, count_TE)

    def plot_loss(self):
        plt.plot(self.lossGRQN_kappa_3, label = "GRQN training loss")
        plt.xlabel("Training Episodes")
        plt.ylabel("Training Loss")
        plt.legend(loc="upper right")
        plt.show()

    def eval_rewards(self):
        self.rewardsQLearning = [a for a, b, c in self.answerQLearning]
        self.rewardsGRQN_kappa_3 = [a for a, b, c in self.answerGRQN_kappa_3]
        self.rewardsGRQN_kappa_2 = [a for a, b, c in self.answerGRQN_kappa_2]
        self.rewardsGRQN_kappa_1 = [a for a, b, c in self.answerGRQN_kappa_1]
        self.rewards_TE = [a for a, b, c in self.answer_TE] 

    def plot_rewards(self):
        # Absolute Rewards
        plt.plot(self.rewardsGRQN_kappa_3, label = "PFW+GRQN (kappa = 3)")
        plt.plot(self.rewardsGRQN_kappa_2, label = "PFW+GRQN (kappa = 2)")
        plt.plot(self.rewardsGRQN_kappa_1, label = "PFW+GRQN (kappa = 1)")
        plt.plot(self.rewardsQLearning, label = "PFW+RL [13]")
        plt.plot(self.rewards_TE, label = "PFW+RL+TE [13]")
        plt.xlabel("Search Episodes")
        plt.ylabel("Rewards Collected")
        plt.legend(loc="upper left")
        plt.show()

    def eval_num_risky_fcs(self):
        self.cumNumRiskyFaultChainsGRQN_kappa_3 = np.cumsum(self.numRiskyFaultChainsGRQN_kappa_3)
        self.cumNumRiskyFaultChainsGRQN_kappa_2 = np.cumsum(self.numRiskyFaultChainsGRQN_kappa_2)
        self.cumNumRiskyFaultChainsGRQN_kappa_1 = np.cumsum(self.numRiskyFaultChainsGRQN_kappa_1)
        self.cumNumRiskyFaultChainsQLearning = np.cumsum(self.numRiskyFaultChainsQLearning)
        self.cumNumRiskyFaultChains_TE = np.cumsum(self.numRiskyFaultChains_TE)

    def plot_num_risky_fcs(self):
        # Number of risky Fault Chains!
        plt.plot(self.cumNumRiskyFaultChainsGRQN_kappa_3, label = "PFW+GRQN (kappa = 3)")
        plt.plot(self.cumNumRiskyFaultChainsGRQN_kappa_2, label = "PFW+GRQN (kappa = 2)")
        plt.plot(self.cumNumRiskyFaultChainsGRQN_kappa_1, label = "PFW+GRQN (kappa = 1)")
        plt.plot(self.cumNumRiskyFaultChainsQLearning, label = "PFW+RL [13]")
        plt.plot(self.cumNumRiskyFaultChains_TE, label = "PFW+RL+TE [13]")
        plt.xlabel("Search Episodes")
        plt.ylabel("Number of Risky FCs")
        plt.legend(loc="upper left")
        plt.show()

    def eval_fc_risk(self):
        self.cumRewardsGRQN_kappa_3 = np.cumsum(self.rewardsGRQN_kappa_3)
        self.cumRewardsGRQN_kappa_2 = np.cumsum(self.rewardsGRQN_kappa_2)
        self.cumRewardsGRQN_kappa_1 = np.cumsum(self.rewardsGRQN_kappa_1)
        self.cumRewardsQLearning = np.cumsum(self.rewardsQLearning)
        self.cumRewards_TE = np.cumsum(self.rewards_TE)

    def plot_fc_risk(self):
        # Accumulted Risk of Fault Chains!
        plt.plot(self.cumRewardsGRQN_kappa_3, label = "PFW+GRQN (kappa = 3)")
        plt.plot(self.cumRewardsGRQN_kappa_2, label = "PFW+GRQN (kappa = 2)")
        plt.plot(self.cumRewardsGRQN_kappa_1, label = "PFW+GRQN (kappa = 1)")
        plt.plot(self.cumRewardsQLearning , label = "PFW+RL [13]")
        plt.plot(self.cumRewards_TE, label = "PFW+RL+TE [13]")
        plt.xlabel("Search Episodes")
        plt.ylabel("Accumulated Risk")
        plt.legend(loc="upper left")
        plt.show()

    def eval_epsilon_decay(self):
        self.epsilonGRQN_kappa_3 = [b for a, b, c in self.answerGRQN_kappa_3]
        self.epsilonGRQN_kappa_2 = [b for a, b, c in self.answerGRQN_kappa_2]
        self.epsilonGRQN_kappa_1 = [b for a, b, c in self.answerGRQN_kappa_1]
        self.epsilonQLearning = [b for a, b, c in self.answerQLearning]
        self.epsilon_TE = [b for a, b, c in self.answer_TE]

    def plot_epsilon_decay(self):
        # Epsilon Decay
        plt.plot(self.epsilonGRQN_kappa_3, label = "PFW+GRQN (kappa = 3)")
        plt.plot(self.epsilonGRQN_kappa_2, label = "PFW+GRQN (kappa = 2)")
        plt.plot(self.epsilonGRQN_kappa_1, label = "PFW+GRQN (kappa = 1)")
        plt.plot(self.epsilonQLearning, label = "PFW+RL [13]")
        plt.plot(self.epsilon_TE, label = "PFW+RL+TE [13]")
        plt.xlabel("Search Episodes")
        plt.ylabel("Epsilon Decay")
        plt.legend(loc="upper right")
        plt.show()


class EvalutionClassThreeComparitive_time(object):
    def __init__(self, riskyFaultChainDict, LOADING_FACTOR, selectCase, M, dataSetString, 
                 M_episodes_QLearning, EXPLOREQLearning, answerQLearning, repeatedSequencesQLearning, numRiskyFaultChainsQLearning, allFalseCountQLearning, 
                 M_episodes_GRQN, EXPLOREGRQN, answerGRQN_kappa_3, repeatedSequencesGRQN_kappa_3, numRiskyFaultChainsGRQN_kappa_3, allFalseCountGRQN_kappa_3, lossGRQN_kappa_3, 
                 M_episodes_TE, EXPLORE_TE, answer_TE, repeatedSequences_TE, numRiskyFaultChains_TE, allFalseCount_TE, 
                 answerGRQN_kappa_2, repeatedSequencesGRQN_kappa_2, numRiskyFaultChainsGRQN_kappa_2, allFalseCountGRQN_kappa_2, lossGRQN_kappa_2, 
                 answerGRQN_kappa_1, repeatedSequencesGRQN_kappa_1, numRiskyFaultChainsGRQN_kappa_1, allFalseCountGRQN_kappa_1, lossGRQN_kappa_1):
        self.M = M

        self.M_episodes_QLearning = M_episodes_QLearning
        self.EXPLOREQLearning = EXPLOREQLearning
        self.answerQLearning = answerQLearning
        self.repeatedSequencesQLearning = repeatedSequencesQLearning
        self.numRiskyFaultChainsQLearning = numRiskyFaultChainsQLearning
        self.allFalseCountQLearning = allFalseCountQLearning

        self.LOADING_FACTOR = LOADING_FACTOR
        self.selectCase = selectCase
        self.FCsovler = FaultChainSovlerClass(self.LOADING_FACTOR, self.selectCase) # instantiated object!
        
        self.M_episodes_GRQN = M_episodes_GRQN
        self.EXPLOREGRQN = EXPLOREGRQN

        self.answerGRQN_kappa_3 = answerGRQN_kappa_3
        self.repeatedSequencesGRQN_kappa_3 = repeatedSequencesGRQN_kappa_3
        self.numRiskyFaultChainsGRQN_kappa_3 = numRiskyFaultChainsGRQN_kappa_3
        self.allFalseCountGRQN_kappa_3 = allFalseCountGRQN_kappa_3
        self.lossGRQN_kappa_3 = lossGRQN_kappa_3

        self.answerGRQN_kappa_2 = answerGRQN_kappa_2
        self.repeatedSequencesGRQN_kappa_2 = repeatedSequencesGRQN_kappa_2
        self.numRiskyFaultChainsGRQN_kappa_2 = numRiskyFaultChainsGRQN_kappa_2
        self.allFalseCountGRQN_kappa_2 = allFalseCountGRQN_kappa_2
        self.lossGRQN_kappa_2 = lossGRQN_kappa_2

        self.answerGRQN_kappa_1 = answerGRQN_kappa_1
        self.repeatedSequencesGRQN_kappa_1 = repeatedSequencesGRQN_kappa_1
        self.numRiskyFaultChainsGRQN_kappa_1 = numRiskyFaultChainsGRQN_kappa_1
        self.allFalseCountGRQN_kappa_1 = allFalseCountGRQN_kappa_1
        self.lossGRQN_kappa_1 = lossGRQN_kappa_1

        self.M_episodes_TE = M_episodes_TE
        self.EXPLORE_TE = EXPLORE_TE
        self.answer_TE = answer_TE
        self.repeatedSequences_TE = repeatedSequences_TE
        self.numRiskyFaultChains_TE = numRiskyFaultChains_TE
        self.allFalseCount_TE = allFalseCount_TE
        self.riskyFaultChainDict = riskyFaultChainDict
        
        # evaluate all the relevant results!
        self.eval_rewards()
        self.eval_num_risky_fcs()
        self.eval_fc_risk()
        self.riskyCount = self.eval_how_many_risky_fcs()
        
    def display_results(self):
        self.plot_loss()
        self.plot_rewards()
        self.plot_num_risky_fcs()
        self.plot_fc_risk()        

        print(" ")
        print("----Results: Q-Learning-----")
        print("Q-Learning: Found ",  self.riskyCount[0], " Risky FCs in ", self.M_episodes_QLearning, " search trials")
        print("Q-Learning: The cumulative risk is ", sum([sublist[0] for sublist in self.rewardsQLearning]))
        print("Q-Learning: Number of ALL False conditions: ", self.allFalseCountQLearning)
        print(" ")
        print("----Results: GRQN (kappa = 3)-----")
        print("GRQN: Found ",  self.riskyCount[1], " Risky FCs in ", self.M_episodes_GRQN, " search trials")
        print("GRQN: The cumulative risk is ", sum([sublist[0] for sublist in self.rewardsGRQN_kappa_3]))
        print("GRQN: Number of ALL False conditions: ", self.allFalseCountGRQN_kappa_3)
        print(" ")
        print("----Results: GRQN (kappa = 2)-----")
        print("GRQN: Found ",  self.riskyCount[2], " Risky FCs in ", self.M_episodes_GRQN, " search trials")
        print("GRQN: The cumulative risk is ", sum([sublist[0] for sublist in self.rewardsGRQN_kappa_2]))
        print("GRQN: Number of ALL False conditions: ", self.allFalseCountGRQN_kappa_2)
        print(" ")
        print("----Results: GRQN (kappa = 1)-----")
        print("GRQN: Found ",  self.riskyCount[3], " Risky FCs in ", self.M_episodes_GRQN, " search trials")
        print("GRQN: The cumulative risk is ", sum([sublist[0] for sublist in self.rewardsGRQN_kappa_1]))
        print("GRQN: Number of ALL False conditions: ", self.allFalseCountGRQN_kappa_1)
        print(" ")
        print("----Results: Trans Extension-----")
        print("TE: Found ",  self.riskyCount[4], " Risky FCs in ", self.M_episodes_TE, " search trials")
        print("TE: The cumulative risk is ", sum([sublist[0] for sublist in self.rewards_TE]))
        print("TE: Number of ALL False conditions: ", self.allFalseCount_TE)

    def eval_how_many_risky_fcs(self):
        # how many of the risky fault chains are discovered!
        countQLearning = 0
        for episode in self.answerQLearning:
            if tuple(episode[0][2]) in self.riskyFaultChainDict:
                countQLearning += 1

        countGRQN_kappa_3 = 0
        for episode in self.answerGRQN_kappa_3:
            if tuple(episode[0][2]) in self.riskyFaultChainDict:
                countGRQN_kappa_3 += 1

        countGRQN_kappa_2 = 0
        for episode in self.answerGRQN_kappa_2:
            if tuple(episode[0][2]) in self.riskyFaultChainDict:
                countGRQN_kappa_2 += 1

        countGRQN_kappa_1 = 0
        for episode in self.answerGRQN_kappa_1:
            if tuple(episode[0][2]) in self.riskyFaultChainDict:
                countGRQN_kappa_1 += 1 

        count_TE = 0
        for episode in self.answer_TE:
            if tuple(episode[0][2]) in self.riskyFaultChainDict:
                count_TE += 1

        return (countQLearning, countGRQN_kappa_3, countGRQN_kappa_2, countGRQN_kappa_1, count_TE)

    def plot_loss(self):
        plt.plot(self.lossGRQN_kappa_3, label = "GRQN training loss")
        plt.xlabel("Training Episodes")
        plt.ylabel("Training Loss")
        plt.legend(loc="upper right")
        plt.show()

    def eval_rewards(self):
        self.rewardsQLearning = [[a[0], b] for a, b in self.answerQLearning]
        self.rewardsGRQN_kappa_3 = [[a[0], b] for a, b in self.answerGRQN_kappa_3]
        self.rewardsGRQN_kappa_2 = [[a[0], b] for a, b in self.answerGRQN_kappa_2]
        self.rewardsGRQN_kappa_1 = [[a[0], b] for a, b in self.answerGRQN_kappa_1]
        self.rewards_TE = [[a[0], b] for a, b in self.answer_TE] 

    def plot_rewards(self):
        # Absolute Rewards
        plt.plot(self.rewardsGRQN_kappa_3, label = "PFW+GRQN (kappa = 3)")
        plt.plot(self.rewardsGRQN_kappa_2, label = "PFW+GRQN (kappa = 2)")
        plt.plot(self.rewardsGRQN_kappa_1, label = "PFW+GRQN (kappa = 1)")
        plt.plot(self.rewardsQLearning, label = "PFW+RL [13]")
        plt.plot(self.rewards_TE, label = "PFW+RL+TE [13]")
        plt.xlabel("Search Episodes")
        plt.ylabel("Rewards Collected")
        plt.legend(loc="upper left")
        plt.show()

    def eval_num_risky_fcs(self):

        self.cumNumRiskyFaultChainsGRQN_kappa_3 = [[cumNumChain, self.numRiskyFaultChainsGRQN_kappa_3[count][1]] for count, cumNumChain in enumerate(list(np.cumsum([row[0] for row in self.numRiskyFaultChainsGRQN_kappa_3])))]
        self.cumNumRiskyFaultChainsGRQN_kappa_2 = [[cumNumChain, self.numRiskyFaultChainsGRQN_kappa_2[count][1]] for count, cumNumChain in enumerate(list(np.cumsum([row[0] for row in self.numRiskyFaultChainsGRQN_kappa_2])))]        
        self.cumNumRiskyFaultChainsGRQN_kappa_1 = [[cumNumChain, self.numRiskyFaultChainsGRQN_kappa_1[count][1]] for count, cumNumChain in enumerate(list(np.cumsum([row[0] for row in self.numRiskyFaultChainsGRQN_kappa_1])))]
        self.cumNumRiskyFaultChainsQLearning = [[cumNumChain, self.numRiskyFaultChainsQLearning[count][1]] for count, cumNumChain in enumerate(list(np.cumsum([row[0] for row in self.numRiskyFaultChainsQLearning])))]            
        self.cumNumRiskyFaultChains_TE = [[cumNumChain, self.numRiskyFaultChains_TE[count][1]] for count, cumNumChain in enumerate(list(np.cumsum([row[0] for row in self.numRiskyFaultChains_TE])))]
        

    def plot_num_risky_fcs(self):
        # Number of risky Fault Chains!
        plt.plot(self.cumNumRiskyFaultChainsGRQN_kappa_3, label = "PFW+GRQN (kappa = 3)")
        plt.plot(self.cumNumRiskyFaultChainsGRQN_kappa_2, label = "PFW+GRQN (kappa = 2)")
        plt.plot(self.cumNumRiskyFaultChainsGRQN_kappa_1, label = "PFW+GRQN (kappa = 1)")
        plt.plot(self.cumNumRiskyFaultChainsQLearning, label = "PFW+RL [13]")
        plt.plot(self.cumNumRiskyFaultChains_TE, label = "PFW+RL+TE [13]")
        plt.xlabel("Search Episodes")
        plt.ylabel("Number of Risky FCs")
        plt.legend(loc="upper left")
        plt.show()

    def eval_fc_risk(self):
        self.cumRewardsGRQN_kappa_3 = [[cumRisk, self.rewardsGRQN_kappa_3[count][1]] for count, cumRisk in enumerate(list(np.cumsum([row[0] for row in self.rewardsGRQN_kappa_3])))]
        self.cumRewardsGRQN_kappa_2 = [[cumRisk, self.rewardsGRQN_kappa_2[count][1]] for count, cumRisk in enumerate(list(np.cumsum([row[0] for row in self.rewardsGRQN_kappa_2])))]
        self.cumRewardsGRQN_kappa_1 = [[cumRisk, self.rewardsGRQN_kappa_1[count][1]] for count, cumRisk in enumerate(list(np.cumsum([row[0] for row in self.rewardsGRQN_kappa_1])))]
        self.cumRewardsQLearning = [[cumRisk, self.rewardsQLearning[count][1]] for count, cumRisk in enumerate(list(np.cumsum([row[0] for row in self.rewardsQLearning])))]
        self.cumRewards_TE = [[cumRisk, self.rewards_TE[count][1]] for count, cumRisk in enumerate(list(np.cumsum([row[0] for row in self.rewards_TE])))]

    def plot_fc_risk(self):
        # Accumulted Risk of Fault Chains!
        plt.plot(self.cumRewardsGRQN_kappa_3, label = "PFW+GRQN (kappa = 3)")
        plt.plot(self.cumRewardsGRQN_kappa_2, label = "PFW+GRQN (kappa = 2)")
        plt.plot(self.cumRewardsGRQN_kappa_1, label = "PFW+GRQN (kappa = 1)")
        plt.plot(self.cumRewardsQLearning , label = "PFW+RL [13]")
        plt.plot(self.cumRewards_TE, label = "PFW+RL+TE [13]")
        plt.xlabel("Search Episodes")
        plt.ylabel("Accumulated Risk")
        plt.legend(loc="upper left")
        plt.show()


class EvaluationClassIterationMCPlot(object):
    def __init__(self, totalNumRiskyGRQN_kappa_3, totalRiskGRQN_kappa_3, 
                 totalNumRiskyQLearning, totalRiskQLearning, 
                 totalNumRisky_TE, totalRisk_TE, 
                 totalNumRiskyGRQN_kappa_2, totalRiskGRQN_kappa_2, 
                 totalNumRiskyGRQN_kappa_1, totalRiskGRQN_kappa_1):
        
        self.totalNumRiskyGRQN_kappa_3 = totalNumRiskyGRQN_kappa_3
        self.totalNumRiskyGRQN_kappa_2 = totalNumRiskyGRQN_kappa_2
        self.totalNumRiskyGRQN_kappa_1 = totalNumRiskyGRQN_kappa_1
        self.totalNumRiskyQLearning = totalNumRiskyQLearning
        self.totalNumRisky_TE = totalNumRisky_TE

        self.totalRiskGRQN_kappa_3 = totalRiskGRQN_kappa_3
        self.totalRiskGRQN_kappa_2 = totalRiskGRQN_kappa_2
        self.totalRiskGRQN_kappa_1 = totalRiskGRQN_kappa_1
        self.totalRiskQLearning = totalRiskQLearning
        self.totalRisk_TE = totalRisk_TE

    def display_statistics(self):
        print("")
        print("No. of Risky FCs found by GRQN (kappa=3)", np.mean(self.totalNumRiskyGRQN_kappa_3)," +- " , np.std(self.totalNumRiskyGRQN_kappa_3))
        print("Total Risk found by GRQN (kappa=3)", np.mean(self.totalRiskGRQN_kappa_3)," +- " , np.std(self.totalRiskGRQN_kappa_3))
        print("")
        print("No. of Risky FCs found by GRQN (kappa=2)", np.mean(self.totalNumRiskyGRQN_kappa_2)," +- " , np.std(self.totalNumRiskyGRQN_kappa_2))
        print("Total Risk found by GRQN (kappa=2)", np.mean(self.totalRiskGRQN_kappa_2)," +- " , np.std(self.totalRiskGRQN_kappa_2))
        print("")
        print("No. of Risky FCs found by GRQN (kappa=1)", np.mean(self.totalNumRiskyGRQN_kappa_1)," +- " , np.std(self.totalNumRiskyGRQN_kappa_1))
        print("Total Risk found by GRQN (kappa=1)", np.mean(self.totalRiskGRQN_kappa_1)," +- " , np.std(self.totalRiskGRQN_kappa_1))
        print("")
        print("No. of Risky FCs found by QLearning ", np.mean(self.totalNumRiskyQLearning)," +- " , np.std(self.totalNumRiskyQLearning))
        print("Total Risk found by QLearning ", np.mean(self.totalRiskQLearning)," +- " , np.std(self.totalRiskQLearning))
        print("")
        print("No. of Risky FCs found by TE ", np.mean(self.totalNumRisky_TE)," +- " , np.std(self.totalNumRisky_TE))
        print("Total Risk found by TE ", np.mean(self.totalRisk_TE)," +- " , np.std(self.totalRisk_TE))
        print("")

    def plot_fc_risk(self, risky_data_list, min_len):
        tick_spacing = 100
        sns.set_style("whitegrid", {'axes.grid' : True, 'axes.edgecolor':'black'})
        fig = plt.figure()
        plt.clf()
        ax = fig.gca()
        colors = ["blue", "black", "purple", "red", "green"]
        markers = ["*", "p", "v", "s", "o"]
        linestyles =  ['solid', 'dashed', 'dashdot', 'dotted', '-.'] 
        markersizes = [10, 10, 10, 9, 9]
        labels = ["Algorithm$~1\;(\kappa = 3)$", "Algorithm$~1\;(\kappa = 2)$", "Algorithm$~1\;(\kappa = 1)$", "${\sf PFW+RL\;}[12]$", "${\sf PFW+RL+TE\;}[12]$"]
        color_patch = []
        for color, label, data, marker, linestyle, markersize in zip(colors, labels, risky_data_list, markers, linestyles, markersizes):
            #sns.tsplot(time=range(min_len), data=data, color = color, ci=95, linestyle = linestyle, marker=marker,  markersize=30)
            sns.tsplot(time=range(min_len), data=data, color = color, ci=0.1, linestyle = linestyle, markevery=tick_spacing, marker=marker, markersize=markersize)
            #color_patch.append(mpatches.Patch(label=label, linestyle = linestyle))
            color_patch.append(mlines.Line2D([], [], color=color, marker=marker, label=label, linestyle = linestyle, markersize=markersize))

        print(min_len)
        plt.xlim([0, min_len])
        plt.xlabel('Search Iteration $s \in [S]$', fontsize=18)
        plt.ylabel('Average Accumulated TLL', fontsize=18)
        lgd=plt.legend(
        frameon=True, fancybox=True, \
        prop={'size':16}, handles=color_patch, loc="best")
        #plt.title('Title', fontsize=14s)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #ax.set_xticks(list(range(1, min_len+1)))
        #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

        #plt.setp(ax.get_xticklabels(), fontsize=16)
        #plt.setp(ax.get_yticklabels(), fontsize=16)
        sns.despine()
        plt.tight_layout()
        plt.savefig('39bus_1200_risky_data_list.pdf', bbox_inches='tight')
        plt.show()

    def plot_num_risky_fcs(self, num_data_list, min_len):
        tick_spacing = 100
        sns.set_style("whitegrid", {'axes.grid' : True, 'axes.edgecolor':'black'})
        fig = plt.figure()
        plt.clf()
        ax = fig.gca()
        colors = ["blue", "black", "purple", "red", "green"]
        markers = ["*", "p", "v", "s", "o"]
        linestyles =  ['solid', 'dashed', 'dashdot', 'dotted', '-.'] 
        labels = ["Algorithm$~1\;{(\kappa = 3)}$", "Algorithm$~1\;{(\kappa = 2)}$", "Algorithm$~1\;{(\kappa = 1)}$", "${\sf PFW+RL\;}[12]$", "${\sf PFW+RL+TE\;}[12]$"]
        markersizes = [10, 10, 10, 9, 9]
        color_patch = []
        for color, label, data, marker, linestyle, markersize in zip(colors, labels, num_data_list, markers, linestyles, markersizes):
            #sns.tsplot(time=range(min_len), data=data, color=color, ci=95)
            sns.tsplot(time=range(min_len), data=data, color = color, ci=0.1, linestyle = linestyle, markevery=tick_spacing, marker=marker, markersize=markersize)
            #color_patch.append(mpatches.Patch(color=color, label=label))
            color_patch.append(mlines.Line2D([], [], color=color, marker=marker, label=label, linestyle = linestyle, markersize=markersize))

        print(min_len)
        plt.xlim([0, min_len])
        plt.xlabel('Search Iteration $s \in [S]$', fontsize=18)
        plt.ylabel('Average No. of Risky FCs', fontsize=18)
        lgd=plt.legend(
        frameon=True, fancybox=True, \
        prop={'size':16}, handles=color_patch, loc="best")
        #plt.title('Title', fontsize=14s)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #ax.set_xticks(list(range(1, min_len+1)))
        #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

        #plt.setp(ax.get_xticklabels(), fontsize=16)
        #plt.setp(ax.get_yticklabels(), fontsize=16)
        sns.despine()
        plt.tight_layout()
        plt.savefig('39bus_1200_num_data_list.pdf', bbox_inches='tight')
        plt.show()

    def plot_regret(self, dataRegret, min_len):
        tick_spacing = 100
        sns.set_style("whitegrid", {'axes.grid' : True, 'axes.edgecolor':'black'})
        fig = plt.figure()
        plt.clf()
        ax = fig.gca()
        colors = ["blue", "black", "purple", "red", "green"]
        markers = ["*", "p", "v", "s", "o"]
        linestyles =  ['solid', 'dashed', 'dashdot', 'dotted', '-.'] 
        markersizes = [10, 10, 10, 9, 9]
        labels = ["Algorithm$~1\;(\kappa = 3)$", "Algorithm$~1\;(\kappa = 2)$", "Algorithm$~1\;(\kappa = 1)$", "${\sf PFW+RL\;}[12]$", "${\sf PFW+RL+TE\;}[12]$"]
        color_patch = []
        for color, label, data, marker, linestyle, markersize in zip(colors, labels, dataRegret, markers, linestyles, markersizes):
            #sns.tsplot(time=range(min_len), data=data, color = color, ci=95, linestyle = linestyle, marker=marker,  markersize=30)
            sns.tsplot(time=range(min_len), data=data, color = color, ci=0.1, linestyle = linestyle, markevery=tick_spacing, marker=marker, markersize=markersize)
            #color_patch.append(mpatches.Patch(label=label, linestyle = linestyle))
            color_patch.append(mlines.Line2D([], [], color=color, marker=marker, label=label, linestyle = linestyle, markersize=markersize))

        print(min_len)
        plt.xlim([0, min_len])
        plt.xlabel('Search Iteration $s \in [S]$', fontsize=18)
        plt.ylabel('Average Regret (in ${\sf MWs}$)', fontsize=18)
        lgd=plt.legend(
        frameon=True, fancybox=True, \
        prop={'size':16}, handles=color_patch, loc="best")
        #plt.title('Title', fontsize=14s)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #ax.set_xticks(list(range(1, min_len+1)))
        #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

        #plt.setp(ax.get_xticklabels(), fontsize=16)
        #plt.setp(ax.get_yticklabels(), fontsize=16)
        sns.despine()
        plt.tight_layout()
        plt.savefig('118bus_1200_average_regret.pdf', bbox_inches='tight')
        plt.show()

    def plot_precision(self, dataPrecision, min_len):
        tick_spacing = 100
        sns.set_style("whitegrid", {'axes.grid' : True, 'axes.edgecolor':'black'})
        fig = plt.figure()
        plt.clf()
        ax = fig.gca()
        colors = ["blue", "black", "purple", "red", "green"]
        markers = ["*", "p", "v", "s", "o"]
        linestyles =  ['solid', 'dashed', 'dashdot', 'dotted', '-.'] 
        labels = ["Algorithm$~1\;{(\kappa = 3)}$", "Algorithm$~1\;{(\kappa = 2)}$", "Algorithm$~1\;{(\kappa = 1)}$", "${\sf PFW+RL\;}[12]$", "${\sf PFW+RL+TE\;}[12]$"]
        markersizes = [10, 10, 10, 9, 9]
        color_patch = []
        for color, label, data, marker, linestyle, markersize in zip(colors, labels, dataPrecision, markers, linestyles, markersizes):
            #sns.tsplot(time=range(min_len), data=data, color=color, ci=95)
            sns.tsplot(time=range(min_len), data=data, color = color, ci=0.1, linestyle = linestyle, markevery=tick_spacing, marker=marker, markersize=markersize)
            #color_patch.append(mpatches.Patch(color=color, label=label))
            color_patch.append(mlines.Line2D([], [], color=color, marker=marker, label=label, linestyle = linestyle, markersize=markersize))

        print(min_len)
        plt.xlim([0, min_len])
        plt.xlabel('Search Iteration $s \in [S]$', fontsize=18)
        plt.ylabel('Average Precision', fontsize=18)
        lgd=plt.legend(
        frameon=True, fancybox=True, \
        prop={'size':16}, handles=color_patch, loc="best")
        #plt.title('Title', fontsize=14s)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #ax.set_xticks(list(range(1, min_len+1)))
        #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

        #plt.setp(ax.get_xticklabels(), fontsize=16)
        #plt.setp(ax.get_yticklabels(), fontsize=16)
        sns.despine()
        plt.tight_layout()
        plt.savefig('118bus_1200_avg_precision.pdf', bbox_inches='tight')
        plt.show()

    def display_statistics_for_list(self, dataRegret, dataPrecision):
        print("")
        print("Range for Regret found by GRQN (kappa=3)", np.mean(dataRegret[0, :, :], axis=0)[-1]," +- " , np.std(dataRegret[0, :, :], axis=0)[-1], " also in percentage ", 100*(np.std(dataRegret[0, :, :], axis=0)[-1])/(np.mean(dataRegret[0, :, :], axis=0)[-1]) )
        print("Range for precision by GRQN (kappa=3)", np.mean(dataPrecision[0, :, :], axis=0)[-1]," +- " , np.std(dataPrecision[0, :, :], axis=0)[-1], " also in percentage ", 100*(np.std(dataPrecision[0, :, :], axis=0)[-1])/(np.mean(dataPrecision[0, :, :], axis=0)[-1]) )
        print("")
        print("Range for Regret found by GRQN (kappa=2)", np.mean(dataRegret[1, :, :], axis=0)[-1]," +- " , np.std(np.std(dataRegret[1, :, :], axis=0)[-1]), " also in percentage ", 100*(np.std(dataRegret[1, :, :], axis=0)[-1])/(np.mean(dataRegret[1, :, :], axis=0)[-1]) )
        print("Range for precision by GRQN (kappa=2)", np.mean(dataPrecision[1, :, :], axis=0)[-1]," +- " , np.std(dataPrecision[1, :, :], axis=0)[-1], " also in percentage ", 100*(np.std(dataPrecision[1, :, :], axis=0)[-1])/(np.mean(dataPrecision[1, :, :], axis=0)[-1]))
        print("")
        print("Range for Regret found by GRQN (kappa=1)", np.mean(dataRegret[2, :, :], axis=0)[-1]," +- " , np.std(np.std(dataRegret[2, :, :], axis=0)[-1]), " also in percentage ", 100*(np.std(dataRegret[2, :, :], axis=0)[-1])/(np.mean(dataRegret[2, :, :], axis=0)[-1]))
        print("Range for precision by GRQN (kappa=1)", np.mean(dataPrecision[2, :, :], axis=0)[-1]," +- " , np.std(dataPrecision[2, :, :], axis=0)[-1], " also in percentage ", 100*(np.std(dataPrecision[2, :, :], axis=0)[-1])/(np.mean(dataPrecision[2, :, :], axis=0)[-1]) )
        print("")
        print("Range for Regret found by QLearning ", np.mean(dataRegret[3, :, :], axis=0)[-1]," +- " , np.std(np.std(dataRegret[3, :, :], axis=0)[-1]), " also in percentage ", 100*(np.std(dataRegret[3, :, :], axis=0)[-1])/(np.mean(dataRegret[3, :, :], axis=0)[-1]))
        print("Range for precision by QLearning ", np.mean(dataPrecision[3, :, :], axis=0)[-1]," +- " , np.std(dataPrecision[3, :, :], axis=0)[-1], " also in percentage ", 100*(np.std(dataPrecision[3, :, :], axis=0)[-1])/(np.mean(dataPrecision[3, :, :], axis=0)[-1]) )
        print("")
        print("Range for Regret found by TE ", np.mean(dataRegret[4, :, :], axis=0)[-1]," +- " , np.std(np.std(dataRegret[4, :, :], axis=0)[-1]), " also in percentage ", 100*(np.std(dataRegret[4, :, :], axis=0)[-1])/(np.mean(dataRegret[4, :, :], axis=0)[-1]))
        print("Range for precision by TE ", np.mean(dataPrecision[4, :, :], axis=0)[-1]," +- " , np.std(dataPrecision[4, :, :], axis=0)[-1], " also in percentage ", 100*(np.std(dataPrecision[4, :, :], axis=0)[-1])/(np.mean(dataPrecision[4, :, :], axis=0)[-1]) )
        print("")


class EvaluationClassIterationMCPlot_time(object):
    def __init__(self, totalNumRiskyGRQN_kappa_3, totalRiskGRQN_kappa_3, 
                 totalNumRiskyQLearning, totalRiskQLearning, 
                 totalNumRisky_TE, totalRisk_TE, 
                 totalNumRiskyGRQN_kappa_2, totalRiskGRQN_kappa_2, 
                 totalNumRiskyGRQN_kappa_1, totalRiskGRQN_kappa_1):
        
        self.totalNumRiskyGRQN_kappa_3 = totalNumRiskyGRQN_kappa_3
        self.totalNumRiskyGRQN_kappa_2 = totalNumRiskyGRQN_kappa_2
        self.totalNumRiskyGRQN_kappa_1 = totalNumRiskyGRQN_kappa_1
        self.totalNumRiskyQLearning = totalNumRiskyQLearning
        self.totalNumRisky_TE = totalNumRisky_TE

        self.totalRiskGRQN_kappa_3 = totalRiskGRQN_kappa_3
        self.totalRiskGRQN_kappa_2 = totalRiskGRQN_kappa_2
        self.totalRiskGRQN_kappa_1 = totalRiskGRQN_kappa_1
        self.totalRiskQLearning = totalRiskQLearning
        self.totalRisk_TE = totalRisk_TE

    def display_statistics(self):
        print("")
        print("No. of Risky FCs found by GRQN (kappa=3)", np.mean(self.totalNumRiskyGRQN_kappa_3)," +- " , np.std(self.totalNumRiskyGRQN_kappa_3))
        print("Total Risk found by GRQN (kappa=3)", np.mean(self.totalRiskGRQN_kappa_3)," +- " , np.std(self.totalRiskGRQN_kappa_3))
        print("")
        print("No. of Risky FCs found by GRQN (kappa=2)", np.mean(self.totalNumRiskyGRQN_kappa_2)," +- " , np.std(self.totalNumRiskyGRQN_kappa_2))
        print("Total Risk found by GRQN (kappa=2)", np.mean(self.totalRiskGRQN_kappa_2)," +- " , np.std(self.totalRiskGRQN_kappa_2))
        print("")
        print("No. of Risky FCs found by GRQN (kappa=1)", np.mean(self.totalNumRiskyGRQN_kappa_1)," +- " , np.std(self.totalNumRiskyGRQN_kappa_1))
        print("Total Risk found by GRQN (kappa=1)", np.mean(self.totalRiskGRQN_kappa_1)," +- " , np.std(self.totalRiskGRQN_kappa_1))
        print("")
        print("No. of Risky FCs found by QLearning ", np.mean(self.totalNumRiskyQLearning)," +- " , np.std(self.totalNumRiskyQLearning))
        print("Total Risk found by QLearning ", np.mean(self.totalRiskQLearning)," +- " , np.std(self.totalRiskQLearning))
        print("")
        print("No. of Risky FCs found by TE ", np.mean(self.totalNumRisky_TE)," +- " , np.std(self.totalNumRisky_TE))
        print("Total Risk found by TE ", np.mean(self.totalRisk_TE)," +- " , np.std(self.totalRisk_TE))
        print("")

    def plot_fc_risk(self, risky_data_list, min_len):
        tick_spacing = 10
        sns.set_style("whitegrid", {'axes.grid' : True, 'axes.edgecolor':'black'})
        fig = plt.figure()
        plt.clf()
        ax = fig.gca()
        colors = ["blue", "black", "purple", "red", "green"]
        markers = ["*", "p", "v", "s", "o"]
        linestyles =  ['solid', 'dashed', 'dashdot', 'dotted', '-.'] 
        markersizes = [10, 10, 10, 9, 9]
        labels = ["Algorithm$~1\;(\kappa = 3)$", "Algorithm$~1\;(\kappa = 2)$", "Algorithm$~1\;(\kappa = 1)$", "${\sf PFW+RL\;}[12]$", "${\sf PFW+RL+TE\;}[12]$"]
        color_patch = []
        for color, label, data, marker, linestyle, markersize in zip(colors, labels, risky_data_list, markers, linestyles, markersizes):
            #sns.tsplot(time=range(min_len), data=data, color = color, ci=95, linestyle = linestyle, marker=marker,  markersize=30)
            
            y = []
            t = []
            check = True
            for MC_iteration in data:
                y.append([episode[0] for episode in MC_iteration])
                if check:
                    for episode in MC_iteration:
                        t.append(episode[1])
                    check = False 

            sns.tsplot(time=t, data=y, color = color, ci=0.1, linestyle = linestyle, markevery=tick_spacing, marker=marker, markersize=markersize)
            #color_patch.append(mpatches.Patch(label=label, linestyle = linestyle))
            color_patch.append(mlines.Line2D([], [], color=color, marker=marker, label=label, linestyle = linestyle, markersize=markersize))

        print(max(t))
        plt.xlim([0, max(t)])
        plt.xlabel('Time Taken (in seconds)', fontsize=18)
        plt.ylabel('Average Accumulated TLL', fontsize=18)
        lgd=plt.legend(
        frameon=True, fancybox=True, \
        prop={'size':16}, handles=color_patch, loc="best")
        #plt.title('Title', fontsize=14s)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #ax.set_xticks(list(range(1, min_len+1)))
        #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

        #plt.setp(ax.get_xticklabels(), fontsize=16)
        #plt.setp(ax.get_yticklabels(), fontsize=16)
        sns.despine()
        plt.tight_layout()
        plt.savefig('1000_risky_data_list.pdf', bbox_inches='tight')
        plt.show()

    def plot_num_risky_fcs(self, num_data_list, min_len):
        tick_spacing = 10
        sns.set_style("whitegrid", {'axes.grid' : True, 'axes.edgecolor':'black'})
        fig = plt.figure()
        plt.clf()
        ax = fig.gca()
        colors = ["blue", "black", "purple", "red", "green"]
        markers = ["*", "p", "v", "s", "o"]
        linestyles =  ['solid', 'dashed', 'dashdot', 'dotted', '-.'] 
        labels = ["Algorithm$~1\;{(\kappa = 3)}$", "Algorithm$~1\;{(\kappa = 2)}$", "Algorithm$~1\;{(\kappa = 1)}$", "${\sf PFW+RL\;}[12]$", "${\sf PFW+RL+TE\;}[12]$"]
        markersizes = [10, 10, 10, 9, 9]
        color_patch = []
        for color, label, data, marker, linestyle, markersize in zip(colors, labels, num_data_list, markers, linestyles, markersizes):
            #sns.tsplot(time=range(min_len), data=data, color=color, ci=95)

            y = []
            t = []
            check = True
            for MC_iteration in data:
                y.append([episode[0] for episode in MC_iteration])
                if check:
                    for episode in MC_iteration:
                        t.append(episode[1])
                    check = False 
            
            sns.tsplot(time=t, data=y, color = color, ci=0.1, linestyle = linestyle, markevery=tick_spacing, marker=marker, markersize=markersize)
            #color_patch.append(mpatches.Patch(color=color, label=label))
            color_patch.append(mlines.Line2D([], [], color=color, marker=marker, label=label, linestyle = linestyle, markersize=markersize))

        print(max(t))
        plt.xlim([0, max(t)])
        plt.xlabel('Time Taken (in seconds)', fontsize=18)
        plt.ylabel('Average No. of Risky FCs', fontsize=18)
        lgd=plt.legend(
        frameon=True, fancybox=True, \
        prop={'size':16}, handles=color_patch, loc="best")
        #plt.title('Title', fontsize=14s)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #ax.set_xticks(list(range(1, min_len+1)))
        #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

        #plt.setp(ax.get_xticklabels(), fontsize=16)
        #plt.setp(ax.get_yticklabels(), fontsize=16)
        sns.despine()
        plt.tight_layout()
        plt.savefig('1000_num_data_list.pdf', bbox_inches='tight')
        plt.show()


        # Epsilon Decay
        epsilonGRQN = [b for a, b, c in self.answerGRQN]
        epsilonQLearning = [b for a, b, c in self.answerQLearning]
        plt.plot(epsilonGRQN, label = "GRQN Epsilon")
        plt.plot(epsilonQLearning, label = "Q-learning Epsilon")
        plt.xlabel("Search Episodes")
        plt.ylabel("Epsilon Decay")
        plt.legend(loc="upper right")
        plt.show()