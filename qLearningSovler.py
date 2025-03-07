import os
import math
import time 
import torch
import pickle
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
from scipy.special import softmax

import argparse
from faultChainSolver import FaultChainSovlerClass



class QLearningSolverClass(object):
    def __init__(self, M_episodes, NUM_ACTIONS, LOADING_FACTOR, selectCase, EPSILON_1_m, GAMMA, dataSetString, M):
        self.M = M
        self.M_episodes = M_episodes
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_STATES = NUM_ACTIONS*(NUM_ACTIONS-1)*(NUM_ACTIONS-2) + NUM_ACTIONS*(NUM_ACTIONS-1) + NUM_ACTIONS + 1

        self.EPSILON_1_m = EPSILON_1_m

        self.LOADING_FACTOR = LOADING_FACTOR
        self.selectCase = selectCase
        self.FCsovler = FaultChainSovlerClass(self.LOADING_FACTOR, self.selectCase) # instantiated object!

        self.visit_count = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))
        self.initial_state = self.FCsovler.environmentStep([])

        self.q_func = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))
        self.GAMMA = GAMMA

        self.answer = []
        self.repeatedSequences = set()
        self.numRiskyFaultChains = []

        #self.numRiskyFaultChain_5percent = []
        #self.numRiskyFaultChain_10percent = []
        #self.numRiskyFaultChain_15percent = []
        
        self.allFalseCount = 0

        # to keep track of the time elapsed!
        self.answer_time = []
        self.numRiskyFaultChains_time = []

    def calculate_epsilon(self):
        """ 
        returns (float) : epsilon (as per paper)
        """
        # columns of actionDecision: [line indices, capcacity, powerflow]
        actionDecision = self.initial_state[5]
        denominator = np.sum(actionDecision[:, 2])
        
        numerator = np.sum(actionDecision[:, 2]/np.sqrt(self.visit_count[0, :] + 1))
        #print(numerator, denominator)
        return max(numerator/denominator, self.EPSILON_1_m)

    def transition_extension_calculate_epsilon(self, EPSILON_20):
        """ 
        returns (float) : epsilon (as per paper)
        """
        denominator = np.sum(self.q_func[0, :])
        
        numerator = np.sum(self.q_func[0, :]/np.sqrt(self.visit_count[0, :] + 1))
        #print(numerator, denominator)

        return min(EPSILON_20*numerator/denominator, 1)

    def power_flow_based(self, actionMask, current_round, actionDecision, currentState):
        """ 
        actionDecision (np.arr) : (remaining lines, 3) with columns [line indices, capcacity, powerflow] for epsilon calculation   
        Returns (int) : Generates actions based on power flowing in rounds in the current round
        """
        numpyActionMask = np.array(actionMask[:, current_round])
        true_idx = np.argwhere(numpyActionMask) # true_idx reveals what actions are legal 
        # now find those indices that can be actually taken out 
        # indices of relavant lines = find intersection between true_idx[:, 0] & actionDecision[:, 0]
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idx[:, 0])      
        #print(pf_based_actions)
        #print("pf_based_actions shape: ", pf_based_actions.shape)
        #print("actionDecision shape: ", actionDecision.shape)
        #print(actionDecision[:, 0], pf_based_actions)

        relevant_rows = []
        for line_index in pf_based_actions:
            row_index = np.where(actionDecision[:, 0]==line_index)
            relevant_rows.append(int(row_index[0]))

        relevantActionDecision = actionDecision[relevant_rows, :]
        visiting_count_vector = self.visit_count[currentState, relevant_rows]

        #weighted_flow = abs(relevantActionDecision[:, 2])          # did NOT perform well so sticking to what is exactly in the paper!   
        weighted_flow = abs(relevantActionDecision[:, 2]) / np.sqrt(visiting_count_vector + 1)         # softmax( abs(relevantActionDecision[:, 2]) / np.sqrt(visiting_count_vector + 1) )
        action = int(relevantActionDecision[np.argmax(weighted_flow), 0])

        return action

    def epsilon_greedy(self, currentState, actionDecision, actionMask, current_round, epsilon):
        """ 
        Returns an action selected by an epsilon-greedy exploration policy

        Args:
                currentState (int)      : index describing the current state
                actionDecision (np.arr) : (remaining lines, 3) with columns [line indices, capcacity, powerflow] for epsilon calculation
                actionMask (torch)      : initailly, tensor.ones of size (self.NUM_ACTIONS, 3) for filtering out valid actions
                current_round (int)     : 
                epsilon (float)         : the probability of choosing a random command
                
        Returns:
                (int, bool): the indices describing the action to take
        """
        numpyActionMaskCheck = np.array(actionMask[:, current_round])
        true_idxCheck = np.argwhere(numpyActionMaskCheck)
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idxCheck[:, 0])
        
        if len(pf_based_actions)==0:
            """ 
            no more actions to take as sequence already taken!
            """
            return None, True
        else:
            """
            possible to take actions. Now follow e-greedy approach!
            """
            allFalse = False 
            if np.random.uniform() > epsilon:         
                # (exploitation) action according to Q-values
                maxQ = float('-Inf')
                
                numpyActionMask = np.array(actionMask[:, current_round])
                true_idx = np.argwhere(numpyActionMask)
                pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idx[:, 0])    
                print("Based on Q-learning: ", len(pf_based_actions))
                
                for action in list(pf_based_actions):                
                    if maxQ < self.q_func[currentState, action] / np.sqrt(self.visit_count[currentState, action] + 1):
                        action_index = action
                        maxQ = self.q_func[currentState, action] / np.sqrt(self.visit_count[currentState, action] + 1)      
            else:
                # (exploration) action according to prior knowledge
                action_index = self.power_flow_based(actionMask, current_round, actionDecision, currentState)
            
            return action_index, allFalse

    def transition_extension_q_value(self, actionMask, current_round, actionDecision, currentState):

        def function(qvalue):
            new = 1 if qvalue>0 else -1
            return qvalue*new

        if self.selectCase=="39":
            prior_table_path = os.path.join("39-bus Code", "39bus_priorQlearningTable_for_06")
            with open(prior_table_path, "rb") as myFile:
                """
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                priorQlearningTable = pickle.load(myFile)

        elif self.selectCase=="118":
            prior_table_path = os.path.join("118-bus Code", "118bus_priorQlearningTable_for_10")
            with open(prior_table_path, "rb") as myFile:
                """
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                priorQlearningTable = pickle.load(myFile)

        numpyActionMask = np.array(actionMask[:, current_round])
        true_idx = np.argwhere(numpyActionMask) 
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idx[:, 0])      
        maxQ = float('-Inf')
        #print("Based on TE Q-learning: ", len(pf_based_actions))
        
        for action in list(pf_based_actions):                
            if maxQ < function(priorQlearningTable[currentState, action] / np.sqrt(self.visit_count[currentState, action] + 1)):
                action_index = action
                maxQ = function(priorQlearningTable[currentState, action] / np.sqrt(self.visit_count[currentState, action] + 1))

        return action_index

    def transition_extension_epsilon_greedy(self, currentState, actionDecision, actionMask, current_round, epsilon1, epsilon2):
        """ Returns an action selected by an epsilon-greedy exploration policy
        Args:
            currentState (int)      : index describing the current state
            actionDecision (np.arr) : (remaining lines, 3) with columns [line indices, capcacity, powerflow] for epsilon calculation
            actionMask (torch)      : initailly, tensor.ones of size (self.NUM_ACTIONS, 3) for filtering out valid actions
            current_round (int)     : 
            epsilon (float)         : the probability of choosing a random command
            
        Returns:
            (int, bool): the indices describing the action to take
        """
        numpyActionMaskCheck = np.array(actionMask[:, current_round])
        true_idxCheck = np.argwhere(numpyActionMaskCheck)
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idxCheck[:, 0])
        
        if len(pf_based_actions)==0:
            """ no more actions to take as sequence already taken!
            """
            return None, True
        else:
            """ possible to take actions. Now follow e-greedy approach!
            """
            allFalse = False 
            if np.random.uniform() > epsilon1:         
                # (exploitation) action according to Q-values
                maxQ = float('-Inf')
                
                numpyActionMask = np.array(actionMask[:, current_round])
                true_idx = np.argwhere(numpyActionMask)
                pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idx[:, 0])    
                print("Based on Q-learning: ", len(pf_based_actions))
                
                for action in list(pf_based_actions):                
                    if maxQ < self.q_func[currentState, action] / np.sqrt(self.visit_count[currentState, action] + 1):
                        action_index = action
                        maxQ = self.q_func[currentState, action] / np.sqrt(self.visit_count[currentState, action] + 1)      
            else:
                # (exploration) action according to prior knowledge
                if np.random.uniform() > epsilon2:     
                    action_index = self.power_flow_based(actionMask, current_round, actionDecision, currentState)
                else:
                    action_index = self.transition_extension_q_value(actionMask, current_round, actionDecision, currentState)
            
            return action_index, allFalse

    def tabular_q_learning(self, last_state, action, reward, current_state, terminal):
        ALPHA = self.visit_count[last_state, action]
        
        maxQ = 0
        if not terminal:
            for actionIndex in range(self.NUM_ACTIONS):
                maxQ = max(maxQ, self.q_func[current_state, actionIndex])

        self.q_func[last_state, action] = (1 - ALPHA)*self.q_func[last_state, action] + ALPHA*(reward + self.GAMMA*maxQ)

        return None   # This function shouldn't return anything just modify the matrix in place

    def run_epoch(self):

        if self.selectCase == "39":
            state_path = os.path.join("39-bus Code", "state_dict_for_IEEE_39.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-93196
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        elif self.selectCase == "118":
            state_path = os.path.join("118-bus Code", "state_dict_for_IEEE_118.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-120100
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        # training time performance!
        for i_episode in range(self.M_episodes): 
            if i_episode%40==0:
                print("---------------------------")
                print("Episode Number: ", i_episode, " of Collecting Experience")  

            # intially all actions can be taken so all are "True" since we have np.ones!
            actionMask = torch.ones( (self.NUM_ACTIONS, 3),  dtype=bool)                 # be careful when dealing with 118-bus system
            actionSpace = []
            #print("New Episode Action Space: ", actionSpace)

            done = False
            current_round  = 0
            current_return = 0

            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[ tuple(actionSpace) ]
            actionDecision = currentAnswer[5]

            for t in count():
                epsilon = self.calculate_epsilon()

                # add a set which you can mutate and pass so that we can choose available actions!
                action, allFalse = self.epsilon_greedy(last_state, actionDecision, actionMask, current_round, epsilon)

                if allFalse:
                    #print("Go back to previous round number ", current_round - 1, " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False # these lines cannot be removed in the future rounds as already taken
                    
                    last_state = stateDict[tuple(actionSpace)]
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    #print("Current Action Space: ", actionSpace)
                    
                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Oops, action already taken. Try another one!")
                        actionMask[actionSpace[-1], current_round] = False      # make unavailable!
                        actionSpace.pop()
                    else:
                        # finally take the action!

                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1
                        
                        # valid action, go to next round
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)
                        
                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False
                        
                        # collect new states, rewards!
                        reward = nextAnswer[2]          # immediate load shed scalar/reward!
                        done = nextAnswer[3]            # boolean - end of episode?
                        actionDecision = nextAnswer[5]  # updated available lines
                        current_return += reward
                        current_state = stateDict[tuple(actionSpace)]
                        
                        # update Q-function.
                        self.tabular_q_learning(last_state, action, reward, current_state, done)
                        
                        last_state = current_state

                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace))  # to make sure not taking it again!

            # for plotting purpose and evaluating the results!

            # mind you that actionSpace is in the new index dimension which impact 118-bus results (since all components are not considered)
            # my decision here is to convert the riskyFaultChainDict in order to avoid making changes in the actionSpace array.
            
            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[0] else 0
            #self.numRiskyFaultChain_5percent.append(risky)   

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[1] else 0
            #self.numRiskyFaultChain_10percent.append(risky)   

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[2] else 0
            #self.numRiskyFaultChain_15percent.append(risky)   

            self.answer.append((round(current_return, 2), round(epsilon, 5), actionSpace))

            if i_episode>0 and i_episode%200==0:
                print("")
                print("####################")
                print("Q-learning:  Found ",  sum(  self.numRiskyFaultChains  ), "  M% Risky FCs in ", i_episode, " search trials")
                #print("Q-learning: Found ",  sum(  self.numRiskyFaultChain_5percent  ), "  5% Risky FCs in ", i_episode, " search trials")
                #print("Q-learning: Found ",  sum(  self.numRiskyFaultChain_10percent ), " 10% Risky FCs in ", i_episode, " search trials")
                #print("Q-learning: Found ",  sum(  self.numRiskyFaultChain_15percent ), " 15% Risky FCs in ", i_episode, " search trials")
                self.compute_rewards()
                print("Q-learning: The cumulative risk is ", sum(self.rewards))
                print("Q-learning: Number of ALL False conditions: ", self.allFalseCount)
                print("####################")
                print("")

    def run_epoch_time(self, time_taken):

        if self.selectCase == "39":
            state_path = os.path.join("39-bus Code", "state_dict_for_IEEE_39.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-93196
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        elif self.selectCase == "118":
            state_path = os.path.join("118-bus Code", "state_dict_for_IEEE_118.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-120100
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        
        # training time performance!
        self.i_episode = 0
        start = time.perf_counter()

        while time.perf_counter() - start < time_taken:
            self.i_episode += 1
            print("---------------------------")
            print("Episode number: ", self.i_episode)    

            # intially all actions can be taken so all are "True" since we have np.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)

            done = False
            current_round  = 0
            current_return = 0

            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            actionDecision = currentAnswer[5]

            for t in count():
                epsilon = self.calculate_epsilon()

                # add a set which you can mutate and pass so that we can choose available actions!
                action, allFalse = self.epsilon_greedy(last_state, actionDecision, actionMask, current_round, epsilon)

                if allFalse:
                    print("Go back to previous round number ", current_round - 1, " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    print("Current Action Space: ", actionSpace)
                    
                    if tuple(actionSpace) in self.repeatedSequences:
                        print("Oops, action already taken. Try another one!")
                        actionMask[actionSpace[-1], current_round] = False      # make unavailable!
                        actionSpace.pop()
                    else:
                        # finally take the action!

                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1
                        
                        # valid action, go to next round
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)
                        
                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False
                        
                        # collect new states, rewards!
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]
                        current_return += reward
                        current_state = stateDict[tuple(actionSpace)]
                        
                        # update Q-function.
                        self.tabular_q_learning(last_state, action, reward, current_state, done)
                        
                        last_state = current_state

                        if done:
                            break

            stamp = time.perf_counter() - start

            self.repeatedSequences.add(tuple(actionSpace))  # to make sure not taking it again!

            # for plotting purpose and evaluating the results!
            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     
            self.answer.append((round(current_return, 2), round(epsilon, 5), actionSpace))

            # keeping track of the time module!
            self.numRiskyFaultChains_time.append([risky, round(stamp, 3)])
            self.answer_time.append([(round(current_return, 2), round(epsilon, 5), actionSpace), round(stamp, 3)])

    def transition_extension_run_epoch(self, EPSILON_20):

        if self.selectCase == "39":
            state_path = os.path.join("39-bus Code", "state_dict_for_IEEE_39.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-93196
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        elif self.selectCase == "118":
            state_path = os.path.join("118-bus Code", "state_dict_for_IEEE_118.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-120100
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        
        # training time performance!
        for i_episode in range(self.M_episodes): 
            if i_episode%40==0:
                print("---------------------------")
                print("Episode Number: ", i_episode, " of Collecting Experience")  

            # intially all actions can be taken so all are "True" since we have np.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            #print("New Episode Action Space: ", actionSpace)

            done = False
            current_round  = 0
            current_return = 0

            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            actionDecision = currentAnswer[5]

            for t in count():
                epsilon1 = self.calculate_epsilon()
                epsilon2 = self.transition_extension_calculate_epsilon(EPSILON_20)
                #print("eps1: ", epsilon1, "eps2: ", epsilon2)

                # add a set which you can mutate and pass so that we can choose available actions!
                action, allFalse = self.transition_extension_epsilon_greedy(last_state, actionDecision, actionMask, current_round, epsilon1, epsilon2)

                if allFalse:
                    #print("Go back to previous round number ", current_round - 1, " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    #print("Current Action Space: ", actionSpace)
                    
                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Oops, action already taken. Try another one!")
                        actionMask[actionSpace[-1], current_round] = False      # make unavailable!
                        actionSpace.pop()
                    else:
                        # finally take the action!

                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1
                        
                        # valid action, go to next round
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)
                        
                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False
                        
                        # collect new states, rewards!
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]
                        current_return += reward
                        current_state = stateDict[tuple(actionSpace)]
                        
                        # update Q-function.
                        self.tabular_q_learning(last_state, action, reward, current_state, done)
                        
                        last_state = current_state

                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace))  # to make sure not taking it again!

            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[0] else 0
            #self.numRiskyFaultChain_5percent.append(risky)   

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[1] else 0
            #self.numRiskyFaultChain_10percent.append(risky)   

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[2] else 0
            #self.numRiskyFaultChain_15percent.append(risky)   

            self.answer.append((round(current_return, 2), round(epsilon1, 5), actionSpace))

            if i_episode>0 and i_episode%200==0:
                print("")
                print("####################")
                print("GRQN:  Found ",  sum(  self.numRiskyFaultChains  ), "  M% Risky FCs in ", i_episode, " search trials")
                #print("Q-learning: Found ",  sum(  self.numRiskyFaultChain_5percent  ), "  5% Risky FCs in ", i_episode, " search trials")
                #print("Q-learning: Found ",  sum(  self.numRiskyFaultChain_10percent ), " 10% Risky FCs in ", i_episode, " search trials")
                #print("Q-learning: Found ",  sum(  self.numRiskyFaultChain_15percent ), " 15% Risky FCs in ", i_episode, " search trials")
                self.compute_rewards()
                print("Q-learning: The cumulative risk is ", sum(self.rewards))
                print("Q-learning: Number of ALL False conditions: ", self.allFalseCount)
                print("####################")
                print("")

    def transition_extension_run_epoch_time(self, EPSILON_20, time_taken):

        if self.selectCase == "39":
            state_path = os.path.join("39-bus Code", "state_dict_for_IEEE_39.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-93196
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        elif self.selectCase == "118":
            state_path = os.path.join("118-bus Code", "state_dict_for_IEEE_118.txt")
            with open(state_path, "rb") as myFile:
                """ 
                stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-120100
                """
                # actionSpace to state mapping which is of length self.NUM_STATES!
                stateDict = pickle.load(myFile)

        
        # training time performance!
        self.i_episode = 0
        start = time.perf_counter()
        while time.perf_counter() - start < time_taken:
            self.i_episode += 1
            print("---------------------------")
            print("Episode number: ", self.i_episode)    

            # intially all actions can be taken so all are "True" since we have np.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)

            done = False
            current_round  = 0
            current_return = 0

            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            actionDecision = currentAnswer[5]

            for t in count():
                epsilon1 = self.calculate_epsilon()
                epsilon2 = self.transition_extension_calculate_epsilon(EPSILON_20)

                # add a set which you can mutate and pass so that we can choose available actions!
                action, allFalse = self.transition_extension_epsilon_greedy(last_state, actionDecision, actionMask, current_round, epsilon1, epsilon2)

                if allFalse:
                    print("Go back to previous round number ", current_round - 1, " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    print("Current Action Space: ", actionSpace)
                    
                    if tuple(actionSpace) in self.repeatedSequences:
                        print("Oops, action already taken. Try another one!")
                        actionMask[actionSpace[-1], current_round] = False      # make unavailable!
                        actionSpace.pop()
                    else:
                        # finally take the action!

                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1
                        
                        # valid action, go to next round
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)
                        
                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False
                        
                        # collect new states, rewards!
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]
                        current_return += reward
                        current_state = stateDict[tuple(actionSpace)]
                        
                        # update Q-function.
                        self.tabular_q_learning(last_state, action, reward, current_state, done)
                        
                        last_state = current_state

                        if done:
                            break

            stamp = time.perf_counter() - start

            self.repeatedSequences.add(tuple(actionSpace))  # to make sure not taking it again!

            # for plotting purpose and evaluating the results!
            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     
            self.answer.append((round(current_return, 2), round(epsilon1, 5), actionSpace))

            # keeping track of the time module!
            self.numRiskyFaultChains_time.append([risky, round(stamp, 3)])
            self.answer_time.append([(round(current_return, 2), round(epsilon1, 5), actionSpace), round(stamp, 3)])

    def how_many_risky_fcs(self):
        # how many of the risky fault chains are discovered!
        count = 0
        for episode in self.answer:
            if tuple(episode[2]) in self.riskyFaultChainDict:
                count += 1

        return count

    def how_many_risky_fcs_time(self):
        # how many of the risky fault chains are discovered!
        count = 0
        for episode in self.answer_time:
            if tuple(episode[0][2]) in self.riskyFaultChainDict:
                count += 1

        return count

    def compute_rewards(self):
        self.rewards = [a for a, b, c in self.answer]

    def plot_rewards(self):
        # Absolute Rewards
        self.compute_rewards()
        plt.plot(self.rewards, label = "Q-learning Rewards Obtained")
        plt.xlabel("Search Trials")
        plt.ylabel("Rewards")
        plt.legend(loc="upper left")
        plt.show()

    def compute_rewards_time(self):
        self.rewards_time = [[a[0], b] for a, b in self.answer_time]

    def plot_rewards_time(self):
        # Absolute Rewards
        self.compute_rewards_time()
        plt.plot([episode[1] for episode in self.rewards_time], [episode[0] for episode in self.rewards_time], label = "Q-learning Rewards Obtained")        
        plt.xlabel("Time Taken")
        plt.ylabel("Rewards")
        plt.legend(loc="upper left")
        plt.show()

    def plot_num_risky_fcs(self):
        # Number of risky Fault Chains
        plt.plot( np.cumsum(self.numRiskyFaultChains)  , label = "Q-learning Number M%   Risky FCs")
        #plt.plot( np.cumsum(self.numRiskyFaultChain_5percent)  , label = "Q-learning Number 5%   Risky FCs")
        #plt.plot( np.cumsum(self.numRiskyFaultChain_10percent) , label = "Q-learning Number 10%  Risky FCs")
        #plt.plot( np.cumsum(self.numRiskyFaultChain_15percent) , label = "Q-learning Number 15%  Risky FCs")
        plt.xlabel("Search Trials")
        plt.ylabel("Number of risky FCs")
        plt.legend(loc="upper left")
        plt.show()

    def plot_num_risky_fcs_time(self):
        # Number of risky Fault Chains
        temp = [[cumNumChain, self.numRiskyFaultChains_time[count][1]] for count, cumNumChain in enumerate(list(np.cumsum([row[0] for row in self.numRiskyFaultChains_time])))]
        plt.plot([episode[1] for episode in temp], [episode[0] for episode in temp], label = "Q-learning Number Risky FCs")
        plt.xlabel("Time Taken")
        plt.ylabel("Number of risky FCs")
        plt.legend(loc="upper left")
        plt.show()

    def plot_fc_risk(self):
        # Risk of Fault Chains
        plt.plot(np.cumsum(self.rewards), label = "Q-learning Cumulative Risk")
        plt.xlabel("Search Trials")
        plt.ylabel("Total Risk Discovered")
        plt.legend(loc="upper left")
        plt.show()

    def plot_fc_risk_time(self):
        # Risk of Fault Chains
        temp = [[cumRisk, self.rewards_time[count][1]] for count, cumRisk in enumerate(list(np.cumsum([row[0] for row in self.rewards_time])))]
        plt.plot([episode[1] for episode in temp], [episode[0] for episode in temp], label = "Q-learning Cumulative Risk")
        plt.xlabel("Time Taken")
        plt.ylabel("Total Risk Discovered")
        plt.legend(loc="upper left")
        plt.show()

    def plot_epsilon_decay(self):
        # Epsilon Decay
        epsilon = [b for a, b, c in self.answer]
        plt.plot(epsilon, label = "Q-learning Epsilon")
        plt.xlabel("Search Trials")
        plt.ylabel("Epslion")
        plt.legend(loc="upper left")
        plt.show()


def main(select_case, loading_factor, m, iteration_track, m_episodes_qlearning, time_taken, gamma):
    # Automatically selecting NUM_ACTIONS and dataset string based on the select_case and loading_factor
    if select_case == "39":
        num_actions = 46
        dataSetString = "loading055_39bus.h5" if loading_factor == 0.55 else "loading06_39bus.h5"
    elif select_case == "118":
        num_actions = 50
        dataSetString = "118bus_loading06.h5" if loading_factor == 0.6 else "118bus_loading10.h5"
    else:
        raise ValueError("Invalid select_case. Please choose between '39' and '118'.")

    # Instantiate FaultChainSovlerClass (baseline)
    baseline = FaultChainSovlerClass(loading_factor, select_case)

    # Load dataset and process
    dataset_path = os.path.join('Datasets', dataSetString)
    data_set, threshold = baseline.load_dataset(dataset_path)
    baseline.print_datatset_information(data_set, threshold)

    # Find risky fault chains
    risky_fault_chain_dict = baseline.find_risky_fault_chains(data_set, m)

    # Instantiate QLearningSolverClass (baseline)
    baseline_qlearning = QLearningSolverClass(m_episodes_qlearning, num_actions, loading_factor, select_case, 0.01, gamma, dataSetString, m)
    baseline_qlearning.riskyFaultChainDict = risky_fault_chain_dict

    # Run Q-learning based on iteration_track flag
    if iteration_track:
        start = time.time()
        baseline_qlearning.run_epoch()
        #EPSILON_20 = 1.3
        #baseline_qlearning.transition_extension_run_epoch(EPSILON_20)
        end = time.time()
        print("Total Run Time: ", end - start)
        baseline_qlearning.plot_rewards()
        baseline_qlearning.plot_num_risky_fcs()
        baseline_qlearning.plot_fc_risk()
        baseline_qlearning.plot_epsilon_decay()
        print("Q-learning: Found ", baseline_qlearning.how_many_risky_fcs(), " Risky FCs in ", baseline_qlearning.M_episodes, " search trials")
        #print("Q-learning: Found ",  sum( baseline_qlearning.numRiskyFaultChain_5percent  ), "  5% Risky FCs in ", baseline_qlearning.M_episodes, " search trials")
        #print("Q-learning: Found ",  sum( baseline_qlearning.numRiskyFaultChain_10percent ), " 10% Risky FCs in ", baseline_qlearning.M_episodes, " search trials")
        #print("Q-learning: Found ",  sum( baseline_qlearning.numRiskyFaultChain_15percent ), " 15% Risky FCs in ", baseline_qlearning.M_episodes, " search trials")
        print("Q-learning: The cumulative risk is ", sum(baseline_qlearning.rewards))
        print("Q-learning: Number of ALL False conditions: ", baseline_qlearning.allFalseCount)
    else:
        # If iteration_track is 0, use the provided time_taken for fixed time
        start = time.time()
        baseline_qlearning.run_epoch_time(time_taken)
        #EPSILON_20 = 1.3
        #baseline_qlearning.transition_extension_run_epoch_time(EPSILON_20)
        end = time.time()
        print("Total Run Time: ", end - start)
        baseline_qlearning.plot_rewards_time()
        baseline_qlearning.plot_fc_risk_time()
        baseline_qlearning.plot_num_risky_fcs_time()
        print("Q-learning: Found ", baseline_qlearning.how_many_risky_fcs_time(), " Risky FCs in ", baseline_qlearning.i_episode , " search trials")
        print("Q-learning: The cumulative risk is ", sum([episode[0] for episode in baseline_qlearning.rewards_time]))
        print("Q-learning: Number of ALL False conditions: ", baseline_qlearning.allFalseCount)


if __name__ == '__main__':
    # python qLearningSovler.py --case 39 --load 0.55 --threshold 5 --if_iteration 1 --num_episodes 50 
    # python qLearningSovler.py --case 39 --load 0.55 --threshold 5 --if_iteration 0 --time_taken 60

    parser = argparse.ArgumentParser(description="Baseline Q-learning Fault Chain Solver.")
    parser.add_argument('--case', type=str, choices=['39', '118'], required=True, help="Select the case: '39', or '118'.")
    parser.add_argument('--load', type=float, choices=[0.55, 0.6, 1.0], required=True, help="Loading factor: 0.55, 0.6 for 39-bus and 0.6, 1.0 for 118-bus.")
    parser.add_argument('--threshold', type=int, required=True, help="Risky FC threshold in percentage.")
    parser.add_argument('--if_iteration', type=int, choices=[0, 1], required=True, help="Fixed iterations (1) or fixed time (0).")

    parser.add_argument('--num_episodes', type=int, default=50, help="Number of episodes for Q-learning. Only relevant when iterations = 1.")
    parser.add_argument('--time_taken', type=int, default=60, help="Time duration for fixed time execution when iterations are 0.")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for future rewards.")

    args = parser.parse_args()
    main(args.case, args.load, args.threshold, args.if_iteration, args.num_episodes, args.time_taken, args.gamma)