import copy
import math
import json
import time
import pickle
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

# deep learning stuff!
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to file path
from lib.alegnn.utils import graphML
from lib.alegnn.modules import architecturesTime

import argparse
from faultChainSolver import FaultChainSovlerClass


class GRNNQN(nn.Module):
    def __init__(self, n_actions, numNodes, dimInputSignals, dimOutputSignals, dimHiddenSignals, nFilterTaps, bias, nonlinearityHidden, nonlinearityOutput, nonlinearityReadout, dimReadout, dimEdgeFeatures):       
        super(GRNNQN, self).__init__()
        
        self.n_actions = n_actions
        # batchSize x timeSamples x dimReadout[-1] x numberNodes
        self.grnn = architecturesTime.GraphRecurrentNN_DB_with_hidden(dimInputSignals, dimOutputSignals, dimHiddenSignals, nFilterTaps, bias, nonlinearityHidden, nonlinearityOutput, nonlinearityReadout, dimReadout, dimEdgeFeatures)
        # batchSize x timeSamples x n_actions
        self.out_layer = nn.Linear(numNodes*dimReadout[-1], n_actions)
            
    # x (torch.tensor) : batchSize x timeSamples x dimInputSignals x numberNodes
    # S (torch.tensor) : batchSize x timeSamples (x dimEdgeFeatures) x numberNodes x numberNodes (can ignore dimEdgeFeatures due to internal unqueeze operation!)
    # ensure that "x, S, hidden" are tensors with correct dimensions!
    def forward(self, x, S, hidden = None):
        # make changes to the architecture so that the hidden layer is accessible from outside!
        grnn_out, hidden_out = self.grnn(x, S, hidden)
        # print(grnn_out.shape, hidden_out.shape)
        grnn_out = grnn_out.reshape(grnn_out.shape[0], grnn_out.shape[1], -1)
        
        # q_values : batchSize x timeSamples x n_actions
        q_values = self.out_layer(grnn_out)
        q_values = F.relu(q_values)         # it was suprisingly performing pretty well if you don't include this as well!
        return q_values, hidden_out
        
    # "x, S, hidden" are tensors with correct dimensions!
    # add an extra set which keep tracks of the already taken actions to ensure valid actions!
    # act is for single time dimension and single batch!
    def act(self, actionDecision, x, S, actionMask, current_round, currentState, visit_count, epsilon, hidden):        
        #   q_values (torch.tensor) -  1 x 1 x n_actions
        # hidden_out (torch.tensor) -  1 x 1 x dimHiddenSignals x numberNodes        
        q_values, hidden_out = self.forward(x, S, hidden)
        
        numpyActionMaskCheck = np.array(actionMask[:, current_round])
        true_idxCheck = np.argwhere(numpyActionMaskCheck)
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idxCheck[:, 0])
        
        if len(pf_based_actions)==0:
            # no more actions to take as sequence already taken!
            return None, None, True      
        else:
            allFalse = False
            if np.random.uniform() > epsilon:
                # be careful, should generate only valid actions!
                # I'd like to know if all of actionMask's entries are False?!!
                #print(" ")
                #print("Action taken based on Q-values")
                
                numpyActionMask = np.array(actionMask[:, current_round])
                true_idx = np.argwhere(numpyActionMask)
                pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idx[:, 0]) 
                #print("Based on GRQN: ", len(pf_based_actions)) 

                # this one takes action greedily w.r.t Q values - Do this when do not want to store a huge table!
                """
                lines_indices = torch.tensor(pf_based_actions)
                max_val = torch.max(torch.index_select(q_values[0, 0, :], 0, lines_indices)).item()
                action = (q_values[0, 0, :] == max_val).nonzero(as_tuple = True)[0].item() 
                """
                # this one takes action w.r.t Q values/visit count - Have to store a huge table if implemented naively.
                maxQ = float('-Inf')
                for try_action in list(pf_based_actions):                
                    if maxQ < q_values[0, 0, try_action] / np.sqrt(visit_count[currentState, try_action] + 1):
                        action = try_action
                        maxQ = q_values[0, 0, try_action] / np.sqrt(visit_count[currentState, try_action] + 1)     
                
            else:
                #print(" ")
                #print("Action taken based on Power-Flow prior")
                #action = self.random_action(actionMask, current_round)
                action = self.power_flow_based(actionMask, current_round, actionDecision, currentState, visit_count)
            
            hidden_next = torch.squeeze(hidden_out, 1)
            return action, hidden_next, allFalse


    def transition_extension_visit_decay_epsilon2(self, triallast_x, triallast_S, visit_count, EPSILON_20, NUM_ACTIONS):

        q_values, _ = self.forward(triallast_x, triallast_S)
        numerator = 0
        denominator = 0
        for try_action in range(NUM_ACTIONS):                
            numerator += q_values[0, 0, try_action] / np.sqrt(visit_count[0, try_action] + 1)    
            denominator += q_values[0, 0, try_action]

        return min(EPSILON_20*numerator/denominator, 1)


    def transition_extension_act(self, transition_extension_action, actionDecision, x, S, actionMask, current_round, currentState, visit_count, epsilon, hidden, epsilon2):        
        #   q_values (torch.tensor) -  1 x 1 x n_actions
        # hidden_out (torch.tensor) -  1 x 1 x dimHiddenSignals x numberNodes        
        q_values, hidden_out = self.forward(x, S, hidden)
        
        numpyActionMaskCheck = np.array(actionMask[:, current_round])
        true_idxCheck = np.argwhere(numpyActionMaskCheck)
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idxCheck[:, 0])
        
        if len(pf_based_actions)==0:
            # no more actions to take as sequence already taken!
            return None, None, True      
        else:
            allFalse = False
            if np.random.uniform() > epsilon:
                # be careful, should generate only valid actions!
                # I'd like to know if all of actionMask's entries are False?!!
                #print(" ")
                #print("Action taken based on Q-values")
                
                numpyActionMask = np.array(actionMask[:, current_round])
                true_idx = np.argwhere(numpyActionMask)
                pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idx[:, 0])  

                # this one takes action greedily w.r.t Q values - Do this when do not want to store a huge table!
                """
                lines_indices = torch.tensor(pf_based_actions)
                max_val = torch.max(torch.index_select(q_values[0, 0, :], 0, lines_indices)).item()
                action = (q_values[0, 0, :] == max_val).nonzero(as_tuple = True)[0].item() 
                """
                # this one takes action w.r.t Q values/visit count - Have to store a huge table if implemented naively.
                maxQ = float('-Inf')
                for try_action in list(pf_based_actions):                
                    if maxQ < q_values[0, 0, try_action] / np.sqrt(visit_count[currentState, try_action] + 1):
                        action = try_action
                        maxQ = q_values[0, 0, try_action] / np.sqrt(visit_count[currentState, try_action] + 1)     
                
            else:
                # (exploration) action according to prior knowledge
                if np.random.uniform() > epsilon2:     
                    action = self.power_flow_based(actionMask, current_round, actionDecision, currentState, visit_count)
                else:
                    action = transition_extension_action


            hidden_next = torch.squeeze(hidden_out, 1)
            return action, hidden_next, allFalse



    def transition_extension_q_value(self, x, S, hidden, actionDecision, actionMask, current_round, currentState, visit_count):
        q_values, _ = self.forward(x, S, hidden)
        numpyActionMaskCheck = np.array(actionMask[:, current_round])
        true_idxCheck = np.argwhere(numpyActionMaskCheck)
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idxCheck[:, 0])

        maxQ = float('-Inf')
        for try_action in list(pf_based_actions):                
            if maxQ < q_values[0, 0, try_action] / np.sqrt(visit_count[currentState, try_action] + 1):
                action = try_action
                maxQ = q_values[0, 0, try_action] / np.sqrt(visit_count[currentState, try_action] + 1)  
        return action


    def random_action(self, actionMask, current_round):
        """ Generates RANDOM actions from (0 -- L-1)
        """       
        numpyActionMask = np.array(actionMask[:, current_round])
        true_idx = np.argwhere(numpyActionMask)       
        action = np.random.choice(true_idx[:, 0])
        return action


    def power_flow_based(self, actionMask, current_round, actionDecision, currentState, visit_count):
        """ Generates actions based on power flowing in rounds in the current round.
        """
        numpyActionMask = np.array(actionMask[:, current_round])
        true_idx = np.argwhere(numpyActionMask)
        # true_idx reveals what actions are legal 
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
        visiting_count_vector = visit_count[currentState, relevant_rows]
        
        weighted_flow = abs(relevantActionDecision[:, 2]) / np.sqrt(visiting_count_vector + 1)
        action = int(relevantActionDecision[np.argmax(weighted_flow), 0])
        
        return action

class SequentialExperienceBuffer():
    """ Alternative Experience Buffer that stores sequences of fixed length
    """
    def __init__(self, max_seqs, seq_len):
        self.max_seqs = max_seqs
        self.counter = 0
        self.seq_len = seq_len
        self.storage = [[] for i in range(max_seqs)] # here is where you can load the stored buffer if any - made avaiable offline!

    def write_tuple(self, aoaro):
        if len(self.storage[self.counter]) >= self.seq_len:
            self.counter += 1
        
        # there are many tuples recorded (that are in a sequence) in each sublist!
        self.storage[self.counter].append(aoaro)
    
    def sample(self, batch_size):
        # Sample batches of (action, observation, action, reward, observation, done) tuples;
        # With dimensions (batch_size, seq_len) for rewards/actions/done and (batch_size, seq_len, obs_dim) for last_Ss/last_xs;
        last_Ss = []
        last_xs = []
        actions = []
        rewards = []
        Ss = []
        xs = []
        dones = []

        for i in range(batch_size):
            seq_idx = np.random.randint(self.counter)
                        
            # all of these belong to one episode!
            prevDummy_S, prevDummy_x, act, rew, dummy_S, dummy_x, done = zip(*self.storage[seq_idx])
            
            last_Ss.append(list(prevDummy_S))
            last_xs.append(list(prevDummy_x)) 
            actions.append(list(act))
            rewards.append(list(rew))
            Ss.append(list(dummy_S))
            xs.append(list(dummy_x))
            dones.append(list(done))
           
        return torch.tensor(last_Ss, dtype = torch.float32), torch.tensor(last_xs, dtype = torch.float32), torch.tensor(actions), torch.tensor(rewards).float(), torch.tensor(Ss, dtype = torch.float32), torch.tensor(xs, dtype = torch.float32), torch.tensor(dones)    
    # last_Ss, last_xs      : (batch_size, seq_len, obs_dim)
    # actions/rewards/dones : (batch_size, seq_len)
    
    def lastFewSamples(self, batch_size): 
        last_Ss = []
        last_xs = []
        actions = []
        rewards = []
        Ss = []
        xs = []
        dones = []

        for i in range(batch_size):
            seq_idx = self.counter - (batch_size + 2)
                        
            # all of these belong to one episode!
            prevDummy_S, prevDummy_x, act, rew, dummy_S, dummy_x, done = zip(*self.storage[seq_idx])
            
            last_Ss.append(list(prevDummy_S))
            last_xs.append(list(prevDummy_x)) 
            actions.append(list(act))
            rewards.append(list(rew))
            Ss.append(list(dummy_S))
            xs.append(list(dummy_x))
            dones.append(list(done))
           
        return torch.tensor(last_Ss, dtype = torch.float32), torch.tensor(last_xs, dtype = torch.float32), torch.tensor(actions), torch.tensor(rewards).float(), torch.tensor(Ss, dtype = torch.float32), torch.tensor(xs, dtype = torch.float32), torch.tensor(dones)    
    # last_Ss, last_xs      : (batch_size, seq_len, obs_dim)
    # actions/rewards/dones : (batch_size, seq_len)

class DGRQNSolverClass(object):
    def __init__(self, M_episodes, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, M, EPSILON_1_m, GAMMA, eps, kappa, numNodes, EXPLORE, replay_buffer_size, sample_length, batch_size, learning_rate, dimInputSignals, dimOutputSignals, dimHiddenSignals, nFilterTaps, bias, nonlinearityHidden, nonlinearityOutput, nonlinearityReadout, dimReadout, dimEdgeFeatures):
        self.M_episodes = M_episodes
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_STATES = NUM_ACTIONS*(NUM_ACTIONS-1)*(NUM_ACTIONS-2) + NUM_ACTIONS*(NUM_ACTIONS-1) + NUM_ACTIONS + 1
        self.numNodes = numNodes

        self.EPSILON_1_m = EPSILON_1_m

        self.replay_buffer_size = replay_buffer_size 
        self.sample_length = sample_length
        self.replay_buffer = SequentialExperienceBuffer(self.replay_buffer_size, self.sample_length)

        self.LOADING_FACTOR = LOADING_FACTOR
        self.selectCase = selectCase
        self.FCsovler = FaultChainSovlerClass(self.LOADING_FACTOR, self.selectCase) # instantiated object!

        self.visit_count = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))
        self.initial_state = self.FCsovler.environmentStep([])

        self.dimInputSignals = dimInputSignals
        self.dimOutputSignals = dimOutputSignals
        self.dimHiddenSignals = dimHiddenSignals
        self.nFilterTaps = nFilterTaps
        self.bias = bias
        self.nonlinearityHidden = nonlinearityHidden
        self.nonlinearityOutput = nonlinearityOutput
        self.nonlinearityReadout = nonlinearityReadout 
        self.dimReadout = dimReadout
        self.dimEdgeFeatures = dimEdgeFeatures

        self.eps = eps
        self.EXPLORE = EXPLORE
        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kappa = kappa

        self.answer = []
        self.repeatedSequences = set()
        self.numRiskyFaultChains = []

        #self.numRiskyFaultChain_5percent = []
        #self.numRiskyFaultChain_10percent = []
        #self.numRiskyFaultChain_15percent = []

        self.allFalseCount = 0
        self.hiddenSequence = [None]
        self.loss = []

        # to keep track of the time elapsed!
        self.answer_time = []
        self.numRiskyFaultChains_time = []

        self.TE_grnnqn = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
 

    def exp_decay_epsilon(self, eps_start, eps_end, i_episode, EXPLORE, eps_decay, MULT_DECAY):
        """ float : 
        """
        return eps_end + (eps_start - eps_end) * math.exp((-MULT_DECAY*(i_episode-EXPLORE))/eps_decay)

    def visit_decay_epsilon(self):
        """ float : 
        """
        # [indices, capcacity, powerflow]
        actionDecision = self.initial_state[5]
        denominator = np.sum(actionDecision[:, 2])
        
        numerator = np.sum(actionDecision[:, 2] / np.sqrt(self.visit_count[0, :] + 1))
        #print(numerator, denominator)
        return max(numerator/denominator, self.EPSILON_1_m) 

    def transition_extension_visit_decay_epsilon(self, EPSILON_20):
        """
        """
        pass

    def power_flow_based(self, actionMask, current_round, actionDecision):
        #Generates actions based on power flowing in rounds in the current round.

        numpyActionMaskCheck = np.array(actionMask[:, current_round])
        true_idxCheck = np.argwhere(numpyActionMaskCheck)         # true_idx reveals what actions are legal 
        pf_based_actions = np.intersect1d(np.array(actionDecision[:, 0], dtype=int), true_idxCheck[:, 0])
        if len(pf_based_actions)==0:
            # no more actions to take as sequence already taken!
            return None, True
        else:      
            numpyActionMask = np.array(actionMask[:, current_round])
            true_idx = np.argwhere(numpyActionMask)

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
            weighted_flow = abs(relevantActionDecision[:, 2])
            action = int(relevantActionDecision[np.argmax(weighted_flow), 0])
            
            return action, False

    def transition_extension(self):
        pass

    def fill_experience_buffer(self):

        for i_episode in range(self.EXPLORE): 
            if i_episode%20==0:
                print("---------------------------")
                print("Episode Number: ", i_episode, " of Collecting Experience")    
                    
            # intially all actions can be taken so all are "True" since we have torch.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)
            
            done = False
            current_round  = 0
            current_return = 0
            
            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_S = currentAnswer[0]           # np.array -- Adjacancy matrices!
            last_x = currentAnswer[1]           # np.array -- voltage angles!
            actionDecision = currentAnswer[5]

            for t in count():
                # add a set which you can mutate and pass so that we can choose available actions!
                action, allFalse = self.power_flow_based(actionMask, current_round, actionDecision)

                if allFalse:
                    print("Go back to previous round number ", current_round-1 , " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_S = currentAnswer[0]        # np.array
                    last_x = currentAnswer[1]        # np.array
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    print("Current Action Space: ", actionSpace)

                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Action ", actionSpace[-1], " already taken. Try another...")
                        # in-valid action
                        actionMask[actionSpace[-1], current_round] = False
                        actionSpace.pop()         
                    else: 
                        # valid action                
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)

                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False

                        # collect new observations!
                        S = nextAnswer[0]               # np.array - GSO
                        x = nextAnswer[1]               # np.array - Voltage Angles
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]

                        current_return += reward
                        self.replay_buffer.write_tuple((last_S, last_x, actionSpace[-1], reward, S, x, done))

                        # previous observations!
                        last_S = copy.deepcopy(S)
                        last_x = copy.deepcopy(x)

                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace))

    def train_grqn_already_experience(self):      

        self.repeatedSequences = set()
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

        grnnqn = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target.load_state_dict(grnnqn.state_dict())
        optimizer = torch.optim.Adam(grnnqn.parameters(), lr = self.learning_rate)

        # training time performance!
        for i_episode in range(self.M_episodes): 
            if i_episode%40==0:
                print("---------------------------")
                print("Episode Number: ", i_episode, " of Collecting Experience")     
                    
            # intially all actions are viable, so all of actionMask are "True" since we have torch.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            #print("New Episode Action Space: ", actionSpace)
            
            done = False
            current_round  = 0
            current_return = 0
            
            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            last_S = currentAnswer[0]     # np.array -- Adjacancy matrices!
            last_x = currentAnswer[1]     # np.array -- voltage angles!
            actionDecision = currentAnswer[5]

            # this will terminate for a length of three lines!
            for t in count():
                
                # add a set which you can mutate and pass so that we can choose available actions!
                action, hidden_next, allFalse = grnnqn.act(actionDecision, torch.tensor(last_x).float().view(1, 1, 1, -1), torch.tensor(last_S).float().view(1, 1, self.numNodes, self.numNodes), actionMask, current_round, last_state, self.visit_count, epsilon = self.eps, hidden = self.hiddenSequence[-1])

                if allFalse:
                    #print("Go back to previous round number ", current_round-1 , " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    self.hiddenSequence.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    last_S = currentAnswer[0]        # np.array
                    last_x = currentAnswer[1]        # np.array
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    self.hiddenSequence.append(hidden_next)
                    #print("Current Action Space: ", actionSpace)

                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Action ", actionSpace[-1], " already taken. Try another...")
                        # in-valid action
                        actionMask[actionSpace[-1], current_round] = False
                        actionSpace.pop()
                        self.hiddenSequence.pop()               
                    else: 
                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1

                        # valid action                
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)

                        # mark unavailable for further rounds 
                        actionMask[ actionSpace[-1], list(range(current_round, 3)) ] = False

                        # collect new observations!
                        S = nextAnswer[0]               # np.array - GSO
                        x = nextAnswer[1]               # np.array - Voltage Angles
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]

                        current_return += reward
                        current_state = stateDict[ tuple(actionSpace) ]
                        self.replay_buffer.write_tuple( (last_S, last_x, actionSpace[-1], reward, S, x, done) )

                        # previous observations!
                        last_S = copy.deepcopy(S)
                        last_x = copy.deepcopy(x)
                        last_state = current_state

                        if i_episode < 100:
                            self.eps = self.visit_decay_epsilon()

                        ###### this is where the learning happens!######
                        if i_episode > 100:
                            #start = time.time()
                            ######!######!######!######!######!######!######
                            for _ in range(self.kappa):

                                self.eps = self.visit_decay_epsilon()
                                # last_Ss : batch_size x timeSamples x numberNodes x numberNodes
                                # last_xs : batch_size x timeSamples x dimInputSignals x numberNodes
                                last_Ss, last_xs, actions, rewards, Ss, xs, dones = self.replay_buffer.sample(self.batch_size)

                                last_xs = torch.unsqueeze(last_xs, 2)
                                xs = torch.unsqueeze(xs, 2)

                                # q_values : batchSize x timeSamples x n_actions
                                q_values, _ = grnnqn.forward(last_xs, last_Ss)

                                # actions  : batchSize x timeSamples
                                # select items form "q_values", 
                                q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)

                                # predicted_q_values : batchSize x timeSamples x n_actions
                                predicted_q_values, _ = grnnqn_target.forward(xs, Ss)

                                target_values = rewards + (self.GAMMA * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

                                # Update network parameters
                                optimizer.zero_grad()
                                loss = torch.nn.MSELoss()(q_values , target_values.detach())
                                self.loss.append(loss.item())
                                loss.backward()
                                optimizer.step()     
                            ######!######!######!######!######!######!######
                            #end = time.time() 
                            #print("time taken: ", end - start)
                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace)) 
            
            # for plotting purpose
            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     
            
            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[0] else 0
            #self.numRiskyFaultChain_5percent.append(risky)   

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[1] else 0
            #self.numRiskyFaultChain_10percent.append(risky)   

            #risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict_all[2] else 0
            #self.numRiskyFaultChain_15percent.append(risky)   

            self.answer.append( (round(current_return, 2), round(self.eps, 5), actionSpace) )
            grnnqn_target.load_state_dict(grnnqn.state_dict())
            """ 
            if i_episode==500:
                with open("priorGRQN_for_06", "wb") as fp:
                    pickle.dump((grnnqn.state_dict(), self.hiddenSequence[-1]), fp)
            """ 
            if i_episode>0 and i_episode%200==0:
                print("")
                print("####################")
                print("GRQN:  Found ",  sum(  self.numRiskyFaultChains  ), "  M% Risky FCs in ", i_episode, " search trials")
                #print("GRQN: Found ",  sum(  self.numRiskyFaultChain_5percent  ), "  5% Risky FCs in ", i_episode, " search trials")
                #print("GRQN: Found ",  sum(  self.numRiskyFaultChain_10percent ), " 10% Risky FCs in ", i_episode, " search trials")
                #print("GRQN: Found ",  sum(  self.numRiskyFaultChain_15percent ), " 15% Risky FCs in ", i_episode, " search trials")
                self.compute_rewards()
                print("GRQN: The cumulative risk is ", sum(self.rewards))
                print("GRQN: Number of ALL False conditions: ", self.allFalseCount)
                print("####################")
                print("")

    def train_grqn_already_experience_time(self, time_taken):      

        self.repeatedSequences = set()
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

        grnnqn = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target.load_state_dict(grnnqn.state_dict())
        optimizer = torch.optim.Adam(grnnqn.parameters(), lr = self.learning_rate)

        # training time performance!
        self.i_episode = 0
        start = time.perf_counter()
        while time.perf_counter() - start < time_taken:
            self.i_episode += 1
            print("---------------------------")
            print("Episode number: ", self.i_episode)    
                    
            # intially all actions can be taken so all are "True" since we have torch.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)
            
            done = False
            current_round  = 0
            current_return = 0
            
            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            last_S = currentAnswer[0]     # np.array -- Adjacancy matrices!
            last_x = currentAnswer[1]     # np.array -- voltage angles!
            actionDecision = currentAnswer[5]

            # this will terminate for a length of three lines!
            for t in count():
                
                # add a set which you can mutate and pass so that we can choose available actions!
                action, hidden_next, allFalse = grnnqn.act(actionDecision, torch.tensor(last_x).float().view(1, 1, 1, -1), torch.tensor(last_S).float().view(1, 1, self.numNodes, self.numNodes), actionMask, current_round, last_state, self.visit_count, epsilon = self.eps, hidden = self.hiddenSequence[-1])

                if allFalse:
                    print("Go back to previous round number ", current_round-1 , " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    self.hiddenSequence.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    last_S = currentAnswer[0]        # np.array
                    last_x = currentAnswer[1]        # np.array
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    self.hiddenSequence.append(hidden_next)
                    print("Current Action Space: ", actionSpace)

                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Action ", actionSpace[-1], " already taken. Try another...")
                        # in-valid action
                        actionMask[actionSpace[-1], current_round] = False
                        actionSpace.pop()
                        self.hiddenSequence.pop()               
                    else: 
                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1

                        # valid action                
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)

                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False

                        # collect new observations!
                        S = nextAnswer[0]               # np.array - GSO
                        x = nextAnswer[1]               # np.array - Voltage Angles
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]

                        current_return += reward
                        current_state = stateDict[tuple(actionSpace)]
                        self.replay_buffer.write_tuple((last_S, last_x, actionSpace[-1], reward, S, x, done))

                        # previous observations!
                        last_S = copy.deepcopy(S)
                        last_x = copy.deepcopy(x)
                        last_state = current_state


                        ###### this is where the learning happens!######
                        ######!######!######!######!######!######!######
                        for _ in range(self.kappa):
                            self.eps = self.visit_decay_epsilon()
                            
                            # last_Ss : batch_size x timeSamples x numberNodes x numberNodes
                            # last_xs : batch_size x timeSamples x dimInputSignals x numberNodes
                            last_Ss, last_xs, actions, rewards, Ss, xs, dones = self.replay_buffer.sample(self.batch_size)

                            last_xs = torch.unsqueeze(last_xs, 2)
                            xs = torch.unsqueeze(xs, 2)

                            # q_values : batchSize x timeSamples x n_actions
                            q_values, _ = grnnqn.forward(last_xs, last_Ss)

                            # actions  : batchSize x timeSamples
                            # select items form "q_values", 
                            q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)

                            # predicted_q_values : batchSize x timeSamples x n_actions
                            predicted_q_values, _ = grnnqn_target.forward(xs, Ss)

                            target_values = rewards + (self.GAMMA * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

                            # Update network parameters
                            optimizer.zero_grad()
                            loss = torch.nn.MSELoss()(q_values , target_values.detach())
                            self.loss.append(loss.item())
                            loss.backward()
                            optimizer.step()     
                        ######!######!######!######!######!######!######

                        if done:
                            break

            stamp = time.perf_counter() - start

            self.repeatedSequences.add(tuple(actionSpace)) 
            # for plotting purpose
            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     
            self.answer.append((round(current_return, 2), round(self.eps, 5), actionSpace))
            grnnqn_target.load_state_dict(grnnqn.state_dict())

            # keeping track of the time module!
            self.numRiskyFaultChains_time.append([risky, round(stamp, 3)])
            self.answer_time.append([(round(current_return, 2), round(self.eps, 5), actionSpace), round(stamp, 3)])

    def transition_extension_train_grqn_already_experience(self, EPSILON_20):      

        self.repeatedSequences = set()
        with open("state_dict_for_IEEE_39.txt", "rb") as myFile:
            """ stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-93196
            """
            # actionSpace to state mapping which is of length self.NUM_STATES!
            stateDict = pickle.load(myFile)


        with open("priorGRQN_for_06", "rb") as myFile:
            """
            """
            # actionSpace to state mapping which is of length self.NUM_STATES!
            trans_ext_state, hidden = pickle.load(myFile)
        transition_extension_grnnqn = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        transition_extension_grnnqn.load_state_dict(trans_ext_state)


        grnnqn = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target.load_state_dict(grnnqn.state_dict())
        optimizer = torch.optim.Adam(grnnqn.parameters(), lr = self.learning_rate)

        # training time performance!
        for i_episode in range(self.M_episodes): 
            print("---------------------------")
            print("Episode number: ", i_episode)    
                    
            # intially all actions can be taken so all are "True" since we have torch.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)
            
            done = False
            current_round  = 0
            current_return = 0
            
            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            last_S = currentAnswer[0]     # np.array -- Adjacancy matrices!
            last_x = currentAnswer[1]     # np.array -- voltage angles!
            actionDecision = currentAnswer[5]

            # this will terminate for a length of three lines!
            for t in count():
                
                triallast_S = self.initial_state[0]     # np.array -- Adjacancy matrices!
                triallast_x = self.initial_state[1]     # np.array -- voltage angles!
                eps2 = grnnqn.transition_extension_visit_decay_epsilon2(torch.tensor(triallast_x).float().view(1, 1, 1, -1), torch.tensor(triallast_S).float().view(1, 1, self.numNodes, self.numNodes), self.visit_count, EPSILON_20, self.NUM_ACTIONS)


                transition_extension_action = transition_extension_grnnqn.transition_extension_q_value(torch.tensor(last_x).float().view(1, 1, 1, -1), torch.tensor(last_S).float().view(1, 1, self.numNodes, self.numNodes), None, actionDecision, actionMask, current_round, last_state, self.visit_count)

                # add a set which you can mutate and pass so that we can choose available actions!
                action, hidden_next, allFalse = grnnqn.transition_extension_act(transition_extension_action, actionDecision, torch.tensor(last_x).float().view(1, 1, 1, -1), torch.tensor(last_S).float().view(1, 1, self.numNodes, self.numNodes), actionMask, current_round, last_state, self.visit_count, epsilon = self.eps, hidden = self.hiddenSequence[-1], epsilon2 = eps2)

                if allFalse:
                    print("Go back to previous round number ", current_round-1 , " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    self.hiddenSequence.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    last_S = currentAnswer[0]        # np.array
                    last_x = currentAnswer[1]        # np.array
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    self.hiddenSequence.append(hidden_next)
                    print("Current Action Space: ", actionSpace)

                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Action ", actionSpace[-1], " already taken. Try another...")
                        # in-valid action
                        actionMask[actionSpace[-1], current_round] = False
                        actionSpace.pop()
                        self.hiddenSequence.pop()               
                    else: 
                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1

                        self.eps = self.visit_decay_epsilon()
                        

                        # valid action                
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)

                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False

                        # collect new observations!
                        S = nextAnswer[0]               # np.array - GSO
                        x = nextAnswer[1]               # np.array - Voltage Angles
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]

                        current_return += reward
                        current_state = stateDict[tuple(actionSpace)]
                        self.replay_buffer.write_tuple((last_S, last_x, actionSpace[-1], reward, S, x, done))

                        # previous observations!
                        last_S = copy.deepcopy(S)
                        last_x = copy.deepcopy(x)
                        last_state = current_state

                        ###### this is where the learning happens!######
                        ######!######!######!######!######!######!######
                        for _ in range(3):
                            # last_Ss : batch_size x timeSamples x numberNodes x numberNodes
                            # last_xs : batch_size x timeSamples x dimInputSignals x numberNodes
                            last_Ss, last_xs, actions, rewards, Ss, xs, dones = self.replay_buffer.sample(self.batch_size)

                            last_xs = torch.unsqueeze(last_xs, 2)
                            xs = torch.unsqueeze(xs, 2)

                            # q_values : batchSize x timeSamples x n_actions
                            q_values, _ = grnnqn.forward(last_xs, last_Ss)

                            # actions  : batchSize x timeSamples
                            # select items form "q_values", 
                            q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)

                            # predicted_q_values : batchSize x timeSamples x n_actions
                            predicted_q_values, _ = grnnqn_target.forward(xs, Ss)

                            target_values = rewards + (self.GAMMA * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

                            # Update network parameters
                            optimizer.zero_grad()
                            loss = torch.nn.MSELoss()(q_values , target_values.detach())
                            self.loss.append(loss.item())
                            loss.backward()
                            optimizer.step()     
                        ######!######!######!######!######!######!######

                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace)) 
            # for plotting purpose
            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     
            self.answer.append((round(current_return, 2), round(self.eps, 5), actionSpace))
            grnnqn_target.load_state_dict(grnnqn.state_dict())

    def train_grqn(self):     
        """
        This function is to be used if you want to fill the buffer and train one after the other together. It involves hidden-layer manipulation. (taking it from start to finish)
        """ 

        with open("state_dict_for_IEEE_39.txt", "rb") as myFile:
            """ stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-93196
            """
            # actionSpace to state mapping which is of length self.NUM_STATES!
            stateDict = pickle.load(myFile)

        grnnqn = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target.load_state_dict(grnnqn.state_dict())
        optimizer = torch.optim.Adam(grnnqn.parameters(), lr = self.learning_rate)

        replay_buffer = SequentialExperienceBuffer(self.replay_buffer_size, self.sample_length)

        # training time performance!
        for i_episode in range(self.M_episodes): 
            print("---------------------------")
            print("Episode number: ", i_episode)    
                    
            # intially all actions can be taken so all are "True" since we have torch.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)
            
            done = False
            current_round  = 0
            current_return = 0
            
            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            last_S = currentAnswer[0]     # np.array -- Adjacancy matrices!
            last_x = currentAnswer[1]     # np.array -- voltage angles!
            actionDecision = currentAnswer[5]

            # this will terminate for a length of three lines!
            for t in count():
                
                # add a set which you can mutate and pass so that we can choose available actions!
                action, hidden_next, allFalse = grnnqn.act(actionDecision, torch.tensor(last_x).float().view(1, 1, 1, -1), torch.tensor(last_S).float().view(1, 1, self.numNodes, self.numNodes), actionMask, current_round, last_state, self.visit_count, epsilon = self.eps, hidden = self.hiddenSequence[-1])

                if allFalse:
                    print("Go back to previous round number ", current_round-1 , " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    self.hiddenSequence.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    last_S = currentAnswer[0]        # np.array
                    last_x = currentAnswer[1]        # np.array
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    self.hiddenSequence.append(hidden_next)
                    print("Current Action Space: ", actionSpace)

                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Action ", actionSpace[-1], " already taken. Try another...")
                        # in-valid action
                        actionMask[actionSpace[-1], current_round] = False
                        actionSpace.pop()
                        self.hiddenSequence.pop()               
                    else: 
                        # update the visit_count matrix
                        if i_episode >= self.EXPLORE:
                            self.visit_count[last_state, action] += 1
                        #if i_episode == self.EXPLORE:
                        #    self.visit_count = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))

                        # valid action                
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)

                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False

                        # collect new observations!
                        S = nextAnswer[0]               # np.array - GSO
                        x = nextAnswer[1]               # np.array - Voltage Angles
                        reward = nextAnswer[2]          # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]

                        current_return += reward
                        current_state = stateDict[tuple(actionSpace)]
                        replay_buffer.write_tuple((last_S, last_x, actionSpace[-1], reward, S, x, done))

                        # previous observations!
                        last_S = copy.deepcopy(S)
                        last_x = copy.deepcopy(x)
                        last_state = current_state

                        # this is where the learning happens!
                        if i_episode > self.EXPLORE:

                            #eps = exp_decay_epsilon(eps_start, eps_end, i_episode, EXPLORE, eps_decay, MULT_DECAY)
                            self.eps = self.visit_decay_epsilon()
                            
                            # last_Ss : batch_size x timeSamples x numberNodes x numberNodes
                            # last_xs : batch_size x timeSamples x dimInputSignals x numberNodes
                            last_Ss, last_xs, actions, rewards, Ss, xs, dones = replay_buffer.sample(self.batch_size)

                            last_xs = torch.unsqueeze(last_xs, 2)
                            xs = torch.unsqueeze(xs, 2)

                            # q_values : batchSize x timeSamples x n_actions
                            q_values, _ = grnnqn.forward(last_xs, last_Ss)

                            # actions  : batchSize x timeSamples
                            # select items form "q_values", 
                            q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)

                            # predicted_q_values : batchSize x timeSamples x n_actions
                            predicted_q_values, _ = grnnqn_target.forward(xs, Ss)

                            target_values = rewards + (self.GAMMA * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

                            # Update network parameters
                            optimizer.zero_grad()
                            loss = torch.nn.MSELoss()(q_values , target_values.detach())
                            loss.backward()
                            optimizer.step()     

                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace)) 
            if i_episode >= self.EXPLORE:
                if i_episode == self.EXPLORE:
                    self.repeatedSequences = set()

                # for plotting purpose
                risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
                self.numRiskyFaultChains.append(risky)     
                self.answer.append((round(current_return, 2), round(self.eps, 5), actionSpace))
                grnnqn_target.load_state_dict(grnnqn.state_dict())
            else:
                self.answer.append((0, round(self.eps, 5), []))
                self.numRiskyFaultChains.append(0)

    def numriskyfcs_fill_experience_buffer(self):
        # idea did not work out!

        for i_episode in range(self.EXPLORE): 
            print("---------------------------")
            print("Episode Number: ", i_episode, " of Collecting Experience")    
                    
            # intially all actions can be taken so all are "True" since we have torch.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)
            
            done = False
            current_round  = 0
            current_return = 0
            
            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_S = currentAnswer[0]           # np.array -- Adjacancy matrices!
            last_x = currentAnswer[1]           # np.array -- voltage angles!
            actionDecision = currentAnswer[5]

            for t in count():
                # add a set which you can mutate and pass so that we can choose available actions!
                action, allFalse = self.power_flow_based(actionMask, current_round, actionDecision)

                if allFalse:
                    print("Go back to previous round number ", current_round-1 , " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_S = currentAnswer[0]        # np.array
                    last_x = currentAnswer[1]        # np.array
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # (Still stays the same) reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    print("Current Action Space: ", actionSpace)

                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Action ", actionSpace[-1], " already taken. Try another...")
                        # in-valid action
                        actionMask[actionSpace[-1], current_round] = False
                        actionSpace.pop()         
                    else: 
                        # valid action                
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)

                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False

                        # collect new observations!
                        S = nextAnswer[0]               # np.array - GSO
                        x = nextAnswer[1]               # np.array - Voltage Angles

                        if nextAnswer[4]>=31.27: # make this dynamically change with M
                            reward = 1       # immediate scalar/reward!
                        else:
                            reward = 0       # immediate scalar/reward!

                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]

                        current_return += nextAnswer[2] # actual immediate scalar/reward!
                        self.replay_buffer.write_tuple((last_S, last_x, actionSpace[-1], reward, S, x, done))

                        # previous observations!
                        last_S = copy.deepcopy(S)
                        last_x = copy.deepcopy(x)

                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace))

    def numriskyfcs_train_grqn_already_experience(self):      
        # idea did not work out    
        self.repeatedSequences = set()
        with open("state_dict_for_IEEE_39.txt", "rb") as myFile:
            """ stateDict          : <class 'dict'>
                stateDict.keys()   : tuples of sequences
                stateDict.values() : int 0-93196
            """
            # actionSpace to state mapping which is of length self.NUM_STATES!
            stateDict = pickle.load(myFile)

        grnnqn = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target = GRNNQN(self.NUM_ACTIONS, self.numNodes, self.dimInputSignals, self.dimOutputSignals, self.dimHiddenSignals, self.nFilterTaps, self.bias, self.nonlinearityHidden, self.nonlinearityOutput, self.nonlinearityReadout, self.dimReadout, self.dimEdgeFeatures)
        grnnqn_target.load_state_dict(grnnqn.state_dict())
        optimizer = torch.optim.Adam(grnnqn.parameters(), lr = self.learning_rate)

        # training time performance!
        for i_episode in range(self.M_episodes): 
            print("---------------------------")
            print("Episode number: ", i_episode)    
                    
            # intially all actions can be taken so all are "True" since we have torch.ones!
            actionMask = torch.ones((self.NUM_ACTIONS, 3), dtype=bool) 
            actionSpace = []
            print("New Episode Action Space: ", actionSpace)
            
            done = False
            current_round  = 0
            current_return = 0
            
            # reset the environment
            currentAnswer = self.FCsovler.environmentStep(actionSpace)
            
            last_state = stateDict[tuple(actionSpace)]
            last_S = currentAnswer[0]     # np.array -- Adjacancy matrices!
            last_x = currentAnswer[1]     # np.array -- voltage angles!
            actionDecision = currentAnswer[5]

            # this will terminate for a length of three lines!
            for t in count():
                
                # add a set which you can mutate and pass so that we can choose available actions!
                action, hidden_next, allFalse = grnnqn.act(actionDecision, torch.tensor(last_x).float().view(1, 1, 1, -1), torch.tensor(last_S).float().view(1, 1, self.numNodes, self.numNodes), actionMask, current_round, last_state, self.visit_count, epsilon = self.eps, hidden = self.hiddenSequence[-1])

                if allFalse:
                    print("Go back to previous round number ", current_round-1 , " No more actions left...")
                    self.allFalseCount += 1
                    actionMask[:, list(range(current_round, 3))] = True
                    
                    current_round -= 1
                    actionMask[actionSpace[-1], current_round] = False
                    actionSpace.pop()
                    self.hiddenSequence.pop()
                    
                    currentAnswer = self.FCsovler.environmentStep(actionSpace)
                    actionMask[actionSpace, list(range(current_round, 3))] = False
                    
                    last_state = stateDict[tuple(actionSpace)]
                    last_S = currentAnswer[0]        # np.array
                    last_x = currentAnswer[1]        # np.array
                    actionDecision = currentAnswer[5]
                    current_return = currentAnswer[4]  # reset cumulative reward until current stage!
                else:
                    actionSpace.append(action)
                    self.hiddenSequence.append(hidden_next)
                    print("Current Action Space: ", actionSpace)

                    if tuple(actionSpace) in self.repeatedSequences:
                        #print("Action ", actionSpace[-1], " already taken. Try another...")
                        # in-valid action
                        actionMask[actionSpace[-1], current_round] = False
                        actionSpace.pop()
                        self.hiddenSequence.pop()               
                    else: 
                        # update the visit_count matrix
                        self.visit_count[last_state, action] += 1

                        # valid action                
                        current_round += 1
                        
                        # execute action
                        nextAnswer = self.FCsovler.environmentStep(actionSpace)

                        # mark unavailable for further rounds 
                        actionMask[actionSpace[-1], list(range(current_round, 3))] = False

                        # collect new observations!
                        S = nextAnswer[0]               # np.array - GSO
                        x = nextAnswer[1]               # np.array - Voltage Angles
                        if nextAnswer[4]>=31.27: # make this dynamically change with M
                            reward = 1       # immediate scalar/reward!
                        else:
                            reward = 0       # immediate scalar/reward!
                        done = nextAnswer[3]            # boolean
                        actionDecision = nextAnswer[5]

                        current_return += nextAnswer[2]
                        current_state = stateDict[tuple(actionSpace)]
                        self.replay_buffer.write_tuple((last_S, last_x, actionSpace[-1], reward, S, x, done))

                        # previous observations!
                        last_S = copy.deepcopy(S)
                        last_x = copy.deepcopy(x)
                        last_state = current_state


                        ###### this is where the learning happens!######
                        ######!######!######!######!######!######!######
                        for _ in range(3):
                            self.eps = self.visit_decay_epsilon()
                            
                            # last_Ss : batch_size x timeSamples x numberNodes x numberNodes
                            # last_xs : batch_size x timeSamples x dimInputSignals x numberNodes
                            last_Ss, last_xs, actions, rewards, Ss, xs, dones = self.replay_buffer.sample(self.batch_size)

                            last_xs = torch.unsqueeze(last_xs, 2)
                            xs = torch.unsqueeze(xs, 2)

                            # q_values : batchSize x timeSamples x n_actions
                            q_values, _ = grnnqn.forward(last_xs, last_Ss)

                            # actions  : batchSize x timeSamples
                            # select items form "q_values", 
                            q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)

                            # predicted_q_values : batchSize x timeSamples x n_actions
                            predicted_q_values, _ = grnnqn_target.forward(xs, Ss)

                            target_values = rewards + (self.GAMMA * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

                            # Update network parameters
                            optimizer.zero_grad()
                            loss = torch.nn.MSELoss()(q_values , target_values.detach())
                            self.loss.append(loss.item())
                            loss.backward()
                            optimizer.step()     
                        ######!######!######!######!######!######!######

                        if done:
                            break

            self.repeatedSequences.add(tuple(actionSpace)) 
            # for plotting purpose
            risky = 1 if tuple(actionSpace) in self.riskyFaultChainDict else 0
            self.numRiskyFaultChains.append(risky)     
            self.answer.append((round(current_return, 2), round(self.eps, 5), actionSpace))
            grnnqn_target.load_state_dict(grnnqn.state_dict())

    def how_many_risky_fcs(self):
        # how many of the risky fault chains are discovered!
        countGRQN = 0
        for episode in self.answer:
            if tuple(episode[2]) in self.riskyFaultChainDict:
                countGRQN += 1

        return countGRQN

    def how_many_risky_fcs_time(self):
        # how many of the risky fault chains are discovered!
        countGRQN = 0
        for episode in self.answer_time:
            if tuple(episode[0][2]) in self.riskyFaultChainDict:
                countGRQN += 1

        return countGRQN

    def plot_loss(self):
        plt.plot(self.loss, label = "GRQN Training Loss")
        plt.xlabel("Training Episodes")
        plt.ylabel("Training Loss Loss")
        plt.legend(loc="upper left")
        plt.show()

    def compute_rewards(self):
        self.rewards = [a for a, b, c in self.answer]

    def plot_rewards(self):
        # Absolute Rewards
        self.compute_rewards()
        #plt.plot(self.rewards[self.EXPLORE:], label = "GRQN Rewards Obatained")
        plt.plot(self.rewards, label = "GRQN Rewards Obtained")
        plt.xlabel("Search Trials")
        plt.ylabel("Rewards")
        plt.legend(loc="upper left")
        plt.show()

    def compute_rewards_time(self):
        self.rewards_time = [[a[0], b] for a, b in self.answer_time]

    def plot_rewards_time(self):
        # Absolute Rewards
        self.compute_rewards_time()
        plt.plot([episode[1] for episode in self.rewards_time], [episode[0] for episode in self.rewards_time], label = "GRQN Rewards Obtained")        
        plt.xlabel("Time Taken")
        plt.ylabel("Rewards")
        plt.legend(loc="upper left")
        plt.show()

    def plot_num_risky_fcs(self):
        # Number of risky Fault Chains
        plt.plot( np.cumsum(self.numRiskyFaultChains)  , label = "GRQN Number M%   Risky FCs")
        #plt.plot( np.cumsum(self.numRiskyFaultChain_5percent)  , label = "GRQN Number 5%   Risky FCs")
        #plt.plot( np.cumsum(self.numRiskyFaultChain_10percent) , label = "GRQN Number 10%  Risky FCs")
        #plt.plot( np.cumsum(self.numRiskyFaultChain_15percent) , label = "GRQN Number 15%  Risky FCs")
        plt.xlabel("Search Trials")
        plt.ylabel("Number of risky FCs")
        plt.legend(loc="upper left")
        plt.show()

    def plot_num_risky_fcs_time(self):
        # Number of risky Fault Chains
        temp = [[cumNumChain, self.numRiskyFaultChains_time[count][1]] for count, cumNumChain in enumerate(list(np.cumsum([row[0] for row in self.numRiskyFaultChains_time])))]
        plt.plot([episode[1] for episode in temp], [episode[0] for episode in temp], label = "GRQN Number Risky FCs")
        plt.xlabel("Time Taken")
        plt.ylabel("Number of risky FCs")
        plt.legend(loc="upper left")
        plt.show()

    def plot_fc_risk(self):
        # Risk of Fault Chains
        plt.plot(np.cumsum(self.rewards), label = "GRQN Cumulative Risk")
        plt.xlabel("Search Trials")
        plt.ylabel("Total Risk Discovered")
        plt.legend(loc="upper left")
        plt.show()

    def plot_fc_risk_time(self):
        # Risk of Fault Chains
        temp = [[cumRisk, self.rewards_time[count][1]] for count, cumRisk in enumerate(list(np.cumsum([row[0] for row in self.rewards_time])))]
        plt.plot([episode[1] for episode in temp], [episode[0] for episode in temp], label = "GRQN Cumulative Risk")
        plt.xlabel("Time Taken")
        plt.ylabel("Total Risk Discovered")
        plt.legend(loc="upper left")
        plt.show()

    def plot_epsilon_decay(self):
        # Epsilon Decay
        epsilon = [b for a, b, c in self.answer]
        plt.plot(epsilon, label = "GRQN Epsilon Decay")
        plt.xlabel("Search Trials")
        plt.ylabel("Epslion")
        plt.legend(loc="upper left")
        plt.show()



def load_config(config_file="39-bus Code/config.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


def main(config, args):
    # Load configuration from JSON file
    selectCase = args.case
    LOADING_FACTOR = args.load
    threshold = args.threshold
    iteration_track = args.if_iteration
    TOTAL_EPISODES = args.num_episodes
    time_taken = args.time_taken
    kappa = args.kappa
    

    # Set the working directory
    # print("Current working directory:", os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
    os.chdir(script_dir)                                    # Change the working directory to the script directory
    # print("New working directory:", os.getcwd())

    # Initialize the FaultChainSovler
    FCsovler = FaultChainSovlerClass(LOADING_FACTOR, selectCase) 

    # Configure NUM_ACTIONS and dataset based on selectCase and LOADING_FACTOR
    if selectCase == "39":
        NUM_ACTIONS = 46
        if LOADING_FACTOR == 0.6:
            dataSetString = "loading06_39bus.h5"
            seqBuffer = "39bus_06_OfflineSequentialBuffer250"
        elif LOADING_FACTOR == 0.55:
            dataSetString = "loading055_39bus.h5"
            seqBuffer = "39bus_055_OfflineSequentialBuffer250"
        buffer_path = os.path.join("39-bus Code", seqBuffer)

    elif selectCase == "118":
        NUM_ACTIONS = 50
        if LOADING_FACTOR == 0.6:
            dataSetString = "118bus_loading06.h5"
            seqBuffer = "118bus_06_OfflineSequentialBuffer250"
        elif LOADING_FACTOR == 1.0:
            dataSetString = "118bus_loading10.h5"
        buffer_path = os.path.join("118-bus Code", seqBuffer)

    # Load dataset and process
    dataset_path = os.path.join('Datasets', dataSetString)
    dataSet, threshold = FCsovler.load_dataset(dataset_path)
    FCsovler.print_datatset_information(dataSet, threshold)

    riskyFaultChainDict = FCsovler.find_risky_fault_chains(dataSet, threshold)

    # Initialize parameters for DRQN
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

    # Initialize DGRQN solver with loaded parameters
    ProposedAlgorithm = DGRQNSolverClass(
        TOTAL_EPISODES, NUM_ACTIONS, LOADING_FACTOR, selectCase, dataSetString, threshold, eps_end, GAMMA, eps_start, kappa, **training_params, **grqn_params
    )

    ProposedAlgorithm.riskyFaultChainDict = riskyFaultChainDict

    # Process the buffer (offline or online)
    if config["fill_Buffer"]:
        ProposedAlgorithm.fill_experience_buffer()
        with open(seqBuffer, "wb") as fp:
            pickle.dump((ProposedAlgorithm.replay_buffer.counter, ProposedAlgorithm.replay_buffer.storage), fp)
    else:
        with open(buffer_path, "rb") as fp:
            a, b = pickle.load(fp)

        ProposedAlgorithm.replay_buffer.counter = a
        ProposedAlgorithm.replay_buffer.storage = b

        # Train the model based on the iteration or time-based tracking
        if iteration_track:
            start = time.time()
            ProposedAlgorithm.train_grqn_already_experience()
            end = time.time()
            print("Time taken: ", end - start)
            ProposedAlgorithm.plot_loss()
            ProposedAlgorithm.plot_rewards()
            ProposedAlgorithm.plot_num_risky_fcs()
            ProposedAlgorithm.plot_fc_risk()
            ProposedAlgorithm.plot_epsilon_decay()
            print("GRQN: Found ", sum(ProposedAlgorithm.numRiskyFaultChains), " M%  Risky FCs in ", ProposedAlgorithm.M_episodes, " search trials")
            print("GRQN: The cumulative risk is ", sum(ProposedAlgorithm.rewards))
            print("GRQN: Number of ALL False conditions: ", ProposedAlgorithm.allFalseCount)

        else:
            start = time.time()
            ProposedAlgorithm.train_grqn_already_experience_time(time_taken)
            end = time.time()
            print("Total Run Time: ", end - start)
            ProposedAlgorithm.plot_rewards_time()
            ProposedAlgorithm.plot_fc_risk_time()
            ProposedAlgorithm.plot_num_risky_fcs_time()
            print("GRQN: Found ", ProposedAlgorithm.how_many_risky_fcs_time(), " Risky FCs in ", ProposedAlgorithm.i_episode, " search trials")
            print("GRQN: The cumulative risk is ", sum([episode[0] for episode in ProposedAlgorithm.rewards_time]))
            print("GRQN: Number of ALL False conditions: ", ProposedAlgorithm.allFalseCount)

            """
            with open("priorGRQN_for_06", "rb") as myFile:
            # actionSpace to state mapping which is of length self.NUM_STATES!
            stateDict, hidden = pickle.load(myFile)
            print(stateDict, hidden)

            # fill buffer + training together!
            ProposedAlgorithm.train_grqn()
            ProposedAlgorithm.plot_rewards()
            ProposedAlgorithm.plot_num_risky_fcs()
            ProposedAlgorithm.plot_fc_risk()
            ProposedAlgorithm.plot_epsilon_decay()
            print("GRQN: Found ",  ProposedAlgorithm.how_many_risky_fcs(), " Risky FCs in ", ProposedAlgorithm.M_episodes - ProposedAlgorithm.EXPLORE, " search trials")
            print("GRQN: The cumulative risk is ", sum(ProposedAlgorithm.rewards[ProposedAlgorithm.EXPLORE:]))
            print("GRQN: Number of ALL False conditions: ", ProposedAlgorithm.allFalseCount)
            """


if __name__ == '__main__':

    # python grqnSolver.py --case 39 --load 0.55 --threshold 5 --kappa 1 --if_iteration 1 --num_episodes 50
    # python grqnSolver.py --case 39 --load 0.55 --threshold 5 --kappa 1 --if_iteration 0 --time_taken 60

    # Parsing mandatory arguments
    parser = argparse.ArgumentParser(description="Run the GRNN Fault Chain Solver.")
    parser.add_argument('--case', type=str, choices=['39', '118'], required=True, help="Select the case: '39', or '118'.")
    parser.add_argument('--load', type=float, choices=[0.55, 0.6, 1.0], required=True, help="Loading factor: 0.55, 0.6 for 39-bus and 0.6, 1.0 for 118-bus.")
    parser.add_argument('--threshold', type=int, required=True, help="Risky FC threshold in percentage.")
    parser.add_argument('--kappa', type=int, required=True, help="number of bellman updates per run")
    parser.add_argument('--if_iteration', type=int, choices=[0, 1], required=True, help="Fixed iterations (1) or fixed time (0).")
    
    parser.add_argument('--num_episodes', type=int, default=50, help="Number of episodes for Q-learning. Only relevant when iterations = 1.")
    parser.add_argument('--time_taken', type=int, default=60, help="Time duration for fixed time execution when iterations are 0.")

    # Reading the command line arguments
    args = parser.parse_args()

    # Load the rest of the configuration from the JSON file
    config = load_config()  # Load configuration from the JSON file
    main(config, args)