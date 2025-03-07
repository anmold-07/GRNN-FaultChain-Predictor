import os
import numpy as np

np.set_printoptions(precision=2)

from lib.pypower.api import *
from scipy import sparse
import copy
import h5py

# new additions
import argparse

class FaultChainSovlerClass(object):
    def __init__(self, LOADING_FACTOR, SELECT_CASE):

        self.LOADING_FACTOR = LOADING_FACTOR
        self.selectCase = SELECT_CASE
        
        if self.selectCase == "118":
            # select a subset of lines as action space due to computational constraints for ground-truth generation
            self.sortedIndicesHeavyList = [2, 3, 4, 6, 7, 8, 9, 10, 20, 22, 30, 31, 32, 35, 36, 37, 40, 49, 50, 51, 52, 53, 65, 69, 86, 89, 90, 92, 93, 94, 97, 98, 99, 103, 111, 113, 118, 119, 120, 121, 123, 129, 130, 131, 132, 134, 155, 159, 169, 175]
            self.sortedIndicesHeavyArray = np.array(self.sortedIndicesHeavyList)
            
            self.lineIndextoNewIndexConverter = dict()
        
            for newIndex, lineIndex in enumerate(self.sortedIndicesHeavyList):
                self.lineIndextoNewIndexConverter[lineIndex] = newIndex

    def environmentStep(self, actionSpace):
        """
        Executes a step in the fault-chain environment based on the action indices in the `actionSpace`.

        This function simulates power system behavior, updates network states, and 
        determines whether the simulation has reached its terminal condition. 
        Be cautious of off-by-one errors when comparing results with MATPOWER.

        Parameters:
        ----------
        actionSpace : object
            The set of possible transmission line indices that need to be disconnected.

        Returns:
        -------
        tuple
            A tuple containing the following elements:

            - np.ndarray: Adjacency matrix (N, N) representing network connectivity after all actions are taken.
            - np.ndarray: Voltage angles (shape: (N,)), filled with measurements at each bus.
            - float: Load shed in the final stage (final stage determined by the `actionSpace` array).
            - bool: Whether the simulation horizon (M=3 for all experiments) has been reached.
            - float: Total cumulative load shed so far.
            - np.ndarray: various properties of the **remaining** line indices, structured as:
                - indices:    transmission line indices remaining in the network (after all the actions in `actionSpace` are taken.
                - capacity:   respective line capcities
                - power flow: respective power-flow in each line index
        """

        if self.selectCase == "14":
            ppc = case14()
        elif self.selectCase == "30":
            ppc = case30()
        elif self.selectCase == "39":
            ppc = case39()   # looking at all possible components (transmission lines)
        elif self.selectCase == "118":
            ppc = case118()  # looking at only the 50 most heavily loaded components only

        #print("Simulating Case: " + self.selectCase)
        # print(ppc.keys())

        # Checking the sizes before pruning the testcases (confirmation with MATLAB (1-13));
        branches = ppc["branch"]
        #print("branch: ", branches.shape)
        buses = ppc["bus"]
        #print("bus: ", buses.shape)
        generation = ppc["gen"]
        #print("generation: ", generation.shape)
        #print(branches[0,:])
        #print(generation[0,:])
        #print(buses)

        # Making sure that the dimensions are as expected (as in MATLAB);
        # 0:13 in Python equvivalent to 1:13 in MATLAB! 
        ppc["branch"] = ppc["branch"][:, 0:13]
        ppc["bus"] = ppc["bus"][:, 0:13]
        ppc["gen"] = ppc["gen"][:, 0:21]
        #print(ppc["bus"])

        # Varying the Loading Condition;
        ppc["bus"][:,2] = self.LOADING_FACTOR*ppc["bus"][:,2] # real power demand P
        ppc["bus"][:,3] = self.LOADING_FACTOR*ppc["bus"][:,3] # reactive power demand Q 
        ppc["gen"][:,1] = self.LOADING_FACTOR*ppc["gen"][:,1] # real power output P_gen
        ppc["gen"][:,2] = self.LOADING_FACTOR*ppc["gen"][:,2] # reactive power output Q_gen
        #print(ppc["bus"])

        # Relabeling (for some tests cases the indices do not start from 1) 
        # the Bus Numbers in Increasing Order (Data Pre-Processing);
        bus = ppc["bus"]
        nodeSelectIndex = np.zeros((bus.shape[0], 2))
        for index in range(bus.shape[0]):
            nodeSelectIndex[index, 0] = index + 1      # new bus index 
            nodeSelectIndex[index, 1] = bus[index, 0]  # old bus index 
            bus[index, 0] = index + 1

        bus = bus[bus[:,0].argsort()]
        ppc["bus"] = bus
        #print(ppc["bus"])
        #print(nodeSelectIndex)
        #print(nodeSelectIndex.shape)

        # Accordingly, relabeling the generator buses (Data Pre-Processing);
        gen = ppc["gen"]
        for index in range(gen.shape[0]):
            gen[index, 0] = nodeSelectIndex[np.argwhere(gen[index,0]==nodeSelectIndex[:,1]), 0]
        gen = gen[gen[:,0].argsort()] 
        ppc["gen"] = gen
        #print(ppc["gen"])

        # Since branches are connected to nodes, the "from" and "to" attributes should be relabeled (Data Pre-Processing);
        branch_tmp = ppc["branch"]
        for index in range(branch_tmp.shape[0]):
            # from attribute
            branch_tmp[index, 0] = nodeSelectIndex[np.argwhere(branch_tmp[index,0]==nodeSelectIndex[:,1]), 0]
            # to attribute
            branch_tmp[index, 1] = nodeSelectIndex[np.argwhere(branch_tmp[index,1]==nodeSelectIndex[:,1]), 0]
        ppc["branch"] = branch_tmp
        #print(ppc["branch"])

        # In the case of parallel lines, add their rates/capacities/long-term ratings (Data Pre-Processing);
        # print(ppc["branch"].shape
        # for each branch
        for i in range(ppc["branch"].shape[0]):

            # go through all the other branches
            for j in range(i+1, ppc["branch"].shape[0]):

                if (ppc["branch"][i,0]==ppc["branch"][j,0] and ppc["branch"][i,1]==ppc["branch"][j,1]) or (ppc["branch"][i,0]==ppc["branch"][j,1] and ppc["branch"][i,1]==ppc["branch"][j,0]):

                    ppc["branch"][i, 5] = ppc["branch"][i, 5] + ppc["branch"][j, 5]
                    ppc["branch"][j, :] = 0

        ppc["branch"] = np.delete(ppc["branch"], np.argwhere(ppc["branch"][:, 0]==0), axis=0)
        branch = ppc["branch"]
        reactanceVector = branch[:,3]

        # Ideally, I'd want that GENERATION = DEMAND; So this ensures that the NET power injection sums to 0; 
        # In some test cases, this might not be. So, the following code ensures that the demand and supply remain balanced!
        # this is done by modifying the load in order to match generation!
        powerInj_consume = bus[:,2]                              # real-power consumption vector
        powerInj_supply = np.zeros((powerInj_consume.shape[0]))  # real-power generation vector

        for index in range(ppc["gen"].shape[0]):
            powerInj_supply[np.argwhere(bus[:,0]==gen[index,0])] = powerInj_supply[np.argwhere(bus[:,0]==gen[index,0])] + gen[index, 1]

        # real NET-power injection vector    
        powerInj_vector = powerInj_supply - powerInj_consume 
        #print(np.sum(powerInj_vector))

        # Increading/Decreasing the loads or consumption vector (load data)!
        powerInj_consume = powerInj_consume + np.sum(powerInj_vector)/powerInj_vector.shape[0]

        # NEW real NET-power injection vector  
        powerInj_vector = powerInj_supply - powerInj_consume
        # print(np.sum(powerInj_vector))

        # Print the current generation (and thus, same load) in the network!
        currentLoad = np.sum(ppc["bus"][:,2])
        threshold = 0.01*currentLoad
        #print(currentLoad, threshold)

        # Building the Adjacency, Capacity and Reactance Matrix;
        Adj = np.zeros((bus.shape[0], bus.shape[0]))
        Reactance = np.zeros((bus.shape[0], bus.shape[0]))
        Capacity = np.zeros((bus.shape[0], bus.shape[0]))

        for index in range(branch.shape[0]):
            Adj[ int(branch[index, 0])-1, int(branch[index, 1])-1 ] = 1
            Adj[ int(branch[index, 1])-1, int(branch[index, 0])-1 ] = 1

            Reactance[ int(branch[index, 0])-1, int(branch[index, 1])-1 ] = branch[index, 3]
            Reactance[ int(branch[index, 1])-1, int(branch[index, 0])-1 ] = branch[index, 3]

            if branch[index, 5] > 0:
                Capacity[ int(branch[index,0])-1, int(branch[index,1])-1 ] = branch[index, 5]
                Capacity[ int(branch[index,1])-1, int(branch[index,0])-1 ] = branch[index, 5]
            else:
                Capacity[ int(branch[index,0])-1, int(branch[index,1])-1 ] = 9900
                Capacity[ int(branch[index,1])-1, int(branch[index,0])-1 ] = 9900

        # re-assigning the capacity vector to take into account the new capacity
        if ppc["branch"].shape[0] > 150: # meaning the 118 bus system
            Capacity_vector = np.load('118bus_capacity_vector.npy')
            #print(Capacity_vector)  
            #print( np.absolute( result[0]["branch"][:, 13] )/Capacity_vector )
            #print( "average : ", np.average( np.absolute( result[0]["branch"][:, 13] )/Capacity_vector ) ) 
            #print( "median : ",  np.median( np.absolute( result[0]["branch"][:, 13] )/Capacity_vector ) )   
        else:
            # 39-bus system
            Capacity_vector = branch[:,5]
            #print(Capacity_vector)
            Capacity_vector = np.expand_dims(Capacity_vector, axis=1)
            count_of_zero_capacity_branches = 0
            for index in range(Capacity_vector.shape[0]):
                if Capacity_vector[index, 0] == 0:
                    count_of_zero_capacity_branches += 1
                    Capacity_vector[index, 0] = 9900

        #print("count_of_zero_capacity_branches: ", count_of_zero_capacity_branches)        
        for index in range(branch.shape[0]):
            branch[index,5] = Capacity_vector[index, 0]
        ppc["branch"] = branch

        def conncomp(link_row, link_column):
            """
            A function that runs a DFS on a graph constructed from branches end indices.
            link_row : From indices 
            link_column : To indices
            """
            def graph_construct(link_row, link_column):
                graph = {node: [] for node in range(1, int(ppc["bus"].shape[0])+1)}
                for a, b in zip(list(link_row), list(link_column)):
                    a = a.item()
                    b = b.item()
                    graph[a].append(b)
                    graph[b].append(a)
                return graph

            def explore(graph, current, visited, count, C):
                if current in visited:
                    return False 

                visited.add(current)
                C[current-1] = count+1
                for neighbor in graph[current]:
                    explore(graph, neighbor, visited, count, C)

                return True    

            graph = graph_construct(link_row, link_column)
            C = np.zeros((int(ppc["bus"].shape[0]), 1))
            visited = set()
            count = 0

            for node in graph:
                if explore(graph, node, visited, count, C) == True:
                    count += 1
            return count, C
    
        def b_Pypower_DC_OPA_slack(voltageAngles, ppc, adjacency, capacity, capacity_vector, alpha, initFail, tmp_index, link_flag_vector = np.ones((int(ppc["branch"].shape[0]), 1))):
            """
            voltageAngles (numpy array)    : (modify, Nx1) array should be filled-up with measurements at each bus
            ppc (dict)                     : (modify)      testcase prior to starting the removal process
            adjacency (sparse matrix)      : (modify, NxN) 
            capacity (numpy array)         : (NxN) maximum capacity of each branch as a 2-d array 
            capacity_vector (numpy array)  : (Lx1) maximum capacity of each branch as 1-d array
            alpha (int)                    : 1 
            initFail ()                    : this specifies the buses between which the failure takes place
            tmp_index ()                   : these are line indices that are to be removed 0 -- (L-1) not 1 -- L !! determined by the "actionSpace" vector
            link_flag_vector (numpy array) : (modify, Lx1) To keep track of all the line outages that occur during the course of the simualtion run
            """
            # as of now, adjacency is of type "csr.matrix"!
            
            # --> nominal system ka powerflow and construction of a (N x N) matrix "tempur" storing abs value of PFs of all branches.
            result = rundcpf_my(ppc)

            temp = ppc["branch"][:, 13]                  # powerFlowVector
            temp = np.expand_dims(temp, axis = 1)
            result[0]["branch"] = np.concatenate([result[0]["branch"], temp], axis = 1)

            start_gen = np.sum(result[0]["gen"][:, 1])   # how much generation in the system to start with!
            start_load = np.sum(result[0]["bus"][:, 2])  #  ''   ''       load in the system to start with!

            tempur_vector = abs(result[0]["branch"][:, 13])                            #  (L, )   absoluteVersionofPowerFlowVector   (original ppc)
            tempur = np.zeros((result[0]["bus"].shape[0], result[0]["bus"].shape[0]))  #  (N x N) matrix of the absolute power-flows (original ppc)

            for index in range(tempur_vector.shape[0]):
                tempur[ int(result[0]["branch"][index, 0])-1, int(result[0]["branch"][index, 1])-1 ] = tempur_vector[index]  # (original ppc)
                tempur[ int(result[0]["branch"][index, 1])-1, int(result[0]["branch"][index, 0])-1 ] = tempur_vector[index]  # (original ppc)

                
            # --> makes changes in the "adjacency" matrix to reflect outages in init_fail()   
            for index in range(initFail.shape[0], 1):
                adjacency[ int(initFail[index, 0])-1, int(initFail[index, 1])-1] = 0
                adjacency[ int(initFail[index, 1])-1, int(initFail[index, 0])-1] = 0

            # print(ppc["branch"].shape)                                     # (L x 14)
            # link_flag_vector = np.ones((result[0]["branch"].shape[0], 1))  # (L x  1)  # check again
            link_flag_vector[tmp_index.item(), 0] = 0                                    # altering the link_flag_vector vector to reflect the changes in to be removed lines

            for t in range(tmp_index.shape[0]): # tmp_index always has only one element
                if np.argwhere(ppc["branch"][:,13] == tmp_index[t]).shape[0]==0:
                    # action cannot be executed since already taken!
                    #print("Branch already removed!")
                    gen_shed = start_gen - np.sum(result[0]["gen"][:, 1])
                    load_shed = start_load - np.sum(result[0]["bus"][:, 2])
                    link_row = ppc["branch"][:, 0]                      # branch FROM
                    link_row = np.expand_dims(link_row, axis=1)
                    link_column = ppc["branch"][:, 1]                   # branch TO
                    link_column = np.expand_dims(link_column, axis=1)

                    link_row = link_row.astype(int)
                    link_column = link_column.astype(int)
                    S, C = conncomp(link_row, link_column)    
                    powerFlowVector = result[0]["branch"][:, 13]

                    return (powerFlowVector, ppc, adjacency, S, gen_shed, load_shed, link_flag_vector)
                else:  
                    # delete the row associated with the line index to be removed
                    ppc["branch"] = np.delete(ppc["branch"], (np.argwhere(ppc["branch"][:,13] == tmp_index[t])), axis=0)
            
            # print(result[0]["branch"].shape[0])
            # print(np.sum(link_flag_vector))
            # print(ppc["branch"].shape[0])

            # --> solve power flow for the "new ppc" which is a consequence of deleting the the rows! 
            link_row = ppc["branch"][:, 0]                      # branch FROM
            link_row = np.expand_dims(link_row, axis=1)

            link_column = ppc["branch"][:, 1]                   # branch TO
            link_column = np.expand_dims(link_column, axis=1)

            # BFS traversal for connected components!
            # print(link_row, link_column)
            link_row = link_row.astype(int)
            link_column = link_column.astype(int)
            S, C = conncomp(link_row, link_column)

            # Load Shedding
            if S > 1:
                result_tmp = copy.deepcopy(result)
                result_tmp[0]["bus"] = np.array([])
                result_tmp[0]["gen"] = np.array([])
                result_tmp[0]["branch"] = np.array([])

                for index in range(1, S+1):
                    #print("----------just check----------")
                    #print(index)
                    #print("----------just check----------") 

                    CC_Index = np.argwhere(C == index)      # (-, 2) 2-D indices of numpy array satisfying given condition
                    #print(CC_Index[:, 0])
                    CC_Load = ppc["bus"][CC_Index[:, 0], 2] # (-,  )
                    CC_Gen = np.array([])
                    CC_Gen_Index = np.array([])             # (-,  )


                    # finding generators and generators indices in this island!
                    for j in range(ppc["gen"][:, 0].shape[0]):
                        if np.argwhere(CC_Index[:, 0] == ppc["gen"][j, 0]).shape[0]==1: 
                            CC_Gen = np.append(CC_Gen, ppc["gen"][j, 1])
                            CC_Gen_Index = np.append(CC_Gen_Index, j)

                    CC_Gen_Index = CC_Gen_Index.astype(int)


                    # controlled load-generation balance in each island!
                    if np.sum(CC_Load) < np.sum(CC_Gen):
                        if np.sum(CC_Load) > 0:
                            CC_Gen = CC_Gen*(np.sum(CC_Load)/np.sum(CC_Gen))
                            # CC_Gen = np.reshape(CC_Gen, (-1, 1))
                            ppc["gen"][CC_Gen_Index, 1] = CC_Gen
                        else:                
                            ppc["gen"][CC_Gen_Index, 1] = 0
                            ppc["bus"][CC_Index[:, 0], 2] = 0
                    elif np.sum(CC_Load) > np.sum(CC_Gen):
                        islandCapacity = np.sum(ppc["gen"][CC_Gen_Index, 8])
                        if np.sum(CC_Gen)==0:
                            ppc["gen"][CC_Gen_Index, 1] = 0
                            ppc["bus"][CC_Index[:, 0], 2] = 0
                        elif islandCapacity >= np.sum(CC_Load):
                            CC_Gen = CC_Gen + ppc["gen"][CC_Gen_Index, 8]*(np.sum(CC_Load) - np.sum(CC_Gen))/islandCapacity
                            #CC_Gen = np.reshape(CC_Gen, (-1, 1))
                            ppc["gen"][CC_Gen_Index, 1] = CC_Gen
                        else:
                            CC_Gen = ppc["gen"][CC_Gen_Index, 8]
                            #CC_Gen = np.reshape(CC_Gen, (-1, 1))
                            ppc["gen"][CC_Gen_Index, 1] = CC_Gen

                            CC_Load = CC_Load*(np.sum(CC_Gen)/np.sum(CC_Load))
                            #CC_Load = np.reshape(CC_Load, (-1, 1))
                            ppc["bus"][CC_Index[:, 0], 2] = CC_Load

                    # print(np.sum(CC_Load), np.sum(CC_Gen))

                    ppc_island = copy.deepcopy(ppc)
                    ppc_island["bus"] = ppc["bus"][CC_Index[:, 0], :]
                    ppc_island["gen"] = ppc["gen"][CC_Gen_Index, :]
                    ppc_island["gencost"] = ppc["gencost"][CC_Gen_Index, :]


                    # assigning slack bus to the island if original not present to successfully solve power-flow! 
                    if np.argwhere(ppc_island["bus"][:, 1]==3).shape[0]==0 and ppc_island["bus"][:, 1].shape[0]>1:
                        island_slack_set = np.argwhere(ppc_island["bus"][:, 2] == min(ppc_island["bus"][:, 2]))
                        island_slack = island_slack_set[0].item()
                        ppc_island["bus"][island_slack, 1] = 3

                    # find all branches that belong to this island!
                    CC_Branch_Index = np.array([])
                    for t in range(ppc["branch"].shape[0]):
                        if np.argwhere(ppc_island["bus"][:, 0]==ppc["branch"][t, 0]).shape[0]>0 and np.argwhere(ppc_island["bus"][:, 0]==ppc["branch"][t, 1]).shape[0]>0: 
                            CC_Branch_Index = np.append(CC_Branch_Index, t)

                    CC_Branch_Index = CC_Branch_Index.astype(int)
                    ppc_island["branch"] = ppc["branch"][CC_Branch_Index, :]


                    if ppc_island["bus"].shape[0]!=0 and ppc_island["gen"].shape[0]!=0 and ppc_island["branch"].shape[0]!=0:
                        # if this is a meaningful island!
                        #print("Useful Island")
                        result_island = rundcpf_my(ppc_island)

                        # print(result_island[0]["bus"][:, 7]) # voltage mag, useless in DC power-flow!
                        # print(result_island[0]["bus"][:, 8])   # voltage angles!
                        voltageAngles[CC_Index[:, 0]] = result_island[0]["bus"][:, 8]

                        temp = ppc_island["branch"][:, 13]
                        temp = np.expand_dims(temp, axis = 1)
                        result_island[0]["branch"] = np.concatenate([result_island[0]["branch"], temp], axis = 1)

                        # print(result_island[0]["bus"].shape)  

                        result_tmp[0]["bus"] = np.vstack([result_tmp[0]["bus"], result_island[0]["bus"]]) if result_tmp[0]["bus"].size else result_island[0]["bus"]
                        result_tmp[0]["gen"] = np.vstack([result_tmp[0]["gen"], result_island[0]["gen"]]) if result_tmp[0]["gen"].size else result_island[0]["gen"]                                                  
                        result_tmp[0]["branch"] = np.vstack([result_tmp[0]["branch"], result_island[0]["branch"]]) if result_tmp[0]["branch"].size else result_island[0]["branch"]
                    else:
                        # if this is NOT a meaningful island!
                        #print("Not a useful Island")
                        ppc_island["gen"][:, 1] = 0
                        ppc_island["bus"][:, 2] = 0

                        voltageAngles[CC_Index[:, 0]] = np.array([0]*len(CC_Index[:, 0]))

                        result_tmp[0]["bus"] = np.vstack([result_tmp[0]["bus"], ppc_island["bus"]]) if result_tmp[0]["bus"].size else ppc_island["bus"]
                        result_tmp[0]["gen"] = np.vstack([result_tmp[0]["gen"], ppc_island["gen"]]) if result_tmp[0]["gen"].size else ppc_island["gen"]  

                        temp1 = ppc_island["branch"][:, 0:ppc_island["branch"].shape[1]-1]
                        # temp1 = np.reshape(temp1, (-1, 1))
                        temp2 = np.zeros((ppc_island["branch"].shape[0], 4))
                        # temp2 = np.reshape(temp2, (-1, 1))
                        temp3 = ppc_island["branch"][:, ppc_island["branch"].shape[1]-1]
                        temp3 = np.reshape(temp3, (-1, 1))
                        #print(temp1, temp1.shape, temp2, temp2.shape, temp3, temp3.shape)
                        temp = np.concatenate((temp1, temp2, temp3), axis = 1)
                        #print(temp, temp.shape)
                        result_tmp[0]["branch"] = np.vstack([result_tmp[0]["branch"], temp]) if result_tmp[0]["branch"].size else temp
                        #print(result_tmp[0]["branch"].shape)  

                result[0]["bus"] = result_tmp[0]["bus"][result_tmp[0]["bus"][:, 0].argsort()]
                result[0]["gen"] = result_tmp[0]["gen"][result_tmp[0]["gen"][:, 0].argsort()]
                result[0]["branch"] = result_tmp[0]["branch"][result_tmp[0]["branch"][:, 17].argsort()]                                     
            else:
                result = rundcpf_my(ppc) 
                voltageAngles[:] = result[0]["bus"][:, 8] # no matter what, result[0]["bus"][:, 8] is always of the same size and so, this assignment is fine!

                temp = ppc["branch"][:, 13] # indices
                temp = np.expand_dims(temp, axis = 1)
                result[0]["branch"] = np.concatenate([result[0]["branch"], temp], axis = 1)
                result[0]["branch"] = result[0]["branch"][result[0]["branch"][:, 17].argsort()]

            #print("Completed performing power-flows for each island!")
            # print(result[0]["bus"][:, [0, 1, 2, 3]])
            # print(result[0]["branch"][:, [0, 1, 13]])

            for index in range(result[0]["branch"].shape[0]): 
                if np.isnan(result[0]["branch"][index, 13]):
                    result[0]["branch"][index, 13] = 0
                    result[0]["branch"][index, 15] = 0

            # print(tempur.shape)     
            tempur = np.multiply(tempur, adjacency)  # updated (N x N) matrix of the absolute power-flows (new "modified" ppc) but this is no more needed since will have to develop a new one based on updated powerlow.

            Node_index = np.zeros((adjacency.shape[0], 1))
            for index in range(adjacency.shape[0]):
                Node_index[index, 0] = index
            
            # print(result[0]["branch"].shape[0])
            # print(np.sum(link_flag_vector))
            # print(ppc["branch"].shape[0])
            
            ####################################################################################################
            # --> Checking Overloads
            ####################################################################################################

            #print("Checking for overloads................................")
            flow_vector = np.absolute(result[0]["branch"][:, 13])           # the length of the vector depends on how many lines are there in the current ppc

            powerFlow = np.zeros((ppc["bus"].shape[0], ppc["bus"].shape[0]))
            for index in range(flow_vector.shape[0]):
                powerFlow[ int(result[0]["branch"][index, 0])-1, int(result[0]["branch"][index, 1])-1 ] = flow_vector[index]
                powerFlow[ int(result[0]["branch"][index, 1])-1, int(result[0]["branch"][index, 0])-1 ] = flow_vector[index]        

            # tempur = alpha*np.absolute( powerFlow ) + (1 - alpha)*np.absolute( tempur )
            tempur = alpha*np.absolute( powerFlow )     # updated (N x N) matrix of the absolute power-flows (new "modified" ppc)

            # print((np.absolute( powerFlow ) > capacity).shape)
            # print(np.multiply(np.absolute( powerFlow ) > capacity, sparse.csr_matrix.todense(adjacency)))

            overload = np.max( np.multiply( tempur > capacity, sparse.csr_matrix.todense(adjacency) ) )
            # print(overload)
            
            # basically remove overloads and modify the adjacency matrix!
            adjacency = np.multiply( tempur <= capacity, sparse.csr_matrix.todense(adjacency) )  # update the adjacency matrix to remove overloads
    
            # use this to check for overloads and remove the appropriate lines from ppc!
            relevant_indices = np.argwhere(link_flag_vector == 1)  # original line indices where overloads could potentially occur
            # print(capacity_vector[relevant_indices[:,0]])
            # print(flow_vector.shape, capacity_vector.shape, capacity_vector[relevant_indices[:,0]].shape, np.squeeze(capacity_vector[relevant_indices[:,0]]).shape)

            condition = flow_vector > np.squeeze( capacity_vector[ relevant_indices[:,0] ] ) # coz of the argsort feature applied previously, LHS RHS comparison makes sence.
            # print(condition)

            fail_index = np.argwhere( condition == True )
            fail_index = fail_index.astype(int)           # mind you, the indices obtained from the 'argwhere' function are not the REAL overloaded line indices. They act as input to the 13th column of PPC to figure out which line is ACTUALLY in outage.

            if fail_index.shape[0]==0:
                #print("No overloads! The system is normal after removing the intended lines.") 
                gen_shed = start_gen - np.sum(result[0]["gen"][:, 1])
                load_shed = start_load - np.sum(result[0]["bus"][:, 2])
                
            else:
                #print("Overload detected! Starting the fast process............")
                overload = 1
                while overload:
                    flow_vector = np.absolute(result[0]["branch"][:, 13])

                    powerFlow = np.zeros((ppc["bus"].shape[0], ppc["bus"].shape[0]))
                    for index in range(flow_vector.shape[0]):
                        powerFlow[ int(result[0]["branch"][index, 0])-1, int(result[0]["branch"][index, 1])-1 ] = flow_vector[index]
                        powerFlow[ int(result[0]["branch"][index, 1])-1, int(result[0]["branch"][index, 0])-1 ] = flow_vector[index]        

                    # tempur = alpha*np.absolute( powerFlow ) + (1 - alpha)*np.absolute( tempur )
                    tempur = alpha*np.absolute( powerFlow )

                    # print((np.absolute( powerFlow ) > capacity).shape)
                    # print(np.multiply(np.absolute( powerFlow ) > capacity, sparse.csr_matrix.todense(adjacency)))
                    overload = np.max(np.multiply(tempur > capacity, adjacency))
                    
                    # print(overload)
                    adjacency = np.multiply(tempur <= capacity, adjacency)
                    relevant_indices = np.argwhere(link_flag_vector == 1)

                    # print(capacity_vector[relevant_indices[:,0]])
                    # print(flow_vector.shape, capacity_vector.shape, capacity_vector[relevant_indices[:,0]].shape, np.squeeze(capacity_vector[relevant_indices[:,0]]).shape)

                    condition = flow_vector > np.squeeze(capacity_vector[relevant_indices[:,0]])
                    # print(condition)

                    fail_index = np.argwhere( condition == True )
                    fail_index = fail_index.astype(int)

                    # find the true index associated with the failed transmission line!
                    tmp_index = ppc["branch"][fail_index, 13]
                    tmp_index = tmp_index.astype(int)         # now these represent the original indices to be removed.

                    # all_outages = np.append(all_outages, tmp_index)
                    # print("------")
                    # print(fail_index.shape)
                    # print(fail_index)
                    # print(tmp_index.shape)
                    # print(tmp_index)
                    # print("------")

                    if tmp_index.shape[0]>0:
                        link_flag_vector[tmp_index, 0] = 0

                    for t in range(tmp_index.shape[0]):
                        ppc["branch"] = np.delete(ppc["branch"], (np.argwhere(ppc["branch"][:, 13] == tmp_index[t])), axis = 0)

                    link_row = ppc["branch"][:, 0]                      # branch FROM
                    link_row = np.expand_dims(link_row, axis=1)
                    link_column = ppc["branch"][:, 1]                   # branch TO
                    link_column = np.expand_dims(link_column, axis=1)

                    # BFS traversal for connected components! Python not detecting single node islands! 
                    # Temp fix in the lower half!
                    ################################################################################################################
                    # print(link_row, link_column)
                    link_row = link_row.astype(int)
                    link_column = link_column.astype(int)
                    S, C = conncomp(link_row, link_column)
                    ################################################################################################################

                    # Load Shedding
                    if S > 1:
                        result_tmp = copy.deepcopy(result)
                        result_tmp[0]["bus"] = np.array([])
                        result_tmp[0]["gen"] = np.array([])
                        result_tmp[0]["branch"] = np.array([])

                        for index in range(1, S+1):
                            CC_Index = np.argwhere(C == index)      # (-, 2) 2-D indices of numpy array satisfying given condition
                            #print(CC_Index[:, 0])
                            CC_Load = ppc["bus"][CC_Index[:, 0], 2] # (-,  )
                            CC_Gen = np.array([])
                            CC_Gen_Index = np.array([])             # (-,  )

                            # finding generators and generators indices in this island!
                            for j in range(ppc["gen"][:, 0].shape[0]):
                                if np.argwhere(CC_Index[:, 0] == ppc["gen"][j, 0]).shape[0]==1: 
                                    CC_Gen = np.append(CC_Gen, ppc["gen"][j, 1])
                                    CC_Gen_Index =  np.append(CC_Gen_Index, j)

                            CC_Gen_Index = CC_Gen_Index.astype(int)


                            """
                            # uncontrolled load-generation balance in each island via shedding!
                            if np.sum(CC_Load) > np.sum(CC_Gen):
                                if np.sum(CC_Gen) > 0:
                                    CC_Load = CC_Load*(np.sum(CC_Gen)/np.sum(CC_Load))
                                    #CC_Load = np.reshape(CC_Load, (-1, 1))
                                    ppc["bus"][CC_Index[:, 0], 2] = CC_Load
                                else:    
                                    ppc["gen"][CC_Gen_Index, 1] = 0
                                    ppc["bus"][CC_Index[:, 0], 2] = 0
                            else:
                                if np.sum(CC_Load) < np.sum(CC_Gen):
                                    if np.sum(CC_Load)>0:
                                        CC_Gen = CC_Gen*(np.sum(CC_Load)/np.sum(CC_Gen))
                                        # print(ppc["gen"][CC_Gen_Index, 1].shape, CC_Gen.shape)
                                        # CC_Gen = np.reshape(CC_Gen, (-1, 1))
                                        # print(ppc["gen"][CC_Gen_Index, 1].shape, CC_Gen.shape)
                                        ppc["gen"][CC_Gen_Index, 1] = CC_Gen
                                    else:
                                        ppc["gen"][CC_Gen_Index, 1] = 0
                                        ppc["bus"][CC_Index[:, 0], 2] = 0
                            """    
                            
                            # controlled load-generation balance in each island!
                            if np.sum(CC_Load) < np.sum(CC_Gen):
                                if np.sum(CC_Load) > 0:
                                    CC_Gen = CC_Gen*(np.sum(CC_Load)/np.sum(CC_Gen))
                                    #CC_Gen = np.reshape(CC_Gen, (-1, 1))
                                    ppc["gen"][CC_Gen_Index, 1] = CC_Gen
                                else:                
                                    ppc["gen"][CC_Gen_Index, 1] = 0
                                    ppc["bus"][CC_Index[:, 0], 2] = 0
                            elif np.sum(CC_Load) > np.sum(CC_Gen):
                                islandCapacity = np.sum(ppc["gen"][CC_Gen_Index, 8])
                                if np.sum(CC_Gen)==0:
                                    ppc["gen"][CC_Gen_Index, 1] = 0
                                    ppc["bus"][CC_Index[:, 0], 2] = 0
                                elif islandCapacity >= np.sum(CC_Load):
                                    CC_Gen = CC_Gen + ppc["gen"][CC_Gen_Index, 8]*(np.sum(CC_Load) - np.sum(CC_Gen))/islandCapacity
                                    #CC_Gen = np.reshape(CC_Gen, (-1, 1))
                                    ppc["gen"][CC_Gen_Index, 1] = CC_Gen
                                else:
                                    #print(np.sum(CC_Load), np.sum(CC_Gen), islandCapacity)
                                    CC_Gen = ppc["gen"][CC_Gen_Index, 8]
                                    #CC_Gen = np.reshape(CC_Gen, (-1, 1))
                                    ppc["gen"][CC_Gen_Index, 1] = CC_Gen

                                    CC_Load = CC_Load*(np.sum(CC_Gen)/np.sum(CC_Load))
                                    #print(np.sum(CC_Load)) 
                                    #CC_Load = np.reshape(CC_Load, (-1, 1))
                                    ppc["bus"][CC_Index[:, 0], 2] = CC_Load
                                    #print(np.sum(ppc["bus"][CC_Index[:, 0], 2]))

                            # print(np.sum(CC_Load), np.sum(CC_Gen))


                            ppc_island = copy.deepcopy(ppc)
                            ppc_island["bus"] = ppc["bus"][CC_Index[:, 0], :]
                            ppc_island["gen"] = ppc["gen"][CC_Gen_Index, :]
                            ppc_island["gencost"] = ppc["gencost"][CC_Gen_Index, :]

                            # print(np.sum(ppc_island["bus"][:, 2]), np.sum(ppc_island["gen"][:, 1]))

                            # assigning slack bus to the island if original not present to successfully solve power-flow! 

                            if np.argwhere(ppc_island["bus"][:, 1]==3).shape[0]==0 and ppc_island["bus"][:, 1].shape[0]>1:
                                island_slack_set = np.argwhere(ppc_island["bus"][:, 2] == min(ppc_island["bus"][:, 2]))  
                                #print(island_slack_set.shape)
                                island_slack = island_slack_set[0].item()
                                ppc_island["bus"][island_slack, 1] = 3


                            # print(ppc_island["bus"][:, 1]) 

                            # find all branches that belong to this island!
                            CC_Branch_Index = np.array([])
                            for t in range(ppc["branch"].shape[0]):
                                if np.argwhere(ppc_island["bus"][:, 0]==ppc["branch"][t, 0]).shape[0]>0 and np.argwhere(ppc_island["bus"][:, 0]==ppc["branch"][t, 1]).shape[0]>0: 
                                    CC_Branch_Index = np.append(CC_Branch_Index, t)

                            CC_Branch_Index = CC_Branch_Index.astype(int)
                            ppc_island["branch"] = ppc["branch"][CC_Branch_Index, :]

                            if ppc_island["bus"].shape[0]!=0 and ppc_island["gen"].shape[0]!=0 and ppc_island["branch"].shape[0]!=0:
                                # if this is a meaningful island!!
                                # print(ppc_island["branch"].shape)
                                # print(ppc_island["bus"].shape)
                                # print(ppc_island["gen"])

                                result_island = rundcpf_my(ppc_island)   
                                voltageAngles[CC_Index[:, 0]] = result_island[0]["bus"][:, 8]
                                # print(result_island[0]["success"])

                                temp = ppc_island["branch"][:, 13]
                                temp = np.expand_dims(temp, axis = 1)
                                result_island[0]["branch"] = np.concatenate([result_island[0]["branch"], temp], axis = 1)

                                # print(result_island[0]["bus"].shape)  

                                result_tmp[0]["bus"] = np.vstack([result_tmp[0]["bus"], result_island[0]["bus"]]) if result_tmp[0]["bus"].size else result_island[0]["bus"]
                                result_tmp[0]["gen"] = np.vstack([result_tmp[0]["gen"], result_island[0]["gen"]]) if result_tmp[0]["gen"].size else result_island[0]["gen"]                                                  
                                result_tmp[0]["branch"] = np.vstack([result_tmp[0]["branch"], result_island[0]["branch"]]) if result_tmp[0]["branch"].size else result_island[0]["branch"]
                            
                            else:
                                # if this is NOT a meaningful island!
                                ppc_island["gen"][:, 1] = 0
                                ppc_island["bus"][:, 2] = 0

                                voltageAngles[CC_Index[:, 0]] = np.array([0]*len(CC_Index[:, 0]))

                                result_tmp[0]["bus"] = np.vstack([result_tmp[0]["bus"], ppc_island["bus"]]) if result_tmp[0]["bus"].size else ppc_island["bus"]
                                result_tmp[0]["gen"] = np.vstack([result_tmp[0]["gen"], ppc_island["gen"]]) if result_tmp[0]["gen"].size else ppc_island["gen"]  

                                temp1 = ppc_island["branch"][:, 0:ppc_island["branch"].shape[1]-1]
                                # temp1 = np.reshape(temp1, (-1, 1))
                                temp2 = np.zeros((ppc_island["branch"].shape[0], 4))
                                # temp2 = np.reshape(temp2, (-1, 1))
                                temp3 = ppc_island["branch"][:, ppc_island["branch"].shape[1]-1]
                                temp3 = np.reshape(temp3, (-1, 1))
                                # print(temp1, temp1.shape, temp2, temp2.shape, temp3, temp3.shape)
                                temp = np.concatenate((temp1, temp2, temp3), axis = 1)
                                # print(temp, temp.shape)
                                result_tmp[0]["branch"] = np.vstack([result_tmp[0]["branch"], temp]) if result_tmp[0]["branch"].size else temp
                                # print(result_tmp[0]["branch"].shape)  

                        result[0]["bus"] = result_tmp[0]["bus"][result_tmp[0]["bus"][:, 0].argsort()]
                        result[0]["gen"] = result_tmp[0]["gen"][result_tmp[0]["gen"][:, 0].argsort()]
                        result[0]["branch"] = result_tmp[0]["branch"][result_tmp[0]["branch"][:, 17].argsort()]                                     
                    else:
                        result = rundcpf_my(ppc)        

                        voltageAngles[:] = result[0]["bus"][:, 8]

                        temp = ppc["branch"][:, 13]
                        temp = np.expand_dims(temp, axis = 1)
                        result[0]["branch"] = np.concatenate([result[0]["branch"], temp], axis = 1)
                        result[0]["branch"] = result[0]["branch"][result[0]["branch"][:, 17].argsort()]

                    for index in range(result[0]["branch"].shape[0]): 
                        if np.isnan(result[0]["branch"][index, 13]):
                            result[0]["branch"][index, 13] = 0
                            result[0]["branch"][index, 15] = 0

                    tempur = np.multiply(tempur, adjacency)

                    gen_shed = start_gen - np.sum(result[0]["gen"][:, 1])
                    load_shed = start_load - np.sum(result[0]["bus"][:, 2])

            #print(result[0]["branch"].shape)
            #print(np.round(result[0]["branch"][:, [13, 15, 17]], 2))
            powerFlowVector = result[0]["branch"][:, 13]
            return (powerFlowVector, ppc, adjacency, S, gen_shed, load_shed, link_flag_vector)
    

        powerInjected_ori = powerInj_vector    
        powerInjected_ori = np.expand_dims(powerInjected_ori, axis=1)
        powerInjected_ori = np.transpose(powerInjected_ori)  # 1 x bus[0]
        adjacency_ori = sparse.csr_matrix(Adj)               # sparse adj

        M = np.argwhere(np.triu(sparse.csr_matrix.todense(adjacency_ori))!=0).shape[0]  # non-zero elements of the matrix

        link_row = branch[:, 0]                      # branch FROM
        link_row = np.expand_dims(link_row, axis=1)

        link_column = branch[:, 1]                   # branch TO
        link_column = np.expand_dims(link_column, axis=1)

        # perform this operation in order to identify the true line outage index!
        index_vector = np.expand_dims(np.sort(np.random.permutation(M)), axis = 1)  # 0 -- L-1
        ppc["branch"] = np.concatenate([ppc["branch"], index_vector], axis = 1)

        # --> solving power-flow
        result = rundcpf_my(ppc)
        
        temp = ppc["branch"][:, 13] # indices
        temp = np.expand_dims(temp, axis = 1)
        result[0]["branch"] = np.concatenate([result[0]["branch"], temp], axis = 1)

        alpha = 1
        K = 1
        link_set = np.concatenate([link_row, link_column], axis=1) 

        ##############################################
        # action to be selected --> ( 0 -- L-1 )
        # actionSpace = [14, 5]     # lines to outage! 
        voltageAngles = np.zeros( int(ppc["bus"].shape[0],), dtype = float ) # want to be able to fill this vector during the cascading failure simulation
        ################################################

        FaultChains = np.zeros((K, 13))

        if len(actionSpace)==0:
            #print(" ")
            #print("############################### len(actionSpace) == 0 ###############################")
            #print(" ")
            
            totalLoadShed = 0
            
            # Adj is numpy array, no issues!
            voltageAngles[:] = result[0]["bus"][:, 8]
            powerFlowVector = result[0]["branch"][:, 13]

            # [trueLineIndices, capcacity, powerflow]
            actionDecision = np.c_[ppc["branch"][:, 13], ppc["branch"][:, 5], powerFlowVector]
            #print("----")
            #print("All in one: ", actionDecision)
            #print(ppc["branch"].shape)
            #print("----")

            if self.selectCase=="118":
                #print( actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ].shape )
                check = actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ] 
                #print("check: ", check)
                for key, value in self.lineIndextoNewIndexConverter.items():   # conversion of the indices in the first column to the newIndices
                    check[:, 0][check[:, 0] == key] = value
                #print("new check: ", check)
                return (np.array(Adj, dtype = np.float32), voltageAngles, 0, False, totalLoadShed, check)
            else:
                #                             (Adj Matrix, Voltage Angles, curr_load_shed, is_terminal_stage, total_load_shed, action_decision)
                return (np.array(Adj, dtype = np.float32), voltageAngles, 0, False, totalLoadShed, actionDecision)
    
        elif len(actionSpace)==1:
            #print(" ")
            #print("############################### len(actionSpace) == 1 ###############################")
            #print(" ")

            totalLoadShed = 0

            tmp_index = np.array([actionSpace[0]])   # these are line indices that are to be removed 0 -- (L-1) not 1 -- L !!
            initFail = link_set[tmp_index, :]        # this specifies the buses between which the failure takes place
            powerFlowVector, ppcFirstOutage, adjacencyFirstOutage, S, gen_shed, load_shed, link_flag_vector = b_Pypower_DC_OPA_slack(voltageAngles, copy.deepcopy(ppc), adjacency_ori, Capacity, Capacity_vector, alpha, initFail, tmp_index)
            
            # [indices, capcacity, powerflow]
            actionDecision = np.c_[ppcFirstOutage["branch"][:, 13], ppcFirstOutage["branch"][:, 5], powerFlowVector]
            #print("----")
            #print("Line Indices: ", ppcFirstOutage["branch"][:, 13])
            #print("All in one: ", actionDecision)
            
            #print(ppcFirstOutage["branch"].shape)
            #print("We have ", np.sum(link_flag_vector), " branches.")
            #print("We have ", S, " islands.")
            # print(ppcFirstOutage["branch"].shape)
            FaultChains[0, 0] = S
            FaultChains[0, 1] = gen_shed
            FaultChains[0, 2] = load_shed
            totalLoadShed += FaultChains[0, 2]
            FaultChains[0, 9] = totalLoadShed

            dataType = str(type(adjacencyFirstOutage))
            if "scipy.sparse" in dataType: 
                returnNumpyArray = np.array(adjacencyFirstOutage.toarray(), dtype = np.float32)
            else:
                returnNumpyArray = np.array(adjacencyFirstOutage, dtype = np.float32)
            
            # this is supposed to return the load_shed in the current round!
            #print("----")
            if self.selectCase=="118":
                #print( actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ].shape )
                check = actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ] 
                #print("check: ", check)
                for key, value in self.lineIndextoNewIndexConverter.items():   # conversion of the indices in the first column to the newIndices
                    check[:, 0][check[:, 0] == key] = value
                #print("new check: ", check)
                return (returnNumpyArray.reshape( (int(ppc["bus"].shape[0]), int(ppc["bus"].shape[0])) ), voltageAngles, load_shed, False, totalLoadShed, check) 
            else:
                return (returnNumpyArray.reshape( (int(ppc["bus"].shape[0]), int(ppc["bus"].shape[0])) ), voltageAngles, load_shed, False, totalLoadShed, actionDecision)
    
        elif len(actionSpace)==2:
            #print(" ")
            #print("############################### len(actionSpace) == 2 ###############################")
            #print(" ")

            totalLoadShed = 0

            tmp_index = np.array([actionSpace[0]])   # these are line indices 0 -- (L-1) not 1 -- L !!
            initFail = link_set[tmp_index, :]        # failure indices
            powerFlowVector, ppcFirstOutage, adjacencyFirstOutage, S, gen_shed, load_shed, link_flag_vector = b_Pypower_DC_OPA_slack(voltageAngles, copy.deepcopy(ppc), adjacency_ori, Capacity, Capacity_vector, alpha, initFail, tmp_index)
            
            # [indices, capcacity, powerflow]
            #actionDecision = np.c_[ppcFirstOutage["branch"][:, 13], ppcFirstOutage["branch"][:, 5], powerFlowVector]
            #print("----")
            #print("Line Indices: ", ppcFirstOutage["branch"][:, 13]) 
            #print("All in one: ", actionDecision)
            
            #print(ppcFirstOutage["branch"].shape)
            #print("We have ", np.sum(link_flag_vector), " branches.")
            #print("We have ", S, " islands.")
            # print(ppcFirstOutage["branch"].shape)
            FaultChains[0, 0] = S
            FaultChains[0, 1] = gen_shed
            FaultChains[0, 2] = load_shed
            totalLoadShed += FaultChains[0, 2]
            FaultChains[0, 9] = totalLoadShed
            

            sp_adjacencyFirstOutage = sparse.csr_matrix(adjacencyFirstOutage)              # sparse adj
            tmp_index = np.array([actionSpace[1]])   # these are line indices 0 -- (L-1) not 1 -- L !!
            initFail = link_set[tmp_index, :]        # failure indices
            powerFlowVector, ppcSecondOutage, adjacencySecondOutage, S, gen_shed, load_shed, link_flag_vector = b_Pypower_DC_OPA_slack(voltageAngles, copy.deepcopy(ppcFirstOutage), sp_adjacencyFirstOutage, Capacity, Capacity_vector, alpha, initFail, tmp_index, link_flag_vector)
            
            # [indices, capcacity, powerflow]
            actionDecision = np.c_[ppcSecondOutage["branch"][:, 13], ppcSecondOutage["branch"][:, 5], powerFlowVector]        
            #print("----")
            #print("Line Indices: ", ppcSecondOutage["branch"][:, 13])
            #print("All in one: ", actionDecision)
            
            #print(ppcSecondOutage["branch"].shape)
            #print("We have ", np.sum(link_flag_vector), " branches.")
            #print("We have ", S, " islands.")
            #print(voltageAngles)
            # print(ppcFirstOutage["branch"].shape)
            FaultChains[0, 3] = S
            FaultChains[0, 4] = gen_shed
            FaultChains[0, 5] = load_shed
            totalLoadShed += FaultChains[0, 5]
            FaultChains[0, 9] = totalLoadShed
            
            dataType = str(type(adjacencySecondOutage))
            if "scipy.sparse" in dataType: 
                returnNumpyArray = np.array(adjacencySecondOutage.toarray(), dtype = np.float32)
            else:
                returnNumpyArray = np.array(adjacencySecondOutage, dtype = np.float32)
            
            #print("----")
            if self.selectCase=="118":
                #print( actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ].shape )
                check = actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ] 
                #print("check: ", check)
                for key, value in self.lineIndextoNewIndexConverter.items():   # conversion of the indices in the first column to the newIndices
                    check[:, 0][check[:, 0] == key] = value
                #print("new check: ", check)
                return (returnNumpyArray.reshape( (int(ppc["bus"].shape[0]), int(ppc["bus"].shape[0])) ), voltageAngles, load_shed, False, totalLoadShed, check) 
            else:
                return (returnNumpyArray.reshape( (int(ppc["bus"].shape[0]), int(ppc["bus"].shape[0])) ), voltageAngles, load_shed, False, totalLoadShed, actionDecision)     

        elif len(actionSpace)==3: 
            #print(" ")
            #print("############################### len(actionSpace) == 3 ###############################")
            #print(" ")

            totalLoadShed = 0

            tmp_index = np.array([actionSpace[0]])   # these are line indices 0 -- (L-1) not 1 -- L !!
            initFail = link_set[tmp_index, :]        # failure indices
            powerFlowVector, ppcFirstOutage, adjacencyFirstOutage, S, gen_shed, load_shed, link_flag_vector = b_Pypower_DC_OPA_slack(voltageAngles, copy.deepcopy(ppc), adjacency_ori, Capacity, Capacity_vector, alpha, initFail, tmp_index)
            
            # [indices, capcacity, powerflow]
            #actionDecision = np.c_[ppcFirstOutage["branch"][:, 13], ppcFirstOutage["branch"][:, 5], powerFlowVector]
            #print("----")
            #print("Power Flow in Line: ", powerFlowVector)
            #print("Line Capacity: ", ppcFirstOutage["branch"][:, 5])
            #print("Line Indices: ", ppcFirstOutage["branch"][:, 13])
            #print("All in one: ", actionDecision)
            
            #print(ppcFirstOutage["branch"].shape)
            #print("We have ", np.sum(link_flag_vector), " branches.")
            #print("We have ", S, " islands.")
            FaultChains[0, 0] = S
            FaultChains[0, 1] = gen_shed
            FaultChains[0, 2] = load_shed
            totalLoadShed += FaultChains[0, 2]
            FaultChains[0, 9] = totalLoadShed
            
            
            sp_adjacencyFirstOutage = sparse.csr_matrix(adjacencyFirstOutage)              # sparse adj
            tmp_index = np.array([actionSpace[1]])   # these are line indices 0 -- (L-1) not 1 -- L !!
            initFail = link_set[tmp_index, :]        # failure indices
            powerFlowVector, ppcSecondOutage, adjacencySecondOutage, S, gen_shed, load_shed, link_flag_vector = b_Pypower_DC_OPA_slack(voltageAngles, copy.deepcopy(ppcFirstOutage), sp_adjacencyFirstOutage, Capacity, Capacity_vector, alpha, initFail, tmp_index, link_flag_vector)
            
            # [indices, capcacity, powerflow]
            #actionDecision = np.c_[ppcSecondOutage["branch"][:, 13], ppcSecondOutage["branch"][:, 5], powerFlowVector]
            #print("----")
            #print("Power Flow in Line: ", powerFlowVector)
            #print("Line Capacity: ", ppcSecondOutage["branch"][:, 5])
            #print("Line Indices: ", ppcSecondOutage["branch"][:, 13])
            #print("All in one: ", actionDecision)
            #print(ppcSecondOutage["branch"].shape)
            #print("We have ", np.sum(link_flag_vector), " branches.")
            #print("We have ", S, " islands.")
            #print(voltageAngles)
            #print(ppcFirstOutage["branch"].shape)
            FaultChains[0, 3] = S
            FaultChains[0, 4] = gen_shed
            FaultChains[0, 5] = load_shed
            totalLoadShed += FaultChains[0, 5]
            FaultChains[0, 9] = totalLoadShed    


            sp_adjacencySecondOutage = sparse.csr_matrix(adjacencySecondOutage)              # sparse adj
            tmp_index = np.array([actionSpace[2]])   # these are line indices 0 -- (L-1) not 1 -- L !!
            initFail = link_set[tmp_index, :]        # failure indices
            powerFlowVector, ppcThirdOutage, adjacencyThirdOutage, S, gen_shed, load_shed, link_flag_vector = b_Pypower_DC_OPA_slack(voltageAngles, copy.deepcopy(ppcSecondOutage), sp_adjacencySecondOutage, Capacity, Capacity_vector, alpha, initFail, tmp_index, link_flag_vector)
            
            # [indices, capcacity, powerflow]
            actionDecision = np.c_[ppcThirdOutage["branch"][:, 13], ppcThirdOutage["branch"][:, 5], powerFlowVector]
            #print("----")
            #print("Power Flow in Line: ", powerFlowVector)
            #print("Line Capacity: ", ppcThirdOutage["branch"][:, 5])
            #print("Line Indices: ", ppcThirdOutage["branch"][:, 13])
            #print("All in one: ", actionDecision)
            #print(ppcThirdOutage["branch"].shape)
            #print("We have ", np.sum(link_flag_vector), " branches.")
            #print("We have ", S, " islands.")
            #print(voltageAngles)
            #print(ppcFirstOutage["branch"].shape)
            FaultChains[0, 6] = S
            FaultChains[0, 7] = gen_shed
            FaultChains[0, 8] = load_shed
            totalLoadShed += FaultChains[0, 8]

            dataType = str(type(adjacencyThirdOutage))
            if "scipy.sparse" in dataType: 
                returnNumpyArray = np.array(adjacencyThirdOutage.toarray(), dtype = np.float32)
            else:
                returnNumpyArray = np.array(adjacencyThirdOutage, dtype = np.float32)
            
            #print("----")
            if self.selectCase=="118":
                #print( actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ].shape )
                check = actionDecision[ np.isin(actionDecision[:, 0], self.sortedIndicesHeavyArray) ] 
                #print("check: ", check)
                for key, value in self.lineIndextoNewIndexConverter.items():   # conversion of the indices in the first column to the newIndices
                    check[:, 0][check[:, 0] == key] = value
                #print("new check: ", check)
                return (returnNumpyArray.reshape( (int(ppc["bus"].shape[0]), int(ppc["bus"].shape[0])) ), voltageAngles, load_shed, True, totalLoadShed, check) 
            else:
                return (returnNumpyArray.reshape((int(ppc["bus"].shape[0]), int(ppc["bus"].shape[0]))), voltageAngles, load_shed, True, totalLoadShed, actionDecision)     
        
    def load_dataset(self, DATASET_STRING):
        """
        Loads the specified ground truth dataset and sets the 1% threshold required for dataset information to display.

        Parameters:
        ----------
        DATASET_STRING : str
            Identifier for the dataset to be loaded.

        Returns:
        -------
        np.ndarray
            The loaded dataset with shape (numSeq, 4).
            numSeq = L * (L-1) * (L-2) * ... * (L-M+1)

        float
            threshold
        """
        self.dataSetString = DATASET_STRING
        print(f"this is dataSetString: {self.dataSetString}")

        if self.selectCase == "39":
            # threshold according to the 1% rule
            if "05" in self.dataSetString:
                self.threshold = 31.27
            elif "055" in self.dataSetString:
                self.threshold = 34.3982
            elif "06" in self.dataSetString: 
                self.threshold = 37.5253
            elif "07" in self.dataSetString:   
                self.threshold = 43.7796

        elif self.selectCase == "118":
            # threshold according to the 1% rule
            if "06" in self.dataSetString:
                self.threshold = 25.4519
            elif "10" in self.dataSetString:
                self.threshold = 42.42

        with h5py.File(DATASET_STRING, 'r') as hf:
            # Use os.path.basename to get just the filename without the path
            key = os.path.basename(DATASET_STRING)[:-3]
            dataSet = hf[key][:]

        threshold = self.threshold

        return dataSet, threshold

    def print_datatset_information(self, dataSet, threshold):
        """
        Prints summary information about the ground truth dataset of all fault chains.

        Parameters:
        ----------
        dataSet : np.ndarray
            The dataset to be analyzed. Type=<class 'numpy.ndarray'>, 
            Shape=(L * (L-1) * (L-2) * ... * (L-M+1), 4).
        """
        print("number of FCs:                           ", dataSet.shape[0])
        print("number of fault FCs ( 1% load shed thd): ", np.sum(dataSet[:, 3] > threshold * 1))   
        print("number of fault FCs ( 5% load shed thd): ", np.sum(dataSet[:, 3] > threshold * 5))
        print("number of fault FCs (10% load shed thd): ", np.sum(dataSet[:, 3] > threshold * 10))
        print("number of fault FCs (15% load shed thd): ", np.sum(dataSet[:, 3] > threshold * 15))
        print("number of fault FCs (20% load shed thd): ", np.sum(dataSet[:, 3] > threshold * 20))
        print("maximum total load shed among all FCs:   ", max(dataSet[:, 3]))

    def find_risky_fault_chains(self, dataSet, M):
        """
        Identifies and extracts risky fault chains based on a threshold.

        This function filters fault chains where the risk value (fourth column of `dataSet`) 
        exceeds `threshold * M`. It then constructs a dictionary mapping fault chain indices 
        to their corresponding risk values.

        Parameters:
        ----------
        dataSet : np.ndarray
            The dataset containing fault chains, with shape (L * (L-1) * ... * (L-M+1), 4).
        M : int
            % threshold considered.

        Returns:
        -------
        dict
            A dictionary where:
            - Keys: Tuples representing fault chain indices.
            - Values: Corresponding risk values.
        """
        riskyFaultChains = dataSet[dataSet[:, 3] > self.threshold*M]
        # more pre-processing for the dataset if 118-bus is considered since newLine indices are different than trueLineIndices            
        riskyFaultChainDict = {}

        for index in range(riskyFaultChains.shape[0]):
            if self.selectCase=="118":
                riskyFaultChainDict[tuple( [self.lineIndextoNewIndexConverter[key] for key in riskyFaultChains[index, [0, 1, 2]]] )] = riskyFaultChains[index, 3]
            else:
                riskyFaultChainDict[ tuple(riskyFaultChains[index, [0, 1, 2]]) ] = riskyFaultChains[index, 3]

        return riskyFaultChainDict

    def find_risky_fault_chains_all(self, dataSet):

        riskyFaultChains_all = [ dataSet[dataSet[:, 3] > self.threshold*5], dataSet[dataSet[:, 3] > self.threshold*10], dataSet[dataSet[:, 3] > self.threshold*15] ]
        # more pre-processing for the dataset if 118-bus is considered since newLine indices are different than trueLineIndices            
        riskyFaultChainDict_all = [ {}, {}, {} ]

        for position, riskyFaultChain in enumerate(riskyFaultChains_all):
            for index in range(riskyFaultChain.shape[0]):
                if self.selectCase=="118":
                    riskyFaultChainDict_all[position][tuple( [self.lineIndextoNewIndexConverter[key] for key in riskyFaultChain[index, [0, 1, 2]]] )] = riskyFaultChain[index, 3]
                else:
                    riskyFaultChainDict_all[position][tuple(riskyFaultChain[index, [0, 1, 2]])] = riskyFaultChain[index, 3]

        return riskyFaultChainDict_all
        
    def print_current_answer_details(self, currentAnswer):
        """
        Prints various essential variables returned by the environmentStep function.
        """
        #print(f"Adjacency Matrix: Type={type(currentAnswer[0])}, Shape={currentAnswer[0].shape}, Value={currentAnswer[0]}\n")
        #print(f"Voltage Angles: Type={type(currentAnswer[1])}, Shape={currentAnswer[1].shape if hasattr(currentAnswer[1], 'shape') else 'N/A'}, Value={currentAnswer[1]}\n")
        print(f"Load Shed (Final Stage): Type={type(currentAnswer[2])}, Value={currentAnswer[2]}\n")
        print(f"Terminal State: Type={type(currentAnswer[3])}, Value={currentAnswer[3]}\n")
        print(f"Total Load Shed (all stages): {currentAnswer[4]}\n")
        #print(f"Action Decision: Type={type(currentAnswer[5])}, Shape={currentAnswer[5].shape if hasattr(currentAnswer[5], 'shape') else 'N/A'}, Value={currentAnswer[5]}\n")


def get_dataset_string(select_case, loading_factor):
    """Returns the appropriate dataset filename based on the selected case and loading factor."""
    datasets = {
        "39": {0.6: "loading06_39bus.h5", 0.55: "loading055_39bus.h5"},
        "118": {0.6: "118bus_loading06.h5", 1.0: "118bus_loading10.h5"},
    }
    return datasets.get(select_case, {}).get(loading_factor, None)


def main(select_case, loading_factor):
    """Main function to execute the fault chain analysis."""
    
    # Ensure valid inputs
    valid_cases = {"39", "118"}
    valid_factors = {
        "39": {0.6, 0.55},
        "118": {0.6, 1.0}
    }

    if select_case not in valid_cases or loading_factor not in valid_factors.get(select_case, {}):
        raise ValueError(f"Invalid SELECT_CASE '{select_case}' or LOADING_FACTOR '{loading_factor}'. "
                         f"Valid cases: {valid_cases}. "
                         f"Valid factors per case: {valid_factors}")

    # Set working directory
    print(f"Current working directory: {os.getcwd()}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"New working directory: {os.getcwd()}\n")

    # Initialize the solver
    FCsolver = FaultChainSovlerClass(loading_factor, select_case)

    # Determine dataset path
    dataset_string = get_dataset_string(select_case, loading_factor)
    if not dataset_string:
        raise ValueError(f"No dataset found for SELECT_CASE={select_case} and LOADING_FACTOR={loading_factor}")
    
    dataset_path = os.path.join('Datasets', dataset_string)
    print(f"Using dataset: {dataset_path}\n")

    # Environment check
    action_space = [9, 13, 11]  # Example actions
    current_answer = FCsolver.environmentStep(action_space)
    print(f"Removing line indices {action_space} resulted in:")
    FCsolver.print_current_answer_details(current_answer)

    # Load dataset and print details
    data_set, threshold = FCsolver.load_dataset(dataset_path)
    print(f"Ground truth FCs associated with {select_case} bus test case with loading condition {loading_factor} render:")
    FCsolver.print_datatset_information(data_set, threshold)

    # Find risky fault chains
    M = 5  # risky percentage Threshold
    risky_fault_chain_dict = FCsolver.find_risky_fault_chains(data_set, M)
    print(f"Number of FCs greater than {M}%: {len(risky_fault_chain_dict)}")

    risky_fault_chain_dict_all = FCsolver.find_risky_fault_chains_all(data_set)
    print(f"Number of FCs greater than 5, 10, 15% thresholds: {[len(risky_fault_chain) for risky_fault_chain in risky_fault_chain_dict_all]}")


if __name__ == '__main__':

    # python FaultChainSolver.py --case 39 --load 0.55
    
    parser = argparse.ArgumentParser(description="Run Fault Chain Analysis for power grid datasets.")
    parser.add_argument("--case", type=str, choices=["39", "118"], required=True, 
                        help="Select power grid case: '39' or '118'.")
    parser.add_argument("--load", type=float, required=True, 
                        help="Specify loading factor. Allowed: 0.6, 0.55 (for '39'), 0.6, 1.0 (for '118').")

    args = parser.parse_args()
    main(args.case, args.load)