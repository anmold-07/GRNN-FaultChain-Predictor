import pickle
import numpy as np
from Evaluation import EvaluationClassIterationMCPlot
from Evaluation import EvaluationClassIterationMCPlot_time

# importing this so that Regret accuracy metric can be calculated
from FaultChainSolver import FaultChainSovlerClass

iteration_track = 1

three_nine_bus = False

alreadyHaveData = True

if alreadyHaveData:
    with open('118bus_precision_data.npy', 'rb') as f:
        dataPrecision = np.load(f)

    finalPlot = EvaluationClassIterationMCPlot([], [], [], [], [], [], [], [], [], [])
    pivot = 25
    finalPlot.plot_precision(dataPrecision[:, :, pivot:], 1600-pivot)


if three_nine_bus:

    selectCase = "39"   # "14, "30", "39", "118"
    LOADING_FACTOR = 0.55
    dataSetString = "loading055_39bus.h5"
    M = 5  
    FCsovler = FaultChainSovlerClass(LOADING_FACTOR, selectCase) # instantiated object!
    dataSet = FCsovler.load_dataset(dataSetString)
    FCsovler.print_datatset_information(dataSet)
    optimalSequenceRisk = np.sort(dataSet[:, 3])[::-1] # sorting the entire dataset in decreasing order 

    if iteration_track:
        # old-results
        finalPlot = EvaluationClassIterationMCPlot([], [], [], [], [], [], [], [], [], [])
        
        dontPlotOld = False
        with open("39-bus Results/Final Results/1200_risky_data_list", "rb") as fp:
            risky_data_list, M_episodes_qlearning = pickle.load(fp)

        with open("39-bus Results/Final Results/1200_num_data_list", "rb") as fp:
            num_data_list, M_episodes_qlearning = pickle.load(fp)

        if dontPlotOld:
            finalPlot.plot_fc_risk(risky_data_list, M_episodes_qlearning)
            finalPlot.plot_num_risky_fcs(num_data_list, M_episodes_qlearning)

        # new-results (Accuracy Metrics)

        # Cumulative Regret
        checkRisky = np.array(risky_data_list)
        optimalSequenceRisk1200 = np.cumsum( optimalSequenceRisk[:1200] )
        #optimalSequenceRisk1200[-1]    # the sum of the entire sequences of risks
        dataRegret = optimalSequenceRisk1200[-1]*np.ones((checkRisky.shape[1], checkRisky.shape[2])) - checkRisky
        finalPlot.plot_regret(dataRegret, M_episodes_qlearning)
        
        # Precision
        checkNumdata = np.array(num_data_list)
        lengthArray = np.array( range(1, checkNumdata.shape[2]+1) )
        lengthArrayMatrix = np.vstack([lengthArray]*checkNumdata.shape[1])
        dataPrecision = (checkNumdata/lengthArrayMatrix)
        pivot = 6
        finalPlot.plot_precision(dataPrecision[:, :, pivot:], 1200-pivot)

        finalPlot.display_statistics_for_list(dataRegret, dataPrecision)

    else:
        finalPlot = EvaluationClassIterationMCPlot_time([], [], [], [], [], [], [], [], [], [])
        with open("39-bus Results/Final Results/time_risky_data_list", "rb") as fp:
            risky_data_list, M_episodes_qlearning = pickle.load(fp)

        for approach in range(5):

            algorithm = risky_data_list[approach]
            MC_iterations = np.array(algorithm).shape[0]

            numberofFCsList, totalRiskList, regretList = [], [], []
            numRiskyFCs, precisionList = [], []
            
            for iteration in range(MC_iterations):
                temp = np.array( algorithm[iteration] )     # FC Risk, time taken
                numberofFCs, _ = temp.shape              
                #print(" Number of FCs: ", numberofFCs, " Total Risk: ", temp[:, 0][-1], " Optimal Sequence Risk: ", sum(optimalSequenceRisk[:numberofFCs]) )
                
                numberofFCsList.append(numberofFCs)
                cumu = temp[:, 0]
                normalArray = cumu.copy()
                normalArray[1:] = np.diff(cumu)
                #print( sum( normalArray > 34.3982*5 ) )
                    
                numRiskyFCs.append( sum( normalArray >= 34*5 ) )
                precisionList.append( sum( normalArray >= 34*5 )/numberofFCs )
                
                totalRiskList.append( temp[:, 0][-1] )
                regretList.append( sum(optimalSequenceRisk[:numberofFCs]) - temp[:, 0][-1] )

            print("Avg No. of FCs found:         ", np.average( np.array(numberofFCsList) ))
            print("Range Total Risk found:       ", np.average( np.array(totalRiskList) ), "    +- " , np.std( np.array(totalRiskList) ), " also in percentage ", 100*np.std( np.array(totalRiskList) )/np.average( np.array(totalRiskList) ))
            print("Range Regret found:           ", np.average( np.array(regretList) ) ,   "    +- " , np.std( np.array(regretList) ), " also in percentage ", 100*np.std( np.array(regretList) )/np.average( np.array(regretList) ))
            print("Range No. of Risky FCs found: ", np.average( np.array(numRiskyFCs) ),   "    +- " , np.std( np.array(numRiskyFCs) ), " also in percentage ", 100*np.std( np.array(numRiskyFCs) )/np.average( np.array(numRiskyFCs) ))
            print("Range Precision found:        ", np.average( np.array(precisionList) ), "    +- " , np.std( np.array(precisionList) ), " also in percentage ", 100*np.std( np.array(precisionList) )/np.average( np.array(precisionList) ) )   
            print(" ")

        """ 
        for mc_iteration in approach:
            y.append([sublist[0] for sublist in mc_iteration]) # rewards
            x.append([sublist[1] for sublist in mc_iteration]) # time

        a = np.array(x)
        b = np.array(y)
        #print(y)
        print(a.mean(axis=0))
        print(b.mean(axis=0))

        finalPlot.plot_fc_risk(risky_data_list, M_episodes_qlearning)
        """



else:

    selectCase = "118"   # "14, "30", "39", "118"
    LOADING_FACTOR = 0.6
    dataSetString = "118bus_loading06.h5"
    M = 5  

    FCsovler = FaultChainSovlerClass(LOADING_FACTOR, selectCase) # instantiated object!
    dataSet = FCsovler.load_dataset(dataSetString)
    FCsovler.print_datatset_information(dataSet)
    riskyFaultChainDict = FCsovler.find_risky_fault_chains(dataSet, M)
    optimalSequenceRisk = np.sort(dataSet[:, 3])[::-1]           # sorting the entire dataset in decreasing order 

    if iteration_track:

        finalPlot = EvaluationClassIterationMCPlot([], [], [], [], [], [], [], [], [], [])

        with open("118-bus Results/Final Results/First MC/118bus_risky_data_list", "rb") as fp: 
            risky_data_list1, M_episodes_qlearning = pickle.load(fp)

        with open("118-bus Results/Final Results/Second MC/118bus_risky_data_list", "rb") as fp: 
            risky_data_list2, M_episodes_qlearning = pickle.load(fp)

        # Cumulative Regret
        checkRisky1 = np.array(risky_data_list1)
        checkRisky2 = np.array(risky_data_list2)
        checkRisky = np.concatenate((checkRisky1, checkRisky2), axis=1)
        optimalSequenceRisk1600 = np.cumsum( optimalSequenceRisk[:1600] )
        dataRegret = optimalSequenceRisk1600[-1]*np.ones((checkRisky.shape[1], checkRisky.shape[2])) - checkRisky
        finalPlot.plot_regret(dataRegret, M_episodes_qlearning)

        # Precision
        with open("118-bus Results/Final Results/Second MC/118bus_num_data_list", "rb") as fp:
            num_data_list2, M_episodes_qlearning = pickle.load(fp)
            
        checkNumdata = np.array(num_data_list2)
        lengthArray = np.array( range(1, checkNumdata.shape[2]+1) )
        lengthArrayMatrix = np.vstack([lengthArray]*checkNumdata.shape[1])
        dataPrecision = (checkNumdata/lengthArrayMatrix)
        pivot = 6
        finalPlot.plot_precision(dataPrecision[:, :, pivot:], 1600-pivot)

        finalPlot.display_statistics_for_list(dataRegret, dataPrecision)
