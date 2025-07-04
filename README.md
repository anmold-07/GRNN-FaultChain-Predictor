# GRNN-FaultChain-Predictor

Source code accompanying our **2024 IEEE Transactions on Power Systems (TPS)** [paper](https://ieeexplore.ieee.org/abstract/document/10075543) 
and our **2025 ICLR Climate Change AI (CCAI) Workshop** [paper](https://arxiv.org/abs/2503.09775)

Other Artifacts:
- [ICLR Slides](https://iclr.cc/media/iclr-2025/Slides/36676.pdf)
- [ICLR Poster](https://iclr.cc/media/PosterPDFs/ICLR%202025/36676.png?t=1751592032.931365)
- [ICLR Video](https://iclr.cc/virtual/2025/36676)
  
---

## Requirements

This codebase requires **Python 3**. The specific package requirements are listed in **`environment.yml`**.  
We recommend setting up a dedicated **conda environment** using:

```
conda env create -f environment.yml
conda activate env
```

## Running the Developed Fault Chain Solver:

To run the Fault Chain Solver on a selected test case, use:
```
python faultChainSolver.py --case 39 --load 0.55
```
Currently, the supported IEEE test cases are the 39-bus and 118-bus systems, with supported load values of 0.55, 0.6, and 1.0 and upto a prediction horizon of 3.



## Running the Developed GRNN Risky Fault Chain Predictor:

To run the GRNN-based fault chain predictor, use the `grqnSolver.py` script.

Option 1: Run for a Fixed Number of Iterations:
```
python grqnSolver.py --case 39 --load 0.55 --threshold 5 --kappa 3 --if_iteration 1 --num_episodes 50
```

Option 2: Run for a Fixed Time Budget:
```
python grqnSolver.py --case 39 --load 0.55 --threshold 5 --kappa 3 --if_iteration 0 --time_taken 60
```


## Running Baseline Solvers (Q-Learning)

Option 1: Run for a Fixed Number of Iterations
```
python qLearningSovler.py --case 39 --load 0.55 --threshold 5 --if_iteration 1 --num_episodes 50
```

Option 2: Run for a Fixed Time Budget
```
python qLearningSovler.py --case 39 --load 0.55 --threshold 5 --if_iteration 0 --time_taken 60
```

## Running All Experiments for Comparisons

To run all experiments (GRNN, Q-learning, and Q-learning TE), use the main script:
```
39-bus Code/39_bus_main.py
```

## Disclaimer:

Please note that the GRNN architecture in this repository is a slightly constrained version compared to the one detailed in the publication. This adjustment was made to explore different ideas and induces slight differences from the main experiments presented in the paper. To use the exact architecture as described in the paper, you will need to make a minor modification to the time-varying filter of the GRNN, which should be a straightforward change. If you have any questions or need further assistance with these modifications, please feel free to reach out at [dwivea2@rpi.edu](mailto:dwivea2@rpi.edu).

## Citation
If you find this repository useful for your research, please consider citing:
```
@article{Dwivedi-GRNN-2024,
author={Dwivedi, Anmol and Tajer, Ali},
journal={IEEE Transactions on Power Systems},
title={GRNN-Based Real-Time Fault Chain Prediction},
year={2024},
volume={39},
number={1},
pages={934-946},
doi={10.1109/TPWRS.2023.3258740}
}
```
and
```
@misc{Dwivedi-GRNN-2025,
author={Anmol Dwivedi and Ali Tajer},
title={Real-Time Risky Fault-Chain Search using Time-Varying Graph RNNs},
year={2025},
eprint={2503.09775},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2503.09775}
}
```
