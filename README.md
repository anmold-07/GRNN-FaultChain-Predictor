# GRNN-FaultChain-Predictor

### Installation:
For installation instructions, please refer to the environment.yml file.

### Running Experiments:
To run the experiments as described in the paper, execute the 39-busCode/39bus_main.py file.

### Disclaimer:
Please note that the GRNN architecture in this repository is a slightly constrained version compared to the one detailed in the publication. This adjustment was made to explore different ideas and induces slight differences from the main experiments presented in the paper. To use the exact architecture as described in the paper, you will need to make a minor modification to the time-varying filter, which should be a straightforward change. If you have any questions or need further assistance with these modifications, please feel free to reach out at dwivea2@rpi.edu

### Citing:
If you find this repository useful in your work, we kindly request that you cite the following [paper](https://ieeexplore.ieee.org/abstract/document/10075543):
```
@ARTICLE{DwivediGRNN,
  author={Dwivedi, Anmol and Tajer, Ali},
  journal={IEEE Transactions on Power Systems}, 
  title={GRNN-Based Real-Time Fault Chain Prediction}, 
  year={2024},
  volume={39},
  number={1},
  pages={934-946},
  keywords={Power system protection;Power system faults;Load modeling;Computational modeling;Power systems;Real-time systems;Topology;Cascading failures;fault chains;graph recurrent neural networks},
  doi={10.1109/TPWRS.2023.3258740}}
```
