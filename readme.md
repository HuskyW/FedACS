# Federated Analytics Informed Distributed Industrial IoT Learning with Non-IID Data (FedACS)

FedACS is a federated anaytics-based client selection scheme to assist other federated tasks (tpically federated learning). It measures the client skewness (severity of data heterogeneity) and selects the low-skewness clients in each round. The implementation of federated learning backbone is quoted form https://github.com/shaoxiongji/federated-learning. Many thanks to them!

## Abstract
The increasing concerns of communication overheads and data privacy greatly challenge the gather-and-analyze paradigm of data-driven tasks currently adopted by the industrial IoT deployments. The federated paradigm resolves this challenge by performing tasks collaboratively without uploading the raw data. However, the inherent data heterogeneity (skewness) of diverse industrial IoT data holders significantly degrades the performances of all kinds of federated industrial IoT learning tasks. Quantifying this skewness is non-trivial and cannot be solved by the existing federated learning techniques. In this paper, we propose a Federated skewness Analytics and Client Selection mechanism (FedACS) to quantify the data skewness in a privacy preserving way and use this information to help downstream federated learning tasks. FedACS provably estimates the skewness of the clients using the Hoeffding's inequality based on the distilled insights of edge data in the form of gradient. It then gracefully handles the drifting estimation and robustly selects clients with milder skewness using a novel dueling bandit approach. FedACS gains advantages in privacy preservation, infrastructure reuse, and optimized overheads. Extensive experiments on open datasets demonstrate that FedACS reduces the accuracy degradation by $\sim 78.2\%$, and accelerates the FL convergence for $\sim2.4\times$.

## Features of this implementation
- simulation environment of federated learning on CIFAR-10 dataset (image classification)
- different skewness environments, adjustable via **--sampling** argumant (IID, uniform, inverse Pareto, Dirichlet, and few-class)
- many benchmarks adjustable via **--client_sel** and **--cmfl** options: vanilla FL, CMFL, Oort, FedACS with non-stationary bandit, and complete FedACS

## Installation
```
python==3.8
pytorch==1.7.1
torchvision==0.8.2
```


## Run the code

We present some lines to reproduce some of our experiment results.

Run vanilla FL on IID environment

```
python main_fed.py --epochs 1000 --local_ep 5 --local_bs 400 --num_data 2000 --sampling iid --testing 5  --client_sel random
```

Run FedACS on uniform skewness environment

```
python main_fed.py --epochs 1000 --local_ep 5 --local_bs 400  --num_data 2000 --sampling uniform --testing 5  --client_sel fedacs --extension 8 --historical_rounds 5 --light_analytics --faf 1
```

Run CMFL on inverse Pareto skewness environment
```
python main_fed.py --epochs 1000 --local_ep 5 --local_bs 400  --num_data 2000 --sampling ipareto --testing 5  --client_sel random --cmfl
```

Run Oort on Dirichlet skewness environment
```
python main_fed.py --epochs 1000 --local_ep 5 --local_bs 400  --num_data 2000 --sampling dirichlet --testing 5  --client_sel oort
```

Check utils/options.py for more usage

## Citation format

Z. Wang, Y. Zhu, D. Wang, and Z. Han, "Federated Analytics Informed Distributed Industrial IoT Learning with Non-IID Data", in *IEEE Transactions on Network Science and Engineering (TNSE)*, 2022.

Z. Wang, Y. Zhu, D. Wang, and Z. Han, "FedACS: Federated Skewness Analytics in Heterogeneous Decentralized Data Environments", in *Proceedings of IEEE/ACM International Symposium on Quality of Service (IWQoS)*, 2021.

```
@inproceedings{wang2022fedacs,
  title={Federated Analytics Informed Distributed Industrial IoT Learning with Non-IID Data},
  author={Wang, Zibo and Zhu, Yifei and Wang, Dan and Han, Zhu},
  booktitle={IEEE Transactions on Network Science and Engineering (TNSE)},
  year={2022}
}
```
```
@inproceedings{wang2021fedacs,
  title={FedACS: Federated Skewness Analytics in Heterogeneous Decentralized Data Environments},
  author={Wang, Zibo and Zhu, Yifei and Wang, Dan and Han, Zhu},
  booktitle={Proceedings of IEEE/ACM International Symposium on Quality of Service (IWQoS)},
  year={2021}
}
```