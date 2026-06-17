# RG-GNN: Reassembly-Graph Neural Network

This repository contains the official PyTorch implementation and the dataset generation scripts for the **RG-GNN** architecture, as presented in our paper: 
*"RG-GNN: A Reassembly-Graph Neural Network for Robust Detection of Advanced Fragmentation Evasion in Enterprise Networks"*.

## Overview
Traditional Intrusion Detection Systems (IDS) often fail to detect advanced evasion techniques, such as **Tiny Fragmentation** and **TCP MSS Clamping**, because they process packets primarily as linear sequential streams, making them vulnerable to resource exhaustion and oversmoothing. 

RG-GNN addresses this by explicitly modeling the **Reassembly Relations** among fragmented packets. Instead of deep payload inspection or temporal chaining, RG-GNN:
1. Models individual packets as nodes with lightweight header features.
2. Links fragments of a common datagram through reassembly edges (matching Source IP, Destination IP, and IP Identification in ascending offset order).

By processing these relationships through stacked GraphSAGE layers with a residual connection, RG-GNN successfully isolates the structural signature of evasion attacks without degrading live network throughput (designed for out-of-band SPAN monitoring).

## Repository Structure

Based on the current repository tree, the files are organized as follows:

- `data/`
  - `202601311400.pcap`: Sample of normal baseline backbone traffic (from MAWI Working Group).
  - `thgnn_evaluation_dataset.pcap`: The synthesized evaluation dataset containing uniformly injected Tiny Fragmentation and TCP MSS Clamping attacks.
  - `ground_truth.csv` & `dataset.csv`: Labels and extracted tabular features mapped to the captured frames.
- `graph_builder.py`: Script to extract lightweight packet features (Frame Length, IP Offset, IP MF Flag, TCP MSS, TCP Seq) and construct the Reassembly-Graph.
- `train.py`: Main execution script to train the RG-GNN model, apply the 40% Gaussian feature-noise stress test, and evaluate macro-level metrics (F1-score, PR-AUC).
- `th_utils.py`: Helper functions and utilities for data processing, metrics calculation, and evaluation.
- `CITATION.cff`: Citation information for referencing this project.
- `requirements.txt`: Python dependencies required to run the project.

## Installation
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment. Install the required dependencies using:

```bash
pip install -r requirements.txt
```
## Usage
```bash
python train.py
```
## Note on Ablation Study
As demonstrated in the paper, incorporating temporal (flow-adjacency) edges degrades performance through oversmoothing. Therefore, this final implementation strictly utilizes Reassembly Edges to achieve an F1-score of ~0.98 under heavily imbalanced conditions.