# Relational Graph Attention from Scratch

This repository provides Relational (heterogeneous) Graph Attention (***RGAT***) operator implementation from scratch. This implementation is, as the name suggests, meant only for relational (simple/property/attributed) graphs. Here, two schemes have been implemented to compute attention logits $\mathbf{a}^{(r)}_{i,j}$ for each relation type r:-

***Additive attention***
```math
\mathbf{a}^{(r)}_{i,j} = \mathrm{LeakyReLU}(\mathbf{q}^{(r)}_i +
	        \mathbf{k}^{(r)}_j)

```
or ***multiplicative attention***
```math
\mathbf{a}^{(r)}_{i,j} = \mathbf{q}^{(r)}_i \cdot \mathbf{k}^{(r)}_j
```
Two different attention mechanisms have also been provided:-
-	***Within-relation attention mechanism***
```math
\alpha^{(r)}_{i,j} =
	        \frac{\exp(\mathbf{a}^{(r)}_{i,j})}
	        {\sum_{k \in \mathcal{N}_r(i)} \exp(\mathbf{a}^{(r)}_{i,k})}

```
-	***Across-relation attention mechanism***
```math
\alpha^{(r)}_{i,j} =
	        \frac{\exp(\mathbf{a}^{(r)}_{i,j})}
	        {\sum_{r^{\prime} \in \mathcal{R}}
	        \sum_{k \in \mathcal{N}_{r^{\prime}}(i)}
	        \exp(\mathbf{a}^{(r^{\prime})}_{i,k})}

```
To ensure better discriminative power for RGATs, the following options have also been made available:-
-	***additive:***  
```math
\mathbf{x}^{{\prime}(r)}_i =
	        \sum_{j \in \mathcal{N}_r(i)}
	        \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j + \mathcal{W} \odot
	        \sum_{j \in \mathcal{N}_r(i)} \mathbf{x}^{(r)}_j
```
-	***scaled:*** 
```math
\mathbf{x}^{{\prime}(r)}_i =
	        \psi(|\mathcal{N}_r(i)|) \odot
	        \sum_{j \in \mathcal{N}_r(i)} \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j
```
-	***f-additive:*** 
```math
\mathbf{x}^{{\prime}(r)}_i =
	        \sum_{j \in \mathcal{N}_r(i)}
	        (\alpha^{(r)}_{i,j} + 1) \cdot \mathbf{x}^{(r)}_j
```
-	***f-scaled:*** 
```math
\mathbf{x}^{{\prime}(r)}_i =
	        |\mathcal{N}_r(i)| \odot \sum_{j \in \mathcal{N}_r(i)}
	        \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j
```
More in-depth information about this implementation is available on [***PyTorch Geometric Official Website***](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGATConv.html#torch_geometric.nn.conv.RGATConv).
## Requirements
-	`PyTorch`
-	`PyTorch Geometric`
## Usage
### Data
Though the `example.py` file contains the path to one of the relational entities graphs (`AIFB`), this implementation works for other heterogeneous graph datasets such as `MUTAG`, `BGS`, `AM`, etc. The `AIFB` dataset contains no. of nodes (`8285`), edges (`58086`), and classes (`4`).
### Training and Testing
To train and test RGATs on heterogeneous graphs, run `example.py`, and this file, after every epoch, prints `train` as well `test` accuracies.
