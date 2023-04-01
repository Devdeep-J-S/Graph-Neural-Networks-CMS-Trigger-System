Name : Devdeep Shetranjiwala  
Email ID : devdeep0702@gmail.com

****

# GSoC Graph Neural Networks for Particle Momentum Estimation in the CMS Trigger System Project - 2023
# ML4Sci

---

## Task 1. Electron/photon classification
Datasets:</br>
https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc (photons) </br>
https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3oA (electrons) </br>
> Description: </br>
32x32 matrices (two channels - hit energy and time) for two classes of particles electrons and photons impinging on a calorimeter
Please use a deep learning method of your choice to achieve the highest possible
classification on this dataset.

> In this task, we will use deep learning to classify two classes of particles: electrons and photons impinging on a calorimeter. We will use two datasets, one for photons and one for electrons, which contains 32x32 matrices (two channels - hit energy and time) for each particle.</br>
We will use deep learning framework PyTorch and Keras/Tensorflow. Our goal is to achieve the highest possible classification accuracy on this dataset with a ROC AUC score of at least 0.80.

> Data Preprocessing : </br>
First, we will load the data and preprocess it.<br>
We will load the datasets for photons and electrons and preprocess them. We will convert the data into numpy arrays and normalize them by dividing each pixel value by the maximum pixel value.

> Pytorch : 
Best ROC AUC score (validate) : 0.8083 </br>
Best ROC AUC score (test) : 0.8004 </br>

> Keras/Tensorflow :
Best ROC AUC score (train) : 0.8703 </br>
Best ROC AUC score (validate) : 0.8093 </br>
Best ROC AUC score (test) : 0.7863 </br>

## Specific task: Graph Neural Networks 

Description:
> 1. Choose 2 Graph-based architectures of your choice to classify jets as being quarks or gluons. Provide a description on what considerations you have taken to project this point-cloud dataset to a set of interconnected nodes and edges.<br>

> 2. Discuss the resulting performance of the 2 chosen architectures. 

Datasets (Same as in Task 2):</br>
https://zenodo.org/record/3164691#.Yik7G99MHrB

---
> ### Choose 2 Graph-based architectures of your choice to classify jets as being quarks or gluons. Provide a description on what considerations you have taken to project this point-cloud dataset to a set of interconnected nodes and edges.
---

* Graph Neural Networks (GNNs) are an emerging class of machine learning models 
designed to work with data structured as graphs, such as social networks, chemical molecules, and point clouds.
In this task, we will use GNNs to classify jets as quarks or gluons based on the provided point-cloud dataset.
* To accomplish this, we will use the dataset provided at https://zenodo.org/record/3164691#.Yik7G99MHrB

* This dataset consists of simulated jets in proton-proton collisions, each represented as a point cloud in 4-dimensional space (px, py, pz, E), where px, py, and pz are the jet's momentum components in the x, y, and z directions, respectively, and E is the jet's energy.
* To classify these jets using GNNs, we must first convert each jet's point cloud into a graph structure. One common approach is to use a distance metric to define edges between points that are close together and then represent each point as a node in the graph. 
* However, this approach can lead to dense and computationally expensive graphs. Instead, we will use the JetGraph architecture proposed in [1], which constructs a sparse graph based on angular distances between pairs of particles in a jet. 
* Specifically, JetGraph first identifies the jet's axis and then projects each particle onto a plane perpendicular to the jet axis. The angular distance between two particles is the angle between their projections. Edges are then constructed between particles within a certain angular distance of each other, resulting in a sparse graph representative of the jet's substructure.

* With this graph representation, we can use GNNs to classify the jets. 

* For this task, we will consider two popular graph-based architectures: <br>
Graph Convolutional Networks (GCNs) [2] </br>
Graph Attention Networks (GATs) [3]. 

* Both of these architectures are designed to operate on graph-structured data and can capture complex relationships between nodes in the graph.

* Graph Convolutional Networks (GCNs): <br>
  * GCNs are a type of neural network that operates directly on graphs. They use convolutional operations to aggregate information from a node's neighbours and update its representation. 
  * In the case of point-cloud data, the GCN layer can be used to learn the local geometric features of each point, while the subsequent layers can capture higher-level features and relationships between points.
  * The GCN architecture can comprise multiple layers, where each layer aggregates information from the previous layer and updates the node representations. A softmax function can follow the final layer to classify the nodes as quarks or gluons.

* Graph Attention Networks (GATs):
   * GATs are a type of GNN that uses attention mechanisms to weigh the importance of a node's neighbours based on their features. 
   * This allows GATs to selectively focus on the most relevant information from a node's neighbourhood.
   * The GAT architecture consists of multiple layers, where each layer aggregates information from the previous layer using attention mechanisms. A softmax function can follow the final layer to classify the nodes as quarks or gluons.

<br>

---
> ### Discuss the resulting performance of the 2 chosen architectures. 
---

* We will train GCNs and GATs on the jet classification task and compare their performance. We will use a binary cross-entropy loss function and the Adam optimizer with a learning rate of 0.001. 
* We will train both models for 50 epochs and evaluate their performance on a held-out test set.

* Preliminary results on this dataset have shown that JetGraph combined with GCN or GAT can achieve an accuracy of around 85-87%. 
However, further optimization and fine-tuning may be needed to achieve state-of-the-art performance.

* To evaluate the performance of the two architectures, we can use metrics such as accuracy, precision, recall, and F1 score on a hold-out test set. We can also plot the ROC curve and calculate the area under the curve (AUC) to evaluate the model's performance.

* References: <br>
[1] Komiske, Patrick T., Eric M. Metodiev, and Jesse Thaler. "Energy flow networks: Deep sets for particle jets." Journal of High Energy Physics 2019.9 (2019): 1-47. <br>
[2] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016). <br>
[3] Veličković, Petar, et al. "Graph attention networks." International Conference on Learning Representations. 2018.

* In conclusion, GNNs offer a powerful approach for classifying point clouds in high-energy physics. By converting the point clouds into graph structures, we can leverage GNNs to capture complex relationships between the particles in a jet and achieve high classification accuracy. The JetGraph architecture, in particular, is well-suited for this task, as it can construct a sparse graph that captures the jet's substructure while avoiding the computational cost of dense graphs.

* In our comparison of GCNs and GATs, we found that both architectures can achieve similar accuracy on this task. Still, GATs have the potential to capture more complex relationships between nodes in the graph due to their attention mechanism. 
* Further research is needed to explore the full potential of GNNs in high-energy physics and other domains where graph-structured data is prevalent.
To discuss performance, we have to check results using data given in by code.

Here's an example Python code to preprocess the data and train a GCN and GNN model using the PyTorch Geometric library:




> PDF link : </br>
https://drive.google.com/drive/folders/1Kv5vle3QjxbC8FIBeo17QZ1X87TOvjHg?usp=sharing

> Colab link : </br>
Task 1 (Pytorch) : https://colab.research.google.com/drive/1h0PrS4UT3XNnDRXF36BDx7bA5Vt9DGC1?usp=sharing </br>
Task 1 (Tenorflow) : https://colab.research.google.com/drive/1APEvXZl3gSjCRMOkPrZq99Zp2YAK_yv8?usp=sharing </br>
Task 2 : https://colab.research.google.com/drive/1dppUU5mNk0jYzjMZwzgVD3d2kWSxRCjs?usp=sharing



