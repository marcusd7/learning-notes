## Chapter4 Graph Embedding
### 2022.03.17

### 4.1 Introduction

Graph Embedding尝试通过映射函数Mapping Function将一个给定图$\mathcal{G}=\{\mathcal{V},\mathcal{E}\}$中的结点映射到更低维度的空间中。例如可以将给定图中的某一结点映射成一个$d$维的向量，即$f:\mathcal{V}\mapsto R^{N\times d}$，其中$N=|\mathcal{V}|$，而$d$为单个结点通过映射函数映射到低维空间中的向量。**在将图像中的结点映射到低维空间的同时需要在低维空间中也保留有原图中的信息**，要保存什么信息，如何保存这种信息，这便引出了Graph Embedding中最重要的两个问题

在Graph Embedding中最重要的两个问题即是：
1. 要保存什么信息(what infomation to preserve?)
2. 如何保存该信息(how to preserve?)

<img src="./pics/Chapter4-pic1.png" width="450"/>

大致的Graph Embedding框架中有四个重要的组成部分：
* Mapping Function：映射函数将图中的结点映射到Embedding Domain
* Information Extractor：从原图中提取出想要保存到Embedding Domain中的重要信息
* Reconstructor：从映射后的域Embedding Domain中提取出信息，保证原图中的信息没有被破坏（通过Objective Function判断/保存）
* Objective Function：通过Objective Function来学习参数

### 4.2 Graph Embedding on Simple Graph

### 4.2.1 Preserving Node Co-occurrence

Node Co-occurrence即为结点与结点之间的链接关系，若在Graph Domain中结点$v_i,v_j$是相邻接的话，则在映射之后的Embedding Domain中，我们希望两节点映射后得到的向量之间的距离（distance）也是相近的。

**Mapping Function**：最直接的映射函数即使查找表(look-up table)，通过one-hot编码来为每一个图中的结点显式的指定映射后的低维向量。参数即直接是图中各结点映射后的低维向量。$f(v_i)=u_i=$
