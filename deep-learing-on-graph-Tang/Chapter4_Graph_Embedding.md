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

**Mapping Function**：最直接的映射函数即使查找表(look-up table)，通过one-hot编码来为每一个图中的结点显式的指定映射后的低维向量。参数即直接是图中各结点映射后的低维向量。$f(v_i)=u_i=e^T_iW$，其中$e^T_i$是查找表的指示函数（one-hot编码形式），当对应的是图中结点$v_i$的时候，$e_i$中除了第i个元素element为1，其余都为零。即通过Mapping Function可以直接获得图中结点对应的低维向量，如此一来，仅需将$W$中变量设定为参数，既可以通过learning的方法来获得图中结点到低维空间的映射。

**Random Walk**：即对于图中一个给定的结点$v^{(0)}$而言，我们可以随机访问该节点的一个相邻结点。重复访问某结点的相邻结点，直到已经访问了T个结点时停止，这就是一个长度为T的random walk。

标准定义：对于给定connected graph，一个从$v^{(0)}$开始的random walk访问下一结点的概率是$p(v^{(t+1)}|v^{t})=1/d(v^{t}), if\quad v^{(t+1)}\in \mathcal{N}(v^{(t)}), 0 \quad else$。即random walk随机性地访问$v^t$的所有相邻结点，对于$v^t$的所有邻接结点而言，下一次访问的概率都是相同的。即$\mathcal{W}=RW(\mathcal{G},v^{(0)},T)$生成在给定图中，从给定结点开始生成长度为T的random walk。

为了捕获全局的信息，一般对图中每一个结点进行$\gamma$次random walk，即一个给定图的总random walk数为$\gamma\cdot N$。故若两结点之间有邻接关系，这两个结点的tuple$(v^t,v^{t+1})$出现在random walk中的次数也就越多。

这里考虑图中的结点一般会扮演两种角色role: center or context nodes。按照如此定义，我们在每一个条random walk路径中定义$(v_{con},v_{cen})$ tuple，将图中的每一个结点定义为$v_{cen}$并将其余结点设为$v_{con}$，并依次将所有tuple放进从图中提取出来的信息数组中$\mathcal{I}$中。 

<img src="./pics/Chapter4-pic2.png" width="450"/>
