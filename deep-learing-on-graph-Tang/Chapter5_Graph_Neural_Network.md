## Chapter5 Graph Neural Network
### 2022.03.21

GNN可以被看作是图层面的表示学习(representation learning)，对于node-focused task而言，GNN是为了在训练的过程中学习到对于图中结点的更好的表达(Embedding/representation)，而对于graph-focused task而言，GNN可以在训练的过程中更好的学习对于整张图的更好的特征与表达：
1. Node-focused task：在训练的过程中学习对结点更好的表达，learn node features
2. Graph-focused task：在训练的过程中学习对整张图更好的表达/特征，learn graph features

在学习结点特征的过程中，我们希望在学习结点特征的过程中能够同时考虑到结点的特征(node features)和图结构(graph structure)，数学化的表达可以是$\bold{F}^{(of)}=h(\bold{A},\bold{F}^{(if)})$，即仅改变图中结点的特征，而并不改变图中结点的连接（即邻接矩阵，图结构不变），这就是**graph filter**，通常对于node-classification任务而言，使用graph filter就足够了。但是对于graph focused task而言，一般需要进一步提取出整个图结构的特征，因此需要进一步使用**graph pooling**操作，从node features中进一步提取出graph-level feature。

对于**graph filtering**和**graph pooling**：
1. graph filtering: $\bold{F}^{(of)}=h(\bold{A},\bold{F}^{(if)})$，其中$\bold{F}^{(if)}, \bold{F}^{(of)}$分别是经过graph filtering前后输出各结点特征的特征矩阵$\bold{F}^{(if/of)}\in R^{N\times d_{if/of}}$即在经过graph filtering前后结点的特征维度可能会发生变化，但结点数，以及整体的图结构并不会发生变化。对node-focused task而言，graph filtering是足够的。
2. graph pooling: $\bold{A}^{(op)},\bold{F}^{(op)}=h(\bold{A}^{(ip)},\bold{F}^{(ip)})$，其中不仅改变了pooling前后的结点特征，还在网络训练的过程中改变了图结构，直观的改变就是图的邻接矩阵在pooling前后发生了变化，这样便于提取出整个图结构的特征(总结出整个图结构的特征)，对于graph focused task而言更加有效。

### 5.2 General GNN Framework
### 5.2.1 General Framework for Node-focused Tasks

对于Node-focused tasks来说，Graph Filter运算即是可以提取出有效的结点特征，因此仅需要在GNN中堆叠graph filtering层以及activation层。

<img src="./pics/Chapter5-pic1.png" width="450"/>

### 5.2.2 General Framework for Graph-focused Tasks

而对于Graph-focused tasks来说，需要进一步使用Graph Pooling层来从图的结点中进一步提取graph level的特征。因此可以通过堆叠由graph filtering层与activation层交叉堆叠组成的block和pooling层来完成对graph-level features的提取。

<img src="./pics/Chapter5-pic2.png" width="450"/>
<img src="./pics/Chapter5-pic3.png" width="450"/>

### 5.3 Graph Filters

对于graph filter而言，一般有两种思路来设计graph filter：
1. spatial-based filters，空域图滤波器侧重于利用图结构中的信息（如结点之间的邻接信息）来提取更好的特征(feature refining)
2. spectral-based filters，频域图滤波器侧重于通过spectral graph theory来在图频域中提取图中的特征

### 5.3.1 Spectral-based Graph Filters

基于Graph Spectral Theory（即图的Laplacian矩阵，smoothness，图傅里叶变换等方法）来进一步提高图中结点特征的效果。

**Graph Spectral Filtering**：图频域滤波是指通过**调整图中某些频率，来实现去除某些频率分量而保留一部分的频率分量**。因此对于一个给定图结构，首先需要使用Graph Fourier Transform来获得他的图傅里叶参数(Graph Fourier Coefficients)，然后通过调整这些频率分量的权重来在空域中重构整张图(reconstruct the signal in the spatial domain)。

对于一个图中信号$f\in R^{N}$（即假设图中的信号对于每一个结点而言是一个常量scalar），其傅里叶变换参数被定义为$\hat{f}=U^Tf$，其中$U$为从给定图结构的Laplacian矩阵中所提取出来的eigenvector所组成的矩阵。$\hat{f}$中的第$i$个参数对应着图Laplacian矩阵的一个eigenvector，其对于的频率值即为对应的eigenvalue $\lambda_i$。故想要调整图中的各频率分量仅需要对傅里叶变换后的各频率参数进行调整，即$\hat{f}^{'}[i]=\hat{f}[i]\cdot \gamma(\lambda_i)$（**即对于频率为$\lambda_i$的频率分量$\hat{f}[i]$加上一个系数$\gamma(\lambda_i)$来调整一个给定图结构中所有的频率分量的大小**）。矩阵形式$\hat{f}^{'}=\gamma(\Lambda)\cdot \hat{f}=\gamma(\Lambda)\cdot U^Tf$，其中$\Lambda$为给定图拉普拉斯矩阵的特征值的对角矩阵。当调整完图中各频率分量的大小后，既可以使用逆傅里叶变换获得经过过滤后的图信号的值（对应到Node-focused GNN task中即是node features）$f^{'}=U\hat{f}^{'}=U\cdot \gamma(\Lambda)\cdot U^Tf$。由上式可以看作对输入信号/结点特征向量**加上了$U\cdot \gamma(\Lambda)\cdot U^T$操作符**，当$\gamma(\Lambda)$中对应某一频率分量的参数为0的时候，即代表需要在原图中删除该频率分量。整个Graph Filtering Processing可以看作：

<img src="./pics/Chapter5-pic4.png" width="450"/>

其中的$U\cdot \gamma(\Lambda)\cdot U^T$可以看作是graph filtering operator。


