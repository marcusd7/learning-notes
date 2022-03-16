## Chapter2 Foundation of Graphs
### 2022.03.16

### 2.2 Graph Representations
*Definition 2.1 (Graph): A graph can be denoted as $\mathcal{G}=\{\mathcal{V}. \mathcal{E} \}$, where $\mathcal{V}=\{v_1,v_2,...,v_N\}$ is a set of $N=|V|$ nodes and $\mathcal{E}=\{e_1,e_2,...,e_N\}$ is a set of M edges*

定义 2.2（邻接矩阵）： 邻接矩阵用来描述一个给定图$\mathcal{G}=\{\mathcal{V}. \mathcal{E} \}$各结点之间的连接关系，用$A \in \{0,1\}^{N \times N}$来表示，其中$A_{i,j}$为1，即表示结点$i,j$之间有边相连。

由定义可知，邻接矩阵一定是实对称的矩阵，即$A_{i,j}=A_{j,i}$，因此也是正定矩阵。

### 2.3 图的性质与描述方法(Properties and measures)

定义2.4 （度）： 在一个给定图$\mathcal{G}=\{\mathcal{V}. \mathcal{E} \}$中，结点$v_i$的度(degree)为与结点$v_i$通过某一边相连接的结点的个数，数学表达式为：$d(v_i)=\sum_{v_j \in \mathcal{V}} \mathbb{1}_{\mathcal{E}}(\{v_i, v_j\})$。

同样，结点的度也可以通过邻接矩阵来定义：$d(v_i) = \sum_jA_{i,j}$

定义2.5（相邻点）：$N(v_i)$由所有与结点$v_i$相邻接的结点组成，故$d(v_i)=|N(v_i)|$。

由上述结果可以推出：$\sum_ {v_i\in\mathcal{V}}d(v_i) = 2|\mathcal{E}|$，即图中所有结点的度之和为图中所有边的数量的两倍。同样有，邻接矩阵中非零数的数量也是边的两倍（因为其实图中所有结点度的和即为邻接矩阵中非零数的数量）

定义2.10/2.11/2.12（walk, path, trail）：图中的walk定义为从某结点$u$到某结点$v$之间的一段结点，边的集合。path则是在walk中，结点是不重复的，而trail是指在该walk中，边是不重复的

推论2.14：即定义邻接矩阵的n次方$A^n$为如下：$A^n$中的元素$A^n_{i,j}$表示为长度为n的 $i-j walk$的数量是多少。（walk length的定义是walk中结点的个数）

定义2.15（子图）：子图是原图的一部分，子图中的结点属于原图的结点集，子图的边属于原图的边集，且子图中的所有结点必须包含其所有边所涉及到的结点。

定义2.17（connected component，连通分量）：连通分量是原图的一个子图，且连通分量内任意两结点之间至少存在一条path，而**连通分量内的结点**与 **（原图/联通分量）** 内的结点没有关联

定义2.18（连通图）：连通图仅有一个连通分量

定义2.19（最短路径，shorest path）：两节点之间的最短路径，有可能不是唯一的，路径的长度由路径上的结点数量来表示

定义2.22（图的直径，Diameter）：一个图的Diameter意为其图中最大的最短路径。

### 2.3.3 Centrality 

通常Centrality是用来描述一个结点的重要性的，本节介绍了不同的方法来衡量/描述一个结点的重要性。

1. Degree Centrality，即通过一个结点的度来描述其重要性，结点的度越大，意味着该结点越重要，$c_d(v_i) = d(v_i) = \sum_{j=1}^N A_{i,j}$
2. Eigenvector Centrality，在Degree Centrality中假设与某一结点相邻接的其他节点重要性一致，而在Eigenvector Centrality中结点不同的临界点有不同的重要性，于是有$c_e(v_i)=\frac{1}{\lambda}\sum_{j=1}^N A_{i,j}\cdot c_e(v_j)$，该方程可以被重写成$c_e=\frac{1}{\lambda}A\cdot c_e, \lambda\cdot c_e = A\cdot c_e$，即所计算的Eigenvector Centrality是矩阵A的Eigenvector，由于**对于一个所有元素都是正数的实方阵来说，一定有一个唯一的最大特征值且其对应的特征向量中的元素都是正数**，于是对于Eigenvector Centrality而言仅需求出矩阵A的最大特征值，其所对应得特征向量即为$c_e$
3. Katz Centrality，在2的基础上为结点自身增加了一定量的值，$c_k=\alpha A c_k+\beta, (I-\alpha \cdot A)c_k=\beta, c_k = (I-\alpha \cdot A)^{-1}\beta$
4. Betweenness Centrality，1-3所述的重要性都是基于邻接结点而言的。而另一种来描述结点重要性的方法是通过计算结点是否在原图中处于一个重要的位置，故若有很多路径经过一个结点，则该节点是一个较为重要的位置。于是便可以通过如下方式来定义:$c_b(v_i)=\sum_{v_s\neq v_i\neq v_t}\frac{\sigma_{st}(v_i)}{\sigma_{st}}$，该方法是用来衡量图中的所有最短路径中，穿过点$v_i$的占多少，但如此一来该指标会随着图的规模增大而增大，即他不是noramlized，通过增加标准化参数（图中可能的最大betweenness scores）来使指标标准化，$c_nb(v_i) = \frac{2c_b(v_i)}{(N-1)(N-2)}$

### 2.4 Spectral Graph Theory

通过分析图像的Laplacian矩阵的Eigenvector和Eigenvalue来做Spectral上的分析

### 2.4.1 Laplacian Matrix

定义2.28（拉普拉斯矩阵）：对于一个给定图$\mathcal{G}=\{\mathcal{V}. \mathcal{E} \}$而言，其拉普拉斯矩阵为$L=D-A$，其中D为对角矩阵，元素为各结点的度$D=diag(d(v_1),...,d(v_N))$，而$A$为邻接矩阵。

定义2.29（标准拉普拉斯矩阵）：对于给定图而言，标准化的拉普拉斯矩阵可以表示为$L=D^{-1/2}(D-A)D^{-1/2}=I-D^{-1/2}AD^{-1/2}$。

令f表示为与结点相关的值，如$f[i]$是结点$v_i$的值，对于$h=Lf=(D-A)f=Df-Af$，其中$h$中的第i个元素可以被表示为$h[i]=\sum_{v_j \in \mathcal{N}(v_i)}(f[i]-f[j])$，而$f^TLf=\frac{1}{2}\sum_{v_i\in \mathcal{V}}\sum_{v_j\in \mathcal{N}(v_i)}(f[i]-f[j])^2$，其中$h=Lf$可以表示为代表f之间各值的差值，而$f^TLf$为各节点值差值的平方和。

推论2.31： 拉普拉斯矩阵中的零特征值数量等于图中的连通分量数

### 2.5 Graph Signal Processing

传统信号处理是将信号在时间域和频率域中分析，而图信号处理则是将信号在空域和频域中分析，其中实现频率域则是通过拉普拉斯矩阵。

对于一个图而言，$f^TLf$表示smoothness，即为图中结点与其相邻接的结点之间值（这里即为$f$中的值）变化的剧烈程度。通过$f^TLf$的实际定义不难发现，其衡量的就是各结点与相邻接结点差值得平方和。当$f^TLf$大时，我们认为图中结点的不平滑，因为相邻接结点之间差值过大，而相反，当小时，认为图时平滑的。

### 2.5.1 Graph Fourier Transform

图傅里叶变换借鉴了信号中的傅里叶变化，图傅里叶变化可以被表示为$\hat{f}[l]=<f,u_l>=\sum_{i=1}^N f[i]u_l[i]$，故对于图拉普拉斯矩阵的eigenvalue而言，越大的eigenvalue即代表变化越大，即频率越高。相反，频率越小。

### 2.6 Complex Graph

定义2.35 Heterogeneous Graph（异构图）：即在原图的基础上，加上了不同的结点和不同的边都有不同的类型。

定义2.36 Bipartite Graph（二分图）

定义2.37 Multi-dimensional Graph（多维图）：即图中结点有多重连接关系

定义2.38 Signed Graph（符号图）：图中链接关系有正有负，正关系可能是follow，而负关系可能是unfollow之类

定义2.39 Hyper Graph

定义2.40 Dynamic Graph（动态图）：引入时间戳

### 2.7 Computational Tasks on Graph

大多数图上的计算任务分为两类：

1. node-focused: node classification, link prediction
2. graph-foucsed: graph classification 
