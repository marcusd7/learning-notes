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

