## Chapter6 Graph Neural Network
### 2022.03.27

### Attackers' Capacity(攻击者的能力)
* Evasion Attack:在测试阶段攻击模型，在该种攻击模式下GNN的模型是训练好的，在攻击的过程中不能改变模型的参数
* Poisoning Attack:在模型训练前攻击模型，即在攻击的过程中攻击者可以将poisoning data嵌入到训练数据中，**一般也意味着原模型可以re-train，即在优化问题的角度下可以是一个min-max问题**

### Perturbation Type(干扰类型)
* Modifying Node Features
* Adding or deleting edges
* Injecting Nodes


### Attackers' Goal(攻击者的目标)
* Targeted Attack:目标在于降低模型对图中特定结点的分类准度，使模型对图中若干结点误分类。
* Untargeted Attack:目标在于降低模型对整个图中分类等任务的精确度，不拘泥于几个结点。

### Attackers' Knowledge(攻击者已知知识)
* White-box:攻击者可以获得想要攻击模型的所有信息，包括模型的参数，架构以及训练数据(model architecture, model parameters, training data)，**即在攻击的过程中模型可以使用$f_{GNN}(;\Theta)$**
* Gray-box:攻击者不清楚模型的参数以及架构，但可以获取到训练数据(即可以获取到图结构的邻接矩阵)。由于无法获取到模型的参数以及架构，一般会攻击替代模型(surrogate model)并**一般假设若对替代模型攻击有效则该方法有效**
* Black-Box:攻击者不清楚模型的所有信息，包括模型的参数，架构以及训练数据，仅可以通过问询(query)的方式获得信息，**一般建模成RL问题**


**由于该节内容就是对各论文做解读，主要内容放在https://github.com/marcusd7/Graph-Adversarial-Learning/tree/master/Notes中**

