打算从底层用 C++/CUDA 实现较为简单但又高性能的demo/tutorial级别的大模型推理系统。

一、总体目标和设计思路
1. 仅支持推理（inference only）
2. 输入为 token 序列，输出为 logits，暂不考虑token序列的分词处理（即假设输入token为已经分词处理好的embedding序列）
3. 先实现cpu版本，再扩展至cuda版本
4. 不依赖于第三方库（如libtorch，cuBlas等），从头实现
5. 先保证正确性和易用性、可读性，性能可以暂时忽略
6. 支持的模型暂时为decoder-only的经典结构

二、整体模块构建
1. 规定模型文件格式，构建模型结构并加载模型参数（权重）
2. 核心推理计算模块，主要包含LayerNorm，Linear / MatMul，Multi-head Self-Attention，FeedForward
3. 实现CUDA算子，支持多后端（CPU/GPU）推理
4. 支持autoregressive推理、KV-Cache缓存加速和batch推理


初步实现为2周，后续完善1个月。
两周初步实现计划
第一周：基础框架与CPU实现
日1：项目结构设计，创建基本目录，定义模型格式
日2：实现模型参数加载功能
日3：实现LayerNorm和Linear/MatMul模块（CPU版本）
日4：实现Multi-head Self-Attention模块（CPU版本）
日5：实现FeedForward模块（CPU版本）
日6：连接各模块，实现完整模型前向推理
日7：测试CPU版本推理功能，修复bug
第二周：CUDA实现与优化
日8：设计CUDA基础设施，实现基本矩阵操作
日9：实现LayerNorm和Linear/MatMul的CUDA版本
日10：实现Multi-head Self-Attention的CUDA版本
日11：实现FeedForward的CUDA版本
日12：连接模块，完成CUDA版推理流程
日13：实现KV-Cache缓存机制
日14：测试CUDA版本，修复bug，完成初步实现
一个月完善计划
第三周：功能完善与优化
日15-16：实现自回归推理功能
日17-18：添加batch推理支持
日19-21：优化内存使用和计算性能
第四周：高级功能与兼容性
日22-23：添加模型量化支持
日24-25：实现不同规模模型的适配
日26-28：完善文档和示例代码
第五周：测试与收尾
日29-30：全面测试、性能分析和基准测试
日31-32：bug修复
日33-34：最终润色和文档完善
这个计划覆盖了README中提到的所有要点，从基本功能到完善优化都有合理安排。

Day1 
模型格式：decoder-only，且假设输入的tokens序列已经embedding和padding成
        长度length=1024，维度input_dim=128，输入X_1024_x_128
文件格式：
内存/变量                               类型               字节Byte
n_Layers: 层数                          int                4
n_heads:  头的个数                      int                4
act_func: 激活函数                      int                4
layer_hidden_dim: 假设Q,K,V一样         int                4
ffn_expansion: FFN层扩张系数            int                 4

----(layer1参数/张量)----
----(layer1 Wq_input_dim_x_layer_hidden_dim)----
Wq1_1,1                                 float32             4
Wq1_1,2                                 float32             4
...
Wq1_input_dim,layer_hidden_dim          float32             4
----(layer1 Wk_input_dim_x_layer_hidden_dim)----
Wk1_1,1                                 float32             4
Wk1_1,2                                 float32             4
...
Wk1_input_dim,layer_hidden_dim          float32             4
----(layer1 Wv_input_dim_x_layer_hidden_dim)----
Wv1_1,1                                 float32             4
Wv1_1,2                                 float32             4
...
Wv1_input_dim,layer_hidden_dim          float32             4
----(layer1 Wo_layer_hidden_dim_x_layer_hidden_dim)----
Wo1_1,1                                 float32             4
Wo1_1,2                                 float32             4
...
Wo1_input_dim,layer_hidden_dim         float32             4
----(layer1 layernorm: gamma, beta)----
gamma1_1                                float32             4
...
gamma1_layer_hidden_dim                 float32             4
beta1_1                                 float32             4
...
beta1_layer_hidden_dim                  float32             4
----(layer1 ffn参数)----
----(W1_layer_hidden_dim_x_n1, scale layer_hidden_dim to n1 = ffn_expansion * layer_hidden_dim)----
W1_1,1                                  float32             4
W1_1,2                                  float32             4
...
W1_layer_hidden_dim,n1                  float32             4
----(W2_n1_x_layer_hidden_dim, rescale n1 to layer_hidden_dim)----
W2_1,1                                  float32             4
W2_1,2                                  float32             4
...
W2_n1,layer_hidden_dim                  float32             4
----(layer1 layernorm: gamma, beta)----
gamma1_1                                float32             4
...
gamma1_layer_hidden_dim                 float32             4
beta1_1                                 float32             4
...
beta1_layer_hidden_dim                  float32             4

----(layer2参数/张量)----
----(layer2 Wq_layer_hidden_dim_x_layer_hidden_dim)----
...


----(layer{n_Layers}参数/张量)----
----(layer{n_Layers} Wq{n_Layers-1}_{n_Layers-1_hidden_dim}_x_128)----
...