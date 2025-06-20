func: Layer
input: X
output: Y
Layer(X):
	norm1 = layer_norm_->forward(X)
	attn_out = attention_->forward(norm1)
	residual1 = X + attn_out
	norm2 = layer_norm_->forward(residual1)
	ffn_out = ffn_->forward(norm2)      // ffn激活用的是relu
	Y = residual1 + ffn_out


实验记录：
在模型较小（seq_len=1024, hidden_dim=128或1024）情况下，
	开启cublas不如直接调用手写的sgemm算子，
	开启oneflow或手写算子（softmax, layernorm, elementwise等）差距不大
	backend性能略低于pytorch

在模型较大（seq_len=8192, hidden_dim=4096）情况下，
	开启cublas明显优于手写的sgemm
	backend性能略优于pytorch