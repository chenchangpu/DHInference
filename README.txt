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