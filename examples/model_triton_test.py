import torch
import triton
import triton.language as tl
import numpy as np
import time

@triton.jit
def layernorm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_b, stride_x_m, stride_x_n,
    stride_g, stride_b,
    M, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 每个程序处理一行
    row_idx = pid
    if row_idx >= M:
        return
        
    # 计算这一行的均值和方差
    mean = 0.0
    mean_sq = 0.0
    
    for start_n in range(0, N, BLOCK_SIZE):
        n = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n < N
        
        x = tl.load(x_ptr + row_idx * stride_x_m + n * stride_x_n, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
        mean_sq += tl.sum(x * x, axis=0)
    
    mean = mean / N
    mean_sq = mean_sq / N
    var = mean_sq - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # 应用LayerNorm
    for start_n in range(0, N, BLOCK_SIZE):
        n = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n < N
        
        x = tl.load(x_ptr + row_idx * stride_x_m + n * stride_x_n, mask=mask)
        gamma = tl.load(gamma_ptr + n * stride_g, mask=mask)
        beta = tl.load(beta_ptr + n * stride_b, mask=mask)
        
        y = gamma * (x - mean) * rstd + beta
        tl.store(out_ptr + row_idx * stride_x_m + n * stride_x_n, y, mask=mask)

@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_k,
    stride_k_b, stride_k_h, stride_k_n, stride_k_k,
    stride_v_b, stride_v_h, stride_v_n, stride_v_k,
    stride_o_b, stride_o_h, stride_o_m, stride_o_k,
    B, H, M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_h = H
    
    # 解码程序ID
    pid_h = pid // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # 计算这个程序负责的区块
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算注意力得分
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for start_k in range(0, K, BLOCK_SIZE_K):
        k_offs = start_k + offs_k
        
        # 加载Q和K
        q = tl.load(q_ptr + offs_m[:, None] * stride_q_m + k_offs[None, :] * stride_q_k, 
                   mask=(offs_m[:, None] < M) & (k_offs[None, :] < K))
        k = tl.load(k_ptr + offs_n[:, None] * stride_k_n + k_offs[None, :] * stride_k_k,
                   mask=(offs_n[:, None] < N) & (k_offs[None, :] < K))
        
        # 计算QK^T
        acc += tl.dot(q, tl.trans(k))
    
    # 应用scale和softmax
    acc = acc * (1.0 / tl.sqrt(float(K)))
    acc = tl.softmax(acc, axis=1)
    
    # 计算注意力输出
    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for start_k in range(0, K, BLOCK_SIZE_K):
        k_offs = start_k + offs_k
        
        # 加载V
        v = tl.load(v_ptr + offs_n[:, None] * stride_v_n + k_offs[None, :] * stride_v_k,
                   mask=(offs_n[:, None] < N) & (k_offs[None, :] < K))
        
        # 计算attention @ V
        out += tl.dot(acc, v)
    
    # 存储结果
    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K
    tl.store(out_ptr + offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k,
             out, mask=m_mask & k_mask)

@triton.jit
def ffn_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 每个程序处理一行
    row_idx = pid
    if row_idx >= M:
        return
    
    # 计算第一个线性层
    hidden = tl.zeros((K,), dtype=tl.float32)
    
    for start_n in range(0, N, BLOCK_SIZE):
        n = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n < N
        
        x = tl.load(x_ptr + row_idx * N + n, mask=mask, other=0.0)
        for k in range(K):
            w = tl.load(w1_ptr + k * N + n, mask=mask, other=0.0)
            hidden[k] += tl.sum(x * w, axis=0)
    
    # 应用ReLU
    hidden = tl.maximum(hidden, 0.0)
    
    # 计算第二个线性层
    out = tl.zeros((N,), dtype=tl.float32)
    
    for start_k in range(0, K, BLOCK_SIZE):
        k = start_k + tl.arange(0, BLOCK_SIZE)
        mask = k < K
        
        h = tl.load(hidden + k, mask=mask, other=0.0)
        for n in range(N):
            w = tl.load(w2_ptr + n * K + k, mask=mask, other=0.0)
            out[n] += tl.sum(h * w, axis=0)
    
    # 存储结果
    for start_n in range(0, N, BLOCK_SIZE):
        n = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n < N
        tl.store(out_ptr + row_idx * N + n, out, mask=mask)

class TritonModel:
    def __init__(self, n_layers, n_heads, hidden_dim, ffn_expansion):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ffn_expansion = ffn_expansion
        self.head_dim = hidden_dim // n_heads
        
        # 为每一层分配参数
        self.params = []
        for _ in range(n_layers):
            layer_params = {
                # LayerNorm1
                'norm1_gamma': torch.ones(hidden_dim, device='cuda'),
                'norm1_beta': torch.zeros(hidden_dim, device='cuda'),
                
                # Self-attention
                'q_weight': torch.randn(hidden_dim, hidden_dim, device='cuda') * 0.02,
                'k_weight': torch.randn(hidden_dim, hidden_dim, device='cuda') * 0.02,
                'v_weight': torch.randn(hidden_dim, hidden_dim, device='cuda') * 0.02,
                'o_weight': torch.randn(hidden_dim, hidden_dim, device='cuda') * 0.02,
                
                # LayerNorm2
                'norm2_gamma': torch.ones(hidden_dim, device='cuda'),
                'norm2_beta': torch.zeros(hidden_dim, device='cuda'),
                
                # FFN
                'ffn_w1': torch.randn(hidden_dim * ffn_expansion, hidden_dim, device='cuda') * 0.02,
                'ffn_w2': torch.randn(hidden_dim, hidden_dim * ffn_expansion, device='cuda') * 0.02,
            }
            self.params.append(layer_params)
    
    def load_weights(self, model_path):
        """从文件加载权重"""
        with open(model_path, 'rb') as f:
            # 跳过配置参数
            f.seek(5 * 4)
            
            for layer_params in self.params:
                # LayerNorm1
                gamma = np.fromfile(f, dtype=np.float32, count=self.hidden_dim)
                beta = np.fromfile(f, dtype=np.float32, count=self.hidden_dim)
                layer_params['norm1_gamma'] = torch.from_numpy(gamma).cuda()
                layer_params['norm1_beta'] = torch.from_numpy(beta).cuda()
                
                # Self-attention
                for name in ['q_weight', 'k_weight', 'v_weight', 'o_weight']:
                    weight = np.fromfile(f, dtype=np.float32, count=self.hidden_dim * self.hidden_dim)
                    weight = weight.reshape(self.hidden_dim, self.hidden_dim)
                    layer_params[name] = torch.from_numpy(weight).cuda()
                
                # LayerNorm2
                gamma = np.fromfile(f, dtype=np.float32, count=self.hidden_dim)
                beta = np.fromfile(f, dtype=np.float32, count=self.hidden_dim)
                layer_params['norm2_gamma'] = torch.from_numpy(gamma).cuda()
                layer_params['norm2_beta'] = torch.from_numpy(beta).cuda()
                
                # FFN
                expanded_dim = self.hidden_dim * self.ffn_expansion
                w1 = np.fromfile(f, dtype=np.float32, count=self.hidden_dim * expanded_dim)
                w2 = np.fromfile(f, dtype=np.float32, count=expanded_dim * self.hidden_dim)
                
                w1 = w1.reshape(expanded_dim, self.hidden_dim)
                w2 = w2.reshape(self.hidden_dim, expanded_dim)
                
                layer_params['ffn_w1'] = torch.from_numpy(w1).cuda()
                layer_params['ffn_w2'] = torch.from_numpy(w2).cuda()
    
    def forward(self, x):
        # 确保输入在GPU上
        if not x.is_cuda:
            x = x.cuda()
        
        for layer_params in self.params:
            # === Self-attention block ===
            # LayerNorm1
            norm1_out = torch.empty_like(x)
            grid = (x.shape[0],)
            layernorm_kernel[grid](
                x, layer_params['norm1_gamma'], layer_params['norm1_beta'], norm1_out,
                x.stride(0), x.stride(0), 1,
                1, 1,
                x.shape[0], x.shape[1],
                BLOCK_SIZE=128
            )
            
            # Self-attention
            q = torch.empty_like(x)
            k = torch.empty_like(x)
            v = torch.empty_like(x)
            attn_out = torch.empty_like(x)
            
            # 计算Q, K, V
            grid = (x.shape[0],)
            for proj_out, weight in [(q, layer_params['q_weight']),
                                   (k, layer_params['k_weight']),
                                   (v, layer_params['v_weight'])]:
                ffn_kernel[grid](
                    norm1_out, weight, None, proj_out,
                    x.shape[0], x.shape[1], x.shape[1],
                    BLOCK_SIZE=128
                )
            
            # 计算注意力
            grid = ((x.shape[0] * self.n_heads * x.shape[0] + 255) // 256,)
            attention_kernel[grid](
                q, k, v, attn_out,
                0, 0, x.shape[1], 1,
                0, 0, x.shape[1], 1,
                0, 0, x.shape[1], 1,
                0, 0, x.shape[1], 1,
                1, self.n_heads, x.shape[0], x.shape[0], self.head_dim,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32
            )
            
            # 投影回输出维度
            proj_out = torch.empty_like(x)
            grid = (x.shape[0],)
            ffn_kernel[grid](
                attn_out, layer_params['o_weight'], None, proj_out,
                x.shape[0], x.shape[1], x.shape[1],
                BLOCK_SIZE=128
            )
            
            # 残差连接
            x = x + proj_out
            
            # === FFN block ===
            # LayerNorm2
            norm2_out = torch.empty_like(x)
            grid = (x.shape[0],)
            layernorm_kernel[grid](
                x, layer_params['norm2_gamma'], layer_params['norm2_beta'], norm2_out,
                x.stride(0), x.stride(0), 1,
                1, 1,
                x.shape[0], x.shape[1],
                BLOCK_SIZE=128
            )
            
            # FFN
            ffn_hidden = torch.empty(x.shape[0], self.hidden_dim * self.ffn_expansion, device='cuda')
            ffn_out = torch.empty_like(x)
            
            grid = (x.shape[0],)
            ffn_kernel[grid](
                norm2_out, layer_params['ffn_w1'], layer_params['ffn_w2'], ffn_out,
                x.shape[0], x.shape[1], self.hidden_dim * self.ffn_expansion,
                BLOCK_SIZE=128
            )
            
            # 残差连接
            x = x + ffn_out
        
        return x

def main():
    # 模型配置
    n_layers = 2
    n_heads = 4
    hidden_dim = 128
    ffn_expansion = 4
    seq_len = 1024
    
    # 创建模型并加载权重
    model = TritonModel(n_layers, n_heads, hidden_dim, ffn_expansion)
    model_path = "dummy_model_pytorch.bin"  # 使用与PyTorch相同的权重文件
    model.load_weights(model_path)
    
    # 生成随机输入
    np.random.seed(42)
    input_data = np.random.normal(0, 1, (seq_len, hidden_dim)).astype(np.float32)
    input_tensor = torch.from_numpy(input_data).cuda()
    
    print("使用Triton进行推理")
    
    # 预热
    for _ in range(10):
        model.forward(input_tensor)
    
    # 计时推理
    torch.cuda.synchronize()
    start_time = time.time()
    
    output = model.forward(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    duration = (end_time - start_time) * 1000  # 转换为毫秒
    
    # 获取结果
    output = output.cpu().numpy()
    
    # 打印统计信息
    print(f"\n推理时间: {duration:.2f} ms")
    print(f"输入维度: [{seq_len} x {hidden_dim}]")
    print(f"输出维度: [{seq_len} x {hidden_dim}]")
    
    print("\n输出统计信息：")
    print(f"  最大值: {output.max():.6f}")
    print(f"  最小值: {output.min():.6f}")
    print(f"  平均值: {output.mean():.6f}")
    
    # 打印每个序列位置的统计信息
    print("\n每个序列位置的统计信息：")
    for i in range(0, seq_len, seq_len//10):
        pos_data = output[i]
        print(f"  位置 {i}:")
        print(f"    最大值: {pos_data.max():.6f}")
        print(f"    最小值: {pos_data.min():.6f}")
        print(f"    平均值: {pos_data.mean():.6f}")

if __name__ == "__main__":
    main() 