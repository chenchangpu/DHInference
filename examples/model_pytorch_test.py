import torch
import torch.nn as nn
import time
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, n_heads, head_dim]
        q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # [batch_size, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # [batch_size, n_heads, seq_len, head_dim]
        out = torch.matmul(attn, v)
        
        # [batch_size, seq_len, hidden_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.o_proj(out)

class FFN(nn.Module):
    def __init__(self, hidden_dim, expansion_factor):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, ffn_expansion):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, ffn_expansion)
        
    def forward(self, x):
        # Self-attention block
        x = x + self.attn(self.norm1(x))
        # FFN block
        x = x + self.ffn(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_dim, ffn_expansion):
        super().__init__()
        # 保存模型参数
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ffn_expansion = ffn_expansion
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, n_heads, ffn_expansion)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def create_dummy_weights(model_path, n_layers, n_heads, hidden_dim, ffn_expansion):
    """创建与C++版本相同的随机权重"""
    np.random.seed(42)
    weights = []
    
    # 写入模型配置
    weights.extend([n_layers, n_heads, 0, hidden_dim, ffn_expansion])
    
    for _ in range(n_layers):
        # LayerNorm1
        weights.extend(1.0 + np.random.uniform(-0.1, 0.1, hidden_dim) * 0.01)  # gamma
        weights.extend(np.random.uniform(-0.1, 0.1, hidden_dim) * 0.01)        # beta
        
        # Self-attention
        for _ in range(4):  # Q, K, V, O
            weights.extend(np.random.uniform(-0.1, 0.1, hidden_dim * hidden_dim))
            
        # LayerNorm2
        weights.extend(1.0 + np.random.uniform(-0.1, 0.1, hidden_dim) * 0.01)  # gamma
        weights.extend(np.random.uniform(-0.1, 0.1, hidden_dim) * 0.01)        # beta
        
        # FFN
        expanded_dim = hidden_dim * ffn_expansion
        weights.extend(np.random.uniform(-0.1, 0.1, hidden_dim * expanded_dim))
        weights.extend(np.random.uniform(-0.1, 0.1, expanded_dim * hidden_dim))
    
    weights = np.array(weights, dtype=np.float32)
    with open(model_path, 'wb') as f:
        weights.tofile(f)
    print(f"已创建测试模型文件: {model_path}")

def load_weights(model, model_path):
    """加载与C++版本相同的权重到PyTorch模型"""
    with open(model_path, 'rb') as f:
        # 跳过配置参数
        f.seek(5 * 4)  # 跳过5个int32配置参数
        
        # 为每一层加载权重
        for layer in model.layers:
            # LayerNorm1
            gamma = np.fromfile(f, dtype=np.float32, count=model.hidden_dim)
            beta = np.fromfile(f, dtype=np.float32, count=model.hidden_dim)
            layer.norm1.weight.data = torch.from_numpy(gamma)
            layer.norm1.bias.data = torch.from_numpy(beta)
            
            # Self-attention
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                weight = np.fromfile(f, dtype=np.float32, count=model.hidden_dim * model.hidden_dim)
                weight = weight.reshape(model.hidden_dim, model.hidden_dim)
                getattr(layer.attn, name).weight.data = torch.from_numpy(weight)
                
            # LayerNorm2
            gamma = np.fromfile(f, dtype=np.float32, count=model.hidden_dim)
            beta = np.fromfile(f, dtype=np.float32, count=model.hidden_dim)
            layer.norm2.weight.data = torch.from_numpy(gamma)
            layer.norm2.bias.data = torch.from_numpy(beta)
            
            # FFN
            expanded_dim = model.hidden_dim * model.ffn_expansion
            # fc1: hidden_dim -> expanded_dim
            fc1_weight = np.fromfile(f, dtype=np.float32, count=model.hidden_dim * expanded_dim)
            fc1_weight = fc1_weight.reshape(model.hidden_dim, expanded_dim).T
            layer.ffn.fc1.weight.data = torch.from_numpy(fc1_weight)
            
            # fc2: expanded_dim -> hidden_dim
            fc2_weight = np.fromfile(f, dtype=np.float32, count=expanded_dim * model.hidden_dim)
            fc2_weight = fc2_weight.reshape(expanded_dim, model.hidden_dim).T
            layer.ffn.fc2.weight.data = torch.from_numpy(fc2_weight)

def main():
    # 模型配置
    n_layers = 2
    n_heads = 4
    hidden_dim = 1024
    ffn_expansion = 4
    seq_len = 1024
    
    # 创建模型文件
    model_path = "../build/dummy_model_pytorch.bin"
    create_dummy_weights(model_path, n_layers, n_heads, hidden_dim, ffn_expansion)
    
    # 创建模型并加载权重
    model = Model(n_layers, n_heads, hidden_dim, ffn_expansion)
    load_weights(model, model_path)
    
    # 生成随机输入
    np.random.seed(42)
    input_data = np.random.normal(0, 1, (seq_len, hidden_dim)).astype(np.float32)
    input_tensor = torch.from_numpy(input_data).unsqueeze(0)  # 添加batch维度
    
    # 移动到GPU
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        print("使用GPU进行推理")
    else:
        print("使用CPU进行推理")
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
    
    # 计时推理
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        output = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    duration = (end_time - start_time) * 1e6  # 转换为um
    
    # 获取结果
    output = output.squeeze(0)  # 移除batch维度
    if torch.cuda.is_available():
        output = output.cpu()
    output = output.numpy()
    
    # 打印统计信息
    print(f"\n推理时间: {duration:.2f} us")
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