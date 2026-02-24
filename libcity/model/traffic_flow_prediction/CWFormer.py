import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

# ============================================================================
# 【Dependency Check】
# ============================================================================
import pywt
try:
    import ptwt # GPU-accelerated Discrete Wavelet Transform
except ImportError:
    raise ImportError(
        "Please install 'ptwt' for GPU-accelerated Wavelet Transform.\n"
        "Command: pip install ptwt"
    )

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

# ============================================================================
# 【Innovation 1: Asymmetric Causal Graph Generator】(Structure)
#   - Story: 交通流是定向传播的 (Source -> Target)，而非对称关联。
#   - Change: 使用 dict_source 和 dict_target 分离编码，生成有向图。
# ============================================================================
class CausalGraphGenerator(nn.Module):
    def __init__(self, num_nodes, input_dim, dict_dim=64, sparsity_threshold=0.05):
        super().__init__()
        self.num_nodes = num_nodes
        self.dict_dim = dict_dim
        self.threshold = sparsity_threshold

        self.feature_proj = nn.Linear(input_dim, dict_dim)
        
        # 定义源字典和目标字典，打破对称性
        self.dict_source = nn.Parameter(torch.randn(dict_dim, dict_dim))
        self.dict_target = nn.Parameter(torch.randn(dict_dim, dict_dim))
        init.xavier_uniform_(self.dict_source)
        init.xavier_uniform_(self.dict_target)

    def forward(self, x):
        # x: [B, T, N, D] -> [B, N, D] (Pooling over time to get static node features for this batch)
        x_pooled = x.mean(dim=1) 
        x_hidden = self.feature_proj(x_pooled) # [B, N, dict_dim]

        # 编码: 分别投影到 Source 空间和 Target 空间
        # code_s: 节点作为“影响源”时的特征
        # code_t: 节点作为“受影响者”时的特征
        code_s = torch.relu(torch.matmul(x_hidden, self.dict_source.t())) 
        code_t = torch.relu(torch.matmul(x_hidden, self.dict_target.t())) 
        
        # 重构 Loss (可选，仅约束源特征能够还原原始信息)
        recon = torch.matmul(code_s, self.dict_source)
        recon_loss = F.mse_loss(recon, x_hidden)
        
        # 生成非对称邻接矩阵: A_ij = Source_i * Target_j
        # [B, N, D] @ [B, D, N] -> [B, N, N]
        adj = torch.matmul(code_s, code_t.transpose(1, 2))
        
        # 归一化 (Min-Max per batch to stabilize attention)
        adj_max = adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6
        adj = adj / adj_max
        
        # 稀疏化 (Hard Thresholding)
        mask = torch.relu(adj - self.threshold)
        adj = mask * torch.sign(adj) # 保留正负相关性，但在Attention里我们通常只看强度
        
        # ================= 插入开始 =================
        import numpy as np
        import os
        # 只要存一次就行了，避免刷屏
        if not hasattr(self, 'has_saved_matrix'):
            # 1. 提取 batch 里第一个样本的矩阵 [N, N]
            matrix_val = adj[0].detach().cpu().numpy()

            # 2. 保存到当前根目录，名字叫 learned_matrix.npy
            save_path = 'learned_matrix.npy' 
            np.save(save_path, matrix_val)

            print(f"\n{'='*40}")
            print(f">>> 🎯 成功！学习到的邻接矩阵已保存为: {save_path}")
            print(f">>> 矩阵形状: {matrix_val.shape}")
            print(f"{'='*40}\n")

            self.has_saved_matrix = True
        # ================= 插入结束 =================

        # return adj

        return adj, recon_loss

# ============================================================================
# 【Innovation 2: Wavelet Enhancement Module】(Representation)
#   - Story: 傅里叶变换有全局性缺陷，小波能精准定位时域上的高频突变 (Extreme Values)。
# ============================================================================
class WaveletEnhancement(nn.Module):
    def __init__(self, embed_dim, wavelet_name='db4', level=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.wavelet_name = wavelet_name
        self.level = level
        self.wavelet = pywt.Wavelet(wavelet_name)

        # 权重参数: 针对不同频率分量给予自适应权重
        # level分解得到 level+1 个分量
        self.component_weights = nn.Parameter(
            torch.ones(level + 1, 1, 1, embed_dim, 1, dtype=torch.float32)
        )
        init.xavier_uniform_(self.component_weights)
        
        self.project = nn.Linear(embed_dim, embed_dim)
        init.xavier_uniform_(self.project.weight)

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        
        # 1. Reshape for ptwt: [Batch, Time] -> [B*N*D, T]
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, T)
        
        # 2. Forward DWT (GPU accelerated)
        coeffs_list = ptwt.wavedec(x_reshaped, self.wavelet, level=self.level, mode='zero')
        
        # 3. Learnable Weighting via Frequency
        enhanced_coeffs_list = []
        for i, coeff in enumerate(coeffs_list):
            T_scale = coeff.shape[-1]
            # 还原维度以便广播权重: [B, N, D, T_scale]
            coeff_expanded = coeff.reshape(B, N, D, T_scale)
            
            # 权重: [1, 1, D, 1] -> 广播到 time 和 node
            weight = self.component_weights[i].expand(-1, -1, -1, T_scale)
            
            enhanced_coeff = coeff_expanded * weight
            enhanced_coeffs_list.append(enhanced_coeff.reshape(-1, T_scale))
            
        # 4. Inverse DWT
        x_recon = ptwt.waverec(enhanced_coeffs_list, self.wavelet)
        
        # 5. Fix Length (Padding or Truncating)
        if x_recon.shape[-1] > T:
            x_recon = x_recon[..., :T]
        elif x_recon.shape[-1] < T:
             x_recon = F.pad(x_recon, (0, T - x_recon.shape[-1]))
            
        # Restore shape: [B, T, N, D]
        x_recon = x_recon.reshape(B, N, D, T).permute(0, 3, 1, 2)
        
        # 6. Residual Connection
        return x + self.project(x_recon)

# ============================================================================
# Standard Transformer Components (Condensed for brevity but functional)
# ============================================================================

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
    def forward(self, x):
        return self.norm(self.token_embed(x))

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)
    def forward(self, lap_mx):
        return self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)

class DataEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim, lape_dim, adj_mx, drop=0., add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu')):
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.device = device
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += self.spatial_embedding(lap_mx)
        return self.dropout(x)

class STSelfAttention(nn.Module):
    def __init__(self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1):
        super().__init__()
        # ... (Initializing Heads - keeping logic same as provided) ...
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.output_dim = output_dim

        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio

        # Linear projections
        self.pattern_q_linears = nn.ModuleList([nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)])
        self.pattern_k_linears = nn.ModuleList([nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)])
        self.pattern_v_linears = nn.ModuleList([nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)])

        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None, dynamic_adj=None):
        B, T, N, D = x.shape
        
        # --- Temporal Attention ---
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        # --- Geo Attention (Enhanced with Dynamic Graph) ---
        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Pattern injection
        for i in range(self.output_dim):
            pattern_q = self.pattern_q_linears[i](x_patterns[..., i])
            pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])
            pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
            pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
            pattern_attn = pattern_attn.softmax(dim=-1)
            geo_k += pattern_attn @ pattern_v

        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale

        # 【Causal Graph Injection】
        if dynamic_adj is not None:
            # dynamic_adj shape: [B, N, N]
            # geo_attn shape:    [B, T, Heads, N, N]
            # Broadcasting: [B, 1, 1, N, N]
            dynamic_bias = dynamic_adj.unsqueeze(1).unsqueeze(1)
            geo_attn = geo_attn + dynamic_bias

        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))

        # --- Semantic Attention ---
        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn.masked_fill_(sem_mask, float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))

        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class STEncoderBlock(nn.Module):
    def __init__(self, dim, s_attn_size, t_attn_size, geo_num_heads, sem_num_heads, t_num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre", output_dim=1):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(dim, s_attn_size, t_attn_size, geo_num_heads, sem_num_heads, t_num_heads, qkv_bias, attn_drop, drop, device, output_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None, dynamic_adj=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask, sem_mask, dynamic_adj))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask, sem_mask, dynamic_adj)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

# ============================================================================
# 【Main Model Class】
# ============================================================================

class CWFormer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self.dtw_matrix = self.data_feature.get('dtw_matrix')
        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')
        self._logger = getLogger()
        
        # Hyperparameters
        self.embed_dim = config.get('embed_dim', 64)
        self.skip_dim = config.get("skip_dim", 256)
        self.lape_dim = config.get('lape_dim', 8)
        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 3)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "pre")
        self.type_short_path = config.get("type_short_path", "hop")

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)

        # Loss Parameters
        self.lambda_causal = config.get('lambda_causal', 0.001)
        self.lambda_recon = config.get('lambda_recon', 0.05)
        self.tail_weight = config.get('tail_weight', 1.0) # Weight for tail loss

        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)

        # Mask Generation (Static)
        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < config.get('far_mask_delta', 5)] = 1
            self.far_mask = self.far_mask.bool()
        else:
            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= config.get('far_mask_delta', 5)] = 1
            self.geo_mask = self.geo_mask.bool()
            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
            sem_mask = self.dtw_matrix.argsort(axis=1)[:, :config.get('dtw_delta', 5)]
            for i in range(self.sem_mask.shape[0]):
                self.sem_mask[i][sem_mask[i]] = 0
            self.sem_mask = self.sem_mask.bool()

        self.pattern_keys = torch.from_numpy(data_feature.get('pattern_keys')).float().to(self.device)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, self.lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        # =========================================================
        # 【Modules Initialization】
        # =========================================================
        # 1. Wavelet Enhancement
        self.wavelet_enhancement = WaveletEnhancement(
            self.embed_dim, wavelet_name='db4', level=3
        )

        # 2. Causal Graph Generator (replaces TGDL)
        self.dict_dim = config.get('dict_dim', 32)
        self.causal_graph_module = CausalGraphGenerator(
            self.num_nodes,
            self.embed_dim,
            dict_dim=self.dict_dim,
            sparsity_threshold=0.05
        )
        # =========================================================

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=enc_dpr[i], act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                device=self.device, type_ln=type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True)

    def forward(self, batch, lap_mx=None):
        x = batch['X']
        T =  x.shape[1]
        
        # Pattern embedding logic
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim],(0, 0, 0, 0, self.s_attn_size - 1 - i, 0),"constant", 0).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)

        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))
        x_patterns = torch.cat(x_pattern_list, dim=-1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)

        # 1. Embedding
        enc = self.enc_embed_layer(x, lap_mx)

        # 2. Wavelet Enhancement (Contribution 1)
        enc = self.wavelet_enhancement(enc)

        # 3. Causal Graph Learning (Contribution 2)
        dynamic_adj, recon_loss = self.causal_graph_module(enc)
        # dynamic_adj = None
        # recon_loss = torch.tensor(0.0, device=self.device)

        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, x_patterns, pattern_keys, self.geo_mask, self.sem_mask, dynamic_adj=dynamic_adj)
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))

        return skip.permute(0, 3, 2, 1), dynamic_adj, recon_loss

    def get_loss_func(self, set_loss):
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        else:
            lf = loss.masked_mae_torch
        return lf

    # ============================================================================
    # 【Innovation 3: Tail-Aware Weighted Loss】(Optimization)
    #   - Story: 优化对极端值（长尾部分）的预测能力
    # ============================================================================
    def tail_weighted_loss(self, y_pred, y_true, null_val=0, gamma=2.0):
        # 创建 Mask: 去除 0 值 (或其他 null_val)
        mask = (y_true > null_val).float()
        
        # 避免分母为 0
        if mask.sum() == 0:
            return torch.tensor(0.0).to(y_pred.device)

        # 计算基础误差
        error = torch.abs(y_pred - y_true)
        
        # 计算权重: 值越大，权重越大 (Log平滑防止权重爆炸)
        # log(1 + x) 是处理长尾分布的经典 trick
        weight = torch.log(1 + torch.abs(y_true)) ** gamma
        
        # 归一化权重 (使得平均权重接近 1，不改变 Loss 的量级，只改变梯度分布)
        weight = weight / (weight.sum() / mask.sum() + 1e-6)
        
        # 加权 Loss
        loss = (error * weight * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        if isinstance(y_predicted, tuple):
            y_predicted = y_predicted[0]

        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # 严格过滤噪声 (可根据数据集调整)
        y_true[y_true < 1] = 0 

        if self.training:
            # Curriculum Learning 调度
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info(f'Training: task_level increase to {self.task_level}')

            if self.use_curriculum_learning:
                y_pred_curr = y_predicted[:, :self.task_level, :, :]
                y_true_curr = y_true[:, :self.task_level, :, :]
            else:
                y_pred_curr = y_predicted
                y_true_curr = y_true

            # 【Combination】Base Loss + Tail Loss
            base_loss = lf(y_pred_curr, y_true_curr, null_val=0)
            tail_loss = self.tail_weighted_loss(y_pred_curr, y_true_curr, null_val=0)
            
            return base_loss + self.tail_weight * tail_loss
            # return base_loss
        else:
            return lf(y_predicted, y_true, null_val=0)

    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        y_true = batch['y']
        y_predicted, dynamic_adj, recon_loss_val = self.forward(batch, lap_mx)
        
        # 1. Main Loss (MAE + Tail)
        main_loss = self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)
        
        # 2. Graph Regularization (Sparse Causal Constraint)
        sparsity_loss = 0.0
        if dynamic_adj is not None:
            sparsity_loss = torch.sum(torch.abs(dynamic_adj)) / (dynamic_adj.numel() + 1e-6)
            
        if not isinstance(recon_loss_val, torch.Tensor):
            recon_loss_val = torch.tensor(0.0, device=self.device)

        total_loss = main_loss + self.lambda_causal * sparsity_loss + self.lambda_recon * recon_loss_val

        if self.training and batches_seen % 500 == 0:
            self._logger.info(f"Batch {batches_seen}: Main={main_loss.item():.4f}, Reg={sparsity_loss.item():.4f}, Recon={recon_loss_val.item():.4f}")

        return total_loss

    def predict(self, batch, lap_mx=None):
        prediction, _, _ = self.forward(batch, lap_mx)
        return prediction