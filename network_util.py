import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, einsum
import math


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # 计算\sigma
        sigma = (2 / (in_features + out_features)) ** 0.5

        # 没有记录下来输入的几个值，以后有问题再改
        para = torch.empty((out_features, in_features), dtype=dtype, device=device)
        nn.init.trunc_normal_(para, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.weight = nn.Parameter(para)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x @ self.weight.T


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        super().__init__()

        # 计算\sigma
        sigma = 1

        para = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        nn.init.trunc_normal_(para, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.weight = nn.Parameter(para)

        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=torch.long)
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        para = torch.ones(d_model, dtype=dtype, device=device)
        self.weight = nn.Parameter(para)
        self.eps = eps
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(self.device, torch.float32)
        # Your code here performing RMSNorm
        r = torch.rsqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps)

        result = x * r * self.weight
        # Return the result in the original dtype
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inside: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_inside, device, dtype)
        self.w2 = Linear(d_inside, d_model, device, dtype)
        self.w3 = Linear(d_model, d_inside, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        g1 = self.w1(x)
        g3 = self.w3(x)
        silu = torch.sigmoid(g1) * g1
        element_wise_multi = silu * g3

        return self.w2(element_wise_multi)


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        dim: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.dim = dim
        self.max_len = max_seq_len

        token_array = torch.arange(max_seq_len, device=device)

        # 对于每一个向量，i数值是一样的
        numer = repeat(token_array, "len -> len dim", dim=self.dim)

        # 每一个向量的同一个位置，k数值是一样的
        denom = torch.tensor(
            [
                self.theta ** ((2 * k - 2) / self.dim)
                for k in range(1, int(self.dim / 2) + 1)
            ],
            device=device,
        )
        denom = denom.repeat_interleave(2)

        # 全部的theta值
        theta_mat = numer / denom
        cos = torch.cos(theta_mat)
        sin = torch.sin(theta_mat)
        self.register_buffer("theta_mat_cos", cos, persistent=False)
        self.register_buffer("theta_mat_sin", sin, persistent=False)
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        # 根据token_pos选出矩阵
        token_positions = token_positions.to(device=self.device, dtype=torch.long)
        cos_mat = self.theta_mat_cos[token_positions]
        sin_mat = self.theta_mat_sin[token_positions]

        # cosine直接乘上x，但是另一个要变一下形
        x_verse = rearrange(x, "... (n two) -> ... n two", two=2).to(self.device)
        sign = torch.tensor([1, -1], device=self.device)
        x_verse = x_verse * sign
        x_verse = x_verse[..., [1, 0]]
        x_verse = rearrange(x_verse, "... n two -> ... (n two)")

        # 最后计算结果

        return x * cos_mat + x_verse * sin_mat


class softmax(nn.Module):
    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor, dim: int):
        # max_values, _ = torch.max(x, dim=dim, keepdim=True)
        max_values = reduce(x, "... len -> ... 1", "max")
        exp = torch.exp(x - max_values)

        denom = reduce(exp, "... len -> ... 1", "sum")

        return exp / denom


class BaseAttention(nn.Module):
    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.softmax = softmax(device, dtype)
        self.device = device
        self.dtype = dtype

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ):
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        mask = mask.to(self.device)
        qk = einsum(q, k, "... q_len dim, ... k_len dim -> ... q_len k_len")
        denom = math.sqrt(q.shape[-1])
        qk = qk / denom

        mask = torch.where(mask, 0, float("-inf"))
        masked = qk + mask

        masked = self.softmax(x=masked, dim=-1)

        return einsum(masked, v, "... q_len k_len, ... k_len d -> ... q_len d")


class CausalMultiHeadsAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.head_num = num_heads
        self.head_dim = int(d_model / num_heads)
        sigma = (2 / (d_model + self.head_dim)) ** 0.5

        Q = torch.empty((d_model, d_model), dtype=dtype, device=device)
        nn.init.trunc_normal_(Q, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.Q = nn.Parameter(Q)

        K = torch.empty((d_model, d_model), dtype=dtype, device=device)
        nn.init.trunc_normal_(K, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.K = nn.Parameter(K)

        V = torch.empty((d_model, d_model), dtype=dtype, device=device)
        nn.init.trunc_normal_(V, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.V = nn.Parameter(V)

        self.O = Linear(d_model, d_model, device, dtype)
        self.attention = BaseAttention(device, dtype)
        self.rope = RoPE(theta, int(d_model / num_heads), max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positinos: torch.Tensor):
        length = x.shape[-2]

        multi_q = rearrange(self.Q, "... (h hd) d -> ... h hd d", h=self.head_num).to(
            self.device
        )
        multi_q = einsum(x, multi_q, "... len d,... h hd d -> ... h len hd")
        multi_q_rope = self.rope(multi_q, token_positinos)

        multi_k = rearrange(self.K, "... (h hd) d -> ... h hd d", h=self.head_num).to(
            self.device
        )
        multi_k = einsum(x, multi_k, "... len d,... h hd d -> ... h len hd")
        multi_k_rope = self.rope(multi_k, token_positinos)

        multi_v = rearrange(self.V, "... (h hd) d -> ... h hd d", h=self.head_num).to(
            self.device
        )
        multi_v = einsum(x, multi_v, "... len d,... h hd d -> ... h len hd")

        causal_mask = torch.tril(torch.ones((length, length))).to(
            dtype=bool, device=self.device
        )

        multi_out = self.attention(multi_q_rope, multi_k_rope, multi_v, causal_mask)

        # 合并
        multi_out = rearrange(multi_out, "... h l d -> ... l (h d)")

        return self.O(multi_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, 1e-5, device, dtype)
        self.ffn_norm = RMSNorm(d_model, 1e-5, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.attn = CausalMultiHeadsAttention(
            d_model, num_heads, max_seq_len, theta, device, dtype
        )
        # self.device = device

    def forward(self, x: torch.Tensor, token_position: torch.Tensor | None = None):
        res = x
        if token_position is None:
            length = x.shape[-2]
            token_position = torch.arange(length, device=x.device)
        x = self.attn_norm(x)

        x = self.attn(x, token_position)
        x = x + res

        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x)

        return x + res


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.embedding = Embedding(vocab_size, d_model, device, dtype)

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta, device
                )
                for _ in range(num_layers)
            ]
        )

        self.out_norm = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

        self.out_embedding = Linear(d_model, vocab_size, device, dtype)

        self.out_softmax = softmax(device, dtype)

    def forward(self, x: torch.Tensor):

        length = x.shape[-1]
        token_position = torch.arange(length, device=x.device)

        x = self.embedding(x)

        for i in range(self.num_layers):
            x = self.layers[i](x, token_position)

        x = self.out_norm(x)
        x = self.out_embedding(x)

        return x
