import torch
from network_util import *
from BPE_tokenizer import *

# 输入
INPUT = "Once upon a time, there was a"

# 编码器
MERGE_FILE = "data/merge_ts10k.txt"
VOCAB_FILE = "data/vocab_ts10k.json"

# 超参
TEMPERATURE = 0.6
TOPK = 0.9
MAX_LEN = 512

# 模型参数
WEIGHT_FILE = "data/ts_model.pt"
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
EMBEDDING_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 16
FFN_DIM = 1344
ROPE_THETA = 10000


# top-p
def top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=0)
    mask = cdf <= p
    # 保证至少保留一个
    mask[0] = True
    filtered_probs = sorted_probs * mask
    s = filtered_probs.sum()
    if s <= 0:
        return torch.argmax(probs)
    filtered_probs = filtered_probs / s
    choice = torch.multinomial(filtered_probs, 1).squeeze(-1)
    return sorted_idx[choice]


# load model only
def load_model(
    src: str,
    model: torch.nn.Module,
):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    if isinstance(src, str):
        ckpt = torch.load(src, map_location=device)
    else:
        ckpt = torch.load(src, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt["iteration"])


# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载
model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=FFN_DIM,
    rope_theta=ROPE_THETA,
    device=device,
)
model.to(device)
load_model(WEIGHT_FILE, model)
model.eval()

# 辅助工具
tokenizer = Tokenizer.from_files(VOCAB_FILE, MERGE_FILE, ["<|endoftext|>"])
softmax_fn = softmax(device)
eof = tokenizer.encode("<|endoftext|>")
eof = eof[0]
# 生成循环
continue_flag = True
generate_content = tokenizer.encode(INPUT)
generate_content = torch.tensor(generate_content, dtype=torch.long, device=device)

while continue_flag:
    next_word = model(generate_content[-CONTEXT_LENGTH:])
    next_word = next_word[-1] / (TEMPERATURE + 1e-8)
    next_word = softmax_fn(next_word, -1)
    next_word = top_p(next_word, TOPK)
    generate_content = torch.cat([generate_content, next_word.view(1)], dim=0)
    if len(generate_content) >= MAX_LEN or int(next_word.item()) == eof:
        continue_flag = False

generate_text = tokenizer.decode(generate_content.tolist())
print(generate_text)
