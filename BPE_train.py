import os
import json
import base64
from typing import BinaryIO
import regex as re
from collections import Counter
from typing import Hashable
from concurrent.futures import ProcessPoolExecutor, as_completed

# 测量工具
import time
# from memory_profiler import profile
from datetime import datetime
from tqdm import tqdm 
 


# 潜在问题，我需要知道数据的编码格式，我先猜是utf8
ENCODE_TYPE = "utf-8"
# 对于owt数据，每个核心处理多份数据，才能确保不超内存
CHUNK_PER_PROCESS = 16

class IndexedMaxPQ:
    # 初始化

    def __init__(
        self,
        initial: dict[Hashable, int],
    ):
        """
        initial: 一个 dict，key 是任意可哈希对象（如 (bytes, bytes)），value 是数值
        show_progress: 是否在建堆时显示进度条（依赖 tqdm）
        """
        # heap 里存 (value, key)
        self.heap = []
        self.pos = {}  # key -> index in heap

        self._build_heap_from_dict(initial)

    def _build_heap_from_dict(self, data: dict[Hashable, int]):
        """
        用 O(n) Floyd 建堆，从一个 dict 初始化：
        data: {key -> value}
        """
        # 1. 直接把 dict 填进 heap 数组
        #    注意：heap 里是 (value, key)
        self.heap = [(value, key) for key, value in data.items()]

        # 2. 建立 key -> index 映射
        self.pos = {key: idx for idx, (value, key) in enumerate(self.heap)}

        n = len(self.heap)
        if n <= 1:
            return

        # 3. 自底向上做 sift_down：从最后一个非叶子节点开始
        start = (n - 2) // 2

        iterator = tqdm(
            range(start, -1, -1),
            desc="Building IndexedMaxPQ heap",
            total=start + 1,
        )

        for i in iterator:
            self._sift_down(i)


    # ---------- 内部工具方法 ----------

    def _swap(self, i: int, j: int):
        """交换堆中两个位置，并更新 pos"""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        _, key_i = self.heap[i]
        _, key_j = self.heap[j]
        self.pos[key_i] = i
        self.pos[key_j] = j

    def _better(self, i: int, j: int) -> bool:
        """
        返回 heap[i] 是否应该排在 heap[j] 前面（也就是“更大”）。
        规则：
        1. value 降序（越大越好）
        2. value 相等时，key 按字典序降序（越大越好）
        """
        val_i, key_i = self.heap[i]
        val_j, key_j = self.heap[j]

        if val_i != val_j:
            return val_i > val_j  # value 大的更好
        # value 相等时，key 更大的更好（字典序降序）
        return key_i > key_j

    def _sift_up(self, idx: int):
        """向上调整为最大堆"""
        while idx > 0:
            parent = (idx - 1) // 2
            if self._better(idx, parent):
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx: int):
        """向下调整为最大堆"""
        n = len(self.heap)
        while True:
            left = idx * 2 + 1
            right = idx * 2 + 2
            best = idx

            if left < n and self._better(left, best):
                best = left
            if right < n and self._better(right, best):
                best = right

            if best == idx:
                break

            self._swap(idx, best)
            idx = best

    # ---------- 对外接口 ----------

    def push(self, key, value):
        """插入或更新 key 的 value"""
        if key in self.pos:
            self.update(key, value)
            return

        idx = len(self.heap)
        self.heap.append((value, key))
        self.pos[key] = idx
        self._sift_up(idx)

    def update(self, key, value):
        """更新某个 key 的 value，如果不存在则插入"""
        if key not in self.pos:
            self.push(key, value)
            return

        idx = self.pos[key]
        self.heap[idx] = (value, key)
        # 更新后可能需要向上或向下调整
        self._sift_up(idx)
        self._sift_down(idx)

    def pop_max(self):
        """弹出 (key, value)，当前最大的元素"""
        if not self.heap:
            return None

        top_value, top_key = self.heap[0]
        last = len(self.heap) - 1

        # 把最后一个元素换到堆顶
        self._swap(0, last)

        # 删除最后一个（原先的堆顶）
        self.heap.pop()
        del self.pos[top_key]

        # 下沉调整
        if self.heap:
            self._sift_down(0)

        return top_key, top_value

    def peek_max(self):
        """查看当前最大元素但不删除，返回 (key, value)"""
        if not self.heap:
            return None
        value, key = self.heap[0]
        return key, value

    def get(self, key, default=None):
        """查询 key 的当前 value"""
        if key not in self.pos:
            return default
        value, _ = self.heap[self.pos[key]]
        return value

    def add(self, key, delta):
        """不存在→创建，存在→在当前值基础上累加 delta"""
        if key not in self.pos:
            self.push(key, delta)
        else:
            old_value = self.get(key)
            new_value = old_value + delta
            self.update(key, new_value)

    def __contains__(self, key):
        return key in self.pos

    def __len__(self):
        return len(self.heap)



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def remove_special_token(
    content: str,
    special_tokens: list[str],
)-> list[str]:
    # 按照special_tokens从长到短进行分割
    if special_tokens:
        special_tokens = sorted(special_tokens, key=lambda x: (len(x), x), reverse=True)
        sp_tok_pat = "|".join([re.escape(token) for token in special_tokens])

        content_without_sptok = []
        piece_start = 0
        for piece in re.finditer(sp_tok_pat, content):
            # content_without_sptok += " "
            content_without_sptok.append(content[piece_start:piece.start()])
            piece_start = piece.end()
        if piece_start < len(content):
            content_without_sptok.append(content[piece_start:])
    else:
        content_without_sptok = [content]
    
    return content_without_sptok


def pre_tokenization(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
)-> Counter[bytes]:
    with open(input_path, "rb") as f:
        f.seek(start)
        content = f.read(end - start).decode("utf-8", errors="ignore")
    
    content_without_sptok = remove_special_token(content, special_tokens)

    words_count = Counter[bytes]()
    for text in content_without_sptok:
        for word in re.finditer(pattern, text):
            word = word.group(0).encode(ENCODE_TYPE)
            words_count[word] += 1

    return words_count    


# @profile
def run_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 第一步，初始化编码，256个编码+排序后特殊token
    init_start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing vocabulary and special tokens...")
    # 潜在问题：大小端,代码默认大端
    init_dict = {i: i.to_bytes(1,"big") for i in range(256)}

    # 这个部分负责把special tokens加入其中，我认为需要等一下
    special_token_init = {j + 256: special_tokens[j].encode(ENCODE_TYPE) for j in range(len(special_tokens))}

    vocab_num = 256 + len(special_tokens)

    token_dict = special_token_init | init_dict


    init_time = time.time() - init_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initialization done in {init_time:.2f} seconds.")

    # 第二步，分割chunk，去除special tokens，做好词汇个数统计
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pretokenizing input data...")
    pre_start_time = time.time()
    n_proc = os.cpu_count() or 1
    words_count = Counter()
    process = []
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        
        with open(input_path, "rb") as f:
            chunks = find_chunk_boundaries(
                f, n_proc * CHUNK_PER_PROCESS, "<|endoftext|>".encode("utf-8")
            )

        for start, end in zip(chunks[:-1], chunks[1:]):
            process.append(ex.submit(pre_tokenization, input_path, start, end, special_tokens))
        for complete in tqdm(as_completed(process), total=len(process), desc="Processing chunks"):
            words_count.update(complete.result())
    
    pre_time = time.time() - pre_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pretokenization done in {pre_time:.2f} seconds.")

    # 第三步，训练开始，创建一个文件记录中间过程
    # 初始化
    # 把关系做到最简化，除了words_count，还有三对关系
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing BPE training...")
    bpe_init_start = time.time()
    all_pairs_count = Counter[tuple[bytes, bytes]]()
    word_to_pair = {}
    pair_to_word = {}
    for key,value in tqdm(words_count.items(), desc="Initializing..."):
        for idx in range(len(key) - 1):
            pair = tuple([key[idx:idx+1], key[idx+1: idx+2]])
            all_pairs_count[pair] += value
            pair_to_word.setdefault(pair, {}).setdefault(key, []).append(idx)
        word_to_pair[key] = list(range(len(key) + 1))
    # 单独初始化，降一下复杂度
    pairs_count = IndexedMaxPQ(all_pairs_count)
    bpe_init_time = time.time() - bpe_init_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initialization done in {bpe_init_time:.2f} seconds.")

    # 算法开始迭代
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting BPE training...")
    bpe_start_time = time.time()
    merge_history = []
    
    with tqdm(total=vocab_size - vocab_num, desc="BPE Merging Progress") as pbar:
        while vocab_num < vocab_size:

            new_token = pairs_count.pop_max()
            
            # 防止胡乱合并：
            if new_token[1] <= 0:
                break
            new_token = new_token[0]


            merge_history.append((new_token[0], new_token[1]))
            token_dict[vocab_num] = new_token[0] + new_token[1]
            vocab_num += 1
            pbar.update(1)

            all_word_contain = pair_to_word[new_token]
            # 处理全部词汇中包含新组合的问题
            to_del = {}
            to_add = {}
            for word, loc_lst in all_word_contain.items():
                all_pairs = word_to_pair[word]
                pair_idx = 0
                loc_cnt = 0
                loc_lst.sort()
                while pair_idx < len(all_pairs) and loc_cnt < len(loc_lst):
                    
                    # 注意，如果出现了连续字母，可能合并前一个，后一个pair就消失了，所以这里要判断
                    if loc_lst[loc_cnt] < all_pairs[pair_idx]:
                        loc_cnt += 1
                        if loc_cnt == len(loc_lst):
                            continue

                    if all_pairs[pair_idx] == loc_lst[loc_cnt]:
                        # 判断需要增删的左右pair
                        # 左边
                        # 如果左边能够取到更左的位置，有左边的增删

                        if pair_idx - 1 >= 0:
                            # 删
                            start = all_pairs[pair_idx - 1]
                            mid = all_pairs[pair_idx]
                            end = all_pairs[pair_idx + 1]
                            del_token = (word[start:mid], word[mid: end])
                            to_del[del_token] = to_del.get(del_token, 0) + words_count[word]
                            del_idx = pair_to_word[del_token][word].index(start)
                            del pair_to_word[del_token][word][del_idx]
                            
                            # 增
                            end = all_pairs[pair_idx + 2] if pair_idx + 2 < len(all_pairs) else -1
                            add_token = (word[start:mid], word[mid: end])
                            to_add[add_token] = to_add.get(add_token, 0) + words_count[word]
                            pair_to_word.setdefault(add_token, {}).setdefault(word, []).append(start)
                        # 右边
                        # 如果右边能够取到更右的位置，有右边的增删
                        if pair_idx + 3 < len(all_pairs):
                            start = all_pairs[pair_idx + 1]
                            mid = all_pairs[pair_idx + 2]
                            end = all_pairs[pair_idx + 3]
                            del_token = (word[start:mid], word[mid: end])
                            to_del[del_token] = to_del.get(del_token, 0) + words_count[word]
                            del_idx = pair_to_word[del_token][word].index(start)
                            del pair_to_word[del_token][word][del_idx]
                            
                            start = all_pairs[pair_idx]
                            add_token = (word[start:mid], word[mid: end])
                            to_add[add_token] = to_add.get(add_token, 0) + words_count[word]
                            pair_to_word.setdefault(add_token, {}).setdefault(word, []).append(start)  


                        # 增删完成后，删除pair_point,进行合并
                        del all_pairs[pair_idx + 1]
                        # 看下一个组合出现点
                        loc_cnt += 1
    
                    
                    pair_idx += 1

                # 更新word中的pair分割
                word_to_pair[word] = all_pairs

            # 全部更新完毕，更新一下统计数据
            for token, cnt in to_add.items():
                pairs_count.add(token, cnt)
            
            for token, cnt in to_del.items():
                pairs_count.add(token, cnt * -1)

    end_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train BPE done in {(end_time - bpe_start_time):.2f} seconds.")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final vocabulary size: {vocab_num}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total time cost {(end_time - init_start_time):.2f} seconds")        
    return (token_dict, merge_history)

def save_vocab_and_merge(vocab: dict[int, bytes], merge: list[tuple[bytes, bytes]], vocab_path: str, merge_path: str):
    # 保存 vocab 到文件
    with open(vocab_path, 'w') as vocab_file:
        # 将字典中的每个字节值转换为 base64 编码的字符串
        vocab_serializable = {str(k): base64.b64encode(v).decode('utf-8') for k, v in vocab.items()}
        json.dump(vocab_serializable, vocab_file, indent=4)

    # 保存 merge 到文件
    with open(merge_path, 'w') as merge_file:
        # 将每个元组的字节对象转换为 base64 编码的字符串
        merge_serializable = [(base64.b64encode(pair[0]).decode('utf-8'), base64.b64encode(pair[1]).decode('utf-8')) for pair in merge]
        for item in merge_serializable:
            merge_file.write(f"{item[0]} {item[1]}\n")

def load_vocab_and_merge(vocab_path: str, merge_path: str):
    # 读取 vocab 文件
    with open(vocab_path, 'r') as vocab_file:
        vocab_data = json.load(vocab_file)
        # 将 base64 编码的字符串解码为字节
        vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}

    # 读取 merge 文件
    with open(merge_path, 'r') as merge_file:
        merge = []
        for line in merge_file:
            part1, part2 = line.strip().split()
            # 解码 base64 编码的字节数据
            merge.append((base64.b64decode(part1), base64.b64decode(part2)))

    return vocab, merge

if __name__ == "__main__":
    
    data_file = "data/owt_train.txt"
    vocab_size = 32000
    special_token = ['<|endoftext|>']

    
    # 开启训练
    vocab, merge = run_bpe(data_file,vocab_size, special_token)

    # 保存数据到文件
    save_vocab_and_merge(vocab, merge, 'vocab_owt32k.json', 'merge_owt32k.txt')
        