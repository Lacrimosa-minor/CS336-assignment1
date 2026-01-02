import regex as re
from typing import Iterable, Iterator
import json
import base64
from tqdm import tqdm
import numpy as np
import gc


def load_vocab_and_merge(vocab_path: str, merge_path: str):
    # 读取 vocab 文件
    with open(vocab_path, "r") as vocab_file:
        vocab_data = json.load(vocab_file)
        # 将 base64 编码的字符串解码为字节
        vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}

    # 读取 merge 文件
    with open(merge_path, "r") as merge_file:
        merge = []
        for line in merge_file:
            part1, part2 = line.strip().split()
            # 解码 base64 编码的字节数据
            merge.append((base64.b64decode(part1), base64.b64decode(part2)))

    return vocab, merge


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.ENCODE_TYPE = "utf-8"

        self.int_to_token = vocab
        self.token_to_int = {}
        for k, v in vocab.items():
            self.token_to_int[v] = k

        self.merge_order = {}
        for idx, item in enumerate(merges):
            self.merge_order[item] = idx

        self.known_words = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        token_dict, merge_history = load_vocab_and_merge(
            vocab_filepath, merges_filepath
        )
        return cls(token_dict, merge_history, special_tokens)

    def _cut_special_token(
        self,
        content: str,
        special_tokens: list[str] | None,
    ) -> tuple[list[str], bool]:
        # 按照special_tokens从长到短进行分割
        start_with_sptok = False
        if special_tokens:
            special_tokens = sorted(
                special_tokens, key=lambda x: (len(x), x), reverse=True
            )
            sp_tok_pat = "|".join([re.escape(token) for token in special_tokens])

            content_sptok_seq = []
            piece_start = 0
            start_with_sptok = False
            for piece in re.finditer(sp_tok_pat, content):
                # content_without_sptok += " "
                match_token = piece.group()
                if piece.start() == 0:
                    start_with_sptok = True
                    content_sptok_seq.append(match_token)
                    piece_start = piece.end()
                    continue
                content_sptok_seq.append(content[piece_start : piece.start()])
                piece_start = piece.end()
                content_sptok_seq.append(match_token)
            if piece_start < len(content):
                content_sptok_seq.append(content[piece_start:])
        else:
            content_sptok_seq = [content]

        return content_sptok_seq, start_with_sptok

    def _encode_single_word(self, word: bytes) -> list[int]:
        # 如果这个词确实存在，直接给
        already_exist = self.token_to_int.get(word, None)
        if already_exist is not None:
            return [already_exist]

        # 之前计算过的, 直接给
        already_merged = self.known_words.get(word, None)
        if already_merged is not None:
            return already_merged

        # 这里摒弃了最复杂的方案，没有维护堆结构，每次遍历全部可能组合寻找最小值
        # 可能造成效率问题，但是不大，已经免除了诸多复杂运算，复杂度只和单词的长度有关系
        all_possible_merge = []
        for i in range(len(word) - 1):
            possible_merge = (word[i : i + 1], word[i + 1 : i + 2])
            merge_priority = self.merge_order.get(possible_merge, None)
            all_possible_merge.append(
                {"merge": possible_merge, "index": merge_priority}
            )

        while len(all_possible_merge) > 1:
            # 寻找合并谁
            to_merge_loc = None
            min_index = None
            for loc, item in enumerate(all_possible_merge):
                index = item["index"]
                if index is None:
                    continue
                if min_index is None or index <= min_index:
                    to_merge_loc = loc
                    min_index = index

            if to_merge_loc is None:
                break

            # 合并
            min_merge = all_possible_merge[to_merge_loc]
            new_token = min_merge["merge"][0] + min_merge["merge"][1]
            prev = to_merge_loc - 1
            next = to_merge_loc + 1
            if prev >= 0:
                possible_merge = (all_possible_merge[prev]["merge"][0], new_token)
                merge_priority = self.merge_order.get(possible_merge, None)
                all_possible_merge[prev] = {
                    "merge": possible_merge,
                    "index": merge_priority,
                }
            if next < len(all_possible_merge):
                possible_merge = (new_token, all_possible_merge[next]["merge"][1])
                merge_priority = self.merge_order.get(possible_merge, None)
                all_possible_merge[next] = {
                    "merge": possible_merge,
                    "index": merge_priority,
                }

            del all_possible_merge[to_merge_loc]

        id_list = []
        if len(all_possible_merge) == 1 and all_possible_merge[0]["index"]:
            id_list.append(self.token_to_int[word])
        else:
            frist_token = all_possible_merge[0]["merge"][0]
            id_list.append(self.token_to_int[frist_token])
            for item in all_possible_merge:
                next_token = item["merge"][1]
                id_list.append(self.token_to_int[next_token])

        self.known_words[word] = id_list

        return id_list

    def encode(self, text: str) -> list[int]:

        # 这里并没有考虑text过大的问题，你都字符串输入了，内存已经占了，没意义
        # 挑出特殊token
        text_list, is_sptok = self._cut_special_token(text, self.special_tokens)
        token_lst = []
        id_list = []

        for text in tqdm(text_list, desc="Pretokenization Stage:"):
            if is_sptok:
                # 后续，从数据类型判断，如果当前类型是一个int，那么跳过
                token_lst.append(self.token_to_int[text.encode(self.ENCODE_TYPE)])
                is_sptok = False
            else:
                # 对文段进行pre_token，有一定可能，文段过长导致超时，以后再看
                for word in re.finditer(self.PAT, text):
                    word = word.group(0).encode(self.ENCODE_TYPE)
                    token_lst.append(word)
                is_sptok = True
        for pre_token in tqdm(token_lst, desc="Encoding Stage:"):
            if isinstance(pre_token, int):
                id_list.append(pre_token)
                continue
            id_list += self._encode_single_word(pre_token)

        return id_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 姑且直接抄了前面的代码，不确定会不会有啥问题，先这么用着
        for text in iterable:
            text_list, is_sptok = self._cut_special_token(text, self.special_tokens)
            token_lst = []

            for text in text_list:
                if is_sptok:
                    # 后续，从数据类型判断，如果当前类型是一个int，那么跳过
                    token_lst.append(self.token_to_int[text.encode(self.ENCODE_TYPE)])
                    is_sptok = False
                else:
                    # 对文段进行pre_token，有一定可能，文段过长导致超时，以后再看
                    for word in re.finditer(self.PAT, text):
                        word = word.group(0).encode(self.ENCODE_TYPE)
                        token_lst.append(word)
                    is_sptok = True
            for pre_token in token_lst:
                if isinstance(pre_token, int):
                    yield pre_token
                    continue
                id_list = self._encode_single_word(pre_token)
                for id in id_list:
                    yield id

    def decode(self, ids: list[int]) -> str:
        text = []
        for id in ids:
            text.append(self.int_to_token[id])

        text = b"".join(text)
        text = text.decode(self.ENCODE_TYPE, errors="replace")

        return text


if __name__ == "__main__":

    TS_MERGE_FILE = "data/merge_ts10k.txt"
    TS_VOCAB_FILE = "data/vocab_ts10k.json"
    sp_tok = ["<|endoftext|>"]

    ts_tokenizer = Tokenizer.from_files(
        merges_filepath=TS_MERGE_FILE,
        vocab_filepath=TS_VOCAB_FILE,
        special_tokens=sp_tok,
    )

    OWT_MERGE_FILE = "data/merge_owt32k.txt"
    OWT_VOCAB_FILE = "data/vocab_owt32k.json"

    owt_tokenizer = Tokenizer.from_files(
        merges_filepath=OWT_MERGE_FILE,
        vocab_filepath=OWT_VOCAB_FILE,
        special_tokens=sp_tok,
    )

    TS_VALID_FILE = "data/TinyStoriesV2-GPT4-valid.txt"
    OWT_VALID = "data/owt_valid.txt"
    TS_TRAIN_FILE = "data/TinyStoriesV2-GPT4-train.txt"
    OWT_TRAIN = "data/owt_train.txt"

    def encode_file(
        input_file: str, output_path: str, tokenizer: Tokenizer, dtype=np.uint16
    ):
        with open(output_path, "wb") as out, open(
            input_file, "r", encoding="utf-8"
        ) as f:
            # 小缓冲避免频繁系统调用
            pbar = tqdm(total=None, unit="tid", unit_scale=True, dynamic_ncols=True)
            buf = np.empty((32_000_000,), dtype=dtype)  #
            k = 0
            for tid in tokenizer.encode_iterable(f):
                buf[k] = tid
                k += 1
                if k == buf.shape[0]:
                    buf.tofile(out)
                    pbar.update(k)
                    k = 0
            if k:
                buf[:k].tofile(out)
                pbar.update(k)

            pbar.close()

    encode_file(TS_VALID_FILE, "data/TinyStory-valid-uint16.raw", ts_tokenizer)
    encode_file(OWT_VALID, "data/OpenWebText-valid-uint16.raw", owt_tokenizer)
    encode_file(TS_TRAIN_FILE, "data/TinyStory-train-uint16.raw", ts_tokenizer)
    encode_file(OWT_TRAIN, "data/OpenWebText-train-uint16.raw", owt_tokenizer)
    # SAMPLE_NUM = 10

    # sample for test

    # def extract_random_samples(file_path: str, num_samples: int = 10) -> list:
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         # 读取整个文件内容
    #         content = file.read()

    #         # 使用 '<|endoftext|>' 分割文本
    #         samples = content.split('<|endoftext|>')

    #         # 去除空的样本（如果有的话）
    #         samples = [sample.strip() for sample in samples if sample.strip()]

    #         # 随机抽取指定数量的样本
    #         random_samples = random.sample(samples, min(num_samples, len(samples)))

    #         return random_samples

    # ts_samples = extract_random_samples(TS_VALID_FILE, SAMPLE_NUM)
    # owt_samples = extract_random_samples(OWT_VALID, SAMPLE_NUM)

    # def compute_bytes_per_token(text: str, token_sequence: list, encoding: str = 'utf-8') -> float:
    #     # 计算文本的字节数
    #     text_bytes = len(text.encode(encoding))

    #     # 获取 token 数量
    #     num_tokens = len(token_sequence)

    #     # 防止除以零的情况
    #     if num_tokens == 0:
    #         return 0.0

    #     # 计算每个 token 的字节数
    #     bytes_per_token = text_bytes / num_tokens

    #     return bytes_per_token

    # ts_ts_ratio = 0

    # for text in tqdm(ts_samples, desc="tokenizing with ts"):
    #     id_lst = ts_tokenizer.encode(text)
    #     compress_ratio = compute_bytes_per_token(text, id_lst)
    #     ts_ts_ratio += compress_ratio

    # print(f"TinyStory text with TinyStory tokenizer: {ts_ts_ratio}")

    # ts_owt_ratio = 0

    # for text in tqdm(ts_samples, desc="tokenizing with ts"):
    #     id_lst = owt_tokenizer.encode(text)
    #     compress_ratio = compute_bytes_per_token(text, id_lst)
    #     ts_owt_ratio += compress_ratio

    # print(f"TinyStory text with OpenWebText tokenizer: {ts_owt_ratio}")

    # owt_owt_ratio = 0

    # for text in tqdm(owt_samples, desc="tokenizing with owt"):
    #     id_lst = owt_tokenizer.encode(text)
    #     compress_ratio = compute_bytes_per_token(text, id_lst)
    #     owt_owt_ratio += compress_ratio

    # print(f"OpenWebText text with OpenWebText tokenizer: {owt_owt_ratio}")

    # owt_ts_ratio = 0

    # for text in tqdm(owt_samples, desc="tokenizing with ts_vocab"):
    #     id_lst = ts_tokenizer.encode(text)
    #     compress_ratio = compute_bytes_per_token(text, id_lst)
    #     owt_ts_ratio += compress_ratio

    # print(f"OpenWebText text with TinyStory tokenizer: {ts_ts_ratio}")

    # # save IDs
