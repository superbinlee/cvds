import random
import string

import numpy as np
from Levenshtein import distance as levenshtein_distance


def _generate_prefix():
    """生成10位前缀（字母开头，后面字母数字混合）"""
    first_char = random.choice(string.ascii_letters)
    rest_chars = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(9))
    return first_char + rest_chars


def _generate_base_suffix(length=30):
    """生成30位基础后缀（字母数字混合）"""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def _modify_single_char(base_str, pos=None):
    """修改字符串的1位（确保变化）"""
    if pos is None:
        pos = random.randint(0, len(base_str) - 1)
    old_char = base_str[pos]
    new_char = random.choice([c for c in (string.ascii_letters + string.digits) if c != old_char])
    return base_str[:pos] + new_char + base_str[pos + 1:]


def _modify_two_chars(base_str):
    """修改字符串的2位（确保变化）"""
    pos1, pos2 = random.sample(range(len(base_str)), 2)  # 随机选2个不同位置
    modified = _modify_single_char(base_str, pos1)
    modified = _modify_single_char(modified, pos2)
    return modified


def _generate_similar_strings(prefix, n_strings, suffix_length=30):
    """生成一组字符串，每个字符串与基础字符串只有1-2位不同"""
    base_suffix = _generate_base_suffix(suffix_length)
    strings = [prefix + base_suffix]  # 基础字符串

    for _ in range(1, n_strings):
        # 随机决定修改1位还是2位（50%概率）
        if random.random() < 0.5:
            modified = _modify_single_char(base_suffix)
        else:
            modified = _modify_two_chars(base_suffix)
        strings.append(prefix + modified)

    return strings


def generate_string_groups(n_groups, n_strings_per_group):
    """生成多组字符串"""
    groups = []
    for _ in range(n_groups):
        prefix = _generate_prefix()
        group_strings = _generate_similar_strings(prefix, n_strings_per_group)
        groups.append((prefix, group_strings))
    return groups


def calculate_similarity(str1, str2, prefix_length=10, suffix_length=30):
    """计算两字符串的相似度（0.0~1.0）"""
    # 分割前缀和后缀
    prefix1, suffix1 = str1[:prefix_length], str1[prefix_length:]
    prefix2, suffix2 = str2[:prefix_length], str2[prefix_length:]

    # 1. 前缀相似度（权重90%）
    prefix_sim = 1.0 if prefix1 == prefix2 else 0.0

    # 2. 后缀相似度（权重10%，基于编辑距离）
    max_suffix_dist = suffix_length  # 最大可能编辑距离
    suffix_dist = levenshtein_distance(suffix1, suffix2)
    suffix_sim = 1.0 - (suffix_dist / max_suffix_dist)

    # 3. 加权综合
    total_sim = 0.9 * prefix_sim + 0.1 * suffix_sim
    return np.clip(total_sim, 0.0, 1.0)  # 限制在[0, 1]范围内


if __name__ == '__main__':
    groups = generate_string_groups(n_groups=1, n_strings_per_group=1)

    for i, (prefix, strings) in enumerate(groups, 1):
        print(f"Group {i} (Prefix: {prefix}):")
        for j, s in enumerate(strings):
            if j == 0:
                print(f"  [Base] {s}")
            else:
                print(f"  [Var{j}] {s}")
        print()

    print(calculate_similarity('rvUAynpQjVjtm7WMilHZ1FhNqG5N2j4yBO4MpzC0', 'rvUAynpQjVjtm7WMilHZ1IhNqG5N2jeyBO4MpzC0'))
    print(calculate_similarity('rvUAynpQjVjtm7WMilHZ1FhNqG5N2j4yBO4MpzC0', 'rvUAynpQjV3tm7WMilHZ1FhNqG5N2jeyBO4MpzC0'))
    print(calculate_similarity('rvUAynpQjVjtm7WMilHZ1FhNqG5N2j4yBO4MpzC0', 'rvUAynpQjVjtm7WMilHZ1FhNqG5N2jeyBO4MpzC0'))
