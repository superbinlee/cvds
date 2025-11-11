import numpy as np
from scipy.stats import norm, beta
from typing import Dict, List, Tuple, Optional

# ==============================
# 全局维度配置（18维）
# ==============================
DIM = 512
DIMENSIONS = {
    # 人脸 (6)
    "face_shape": {"size": 80, "modal": "face"},
    "skin_tone": {"size": 40, "modal": "face"},
    "eye_nose": {"size": 60, "modal": "face"},
    "lighting": {"size": 30, "modal": "face"},
    "glasses": {"size": 20, "modal": "face"},
    "expression": {"size": 30, "modal": "face"},
    # 人体 (6)
    "body_shape": {"size": 70, "modal": "body"},
    "clothing_color": {"size": 50, "modal": "body"},
    "clothing_style": {"size": 40, "modal": "body"},
    "pose": {"size": 40, "modal": "body"},
    "accessory": {"size": 30, "modal": "body"},
    "occlusion": {"size": 30, "modal": "body"},
    # 车辆 (6)
    "vehicle_type": {"size": 70, "modal": "vehicle"},
    "color": {"size": 60, "modal": "vehicle"},
    "brand_logo": {"size": 50, "modal": "vehicle"},
    "plate_region": {"size": 40, "modal": "vehicle"},
    "damage": {"size": 20, "modal": "vehicle"},
    "angle": {"size": 20, "modal": "vehicle"},
}


# ==============================
# 函数1：向量合成
# ==============================
def synthesize_vectors(
        is_same_target: bool,
        modal_type: str,  # "face", "body", "vehicle", "all"
        num_vectors: int,
        quality: float,  # 0.0~1.0，成像质量
        dist_params: Dict[str, Dict[str, float]] = None
) -> List[Dict]:
    """
    输入：
        is_same_target: 是否同一目标
        modal_type: 人脸/人体/车辆/all
        num_vectors: 生成数量
        quality: 成像质量（影响噪声）
        dist_params: 分布参数
            {
                "same": {"mu": 0.88, "sigma": 0.03},
                "boundary": {"mu": 0.78, "sigma": 0.09},
                "diff_main": {"a": 3, "b": 8, "scale": 0.5, "loc": 0.1},
                "diff_tail": {"a": 0.5, "b": 1.0}
            }
    输出：List[dict(vector, sim, label, dims)]
    """
    if dist_params is None:
        dist_params = {
            "same": {"mu": 0.88, "sigma": 0.03},
            "boundary": {"mu": 0.78, "sigma": 0.09},
            "diff_main": {"a": 3, "b": 8, "scale": 0.5, "loc": 0.1},
            "diff_tail": {"a": 0.5, "b": 1.0}
        }

    # 过滤模态
    active_dims = {k: v for k, v in DIMENSIONS.items() if modal_type in ("all", v["modal"])}

    # 锚点
    anchor = np.random.randn(DIM)
    anchor /= np.linalg.norm(anchor)

    results = []
    for i in range(num_vectors):
        if is_same_target:
            target = "same"
            mu, sigma = dist_params["same"]["mu"], dist_params["same"]["sigma"]
            sim = np.clip(norm(mu, sigma * (1 - quality)).rvs(), 0.75, 0.98)
        else:
            if np.random.rand() < 0.3:  # 30% 临界
                target = "boundary"
                mu, sigma = dist_params["boundary"]["mu"], dist_params["boundary"]["sigma"]
                sim = np.clip(norm(mu, sigma * (1 - quality)).rvs(), 0.65, 0.88)
            elif np.random.rand() < 0.8:  # 50% 主峰
                target = "diff_main"
                p = dist_params["diff_main"]
                sim = beta(p["a"], p["b"]).rvs() * p["scale"] + p["loc"]
            else:  # 20% 长尾
                target = "diff_tail"
                p = dist_params["diff_tail"]
                sim = beta(p["a"], p["b"]).rvs()

        # 构造维度
        dims = {}
        v = np.zeros(DIM)
        start = 0
        for name, config in active_dims.items():
            size = config["size"]
            if target == "same":
                val = norm(0.9, 0.05 * (1 - quality)).rvs()
            elif target == "boundary":
                val = norm(0.8, 0.08 * (1 - quality)).rvs() if name != "accessory" else 1.0
            else:
                val = np.random.beta(2, 5) if target == "diff_main" else np.random.beta(0.6, 1.2)
            val = np.clip(val, 0, 1)
            v[start:start + size] = (val * 2 - 1)
            dims[name] = val
            start += size

        # 调整到目标相似度
        v = v / np.linalg.norm(v)
        v = sim * anchor + np.sqrt(max(0, 1 - sim ** 2)) * v
        v = v / np.linalg.norm(v)
        actual_sim = np.dot(v, anchor)

        results.append({
            "vector": v,
            "sim": float(actual_sim),
            "label": target,
            "dims": dims,
            "modal": modal_type
        })

    return results


if __name__ == "__main__":
    results = synthesize_vectors(
        is_same_target=True,
        modal_type="all",
        num_vectors=10,
        quality=0.9,
        dist_params={
            "same": {"mu": 0.88, "sigma": 0.03},
            "boundary": {"mu": 0.78, "sigma": 0.09},
            "diff_main": {"a": 3, "b": 8, "scale": 0.5, "loc": 0.1},
        }
    )
