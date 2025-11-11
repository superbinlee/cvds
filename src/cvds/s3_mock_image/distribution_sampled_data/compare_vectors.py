# ==============================
# 函数2：向量比对
# ==============================


def compare_vectors(
        vec1: np.ndarray,
        vec2: np.ndarray,
        target_type: Optional[str] = None,
        dist_params: Dict = None
) -> float:
    """
    输入：两个向量 + 可选目标类型
    输出：校准后的相似度（服从分布）
    """
    if dist_params is None:
        dist_params = {
            "same": {"mu": 0.88, "sigma": 0.03},
            "boundary": {"mu": 0.78, "sigma": 0.09},
        }

    raw_sim = np.dot(vec1, vec2)

    if target_type == "same":
        return np.clip(norm(dist_params["same"]["mu"], dist_params["same"]["sigma"]).ppf(raw_sim), 0.75, 0.98)
    elif target_type == "boundary":
        return np.clip(norm(dist_params["boundary"]["mu"], dist_params["boundary"]["sigma"]).ppf(raw_sim), 0.65, 0.88)
    else:
        return raw_sim  # diff 不校准