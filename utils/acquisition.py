import numpy as np

# Least Confidence (lc)
def least_confidence(scores):
    return 1 - np.max(scores)

# Minimum Margin (margin)
def minimum_margin(scores):
    top2_scores = np.partition(scores, -2)[-2:]
    return top2_scores[1] - top2_scores[0]

# Token Entropy (te)
def token_entropy(scores):
    return -np.sum(scores * np.log2(scores))

# Total Token Entropy (tte)
def total_token_entropy(scores):
    return -np.sum(np.sum(scores, axis=0) * np.log2(np.sum(scores, axis=0)))

# Decay Logarithm Frequency (delfy)
def decay_log_frequency(scores, decay_factor=0.9):
    token_weights = np.arange(1, scores.shape[0] + 1)
    weighted_scores = scores * (decay_factor ** token_weights)
    return np.sum(weighted_scores)


if __name__ == "__main__":
    # 가상의 점수 배열 (예: 모델의 확률 출력)
    scores = np.array([0.1, 0.3, 0.2, 0.4])

    lc_value = least_confidence(scores)
    margin_value = minimum_margin(scores)
    te_value = token_entropy(scores)
    tte_value = total_token_entropy(scores)
    delfy_value = decay_log_frequency(scores)

    print("Least Confidence:", lc_value)
    print("Minimum Margin:", margin_value)
    print("Token Entropy:", te_value)
    print("Total Token Entropy:", tte_value)
    print("Decay Logarithm Frequency:", delfy_value)