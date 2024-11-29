import torch
import torch.nn as nn
import torch.nn.functional as F
from env import *

def compute_attention_scores(document_sections, d_model):
    """
    Compute attention scores for the given document sections without a prompt.
    This is a simplified version to check behavior.

    Args:
        document_sections (torch.Tensor): Tensor of shape (num_sections, d_model).
        d_model (int): Dimension of the embedding.

    Returns:
        torch.Tensor: Attention scores and weighted values.
    """
    # Keys and Values are derived from document sections
    keys = document_sections  # Shape: (num_sections, d_model)
    values = document_sections  # Shape: (num_sections, d_model)

    # Assume self-attention where query = key
    query = document_sections.mean(dim=0, keepdim=True)  # Shape: (1, d_model)

    # 1. Attention Score Calculation: Q x K^T
    attention_scores = torch.matmul(query, keys.T)  # Shape: (1, num_sections)

    # 2. Scaling and Softmax
    attention_scores = F.softmax(attention_scores / (d_model ** 0.5), dim=-1)  # Shape: (1, num_sections)

    # 3. Weighted Sum of Values
    weighted_values = torch.matmul(attention_scores, values)  # Shape: (1, d_model)

    return attention_scores, weighted_values


def compute_self_attention(tokens, d_model):
    """
    일반적인 Self-Attention 계산 (i->j) 형태의 어텐션 스코어를 구합니다.

    Args:
        tokens (torch.Tensor): 입력 토큰 임베딩 (shape: [num_tokens, d_model]).
        d_model (int): 임베딩 차원.

    Returns:
        torch.Tensor: Self-Attention 스코어 행렬 (shape: [num_tokens, num_tokens]).
    """
    # Q, K, V 생성 (Self-Attention에서 Q, K, V는 동일한 tokens에서 나옴)
    queries = tokens  # Shape: [num_tokens, d_model]
    keys = tokens  # Shape: [num_tokens, d_model]
    values = tokens  # Shape: [num_tokens, d_model]

    # 1. 어텐션 점수 계산 (Q x K^T)
    attention_scores = torch.matmul(queries, keys.T) / (d_model ** 0.5)  # Shape: [num_tokens, num_tokens]
    
    # 2. softmax를 통해 어텐션 값 정규화
    attention_probs = F.softmax(attention_scores, dim=-1)  # Shape: [num_tokens, num_tokens]

    # 3. 어텐션 값과 V를 곱하여 최종 어텐션 결과 계산
    attention_output = torch.matmul(attention_probs, values)  # Shape: [num_tokens, d_model]

    return attention_scores, attention_output
