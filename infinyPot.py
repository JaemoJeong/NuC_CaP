import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from env import *
from self_attention import *

model_name="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def tokens_to_text(input_ids, tokens=list(range(0,5))):
    """
    Convert a list of token indices to the corresponding text.

    Args:
        tokens (list[int]): List of token indices.
        tokenizer: Tokenizer object to decode the tokens.

    Returns:
        str: Corresponding text from the token indices.
    """
    # 토큰 인덱스를 텍스트로 변환
    token_ids = []
    id_li = input_ids[0].tolist()
    # print("id_li ", id_li, tokens)
    for token in tokens:
        token_ids.append(id_li[token])
    decoded_text = tokenizer.decode(torch.tensor(token_ids), skip_special_tokens=True)
    decoded_text = decoded_text.replace("[unused]", "")
    return decoded_text


def text_to_embedding(text):
    """
    Convert text to a fixed-size embedding using a pre-trained language model.

    Args:
        text (str): Input text (e.g., prompt).
        model_name (str): Name of the pre-trained model to use.

    Returns:
        torch.Tensor: Embedding of the input text (shape: [1, original_d_model]).
    """
    # Load tokenizer and model

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Pass the input through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token's embedding as a representation of the entire text
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [1, original_d_model]
    return cls_embedding


def text_tokening(text, model_name="bert-base-uncased"):
    """
    Convert text to token-level embeddings using a pre-trained language model.

    Args:
        text (str): Input text (e.g., prompt).
        model_name (str): Name of the pre-trained model to use.

    Returns:
        torch.Tensor: Embeddings of the input text at token level (shape: [num_tokens, d_model]).
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Pass the input through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get all token embeddings from the last hidden state
    token_embeddings = outputs.last_hidden_state  # Shape: [1, num_tokens, d_model]
    
    # Remove batch dimension to get the embeddings for tokens
    token_embeddings = token_embeddings.squeeze(0)  # Shape: [num_tokens, d_model]
    return inputs.input_ids, outputs, token_embeddings


class AttentionWithTokenInteraction(nn.Module):
    def __init__(self, original_d_model, d_model):
        """
        프롬프트와 토큰 간 상호작용을 포함한 어텐션 계산.
        
        Args:
            original_d_model (int): 원본 임베딩 차원 (e.g., BERT의 768).
            d_model (int): 변환 후의 차원.
        """
        super().__init__()
        self.projection = nn.Linear(original_d_model, d_model)

    def forward(self, prompt, document_tokens):
        """
        프롬프트와 섹션의 각 토큰 간의 어텐션 계산.

        Args:
            prompt (torch.Tensor): 프롬프트 임베딩 (shape: [1, original_d_model]).
            document_tokens (torch.Tensor): 문서 토큰 (shape: [num_tokens, d_model]).

        Returns:
            torch.Tensor: Attention scores (shape: [num_tokens]),
                          Weighted result (shape: [1, d_model]).
        """
        # Project prompt embedding to d_model
        query = self.projection(prompt)  # Shape: [1, d_model]

        # Project document tokens
        keys = document_tokens  # Shape: [num_tokens, d_model]
        values = document_tokens  # Shape: [num_tokens, d_model]

        # 1. Attention Score Calculation: Q x K^T
        attention_scores = torch.matmul(query, keys.T)  # Shape: [1, num_tokens]

        # 2. Scaling and Softmax
        attention_scores = F.softmax(attention_scores / (keys.size(-1) ** 0.5), dim=-1)  # Shape: [1, num_tokens]

        # 3. Weighted Sum of Values
        weighted_values = torch.matmul(attention_scores, values)  # Shape: [1, d_model]

        return attention_scores, weighted_values
    
class AttentionWithPrompt(nn.Module):
    def __init__(self, original_d_model, d_model):
        """
        Initialize the attention model with dimension adjustments.
        
        Args:
            original_d_model (int): Dimension of the original embeddings (e.g., 768 for BERT).
            d_model (int): Desired dimension for calculations.
        """
        super().__init__()
        self.projection = nn.Linear(original_d_model, d_model)  # BERT 임베딩 크기를 d_model로 변환하는 선형 변환

    def forward(self, prompt, document_sections):
        """
        Calculate attention scores based on the given prompt and document sections.

        Args:
            prompt (torch.Tensor): Prompt embedding of shape (1, original_d_model).
            document_sections (torch.Tensor): Section embeddings of shape (num_sections, d_model).

        Returns:
            torch.Tensor: Attention scores and weighted values.
        """
        # Prompt embedding을 d_model 차원으로 변환
        query = self.projection(prompt)  # Shape: (1, d_model)

        # Keys and Values are derived from document sections
        keys = document_sections  # Shape: (num_sections, d_model)
        values = document_sections  # Shape: (num_sections, d_model)

        # Attention Score Calculation: Q x K^T
        attention_scores = torch.matmul(query, keys.T)  # Shape: (1, num_sections)

        # Scaling and Softmax
        attention_scores = F.softmax(attention_scores / (keys.size(-1) ** 0.5), dim=-1)  # Shape: (1, num_sections)

        # Weighted Sum of Values
        weighted_values = torch.matmul(attention_scores, values)  # Shape: (1, d_model)

        return attention_scores, weighted_values

# Output
def compute_prompt_attention(prompt_text, tokens, d_model):
    """
    프롬프트 텍스트를 고려한 토큰 간 어텐션 스코어 계산 (i->j).

    Args:
        prompt_text (str): 프롬프트 텍스트 (예: "Summarize this section.").
        tokens (torch.Tensor): 입력 토큰 임베딩 (shape: [num_tokens, d_model]).
        d_model (int): 임베딩 차원.

    Returns:
        torch.Tensor: 어텐션 스코어 행렬 (shape: [num_tokens, num_tokens]).
    """
    # 프롬프트 텍스트를 임베딩으로 변환
    prompt = text_to_embedding(prompt_text)  # Shape: [1, 768]

    # 프롬프트를 d_model 차원으로 변환
    # prompt = prompt.view(1, -1)  # BERT 임베딩 차원에서 [1, 768] 형태로 변환
    # projection = nn.Linear(768, d_model)  # 768을 d_model로 변환하는 선형 변환
    # prompt = projection(prompt)  # Shape: [1, d_model]

    # Q, K 생성
    queries = tokens  # Shape: [num_tokens, d_model]
    keys = tokens  # Shape: [num_tokens, d_model]

    # 1. 토큰 간 어텐션 계산 (Self-Attention)
    token_attention_scores = torch.matmul(queries, keys.T) / (d_model ** 0.5)  # Shape: [num_tokens, num_tokens]
    
    # 2. 프롬프트와 각 토큰 간 어텐션 계산
    prompt_scores = torch.matmul(prompt, keys.T) / (d_model ** 0.5)  # Shape: [1, num_tokens]
    prompt_attention = F.softmax(prompt_scores, dim=-1)  # Shape: [1, num_tokens]
    
    # 3. 토큰 간 어텐션에 프롬프트 영향 반영
    weighted_token_attention = token_attention_scores * prompt_attention  # Broadcasting 적용
    final_attention_scores = F.softmax(weighted_token_attention, dim=-1)  # Normalize

    return final_attention_scores

def cap(flat_tokens):
    u_t = []
    for token in flat_tokens:
        value = 0
        for i in range(M-P, M):
            value += token[i]
        u_t.append(value.item())
        #print("value: ", value)
    
    return torch.tensor(u_t)
    
def nuc(input_ids, outputs, context_indices):
    """
    특정 `target_indices` 내 토큰들에 대해,
    해당 토큰의 로그 확률을 이전의 `target_indices` 내 토큰들로만 예측하도록 NuC 점수를 계산합니다.

    Args:
        input_ids (torch.Tensor): 입력 시퀀스 ([batch_size, seq_len]).
        target_indices (list[int]): NuC 점수를 계산할 토큰 인덱스 리스트.
        model: 언어 모델 객체.
        tokenizer: 토크나이저 객체.

    Returns:
        list[float]: 선택된 NuC 점수 리스트.
    """
    # 모델 출력 계산
    hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    vocab_size = tokenizer.vocab_size
    lm_head = torch.nn.Linear(model.config.hidden_size, vocab_size)
    logits = lm_head(hidden_states)

    # 확률 분포 계산
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]

    nt_scores = []

    for t in range(input_ids.size(1)):  # 각 토큰에 대해 계산
        token_id = input_ids[0][t]

        if t > len(context_indices):
            # t > |C|: 컨텍스트 + 시퀀스 일부 사용
            past_indices = context_indices + list(range(len(context_indices), t))
        else:
            # t <= |C|: 컨텍스트 내부 계산
            past_indices = context_indices[:t]

        # `past_indices`에 포함된 토큰 ID만 활성화된 마스크 생성
        context_token_ids = input_ids[0][past_indices]
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[context_token_ids] = True

        # 마스킹 처리한 로그 확률 계산
        masked_log_probs = log_probs[0, t].clone()
        masked_log_probs[~mask] = float('-inf')

        # NuC 점수 계산 (-log 확률)
        log_prob = masked_log_probs[token_id]
        nt_scores.append(-log_prob.item())

    return nt_scores


def infiniPot(document_sections, origin_tokens, outputs, input_ids):
    survived_tokens = list(range(0, M))  # List to store indices of surviving tokens
    num_sections = document_sections.size(0)  # Total number of sections
    start_idx = M-1
    cap_tokens = []
    nuc_tokens = []
    if(start_idx >= num_sections):
        return_tokens = list(range(0, num_sections))
        return return_tokens, return_tokens, return_tokens
    
    while start_idx < num_sections-1:
        # P개의 섹션을 처리 (남은 섹션이 P보다 적다면 적은 섹션만 선택)
        end_idx = min(start_idx + M-C, num_sections-1)
        
        # survived_tokens로 선택된 토큰 인덱스들만 가져오기
        selected_indices = torch.tensor(survived_tokens, dtype=torch.long).view(-1)  # 원하는 토큰 인덱스들
        
    
        current_sections = torch.index_select(document_sections, dim=0, index=selected_indices)
        
        
        # 단일 토큰 크기로 펼쳐서 점수 계산
        flat_tokens = current_sections
        #.view(-1, d_model)  # Shape: [P * 1, d_model]

        # 사용자 정의 점수 계산 함수 (예시)
        scores = cap(flat_tokens)  # Shape: [P * num_tokens]
        
        nt_scores = nuc(input_ids, outputs, survived_tokens)
        #print("nt Scores:", nt_scores)

        # 점수를 기반으로 높은 점수의 토큰 top_k개를 선택
        _, top_k_indices_by_cap = torch.topk(scores, k=T, largest=True)  # Shape: [top_k]
        top_k_indices_by_nuc = []
        for i, nt in enumerate(nt_scores):
            top_k_indices_by_nuc.append((nt, i))
        top_k_indices_by_nuc.sort()
        survived_tokens = []
        cap_tokens = []
        nuc_tokens = []
        for top in top_k_indices_by_cap:
            survived_tokens.append(top.item())
            cap_tokens.append(top.item())
        for top, idx in top_k_indices_by_nuc:
            if len(survived_tokens) >= C:
                break
            if idx not in survived_tokens:
                survived_tokens.append(idx)
                nuc_tokens.append(idx)

        #print("now sur: ", survived_tokens)
        while start_idx < end_idx:
            start_idx+=1
            survived_tokens.append(start_idx)
        survived_tokens.sort()
        #print("sur ", survived_tokens)

    return cap_tokens, nuc_tokens, survived_tokens
