import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from env import *
from self_attention import *
from infinyPot import *
from data import *
from plot import *

import torch

for txt in text:

    input_ids, outputs, tokens = text_tokening(txt)
    prompt_tokens = []
    for key, prmt in prompt.items():  # 프롬프트 텍스트
        # 프롬프트를 고려한 어텐션 계산
        prompt_attention_scores = compute_prompt_attention(prmt, tokens, d_model)
        cap_tokens, nuc_tokens, survived_tokens = infiniPot(prompt_attention_scores, tokens, outputs, input_ids)

        # 결과 출력
        print("Prompt-Attention Scores: ", key)
        # # print(prompt_attention_scores)
        # print("cap text: ", tokens_to_text(input_ids, cap_tokens))
        # print("nuc text: ", tokens_to_text(input_ids, nuc_tokens))
        # print("survived text: ", tokens_to_text(input_ids, survived_tokens))
        print("cap tokens: ", cap_tokens)
        print("nuc tokens: ", nuc_tokens)
        print("total tokens: ", survived_tokens)
        prompt_tokens.append([key, [cap_tokens, nuc_tokens, survived_tokens]])

    general_attention_scores, _ = compute_attention_scores(tokens, d_model)
    #print("general_important_tokens: ", torch.topk(general_attention_scores, k=M, largest=True))
    sur_tokens = torch.topk(general_attention_scores, k=C if len(general_attention_scores[0]) >= M else len(general_attention_scores[0]), largest=True)
    #print("general; ", general_attention_scores)
    gen_survived = sorted(sur_tokens.indices.tolist()[0])
    print("sur tokens in gen: ", gen_survived)

    decoded_text = tokens_to_text(input_ids, gen_survived)  # input_ids[0]은 배치 차원이므로 첫 번째 항목을 선택
    total_token_length = survived_tokens[-1]+1
    print("total token length: ", total_token_length)
    #print(f"original text: {txt}")
    #print(f"Decoded text: {decoded_text}")
    print("\n")
    hit_plot(prompt_tokens, gen_survived, total_token_length)

plt.tight_layout()
plt.show()




# general_attention_scores, _ = compute_self_attention(tokens, d_model)
# print("i_j attn; ", general_attention_scores)
# li=[]
# for vec in general_attention_scores:
#     li.append(sum(vec))
# for ui in li:
#     print("sum: ", ui/sum(li))
# Print result

