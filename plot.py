import matplotlib.pyplot as plt
from env import *

# hit ratio 계산 함수
def calculate_hit_ratio(sur_tokens, tokens, total_length):
    intersection = len(set(sur_tokens) & set(tokens))
    return intersection / total_length

def hit_plot(prompt_tokens, gen_tokens, total_token_length):
    global table

    for key, prompt in prompt_tokens:
        cap_tokens, nuc_tokens, total_tokens = prompt
            
        # G1, G2 hit ratio 계산
        cap_hit_ratio = calculate_hit_ratio(cap_tokens, gen_tokens, total_token_length)
        nuc_hit_ratio = calculate_hit_ratio(nuc_tokens, gen_tokens, total_token_length)
        total_hit_ratio = calculate_hit_ratio(total_tokens, gen_tokens, total_token_length)
        ax = plots[key]
        blue_total_token_length = total_token_length
        orange_total_token_length = total_token_length
        red_total_token_length = total_token_length

        if cap_hit_ratio == nuc_hit_ratio or cap_hit_ratio == total_hit_ratio or nuc_hit_ratio == total_hit_ratio:
            blue_total_token_length = total_token_length+offset
            orange_total_token_length = total_token_length-offset
            
        # plot 생성
        if not table[key]:
            ax.plot([blue_total_token_length], [cap_hit_ratio], 'o-', label=f"{key} (cap)", color="blue")
            ax.plot([orange_total_token_length], [nuc_hit_ratio], 'o-', label=f"{key} (nuc)", color="orange")
            ax.plot([red_total_token_length], [total_hit_ratio], 'o-', label=f"{key} (cap + nuc)", color="red")
            table[key] = True
        else:
            ax.plot([blue_total_token_length], [cap_hit_ratio], 'o-', color="blue")
            ax.plot([orange_total_token_length], [nuc_hit_ratio], 'o-', color="orange")
            ax.plot([red_total_token_length], [total_hit_ratio], 'o-', color="red")
            # 그래프 설정
        # ax.xlabel("Total Token Length", fontsize=12)
        # ax.ylabel("Hit Ratio", fontsize=12)
        ax.legend(fontsize=12, loc="best")
        plt.grid(alpha=0.5)

        # 그래프 출력
