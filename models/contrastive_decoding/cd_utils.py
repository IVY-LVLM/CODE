import torch
from torch import nn


def contrastive_decoding (processor, model_kwargs, next_token_logits, next_token_logits_cd, results, description, is_beam=False):
    cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.2
    cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.25
                        
    p_v = nn.functional.softmax(next_token_logits, dim=-1)
    p_d = nn.functional.softmax(next_token_logits_cd, dim=-1)
    
    kl_d = 0.5 * ((torch.log2(torch.abs(p_v - p_d) ** cd_alpha + 1)) * (p_v + p_d)).sum(dim=-1).unsqueeze(-1)

    kld_alpha = 1 - kl_d 
    
    cutoff = kl_d * p_v.max(dim=-1, keepdim=True).values

    ##############################
    diffs = (1 + kld_alpha) * next_token_logits - kld_alpha * next_token_logits_cd
    cd_logits = diffs.masked_fill(p_v < cutoff, -float("inf"))

    next_token_logits = cd_logits
                    
    if processor is not None:
        final_probs = nn.functional.softmax(cd_logits, dim=-1)    
        next_img_probs = nn.functional.softmax(next_token_logits, dim=-1)
        next_desc_probs = nn.functional.softmax(next_token_logits_cd, dim=-1)
        
        final_tokens = torch.argmax(final_probs, dim=-1)
        from_img_tokens = torch.argmax(next_img_probs, dim=-1)
        from_desc_tokens = torch.argmax(next_desc_probs, dim=-1)
        
        img_token = processor.decode(from_img_tokens, skip_special_tokens=True)
        desc_token = processor.decode(from_desc_tokens, skip_special_tokens=True)
        final_token = processor.decode(final_tokens, skip_special_tokens=True)
        
        img_prob1 = next_img_probs[0, from_img_tokens].item()
        img_prob2 = next_img_probs[0, from_desc_tokens].item()
        img_prob3 = next_img_probs[0, final_tokens].item()
        
        desc_prob1 = next_desc_probs[0, from_img_tokens].item()
        desc_prob2 = next_desc_probs[0, from_desc_tokens].item()
        desc_prob3 = next_desc_probs[0, final_tokens].item()

        final_prob1 = final_probs[0, from_img_tokens].item()
        final_prob2 = final_probs[0, from_desc_tokens].item()
        final_prob3 = final_probs[0, final_tokens].item()
        
        results.append({
            "beta":kl_d.item(),"img_token": img_token, "img_prob1": img_prob1, "img_prob2": img_prob2, "img_prob3": img_prob3,
            "desc_token": desc_token,  "desc_prob1": desc_prob1, "desc_prob2": desc_prob2, "desc_prob3": desc_prob3,
            "final_token": final_token, "final_prob1": final_prob1, "final_prob2": final_prob2, "final_prob3": final_prob3,
        })
        description.append(final_token)
        # print(final_token)

    return cd_logits, results, description
