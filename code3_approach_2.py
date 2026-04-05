import torch
from tqdm import tqdm
from utils import PromptUtils
import random


def get_query_span(inputs, tokenizer, question, putils):
    """
    Identify the token span corresponding to the query.
    Moved here from run3.py to break the circular import.
    """
    query_prompt = f"Query: {question}" + "\nCorrect tool_id:"
    query_tokens = tokenizer(query_prompt, add_special_tokens=False).input_ids
    query_len = len(query_tokens)
    total_length = inputs.input_ids.shape[1]  # shape is [1, sequence_length]

    end_idx = total_length - putils.prompt_suffix_length
    start_idx = end_idx - query_len
    return (start_idx, end_idx)


def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns (query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # Accumulate scores per (layer, head)
    head_scores = torch.zeros(num_layers, num_heads, device=device)

    for qix in tqdm(range(len(train_queries))):
        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)

        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=tool_ids,
            dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            attentions = model(**inputs).attentions  # tuple of num_layers x [1, num_heads, N, N]

        query_span = get_query_span(inputs, tokenizer, question, putils)
        gold_tool_span = item_spans[map_docname_id[gold_tool_name]]

        # Score each (layer, head) by how much attention the query pays to the gold tool
        for layer_idx, layer_attn in enumerate(attentions):
            query_attn = layer_attn[0, :, query_span[0]:query_span[1], :]  # [heads, q_len, N]
            
            gold_attn = query_attn[:, :, gold_tool_span[0]:gold_tool_span[1]].sum(dim=(1, 2))  # [heads]
            
            # Sum attention to ALL tools (not the full sequence, just the tool spans)
            total_attn = torch.zeros(num_heads, device=device)
            for span in item_spans:
                total_attn += query_attn[:, :, span[0]:span[1]].sum(dim=(1, 2))
            
            # Fraction of attention going to gold tool
            head_scores[layer_idx] += gold_attn / (total_attn + 1e-9)

    # Select top max_heads (layer, head) pairs by score
    selected_heads = []
    top_values, top_indices = torch.topk(head_scores.view(-1), max_heads)
    for idx in top_indices:
        idx_val = idx.item()
        layer = idx_val // num_heads
        head = idx_val % num_heads
        selected_heads.append((layer, head))

    # expected format: [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads