import torch
from tqdm import tqdm
from run3 import get_query_span
from utils import PromptUtils
import random 

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
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
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions 

        # Add your head scoring logic after this line
        query_span = get_query_span(inputs, tokenizer, question, putils)
        gold_tool_span = item_spans[map_docname_id[gold_tool_name]]

        query_attn = attentions[-1][:, :, query_span[0]:query_span[1], :]
        gold_tool_attn = query_attn[:, :, :, gold_tool_span[0]:gold_tool_span[1]]
        head_scores += gold_tool_attn.sum(dim=(3, 4)).squeeze(1)

    # TODO: select top heads
    selected_heads = []
    
    top_values, top_indices = torch.topk(head_scores.view(-1), max_heads)
    for idx in top_indices:
        idx_val = idx.item()
        layer = idx_val // num_heads
        head = idx_val % num_heads
        selected_heads.append((layer, head))

    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads