def prepare_train_features(examples,tokenizer,pad_on_right=True,max_length=512,doc_stride=128):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    
    #g = examples['para'].split("</s> ")
    #p ="</s> ".join(g[1:])
    #print(p)
    tokenized_examples = tokenizer(
            examples['para'],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    
    

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    remove = []
    # print(tokenized_examples)
    #print(offset_mapping)
    #print(len(offset_mapping))
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        
        input_ids = tokenized_examples["input_ids"][i]
        
        cls_index = input_ids.index(tokenizer.eos_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
#         answer = examples["answers"][sample_index]
        answer = examples['target_text']
        # If no answers are given, set the cls_index as answer.
#         try:
        # Start/end character index of the answer in the text.
        context = examples['para']
        #context = p
        
        start_char = context.lower().find(answer.lower())
        if start_char == -1: # not find
#             tokenized_examples["start_positions"].append(cls_index)
#             tokenized_examples["end_positions"].append(cls_index)
            remove.append(i)
            continue
            
        
        end_char = start_char + len(answer)

        # Start token index of the current span in the text.
        token_start_index = 0
        
        while sequence_ids[token_start_index] != 0:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] !=0:
            token_end_index -= 1
        
        #print(token_start_index, token_end_index)
        # Detect if the answer is out of the span (in which case we remove this split).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            remove.append(i)
        
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
    new_input_ids = []
    new_target_ids = []
    for i,(ids, attn) in enumerate(zip(tokenized_examples['input_ids'],tokenized_examples['attention_mask'])):
        if i not in remove:
            new_input_ids.append(ids)
            new_target_ids.append(attn)
    tokenized_examples['input_ids'] = new_input_ids
    tokenized_examples['attention_mask']=new_target_ids
    return tokenized_examples