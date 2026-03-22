full_special_tokens_map = {
    'Qwen': {
        'latent_mention_token': '<|fim_middle|>',
        'latent_entity_token': '<|fim_pad|>',
    },
    'Llama': {
        'latent_mention_token': '<|reserved_special_token_0|>',
        'latent_entity_token': '<|reserved_special_token_1|>',
    }
}

mention_start_token = '<MENTION>'
mention_end_token = '</MENTION>'

prompt_templates = {
    'mention': '{input_text}\n\nThe mention "{mention}" between <MENTION> and </MENTION> could refer to an entity with the name of ',
    'entity': '{description_text}\n\nThe text is describing an entity named ',
}