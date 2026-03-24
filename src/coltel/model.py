import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model


class ColtelTokenizer(AutoTokenizer):
    def __init__(
        self,
    ):
        super().__init__()
        raise OSError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, special_tokens_map, *inputs, **kwargs):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            *inputs,
            **kwargs,
        )

        tokenizer.latent_mention_token = special_tokens_map['latent_mention_token']
        tokenizer.latent_entity_token = special_tokens_map['latent_entity_token']

        tokenizer.latent_mention_token_id = tokenizer.convert_tokens_to_ids(tokenizer.latent_mention_token)
        tokenizer.latent_entity_token_id = tokenizer.convert_tokens_to_ids(tokenizer.latent_entity_token)

        return tokenizer


class ColtelDecoder(nn.Module):
    """Latent decoder model with a Transformer structure."""
    def __init__(
        self,
        backbone,
        adapter_name,
        projector_type='linear',
    ):
        super().__init__()
        self.config = backbone.config
        self.backbone = backbone
        self.adapter_name = adapter_name

        if projector_type == 'linear':
            self.projector_type = 'linear'
            self.projector = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        elif projector_type == '2xMLP':
            self.projector_type = '2xMLP'
            self.projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False),
                nn.LayerNorm(self.config.hidden_size),
            )
        else:
            raise NotImplementedError('Unsupported projector type')

    def forward(
        self,
        seed_embeddings,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        """nn.Module forward()."""
        self.backbone.set_adapter(self.adapter_name)
        _ = (position_ids, past_key_values, use_cache) # parameter sink

        # seed_embedding_shape: [bsz, hidden_dim]
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            embed_layer = self.backbone.get_input_embeddings()
            inputs_embeds: torch.Tensor = embed_layer(input_ids)

        batch_size, seed_len = seed_embeddings.shape[0], seed_embeddings.shape[1]

        projected_seed = self.projector(seed_embeddings) # [bsz, seed_len, hidden_dim]
        inputs_embeds = torch.cat([projected_seed, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [torch.ones((batch_size, seed_len), device=attention_mask.device), attention_mask],
            dim=1,
        )
        labels = torch.cat(
            [torch.full((batch_size, seed_len), fill_value=-100, device=labels.device), labels],
            dim=1,
        )

        return self.backbone(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=None,
            cache_position=None,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        seed_embeddings,
        llm_generation_config,
        **kwargs,
    ):
        """Generates texts based on seed embeddings."""
        self.backbone.set_adapter(self.adapter_name)
        batch_size, seed_len = seed_embeddings.shape[0], seed_embeddings.shape[1]

        projected_seed = self.projector(seed_embeddings)
        attention_mask = torch.ones(
            (batch_size, seed_len),
            dtype=torch.long,
            device=projected_seed.device,
        )

        generation_result = self.backbone.generate(
            inputs_embeds=projected_seed,
            attention_mask=attention_mask,
            generation_config=llm_generation_config,
            **kwargs,
        )
        return generation_result


class ColtelModel(nn.Module):
    """The main CoLT model."""
    def __init__(
        self,
        llm,
        tokenizer,
        lora_configs,
        projector_type='linear',
        seed_len=1,
        **_kwargs,
    ):
        super(ColtelModel, self).__init__()

        if isinstance(llm, str):
            self.backbone = AutoModelForCausalLM.from_pretrained(llm)
        else:
            self.backbone = llm
        self.tokenizer = tokenizer

        if tokenizer.pad_token_id is None:
            self.pad_token = tokenizer.eos_token
            self.pad_token_id = tokenizer.eos_token_id
        else:
            self.pad_token = tokenizer.pad_token
            self.pad_token_id = tokenizer.pad_token_id

        self.latent_mention_token = tokenizer.latent_mention_token
        self.latent_mention_token_id = tokenizer.latent_mention_token_id
        self.latent_entity_token = tokenizer.latent_entity_token
        self.latent_entity_token_id = tokenizer.latent_entity_token_id

        self.llm_hidden_dim = llm.config.hidden_size
        self.seed_length = seed_len

        # register adapters
        config_decoder = lora_configs['decoder']
        config_mention = lora_configs['mention']
        config_entity = lora_configs['entity']

        self.backbone = get_peft_model(self.backbone, config_decoder, adapter_name="adapter_decoder")
        self.backbone.add_adapter(adapter_name="adapter_mention", peft_config=config_mention)
        self.backbone.add_adapter(adapter_name="adapter_entity", peft_config=config_entity)

        for name, param in self.backbone.named_parameters():
            if "embed_tokens" in name:
                param.requires_grad = True

        self.mention_decoder = ColtelDecoder(
            backbone=self.backbone,
            adapter_name="adapter_decoder",
            projector_type=projector_type,
        )
        self.entity_decoder = ColtelDecoder(
            backbone=self.backbone,
            adapter_name="adapter_decoder",
            projector_type=projector_type,
        )
    
    def set_adapter_trainable(self, adapter_name, trainable=True):
        for name, param in self.named_parameters():
            if adapter_name in name:
                param.requires_grad = trainable

    @property
    def device(self):
        """The device of model parameters."""
        return next(self.parameters()).device

    def get_optimizer(
        self,
        learning_rate=1e-5,
        weight_decay=0.01,
    ):
        """Get an optimizer for the CoLT model."""
        params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay
        )

        return optimizer

    def forward(
        self,
        batch,
        **_kwargs,
    ):
        """nn.Module.forward()."""
        # 1. CE loss
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        input_type = batch['input_type']
        batch_size = input_ids.shape[0]

        if (input_type == 0): # mention data
            self.backbone.set_adapter("adapter_mention")
        elif (input_type == 1): # entity data
            self.backbone.set_adapter("adapter_entity")

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        ce_loss = outputs.loss

        # 2. Latent loss
        last_hidden_states = outputs.hidden_states[-1]

        # 2.1. Mention loss
        mention_mask = (input_ids == self.latent_mention_token_id)
        if (mention_mask.sum() == 0):
            mention_loss = last_hidden_states.view(-1)[0] * 0.0 # dummy loss
            mention_latents = None
        else:
            seed_embeddings = last_hidden_states[mention_mask].view(-1, self.seed_length, self.llm_hidden_dim)
            decoder_outputs = self.mention_decoder(
                seed_embeddings=seed_embeddings,
                input_ids=batch['decoder_input_ids'],
                attention_mask=batch['decoder_attention_mask'],
                labels=batch['decoder_labels'],
            )
            mention_loss = decoder_outputs.loss
            mention_latents = seed_embeddings

        # 2.2. Entity loss
        entity_mask = (input_ids == self.latent_entity_token_id)
        if (entity_mask.sum() == 0):
            entity_loss = last_hidden_states.view(-1)[0] * 0.0 # dummy loss
            entity_latents = None
        else:
            seed_embeddings = last_hidden_states[entity_mask].view(-1, self.seed_length, self.llm_hidden_dim)
            decoder_outputs = self.mention_decoder(
                seed_embeddings=seed_embeddings,
                input_ids=batch['decoder_input_ids'],
                attention_mask=batch['decoder_attention_mask'],
                labels=batch['decoder_labels'],
            )
            entity_loss = decoder_outputs.loss
            entity_latents = seed_embeddings

        loss = ce_loss + mention_loss + entity_loss

        return {
            'loss': loss,
            'ce_loss': ce_loss,
            'mention_loss': mention_loss,
            'mention_latents': mention_latents.flatten(start_dim=1) if mention_latents is not None else None,
            'entity_loss': entity_loss,
            'entity_latents': entity_latents.flatten(start_dim=1) if entity_latents is not None else None,
        }
    
    def predict(
        self,
        batch,
        llm_generation_config,
        **kwargs,
    ):
        """Generates the reasoning trail and the final answer with possible latent tool calls."""
        tokenize_results = self.tokenizer(
            batch['query'] + batch['seed_tokens'],
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
            padding_side='left',
        )
        question_input_ids = tokenize_results['input_ids'].to(self.device)
        question_attention_mask = tokenize_results['attention_mask'].to(self.device)

        input_type = batch['input_type']
        if (input_type == 0): # mention data
            self.backbone.set_adapter("adapter_mention")
        elif (input_type == 1): # entity data
            self.backbone.set_adapter("adapter_entity")

        outputs = self.backbone(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1]

        if (input_type == 0):
            mention_mask = (question_input_ids == self.latent_mention_token_id)
            seed_embeddings = last_hidden_states[mention_mask].view(-1, self.seed_length, self.llm_hidden_dim)
        elif (input_type == 1):
            entity_mask = (question_input_ids == self.latent_entity_token_id)
            seed_embeddings = last_hidden_states[entity_mask].view(-1, self.seed_length, self.llm_hidden_dim)

        decoder_outputs = self.mention_decoder.generate(
            seed_embeddings=seed_embeddings,
            llm_generation_config=llm_generation_config,
            **kwargs,
        )

        return decoder_outputs