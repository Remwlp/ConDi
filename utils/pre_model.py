# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """

import math

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torch.nn.functional as F

from transformers.activations import ACT2FN, gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.roberta.configuration_roberta import RobertaConfig


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


# class RobertaEmbeddings(nn.Module):
#     """
#     Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
#     """
#
#     # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
#
#         # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
#         # any TensorFlow checkpoint file
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # position_ids (1, len position emb) is contiguous in memory and exported when serialized
#         self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
#         self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
#         if version.parse(torch.__version__) > version.parse("1.6.0"):
#             self.register_buffer(
#                 "token_type_ids",
#                 torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
#                 persistent=False,
#             )
#
#         # End copy
#         self.padding_idx = config.pad_token_id
#         self.position_embeddings = nn.Embedding(
#             config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
#         )
#
#     def forward(
#         self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
#     ):
#         if position_ids is None:
#             if input_ids is not None:
#                 # Create the position ids from the input token ids. Any padded tokens remain padded.
#                 position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
#             else:
#                 position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
#
#         if input_ids is not None:
#             input_shape = input_ids.size()
#         else:
#             input_shape = inputs_embeds.size()[:-1]
#
#         seq_length = input_shape[1]
#
#         # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
#         # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
#         # issue #5664
#         if token_type_ids is None:
#             if hasattr(self, "token_type_ids"):
#                 buffered_token_type_ids = self.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
#
#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#
#         embeddings = inputs_embeds + token_type_embeddings
#         if self.position_embedding_type == "absolute":
#             position_embeddings = self.position_embeddings(position_ids)
#             embeddings += position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings
#
#     def create_position_ids_from_inputs_embeds(self, inputs_embeds):
#         """
#         We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
#
#         Args:
#             inputs_embeds: torch.Tensor
#
#         Returns: torch.Tensor
#         """
#         input_shape = inputs_embeds.size()[:-1]
#         sequence_length = input_shape[1]
#
#         position_ids = torch.arange(
#             self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
#         )
#         return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta

class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]

            # if self.config.is_decoder:
            #     batch_size, seq_length = input_shape
            #     seq_ids = torch.arange(seq_length, device=device)
            #     causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            #     # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            #     # causal and attention masks must have same type with pytorch version < 1.3
            #     causal_mask = causal_mask.to(attention_mask.dtype)
            #
            #     if causal_mask.shape[1] < attention_mask.shape[1]:
            #         prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            #         causal_mask = torch.cat(
            #             [
            #                 torch.ones(
            #                     (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
            #                 ),
            #                 causal_mask,
            #             ],
            #             axis=-1,
            #         )
            #
            #     extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            # else:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

############################################################################################

import copy

class BertPooler_i2t(nn.Module):
    def __init__(self):
        super(BertPooler_i2t, self).__init__()
        self.dense = nn.Linear(768,768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPooler(nn.Module):
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]   
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCoAttention(nn.Module):
    def __init__(self):
        super(BertCoAttention, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        # s2_attention_mask  b*1*1*49
        mixed_query_layer = self.query(s1_hidden_states)  # b*75*768
        mixed_key_layer = self.key(s2_hidden_states)  # b*49*768
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b*12*75*64
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b*12*49*64
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b*12*49*64
        # print('q',query_layer.size())
        # print('k',key_layer.size())
        # print('v',value_layer.size())

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1, -2))  # b*12*75*49
        # print('qk',attention_scores.size())
        # print('att', s2_attention_mask.size())
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # b*12*75*49
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask
        # attention_scores b*12*75*49
        # Normalize the attention scores to probabilities.
        # b*12*75*49
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # aa = attention_probs.cpu().numpy()
        # aa = numpy.sum(aa,axis=1,keepdims=True)
        # aa = numpy.sum(aa,axis=2,keepdims=True)
        # aa = numpy.linalg.norm(aa, ord=None, axis=-2, keepdims=True)
        # with open('/data/qiaoyang/ms_data/aa.pkl', 'wb') as f:
        #     pickle.dump(aa, f)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)      # (32,12,197,64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()      # (32,197,12,64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)     # (32,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer b*75*768
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttention(nn.Module):
    def __init__(self):
        super(BertCrossAttention, self).__init__()
        self.bertCoAttn = BertCoAttention()
        self.output = BertSelfOutput()

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self):
        super(BertCrossAttentionLayer, self).__init__()
        self.bertCorssAttn = BertCrossAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.bertCorssAttn(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        # b*75*768
        intermediate_output = self.intermediate(attention_output)
        # b*75*3072
        layer_output = self.output(intermediate_output, attention_output)
        # b*75*3072
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(3)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        return s1_hidden_states

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


# class CrossPreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm_source = nn.LayerNorm(dim)
#         self.norm_target = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, source_x, target_x, **kwargs):
#         source_x = self.norm_source(source_x)
#         target_x = self.norm_target(target_x)
#
#         return self.fn(source_x,target_x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        # print("1", q.shape, k.shape, v.shape)
        b, n, _, h = *q.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim = -1)

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # print("2", q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, x, x) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, tgt, memory):
        for attn1, attn2, ff in self.layers:
            tgt = attn1(tgt, tgt, tgt) + tgt
            tgt = attn1(tgt, memory, memory) + tgt
            tgt = ff(tgt) + tgt
        return tgt



class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x = attn(target_x, source_x, source_x) + target_x
            target_x = ff(target_x) + target_x
        return target_x



class Transformer(nn.Module):
    def __init__(self, *, num_frames, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, num_frames, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.decoder = TransformerDecoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, input):
        # x = self.to_patch_embedding(img)
        b, n, _ = input.shape

        tgt = input + self.pos_embedding[:, :n]
        x = self.dropout(tgt)
        x = self.encoder(x)
        # x_rec = self.decoder(torch.flip(input + self.pos_embedding_decoder[:, :n], dims=[1]), x)

        return x

    
class ProjetTransformer(nn.Module):
    def __init__(self, num_frames, dim, depth, heads, mlp_dim, dim_head = 64, len_invariant = 8, len_specific = 8, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.cls_token = nn.Parameter(torch.randn(1, len_invariant+len_specific, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames+len_invariant+len_specific, dim))

        self.dropout = nn.Dropout(emb_dropout)


    def forward(self, input):
        # x = self.to_patch_embedding(img)
        b, n, _ = input.shape
        projet_tokens = repeat(self.cls_token, '1 f d -> b f d', b = b)
        x = torch.cat((projet_tokens, input), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        return x

class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        source_x = source_x + self.pos_embedding_s[:, : n_s]
        target_x = target_x + self.pos_embedding_t[:, : n_t]
        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t
