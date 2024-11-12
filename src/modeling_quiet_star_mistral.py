# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mistral model."""
import math
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np
from matplotlib.colors import  LogNorm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import MistralPreTrainedModel, MistralModel
from transformers.utils import (
	add_start_docstrings_to_model_forward,
	logging,
	replace_return_docstrings,
)

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "QuietStarMistralConfig"

def save_tokens_with_rewards_to_pdf(input_ids, token_rewards, tokenizer, output_file="text.pdf", eps=0.2, eps2=0.5):
	c = canvas.Canvas(output_file, pagesize=letter)
	c.setFont("Courier", 8)
	x, y = 50, 750
	previous_text = ""
	current_text = ""
	for token_idx, reward in enumerate(token_rewards):
		current_text = tokenizer.decode(input_ids[: token_idx + 1])
		if current_text != previous_text:
			diff_text = current_text[len(previous_text) :]
			if "\n" in diff_text:
				lines = diff_text.split("\n")
				for line_idx, line in enumerate(lines):
					if line_idx > 0:
						x = 50
						y -= 12
					if abs(reward) < eps:
						opacity = 0
					elif abs(reward) > eps2:
						opacity = 0.8
					else:
						opacity = 0.8 * (abs(reward) - eps) / (eps2 - eps)
					text_width = c.stringWidth(line)
					if reward > 0:
						highlight_color = HexColor("#4CCD99")
					else:
						highlight_color = HexColor("#FFC700")
					highlight_color.alpha = opacity
					c.setFillColor(highlight_color)
					c.rect(x, y - 2, text_width, 10, fill=True, stroke=False)
					c.setFillColor(HexColor("#000000"))
					c.drawString(x, y, line)
					x += text_width
			else:
				if abs(reward) < eps:
					opacity = 0
				elif abs(reward) > eps2:
					opacity = 0.8
				else:
					opacity = 0.8 * (abs(reward) - eps) / (eps2 - eps)
				text_width = c.stringWidth(diff_text)
				if reward > 0:
					highlight_color = HexColor("#4CCD99")
				else:
					highlight_color = HexColor("#FFC700")
				highlight_color.alpha = opacity
				c.setFillColor(highlight_color)
				c.rect(x, y - 2, text_width, 10, fill=True, stroke=False)
				c.setFillColor(HexColor("#000000"))
				c.drawString(x, y, diff_text)
				x += text_width
			if x > 550:
				x = 50
				y -= 12
			if y < 50:
				c.showPage()
				y = 750
				x = 50
			previous_text = current_text
	c.showPage()
	c.save()

def nonzero_mean(x, axis=None):
	if axis is not None:
		return x.sum(axis) / (x != 0).sum(axis)
	return x.sum() / (x != 0).sum()

def loss_mean(x):
	return x.sum() / (x != 0).sum()


MISTRAL_INPUTS_DOCSTRING = r"""
	Args:
		input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
			Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
			it.

			Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.

			[What are input IDs?](../glossary#input-ids)
		attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
			Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.

			[What are attention masks?](../glossary#attention-mask)

			Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.

			If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
			`past_key_values`).

			If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
			and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
			information on the default strategy.

			- 1 indicates the head is **not masked**,
			- 0 indicates the head is **masked**.
		position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
			config.n_positions - 1]`.

			[What are position IDs?](../glossary#position-ids)
		past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
			Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
			blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
			returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

			Two formats are allowed:
			- a [`~cache_utils.Cache`] instance;
			- Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
			shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
			cache format.

			The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
			legacy cache format will be returned.

			If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
			have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
			of shape `(batch_size, sequence_length)`.
		inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
			is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
			model's internal embedding lookup matrix.
		use_cache (`bool`, *optional*):
			If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
			`past_key_values`).
		output_attentions (`bool`, *optional*):
			Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
			tensors for more detail.
		output_hidden_states (`bool`, *optional*):
			Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
			more detail.
		return_dict (`bool`, *optional*):
			Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class QuietMistralForCausalLM(MistralPreTrainedModel):
	_tied_weights_keys = ["lm_head.weight"]

	def __init__(self, config):
		super().__init__(config)
		self.model = MistralModel(config)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		self.max_thoughts = config.max_thoughts
		self.merged_lm_and_talk_heads = config.merged_lm_and_talk_heads
		self.use_concat_talk_head = config.use_concat_talk_head
		self.use_shallow_talk = config.use_shallow_talk
		self.use_complex_talk_head = config.use_complex_talk_head
		self.use_weighted_talk_head = config.use_weighted_talk_head
		# the weighted head will output a single value, so it can't be passed to the lm head
		assert not (self.use_weighted_talk_head and self.use_shallow_talk)

		self.n_ahead = 1
		self.n_ahead_talk = 1
		self.n_passes = 1
		self.n_tokens_print = 1
		self.gradient_accumulation_steps = 1
		self.training_steps = 0
		self.tokenizer = None
		self.start_token_id = None
		self.end_token_id = None
		self.rm_initialized = False
		self.residual_talk_head = True
		self.thought_init_std_scale = 1e-2

		self.final_only_mode = False
		self.first_and_last_mode = True
		self.first_only = False
		self.original_loss_weight = 0.5

		self.cumulative_residual = False
		self.clever_residual = False
		self.skip_residual = False
		self.no_residual = True

		self.optimize_lm_head_only_at_start = False
		self.optimize_model_only_at_start = False

		if self.optimize_model_only_at_start:
			raise NotImplementedError
		self.train_only_thinking_embedding = False
		self.weighted_embeddings = False
		self.use_start_thought_token = True
		self.use_end_thought_token = True
		self.initialize_thought_embedding_to_normal = False
		self.initial_start_token = "---"
		self.initial_end_token = "---"
		self.output_logits_at_the_end = True

		self.wandb_enabled = False
		self.gumbel_temperature = 0.001

		self.use_policy_loss = True
		self.include_policy_loss = True
		self.trice_mode = True
		self.remove_negative_rewards = True
		self.use_policy_loss_for_end_thought = True

		self.base_original_mode = False
		self.original_mode = False

		self.thought_prefix = "(Let's think step by step"
		self.tokenized_thought_prefix = None
		self.log_dict = defaultdict(int)
		self.eval_log_dict = defaultdict(int)
		self.print_final_only = True
		self.loss_mean = loss_mean
		self.all_rewards = []
		self.all_unreduced_losses = []
		self.kill_after = 100

		self.start_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))
		self.end_embedding = nn.Parameter(torch.zeros(2, self.model.config.hidden_size))

		self.policy_loss_beta = 1e6
		self.embedding_scale = 1e2
		self.reinforce_temperature = 3
		self.base_loss_beta = 1

		# Not used in the paper:
		self.use_thought_prefix = False
		self.use_reparam_for_thought_embeddings = False
		self.use_upper_triangular = False
		self.subtract_mean_reward = False
		self.comparison_mode = False
		self.gumbel_detach = True

		# For visualization
		self.eval_mode = False

		talk_input_dim = config.hidden_size if not self.use_concat_talk_head else config.hidden_size * 2
		if self.use_weighted_talk_head:
			talk_output_dim = 1
		else:
			talk_output_dim = config.hidden_size if self.use_shallow_talk else config.vocab_size

		if not self.merged_lm_and_talk_heads:
			if self.use_complex_talk_head:
				self.talk_head = nn.ModuleList([nn.Sequential(
					nn.Linear(talk_input_dim, config.hidden_size),
					nn.ReLU(),
					nn.Linear(config.hidden_size, config.hidden_size),
					nn.ReLU(),
					nn.Linear(config.hidden_size, talk_output_dim, bias=False)
				)])
			else:
				self.talk_head = nn.ModuleList([nn.Sequential(
					nn.Linear(talk_input_dim, talk_output_dim, bias=False)
				)])

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def set_input_embeddings(self, value):
		self.model.embed_tokens = value

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def set_decoder(self, decoder):
		self.model = decoder

	def get_decoder(self):
		return self.model

	@torch.no_grad()
	def infer(
		self,
		input_ids: torch.LongTensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	):
		batch_size, seq_len = input_ids.shape

		# Save the original input_ids and attention_mask for later use
		original_input_ids = input_ids.clone()
		original_attention_mask = attention_mask.clone() if attention_mask is not None else None

		# Append the start thought token to the input sequence
		start_thought_token_id = self.tokenizer.convert_tokens_to_ids("<|startthought|>")
		input_ids = torch.cat([input_ids, torch.tensor([[start_thought_token_id]] * batch_size).to(input_ids.device)], dim=-1)
		seq_len += 1

		# Update the attention mask
		if attention_mask is not None:
			attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1)).to(attention_mask.device)], dim=-1)

		# Generate the continuation
		continuation_length = self.n_ahead - 2
		new_key_values = past_key_values

		for continuation_idx in range(continuation_length):
			outputs = self.model(
				input_ids=input_ids if continuation_idx == 0 else next_token_id.unsqueeze(-1).to(input_ids.device),
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_values=new_key_values,
				inputs_embeds=inputs_embeds,
				use_cache=True,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)
			new_key_values = outputs.past_key_values

			hidden_states = outputs[0]

			logits = self.lm_head(hidden_states)
			logits = logits[:, -1, :]  # Only consider the last token

			# Apply Gumbel-Softmax to the logits
			next_token_logits = F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=True, dim=-1)
			next_token_id = torch.argmax(next_token_logits, dim=-1)

			# Append the generated token to the input sequence
			input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1).to(input_ids.device)], dim=-1)
			seq_len += 1

			# Update the attention mask
			if attention_mask is not None:
				attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1)).to(attention_mask.device)], dim=-1)

		# Append the end thought token to the input sequence
		end_thought_token_id = self.tokenizer.convert_tokens_to_ids("<|endthought|>")
		input_ids = torch.cat([input_ids, torch.tensor([[end_thought_token_id]] * batch_size).to(input_ids.device)], dim=-1)
		seq_len += 1

		# Update the attention mask
		if attention_mask is not None:
			attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1)).to(attention_mask.device)], dim=-1)

		# Get the hidden states before and after the thought
		outputs_before = self.model(
			input_ids=original_input_ids,
			attention_mask=original_attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states_before = outputs_before[0][:, -1:, :]

		# two new tokens: last continuation token and end thought token
		outputs_after = self.model(
			input_ids=torch.cat([next_token_id.unsqueeze(-1).to(input_ids.device), torch.tensor(end_thought_token_id).unsqueeze(-1).unsqueeze(-1).to(input_ids.device)], dim=-1),
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=new_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states_after = outputs_after[0][:, -1:, :]

		# Apply the talk head to get the mixing weight
		mixing_weight = self.talk_head[0](torch.cat([hidden_states_before, hidden_states_after], dim=-1))

		# Apply the mixing weight to the hidden states
		mixed_hidden_states = (1 - mixing_weight) * hidden_states_before + mixing_weight * hidden_states_after

		# Apply the language model head to get the final logits
		logits = self.lm_head(mixed_hidden_states)
		return logits

	@add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
	@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, CausalLMOutputWithPast]:
		r"""
		Args:
			labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
				Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
				config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
				(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

		Returns:

		Example:

		```python
		>>> from transformers import AutoTokenizer, MistralForCausalLM

		>>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
		>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

		>>> prompt = "Hey, are you conscious? Can you talk to me?"
		>>> inputs = tokenizer(prompt, return_tensors="pt")

		>>> # Generate
		>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
		>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
		```"""
		log_dict = self.log_dict if self.training else self.eval_log_dict

		if self.training and self.kill_after is not None and self.training_steps // self.gradient_accumulation_steps > self.kill_after:
			raise ValueError("Killed after")

		if not self.training:
			n_ahead_talk_to_restore = self.n_ahead_talk
			n_passes_to_restore = self.n_passes
			self.n_ahead_talk = 1
			self.n_passes = 1

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		assert self.cumulative_residual or self.clever_residual or self.skip_residual or self.no_residual
		assert not (self.skip_residual and self.use_policy_loss)

		if self.tokenized_thought_prefix is None and self.use_thought_prefix:
			self.tokenized_thought_prefix = self.tokenizer(self.thought_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]

		def apply_head(head, states, detach=False):
			if detach:
				head_weight = head.weight.detach()
			else:
				head_weight = head.weight
			head_weight = head_weight.to(states.device)
			return (head_weight @ states.transpose(-1, -2)).transpose(-1, -2).contiguous()

		def idx_if_sequential(head, idx=0):
			if isinstance(head, nn.Sequential) or isinstance(head, nn.ModuleList):
				return idx_if_sequential(head[idx], idx=idx)
			return head

		def none_repeat_interleave(x, n):
			if x is None:
				return x
			return x.repeat_interleave(n, dim=0)

		if self.n_passes > 1:
			input_ids = none_repeat_interleave(input_ids, self.n_passes)
			attention_mask = none_repeat_interleave(attention_mask, self.n_passes)
			position_ids = none_repeat_interleave(position_ids, self.n_passes)
			inputs_embeds = none_repeat_interleave(inputs_embeds, self.n_passes)
			labels = none_repeat_interleave(labels, self.n_passes)
			if past_key_values is not None:
				past_key_values = [none_repeat_interleave(p, self.n_passes) for p in past_key_values]

		tokenizer_has_start_thought_token = True
		tokenizer_has_end_thought_token = True
		if self.start_token_id is None:
			self.start_token_id = self.tokenizer.convert_tokens_to_ids("<|startthought|>")
			if self.start_token_id == 0:
				self.start_token_id = self.tokenizer.bos_token_id
				tokenizer_has_start_thought_token = False
			elif self.use_start_thought_token:
				# base_start_id = self.tokenizer.convert_tokens_to_ids(self.initial_start_token)
				base_start_id = self.tokenizer.encode(self.initial_start_token, add_special_tokens=False)[0]
				if self.initialize_thought_embedding_to_normal:
					self.start_embedding.data = torch.zeros_like(self.start_embedding.data)
				else:
					self.start_embedding.data[0] = self.model.embed_tokens.weight.data[base_start_id].clone().detach() / self.embedding_scale
				self.start_embedding.data[1] = torch.log(self.model.embed_tokens.weight.data.std(dim=0) * self.thought_init_std_scale / self.embedding_scale)
		if self.end_token_id is None:
			self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|endthought|>")
			if self.end_token_id == 0:
				self.end_token_id = self.tokenizer.eos_token_id
				tokenizer_has_end_thought_token = False
			elif self.use_end_thought_token:
				# base_end_id = self.tokenizer.convert_tokens_to_ids(self.initial_end_token)
				base_end_id = self.tokenizer.encode(self.initial_end_token, add_special_tokens=False)[0]
				if self.initialize_thought_embedding_to_normal:
					self.end_embedding.data = torch.zeros_like(self.end_embedding.data)
				else:
					self.end_embedding.data[0] = self.model.embed_tokens.weight.data[base_end_id].clone().detach() / self.embedding_scale
				self.end_embedding.data[1] = torch.log(self.model.embed_tokens.weight.data.std(dim=0) * self.thought_init_std_scale / self.embedding_scale)

		if not self.rm_initialized and (self.n_ahead > 1 or not self.base_original_mode):
			self.rm_initialized = True
			if not self.use_shallow_talk:
				head = self.talk_head[0]
				cur_head = head[-1] if isinstance(head, nn.Sequential) else head
				talk_input_dim = cur_head.weight.data.shape[1]
				talk_output_dim = 1 if self.use_weighted_talk_head else self.lm_head.weight.data.shape[0]
				cur_head.weight.data = torch.zeros(talk_output_dim, talk_input_dim, device=cur_head.weight.device, dtype=cur_head.weight.dtype)
			else:
				# convert to identity transform
				def lambda_transform(cur_head):
					if cur_head.weight.data.shape[0] != cur_head.weight.data.shape[1]:
						return torch.cat([
						torch.eye(
							cur_head.weight.data.shape[0],
							device=cur_head.weight.device,
							dtype=cur_head.weight.dtype
						),
						torch.zeros(
							cur_head.weight.data.shape[0],
							cur_head.weight.data.shape[1] - cur_head.weight.data.shape[0],
							device=cur_head.weight.device,
							dtype=cur_head.weight.dtype
						)], dim=1)
					return torch.eye(
						cur_head.weight.data.shape[0],
						device=cur_head.weight.device,
						dtype=cur_head.weight.dtype
					)
				if isinstance(self.talk_head[0], nn.Sequential):
					for cur_head in self.talk_head[0]:
						# if it has weights
						if hasattr(cur_head, "weight"):
							cur_head.weight.data = lambda_transform(cur_head)
				else:
					self.talk_head[-1].weight.data = lambda_transform(self.talk_head[0])

		loss = None
		prev_sample_probs = None
		did_skip_sampling = None
		skip_sampling = None
		sample_probs = None
		hidden_states = None
		logits = None
		rm_logits = None
		residual_logits = None
		probabilities_2d = None
		prev_probabilities_2d = None
		policy_reward = None
		batch_size, seq_len = input_ids.shape
		loss_list = []
		dqn_loss_list = []
		sampled_token_history = []
		action_loglikelihoods_list = []

		start_embedding = None
		end_embedding = None

		if self.use_end_thought_token or self.use_start_thought_token:
			if not self.use_reparam_for_thought_embeddings:
				start_embedding = self.start_embedding[0].unsqueeze(0) * self.embedding_scale
				end_embedding = self.end_embedding[0].unsqueeze(0) * self.embedding_scale
			else:
				start_embedding = self.start_embedding * self.embedding_scale
				end_embedding = self.end_embedding * self.embedding_scale
			base_embeddings = self.model.embed_tokens.weight
			if self.train_only_thinking_embedding:
				base_embeddings = base_embeddings.detach()
		# # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
		fwd_iters = 1 if self.original_mode else self.n_ahead + self.n_ahead_talk - 1
		for ahead_idx in range(fwd_iters):
			past_key_values_length = 0
			if past_key_values is not None:
				use_legacy_cache = not isinstance(past_key_values, Cache)
				if use_legacy_cache:
					past_key_values = DynamicCache.from_legacy_cache(past_key_values)
				past_key_values_length = past_key_values.get_usable_length(seq_len)

			if position_ids is None:
				device = input_ids.device if input_ids is not None else inputs_embeds.device
				position_ids = torch.arange(
					past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
				)
				position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
			else:
				position_ids = position_ids.view(-1, seq_len).long()

			if inputs_embeds is None:
				contains_start = self.use_start_thought_token and (input_ids == self.start_token_id).any() and start_embedding is not None
				contains_end = self.use_end_thought_token and (input_ids == self.end_token_id).any() and end_embedding is not None
				contains_thought = contains_start or contains_end
				if contains_thought:
					cur_thought_embedding = start_embedding if contains_start else end_embedding
					if self.use_reparam_for_thought_embeddings:
						inputs_embeds = torch.randn(batch_size, seq_len, self.model.config.hidden_size, device=input_ids.device, dtype=cur_thought_embedding.dtype)
						inputs_embeds = inputs_embeds.detach() * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
						if contains_start:
							sampled_start = inputs_embeds.clone().detach()
						if contains_end:
							sampled_end = inputs_embeds.clone().detach()
					else:
						inputs_embeds = cur_thought_embedding.unsqueeze(0).repeat(batch_size, seq_len, 1)
				else:
					with torch.set_grad_enabled(not self.train_only_thinking_embedding):
						inputs_embeds = self.model.embed_tokens(input_ids)

			if self.n_ahead != 1 or self.n_ahead_talk != 1 or self.comparison_mode:
				if attention_mask is None:
					base_attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).to(input_ids.device)
					base_attention_mask = base_attention_mask.view(1, 1, seq_len, seq_len)
					base_attention_mask = base_attention_mask.repeat(input_ids.shape[0], 1, 1, 1)
					attention_mask = base_attention_mask
					# breakpoint()
				elif attention_mask.dim() == 2:
					if seq_len + past_key_values_length != attention_mask.shape[-1]:
						# breakpoint()
						attention_mask = torch.cat(
							[torch.ones((attention_mask.shape[0], past_key_values_length), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask],
							dim=-1
						)
					# # if the attention mask
					attention_mask = _prepare_4d_causal_attention_mask(
						attention_mask,
						(batch_size, seq_len),
						inputs_embeds,
						past_key_values_length,
						sliding_window=self.config.sliding_window,
					)

			outputs = self.model(
				# input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
				inputs_embeds=inputs_embeds,
				use_cache=use_cache,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)

			hidden_states = outputs[0]

			if ahead_idx == 0:
				hidden_states_lm = hidden_states
				logits = self.lm_head(hidden_states_lm)
				base_hidden_states = hidden_states.clone()
				initial_loss_logits = logits.clone()
				if self.optimize_lm_head_only_at_start or self.optimize_model_only_at_start:
					logits = logits.detach()
					base_hidden_states = base_hidden_states.detach()
				if self.optimize_model_only_at_start:
					hidden_states = hidden_states.detach()
				base_logits = logits.clone()
			else:
				talk_hidden_states = hidden_states
				if self.merged_lm_and_talk_heads:
					assert self.no_residual
					residual_logits = self.lm_head(hidden_states)
					talk_hidden_states = hidden_states
				else:
					if ahead_idx > self.n_ahead - 1:
						cur_base_hidden = torch.cat([
							base_hidden_states[..., ahead_idx - self.n_ahead + 1:, :],
							base_hidden_states[..., :ahead_idx - self.n_ahead + 1, :]
						], dim=-2)
					else:
						cur_base_hidden = base_hidden_states

					if self.use_concat_talk_head:
						# concatenate the hidden states with the original hidden states
						head_input_hidden_states = torch.cat([cur_base_hidden, talk_hidden_states], dim=-1)
					else:
						head_input_hidden_states = talk_hidden_states

					residual_logits = self.talk_head[0](head_input_hidden_states)
					if self.use_shallow_talk:
						residual_logits = apply_head(self.lm_head, residual_logits, detach=self.optimize_lm_head_only_at_start)
					residual_logits = residual_logits.to(logits.device)
					if self.use_weighted_talk_head:
						# combine the cur_base_hidden with the talk_hidden_states according to the weighted head
						residual_logits = cur_base_hidden * (1 - residual_logits) + talk_hidden_states * residual_logits
						residual_logits = apply_head(self.lm_head, residual_logits, detach=self.optimize_lm_head_only_at_start)

				assert sum([self.cumulative_residual, self.clever_residual, self.skip_residual, self.no_residual]) == 1
				if self.clever_residual:
					if ahead_idx >= self.n_ahead - 1:
						# get the logits shifted according to the current talk ahead
						cur_base_logits = torch.cat([
							base_logits[..., ahead_idx - self.n_ahead + 1:, :],
							base_logits[..., :ahead_idx - self.n_ahead + 1, :]
						], dim=-2)
						if self.optimize_lm_head_only_at_start:
							cur_base_logits = cur_base_logits.detach()
						logits = cur_base_logits + residual_logits
					else:
						logits += residual_logits / self.n_ahead
				elif self.cumulative_residual:
					if self.residual_talk_head:
						if ahead_idx < self.n_ahead:
							logits += residual_logits
						else:
							# get the logits shifted according to the current talk ahead
							cur_base_logits = torch.cat([
								base_logits[..., ahead_idx - self.n_ahead + 1:, :],
								base_logits[..., :ahead_idx - self.n_ahead + 1, :]
							], dim=-2)
							if self.optimize_lm_head_only_at_start:
								cur_base_logits = cur_base_logits.detach()
							logits = cur_base_logits + residual_logits
					else:
						if ahead_idx < self.n_ahead:
							logits += residual_logits
						else:
							logits = residual_logits
				elif self.skip_residual:
					if ahead_idx >= self.n_ahead:
						# get the logits shifted according to the current talk ahead
						cur_base_logits = torch.cat([
							base_logits[..., ahead_idx - self.n_ahead + 1:, :],
							base_logits[..., :ahead_idx - self.n_ahead + 1, :]
						], dim=-2)
						if self.optimize_lm_head_only_at_start:
							cur_base_logits = cur_base_logits.detach()
						logits = cur_base_logits
				elif self.no_residual:
					logits = residual_logits
				else:
					logits = base_logits + residual_logits

			attempted = False
			talk_loss_list = []
			if self.original_mode or (self.n_ahead == 1) or (self.comparison_mode and ahead_idx == 0):# or (self.optimize_lm_head_only_at_start and ahead_idx == 0):
				loss = None
				attempted = True

				if labels is not None:
					for shift_amount in range(self.n_ahead_talk):
						# Shift so that tokens < n predict n
						#  ab[cde]f
						# abc[def]
						if ahead_idx == 0 and self.optimize_lm_head_only_at_start:
							loss_logits = initial_loss_logits
						else:
							loss_logits = logits
						shift_logits = loss_logits[..., shift_amount:-1, :].contiguous()
						shift_labels = labels[..., 1 + shift_amount:].contiguous()
						# Flatten the tokens
						loss_fct = CrossEntropyLoss(reduction="none")
						shift_logits = shift_logits.view(-1, self.config.vocab_size)
						shift_labels = shift_labels.view(-1).clone()
						# Enable model parallelism
						shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
						shift_labels = shift_labels.to(shift_logits.device)
						loss = loss_fct(shift_logits, shift_labels)
						if not self.comparison_mode and not (self.optimize_lm_head_only_at_start and (self.n_ahead + self.n_ahead_talk > 2)) or self.original_mode:
							loss_list.append(loss)
						talk_loss_list.append(nonzero_mean(loss).detach())

			if not attempted or self.comparison_mode:
				rm_hidden_states = hidden_states
				# print("Magnitude of RM hidden states before RM head", rm_hidden_states.norm())
				rm_logits = apply_head(self.lm_head, rm_hidden_states, detach=self.optimize_lm_head_only_at_start)

				# don't allow it to predict the thinking token
				if tokenizer_has_start_thought_token:
					rm_logits[..., self.start_token_id] = torch.finfo(rm_logits.dtype).min
				if tokenizer_has_end_thought_token:
					rm_logits[..., self.end_token_id] = torch.finfo(rm_logits.dtype).min
				probabilities = rm_logits
				if probabilities_2d is not None:
					prev_probabilities_2d = probabilities_2d.clone()
				probabilities_2d = probabilities.view(-1, probabilities.size(-1))

				did_skip_sampling = skip_sampling
				skip_sampling = False
				if ahead_idx == 0 and self.use_start_thought_token:
					override_token = self.start_token_id
				elif self.use_thought_prefix and ahead_idx < self.tokenized_thought_prefix.shape[-1]:
					override_token = self.tokenized_thought_prefix[..., ahead_idx]
				elif ahead_idx == self.n_ahead - 2 and self.use_end_thought_token:
					override_token = self.end_token_id
				else:
					override_token = None
				if override_token is not None and self.n_ahead > 1:
					# always start with the start token
					probabilities_2d = torch.zeros_like(probabilities_2d)
					probabilities_2d[:, override_token] = 1.0
					skip_sampling = True
				elif ahead_idx >= self.n_ahead - 1:
					if labels is not None:  # we're in the talk phase
						cur_talk_n = ahead_idx - (self.n_ahead - 1) + 1
						# print("Setting rm to labels", cur_talk_n, "during", ahead_idx)
						shift_labels = labels[..., cur_talk_n:].contiguous().to(probabilities_2d.device)
						padding = torch.full_like(
							labels[..., :cur_talk_n],
							self.tokenizer.pad_token_id,
							dtype=torch.long,
							device=shift_labels.device
						)
						new_rm_tokens = torch.cat(
							[shift_labels, padding],
							dim=-1
						)
						# convert rm tokens to one-hot
						probabilities_2d = F.one_hot(new_rm_tokens, num_classes=self.vocab_size).reshape(-1, self.vocab_size).to(probabilities_2d.dtype)
						skip_sampling = True
					else:
						continue
				temperature = self.gumbel_temperature if self.training else 0.001
				prev_sample_probs = sample_probs
				sample_probs = probabilities_2d
				if ahead_idx < self.n_ahead - 1 and not skip_sampling:
					probabilities_2d = F.gumbel_softmax(sample_probs, tau=temperature, hard=True, dim=-1)
					if self.gumbel_detach:
						probabilities_2d = probabilities_2d.detach()
				sampled_token_history.append(probabilities_2d.argmax(dim=-1).detach().cpu())
				# convert rm logits directly to embeddings
				contains_start = self.use_start_thought_token and (probabilities_2d[..., self.start_token_id].sum() > 0)
				contains_end = self.use_end_thought_token and (probabilities_2d[..., self.end_token_id].sum() > 0)
				contains_thought = contains_start or contains_end

				if not contains_thought:
					with torch.set_grad_enabled(not self.train_only_thinking_embedding):
						inputs_embeds = probabilities_2d @ (self.model.embed_tokens.weight.to(probabilities.device).to(probabilities.dtype))
				else:
					cur_thought_embedding = start_embedding if contains_start else end_embedding
					if self.use_reparam_for_thought_embeddings:
						inputs_embeds = torch.randn(batch_size, seq_len, self.model.config.hidden_size, device=input_ids.device, dtype=cur_thought_embedding.dtype)
						inputs_embeds = inputs_embeds * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
						if contains_start:
							sampled_start = inputs_embeds.clone().detach()
						else:
							sampled_end = inputs_embeds.clone().detach()
					else:
						inputs_embeds = cur_thought_embedding.unsqueeze(0).repeat(batch_size, seq_len, 1)
						inputs_embeds = inputs_embeds.view(probabilities.size(0), probabilities.size(1), -1).to(self.model.embed_tokens.weight.dtype)
				inputs_embeds = inputs_embeds.view(probabilities.size(0), probabilities.size(1), -1).to(self.model.embed_tokens.weight.dtype)

				if len(attention_mask.shape) == 2:
					# breakpoint()
					# TODO: fix this
					pass
				else:
					original_attention = attention_mask[..., :attention_mask.shape[-2]]
					if self.use_upper_triangular:
						new_attention = original_attention
					else:
						original_attention = original_attention == attention_mask.max()
						# because eye isn't implemented for BF16, we need to handle the case
						if not attention_mask.dtype == torch.bfloat16:
							new_attention = torch.eye(
								seq_len, dtype=attention_mask.dtype, device=attention_mask.device
							)
						else:
							new_attention = torch.eye(
								seq_len, dtype=torch.float32, device=attention_mask.device
							).to(attention_mask.dtype)

						new_attention = new_attention.view(1, 1, seq_len, seq_len).repeat(input_ids.shape[0], 1, 1, 1)
						new_attention = new_attention * original_attention
						new_attention[new_attention == 0] = attention_mask.min()
						new_attention[new_attention == 1] = attention_mask.max()
					attention_mask = torch.cat([attention_mask, new_attention], dim=-1)
				past_key_values = outputs.past_key_values
				position_ids = position_ids + 1

				if labels is not None and (self.n_ahead > 1 or not self.base_original_mode):
					# Shift so that tokens < n predict n
					# logits: abcdef -> bcdef? -> cdef??
					# labels: abcdef -> ?bcdef -> ??cdef
					if ahead_idx == 0 and self.optimize_lm_head_only_at_start:
						loss_logits = initial_loss_logits
					else:
						loss_logits = logits
					shift_idx = 1 + max(0, ahead_idx - (self.n_ahead - 1))
					shift_logits = loss_logits[..., :-shift_idx, :].contiguous()
					shift_labels = labels[..., shift_idx:].contiguous()
					# Flatten the tokens
					loss_fct = CrossEntropyLoss(reduction="none")
					shift_logits = shift_logits.view(-1, self.config.vocab_size)
					shift_labels = shift_labels.view(-1)
					# Enable model parallelism
					shift_labels = shift_labels.to(shift_logits.device)
					# if shift_labels.min() == self.tokenizer.pad_token_id:
					shift_labels = torch.where(shift_labels == self.tokenizer.pad_token_id, -100, shift_labels)
					unreduced_loss = loss_fct(shift_logits, shift_labels)
					if torch.any(unreduced_loss != unreduced_loss):
						raise ValueError("NaN loss")
					unreduced_loss = unreduced_loss.reshape(logits.shape[0], -1)
					loss_list.append(unreduced_loss)


					if self.use_policy_loss and ahead_idx > 0 and (ahead_idx > 1 or not self.use_start_thought_token):
						# we treat the change in loss as the reward
						previous_loss = loss_list[-2]
						# for example, suppose n_ahead = 3 and n_ahead_talk = 2
						# note that we end at self.n_ahead + self.n_ahead_talk - 2
						# in this case, 5 - 2 = 3, so we end at ahead_idx = 3
						# we also predict the next token at ahead_idx = 2
						# when we get to ahead_idx = 2, we predict ahead
						# so we shift by 1
						# note that this is ahead_idx = n_ahead - 1
						# when we get to ahead_idx = 3, we predict ahead
						# so we shift by 2
						# note that this is ahead_idx = n_ahead
						if ahead_idx < self.n_ahead - 1:
							shift_amount = 0
							original_dqn_reward = (previous_loss - unreduced_loss).detach()
							if self.first_and_last_mode:
								original_dqn_reward = original_dqn_reward * 0.0
						else:
							# logits vs cur_policy_shift_logits
							# let's look at rm_logits and prev_rm_logits
							shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
							# let's say shift_amount = 2
							# abcdefg -> bcdefg? -> cdefg??
							# logits = [a b]c d e f[g]
							# labels = [a b c]d e f g
							cur_policy_shift_logits = initial_loss_logits[..., shift_amount:-1, :].contiguous().detach()
							cur_policy_shift_labels = labels[..., 1 + shift_amount:].contiguous()
							# Flatten the tokens
							cur_policy_shift_logits = cur_policy_shift_logits.view(-1, self.config.vocab_size)
							cur_policy_shift_labels = cur_policy_shift_labels.view(-1).clone()
							# Enable model parallelism
							cur_policy_shift_labels[cur_policy_shift_labels == self.tokenizer.pad_token_id] = -100
							cur_policy_shift_labels = cur_policy_shift_labels.to(cur_policy_shift_labels.device)
							cur_policy_reward_base_loss = loss_fct(
								cur_policy_shift_logits, cur_policy_shift_labels.to(cur_policy_shift_logits.device)
							).reshape(logits.shape[0], -1)
							original_dqn_reward = cur_policy_reward_base_loss.detach() - unreduced_loss

						if not did_skip_sampling:
							nonzero_indices = prev_probabilities_2d.nonzero()
							action_loglikelihoods = F.log_softmax(prev_sample_probs / self.reinforce_temperature, dim=-1)[nonzero_indices[:, 0], nonzero_indices[:, 1]]
							action_loglikelihoods_2d = action_loglikelihoods.reshape(batch_size, -1)[:, :-1 - shift_amount]
							action_loglikelihoods_list.append(action_loglikelihoods_2d)
						if policy_reward is None:
							policy_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
						else:
							if self.n_ahead_talk > shift_amount:
								added_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
							else:
								added_reward = original_dqn_reward
							policy_reward += added_reward

					if self.use_policy_loss and ahead_idx == self.n_ahead + self.n_ahead_talk - 2:
						# only compute during the thinking phase
						if self.use_reparam_for_thought_embeddings and (self.use_start_thought_token or self.use_end_thought_token):
							# sampled_start, sampled_end
							# calculate the log likelihood of the start and end embeddings sampled from a multivariate normal distribution
							# with mean start_embedding[0] and standard deviation start_embedding[1]
							if self.use_start_thought_token:
								exp_start_std = torch.exp(start_embedding[1])
								# FIXME: Spooky action at a distance
								# pylint: disable=used-before-assignment
								start_loglikelihood = -0.5 * (sampled_start.detach() - start_embedding[0]) ** 2 / exp_start_std ** 2 - start_embedding[1] - 0.5 * math.log(2 * math.pi)
								start_loglikelihood = start_loglikelihood.mean(dim=-1)
							if self.use_end_thought_token:
								exp_end_std = torch.exp(end_embedding[1])
								# pylint: disable=used-before-assignment
								end_loglikelihood = -0.5 * (sampled_end.detach() - end_embedding[0]) ** 2 / exp_end_std ** 2 - end_embedding[1] - 0.5 * math.log(2 * math.pi)
								end_loglikelihood = end_loglikelihood.mean(dim=-1)
							# we use the mean instead of the sum to prevent dependence on the dimensionality of the embeddings
							if self.use_end_thought_token and self.use_policy_loss_for_end_thought:
								action_loglikelihoods_list.append(end_loglikelihood)
							if self.use_start_thought_token:
								action_loglikelihoods_list.append(start_loglikelihood)

						if ahead_idx == self.n_ahead + self.n_ahead_talk - 2 and self.eval_mode:
							with torch.no_grad():
								# calculate the 0.75 quantile of the rewards
								filtered_tokens = input_ids[:, :policy_reward.shape[-1]].cpu().detach().numpy().flatten()
								filtered_tokens_mask = filtered_tokens != self.tokenizer.pad_token_id
								filtered_tokens = filtered_tokens[filtered_tokens_mask]
								filtered_rewards = policy_reward.float().cpu().detach().numpy()[:, :seq_len - self.n_ahead_talk].flatten()
								filtered_rewards = filtered_rewards[filtered_tokens_mask]

								abs_reward_list = np.abs(policy_reward.float().cpu().detach().numpy()[:, :seq_len - self.n_ahead_talk].flatten())
								abs_reward_list = abs_reward_list[filtered_tokens_mask]
								medium_quantile = np.quantile(abs_reward_list, 0.5)
								upper_quantile = np.quantile(abs_reward_list, 0.95)

								save_tokens_with_rewards_to_pdf(
									filtered_tokens,
									[0] + filtered_rewards.tolist(),
									self.tokenizer,
									output_file=f"texts/rewards_talk_{self.n_ahead_talk}_{self.training_steps}.pdf",
									eps=medium_quantile,
									eps2=upper_quantile,
								)

								def plot_kde(data, losses):
									sns.set(style="whitegrid")
									# Create the KDE plot
									sns.kdeplot(data, fill=True)
									# Set the plot title and labels
									plt.title("KDE Plot")
									plt.xlabel("Value")
									plt.ylabel("Density")
									# Save the plot
									plt.savefig(f"texts/kde_talk_{self.n_ahead_talk}_{self.training_steps}.pdf")
									# Close the plot
									plt.close()

									# Step 1: Create a base color palette
									log_norm = LogNorm(vmin=1e-3, vmax=10)

									sns.kdeplot(x=data, y=losses, fill=True, levels=20, norm=log_norm, cut=0, linewidths=0)
									# limit y to 0 to 25 and x to -1 to 1
									plt.xlim(-1, 1)
									plt.ylim(0, 25)
									plt.savefig(f"texts/jointer_talk_{self.n_ahead_talk}_{self.training_steps}.pdf")
									plt.close()

								self.all_rewards.extend(filtered_rewards)
								self.all_unreduced_losses.extend(unreduced_loss[:, :-1].flatten()[filtered_tokens_mask].float().flatten().cpu().detach().numpy())
								plot_kde(self.all_rewards, self.all_unreduced_losses)

						for action_loglikelihoods_2d in action_loglikelihoods_list:
							train_policy_reward = policy_reward

							# discard rewards below the mean
							if self.trice_mode and self.n_passes > 1:
								batched_policy_reward = train_policy_reward.reshape(-1, self.n_passes, train_policy_reward.shape[-1])
								# average over the passes
								train_policy_reward = batched_policy_reward - batched_policy_reward.mean(dim=1, keepdim=True)
								train_policy_reward = train_policy_reward.reshape(-1, train_policy_reward.shape[-1])

							if self.subtract_mean_reward:
								train_policy_reward = train_policy_reward - train_policy_reward.mean()
							if self.remove_negative_rewards:
								fixed_policy_reward = train_policy_reward.detach().clamp(min=0)
							else:
								fixed_policy_reward = train_policy_reward.detach()
							actor_loss = -fixed_policy_reward * action_loglikelihoods_2d[:, :policy_reward.shape[-1]].to(policy_reward.device)
							if action_loglikelihoods_2d.mean() < -1e4 and not self.use_policy_loss_just_for_thoughts:
								# This will only happen when we force the next token to be the end of thought token
								break
							dqn_loss_list.append(actor_loss.mean())

		if loss_list:
			if self.first_and_last_mode:
				loss = sum(
					self.loss_mean(loss_list[-(i + 1)]) for i in range(self.n_ahead_talk)
				) * (1 - self.original_loss_weight) / self.n_ahead_talk
				loss = loss + self.loss_mean(loss_list[0]) * self.original_loss_weight
				# Let's NaN out the others
				# e.g. if n_ahead_talk = 2 and the list is 5 long, we want to NaN out 1, 2 but keep 0, 3, 4
				for i in range(1, len(loss_list) - self.n_ahead_talk):
					loss_list[i] = loss_list[i] * math.nan
			elif self.first_only:
				loss = self.loss_mean(loss_list[0])
			elif self.final_only_mode:
				loss = sum(
					self.loss_mean(loss_list[-i]) for i in range(1, self.n_ahead_talk + 1)
				) / self.n_ahead_talk
			else:
				loss = None
				# FIXME
				# pylint: disable=consider-using-enumerate
				for i in range(len(loss_list)):
					cur_loss = self.loss_mean(loss_list[i])
					if loss is not None:
						loss = loss + cur_loss.to(loss.device)
					else:
						loss = cur_loss
				loss = loss / len(loss_list)

			loss = loss * self.base_loss_beta

		if dqn_loss_list:
			dqn_loss = sum(dqn_loss_list) / len(dqn_loss_list)
			if self.include_policy_loss:
				if loss is not None:
					loss += dqn_loss * self.policy_loss_beta
				else:
					loss = dqn_loss * self.policy_loss_beta

		if not return_dict:
			output = (logits,) + outputs[1:]
			return (loss,) + output if loss is not None else output

		base_log_dict = {
			f"loss_{i}": nonzero_mean(loss_list[i]) for i in range(len(loss_list))
		}

		if loss is not None:
			base_log_dict["loss_train"] = loss.item()

		for loss_key, loss_val in base_log_dict.items():
			log_dict[loss_key] += loss_val / self.n_tokens_print

		if self.use_policy_loss and policy_reward is not None:
			log_dict["policy_loss"] += dqn_loss / self.n_tokens_print
			log_dict["policy_reward"] += policy_reward.mean() / self.n_tokens_print

		if not loss_list:
			if loss is not None:
				log_dict["loss_0"] += loss / self.n_tokens_print
		else:
			log_dict["loss_final"] += nonzero_mean(loss_list[-1]) / self.n_tokens_print
			log_dict["loss_talk"] += sum(nonzero_mean(cur_loss_item) for cur_loss_item in loss_list[-self.n_ahead_talk:]) / self.n_ahead_talk / self.n_tokens_print

		# also log relative losses to loss_0
		if loss_list:
			# FIXME
			# pylint: disable=consider-using-enumerate
			for i in range(len(loss_list)):
				talk_idx = min(max(i - (self.n_ahead - 1), 0), len(talk_loss_list) - 1)
				if not talk_loss_list:
					cur_talk_loss = nonzero_mean(loss_list[0])
				else:
					cur_talk_loss = talk_loss_list[talk_idx]
				log_dict[f"rel_loss_{i}"] += (nonzero_mean(loss_list[i]) - cur_talk_loss) / self.n_tokens_print
		if self.training:
			self.training_steps += 1
		try:
			# if self.training_steps % (self.gradient_accumulation_steps * 256) == 0:
			if self.wandb_enabled:
				if self.training_steps % (self.n_tokens_print) == 0 or not self.training:# and "0" in str(loss.device):
					if not self.training:
						new_log_dict = {}
						for key in list(log_dict.keys()):
							new_log_dict["eval_" + key] = log_dict[key]
						log_dict = new_log_dict
					log_dict["training_steps"] = self.training_steps
					log_dict["batch_size"] = batch_size
					log_dict["example_steps"] = self.training_steps * batch_size * self.gradient_accumulation_steps
					if self.n_ahead > 1:
						log_dict["compute_steps"] = self.training_steps * batch_size * (self.n_ahead + self.n_ahead_talk - 1) * self.gradient_accumulation_steps
					else: # There's no overhead for talk tokens if there's no thinking
						log_dict["compute_steps"] = self.training_steps * batch_size * self.gradient_accumulation_steps
					# remove all nans
					for key in list(log_dict.keys()):
						if log_dict[key] != log_dict[key]:
							del log_dict[key]
					if self.training:
						wandb.log(log_dict)
					if self.training:
						self.log_dict = defaultdict(int)
					else:
						self.eval_log_dict = defaultdict(int)
		# FIXME: broad exception
		# pylint: disable=broad-except
		except Exception as _:
			pass

		if not self.training:
			self.n_ahead_talk = n_ahead_talk_to_restore
			self.n_passes = n_passes_to_restore
		return CausalLMOutputWithPast(
			loss=loss if loss is not None else None,
			logits=(rm_logits if self.n_ahead > 1 else logits) if not self.output_logits_at_the_end else logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


	def prepare_inputs_for_generation(
		self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
	):
		# Omit tokens covered by past_key_values
		if past_key_values is not None:
			if isinstance(past_key_values, Cache):
				cache_length = past_key_values.get_seq_length()
				past_length = past_key_values.seen_tokens
				max_cache_length = past_key_values.get_max_length()
			else:
				cache_length = past_length = past_key_values[0][0].shape[2]
				max_cache_length = None

			# Keep only the unprocessed tokens:
			# 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
			# some of the inputs are exclusively passed as part of the cache (e.g. when passing inputs_embeds as
			# input)
			if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
				input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
			# 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
			# input_ids based on the past_length.
			elif past_length < input_ids.shape[1]:
				input_ids = input_ids[:, past_length:]
			# 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

			# If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
			if (
				max_cache_length is not None
				and attention_mask is not None
				and cache_length + input_ids.shape[1] > max_cache_length
			):
				attention_mask = attention_mask[:, -max_cache_length:]

		position_ids = kwargs.get("position_ids", None)
		if attention_mask is not None and position_ids is None:
			# create position_ids on the fly for batch generation
			position_ids = attention_mask.long().cumsum(-1) - 1
			position_ids.masked_fill_(attention_mask == 0, 1)
			if past_key_values:
				position_ids = position_ids[:, -input_ids.shape[1] :]

		# if `inputs_embeds` are passed, we only want to use them in the 1st generation step
		if inputs_embeds is not None and past_key_values is None:
			model_inputs = {"inputs_embeds": inputs_embeds}
		else:
			model_inputs = {"input_ids": input_ids}

		model_inputs.update(
			{
				"position_ids": position_ids,
				"past_key_values": past_key_values,
				"use_cache": kwargs.get("use_cache"),
				"attention_mask": attention_mask,
			}
		)
		return model_inputs

	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		reordered_past = ()
		for layer_past in past_key_values:
			reordered_past += (
				tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
			)
		return reordered_past
