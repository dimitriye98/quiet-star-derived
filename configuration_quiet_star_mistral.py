# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
""" Mistral model configuration"""

from transformers import MistralConfig
from transformers import PretrainedConfig
from transformers import logging


logger = logging.get_logger(__name__)

class QuietStarMistralConfig(MistralConfig):
    model_type = "quiet-star-mistral"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        max_thoughts=16,
        merged_talk_heads=True,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        **kwargs,
    ):
        self.max_thoughts = max_thoughts
        self.merged_talk_heads = merged_talk_heads
        self.merged_lm_and_talk_heads = merged_lm_and_talk_heads
        self.merged_lm_and_think_heads = merged_lm_and_think_heads
        self.use_concat_talk_head = use_concat_talk_head
        self.use_shallow_think = use_shallow_think
        self.use_shallow_talk = use_shallow_talk
        self.use_complex_think_head = use_complex_think_head
        self.use_complex_talk_head = use_complex_talk_head
        self.use_weighted_talk_head = use_weighted_talk_head

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
