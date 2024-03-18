# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoTokenizer

config = GPTNeoXConfig()


def get_model():
    config = GPTNeoXConfig(
        vocab_size=5000,
        num_attention_heads=10,
        hidden_size=600,
        intermediate_size=2400,
        num_hidden_layers=12,
        max_position_embeddings=2048,
    )
    return GPTNeoXForCausalLM(config)


def get_tokenizer(cache_dir: str = None):
    """Returns the tokenizer used by SN 9."""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
