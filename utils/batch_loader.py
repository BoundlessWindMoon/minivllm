"""Load prompts from assets and construct Request objects for batch inference."""

import json
import os
import random
import uuid

from engine.request import Request
from engine.schema import SamplingParams
from utils.config import GlobalConfig


def load_batch_requests(cfg: GlobalConfig, tokenizer) -> list[Request]:
    """Read prompts from the configured asset file and return a Request list.

    Prompt source (cfg.batch.prompts):
      random: false  → first num_requests lines, in file order
      random: true   → seeded random sample of num_requests lines
    """
    pcfg  = cfg.batch.prompts
    asset = os.path.expanduser(pcfg.asset)

    if not os.path.exists(asset):
        items = [{"prompt": cfg.generation.prompt,
                  "max_new_tokens": cfg.generation.max_new_tokens}]
    else:
        with open(asset, encoding="utf-8") as f:
            items = [json.loads(l) for l in f if l.strip()]
        if pcfg.random:
            items = random.Random(pcfg.seed).sample(items, min(pcfg.num_requests, len(items)))
        else:
            items = items[:pcfg.num_requests]

    for item in items:
        item.setdefault("max_new_tokens", cfg.generation.max_new_tokens)

    reqs = []
    for item in items:
        prompt = item["prompt"]
        if cfg.generation.use_chat_template and getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=cfg.generation.use_thinking,
                return_tensors="pt",
            )["input_ids"][0].tolist()
        else:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        reqs.append(Request(
            request_id       = str(uuid.uuid4())[:8],
            prompt_token_ids = ids,
            prompt_text      = prompt,
            sampling_params  = SamplingParams(
                temperature    = cfg.generation.sampling.temperature,
                top_k          = cfg.generation.sampling.topk,
                top_p          = cfg.generation.sampling.topp,
                max_new_tokens = item["max_new_tokens"],
                stop_on_eos    = cfg.generation.stop_on_eos,
            ),
        ))
    return reqs
