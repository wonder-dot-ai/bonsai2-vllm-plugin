"""Load the complete M3 text checkpoint in vLLM and save a short greedy smoke."""

import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="converted/bonsai2-pq2",
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_backend="FLASH_ATTN",
    )
    prompts = ["The capital of France is", "1 + 1 =", "안녕하세요."]
    results = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8, logprobs=10))
    records = []
    for result in results:
        completion = result.outputs[0]
        records.append(
            {
                "prompt": result.prompt,
                "prompt_token_ids": result.prompt_token_ids,
                "text": completion.text,
                "token_ids": completion.token_ids,
                "logprobs": [
                    {str(k): {"logprob": v.logprob, "rank": v.rank} for k, v in row.items()}
                    for row in completion.logprobs
                ],
            }
        )
    Path("artifacts/m3-smoke.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
