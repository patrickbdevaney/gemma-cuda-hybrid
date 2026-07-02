# gemma-4 + DFlash pure-CUDA OpenAI server (Phase 1c)
Build: `bash scripts/build.sh` → `build/forward`. Serve: `SERVE=1 PORT=8080 ./build/forward [ckpt]`.
Endpoints: `POST /v1/chat/completions` (stream + non-stream), `GET /v1/models`.
Pure C++ single binary (cpp-httplib + nlohmann/json + include/tokenizer.h BPE). Single-instance (mutex-serialized).
Chat: gemma-4 `<|turn>role\n...<turn|>` format, thinking channel pre-filled (non-thinking), stops [1,106,50].
Tokenizer validated 5/5 vs HF (server/tok_test.cpp). Decode filters <|X>/<X|> control tokens.
Params honored: messages, max_tokens, stream, temperature (lossless Gumbel-max spec sampling: temp=0 exact-greedy, temp>0 sampled+varied). top_p accepted but v1 applies temperature only (nucleus sort = TODO).

## Phase 2: prefix caching
Single-slot LCP cache (g_cached_ids): each request reuses the longest common prefix of KV with the previous
request, re-prefilling only the new suffix at base=p (bit-exact — LCP guarantees the cached KV[0:p) is identical).
Verified: outputs unchanged regardless of cache state; system prompt reused across turns (e.g. 41/55 tokens).
Scope: single-instance sequential (ideal for one agentic conversation). Multi-user interleaving thrashes the
single slot (a radix/block cache like vLLM would generalize it — future work). DFlash taps reused alongside KV.

## Phase 3: FP8 KV cache + configurable context (default 64K)
KV size + dtype are launch-config (the "dynamic reset" interface — set per run, no recompile):
  SERVE=1 CTX=65536 FP8KV=1 ./build/forward   # defaults: CTX=65536 (64K), FP8KV=1 (on)
- CTX=<n>: KV context capacity (default 64K). FP8KV=1/0: FP8 e4m3 KV (half bytes, lossy, default ON) vs BF16.
- FP8 KV (g_fp8kv) is server-default; the benchmark/gate stays BF16 (bit-exact — verified). k_store_kv/sdpa
  branch on the flag; Session allocs 1B/elem (fp8) or 2B/elem (bf16). Verified coherent at 64K FP8 ("Paris"; fruit list).
- Memory @64K FP8 ≈ 7GB KV + 4.3GB taps + 16GB model (fits). All layers alloc full CAP (sliding-window ring buffer
  = future optimization to shrink the 25 sliding layers to SWIN=1024). Runtime /admin resize endpoint = future work.
