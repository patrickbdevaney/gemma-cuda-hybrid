# gemma-4 + DFlash pure-CUDA OpenAI server (Phase 1c)
Build: `bash scripts/build.sh` → `build/forward`. Serve: `SERVE=1 PORT=8080 ./build/forward [ckpt]`.
Endpoints: `POST /v1/chat/completions` (stream + non-stream), `GET /v1/models`.
Pure C++ single binary (cpp-httplib + nlohmann/json + include/tokenizer.h BPE). Single-instance (mutex-serialized).
Chat: gemma-4 `<|turn>role\n...<turn|>` format, thinking channel pre-filled (non-thinking), stops [1,106,50].
Tokenizer validated 5/5 vs HF (server/tok_test.cpp). Decode filters <|X>/<X|> control tokens.
Params honored: messages (system/user/assistant), max_tokens, stream. TODO(1e): temperature/top-p (greedy for now).
