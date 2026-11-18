# Local IISc chat

Follow [README.md](README.md) from the cloned asks folder. Python 3.12 is tested by CI.

~~~bash
cp config/local.example.json config/local.json
~~~

The example selects model discovery, public chat and data/local. Set VLLM_URL to your running OpenAI-compatible model server. SERVED_MODEL=auto discovers its model. For a separately managed MLX or Ollama server, use the name returned by /v1/models. The switch script manages the Llama profiles in [MODELS.md](MODELS.md).

~~~bash
export ASKIISC_SOURCE_DEPARTMENTS=CDS,CSA,ECE
./run-local.sh --prepare --auth off
./run-local.sh --auth off --port 8100
~~~

Use --port 8101 when 8100 is occupied. ASKIISC_DATA_DIR overrides the profile directory. Stop serving before rebuilding it. The source list and department selection determine coverage.

Public chat serves public evidence. Enable OTP with --auth on and private SMTP settings for role-specific documents. PUBLIC_CHAT=0 requires credentials for every query.

The UI supports source links, follow-ups, Stop and New conversation. It releases answers after validation. Model startup and shutdown are separate from the chat process.
