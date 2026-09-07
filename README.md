# AI Support Agent

A small Streamlit-based ServiceNow support agent that summarizes tickets, checks SLA risk, and refuses out-of-scope questions.

## Features

- Streamlit chat UI
- Azure AI / Foundry model integration
- Content safety and PII/jailbreak checks
- SLA prioritization logic

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your environment variables in a `.env` file:

```env
AOAI_ENDPOINT="https://<your-resource>.services.ai.azure.com"
AOAI_API_KEY="<your-api-key>"
AOAI_DEPLOYMENT="<your-model-deployment>"
```

4. Run the app:

```bash
python -m streamlit run aisupportagent.py --server.port=8509 --server.address=0.0.0.0
```

## Notes

- `.env` is intentionally ignored by Git and should not be published.
- This repository intentionally contains only this project.
