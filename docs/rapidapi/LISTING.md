# Listing the API on RapidAPI

RapidAPI sits in front of the Modal endpoint. Consumers use **their own** `X-RapidAPI-Key`;
your provider key is never shared. The one secret between RapidAPI and this server is the
**proxy secret**, which RapidAPI adds to every call and `deploy/api.py::_auth` checks.
Account, login and plan creation are done by the owner in the RapidAPI dashboard.

## Before listing (once)
1. **Modal workspace budget** — Settings -> Usage & Billing. This is the hard outer cap; the
   spend guards (`deploy/spend_guard.py`) are counters inside it.
2. **Proxy secret** — generate one, then put it in the Modal secret *and* in RapidAPI:
   ```bash
   modal secret create dubbing-secrets GEMINI_API_KEY=... HF_TOKEN=... RAPIDAPI_PROXY_SECRET=<new random value>
   ```
   It must **differ** from `DEMO_ACCESS_CODE` (the demo code would otherwise open `/v1`).
   Without it `/v1` answers 503 by design.
3. Deploy (GitHub Actions -> Deploy, `production` environment) and copy the base URL
   (`https://<workspace>--indic-dubbing-fastapi-app.modal.run`).

## In the RapidAPI dashboard
1. **My APIs -> Add API.** Import `docs/rapidapi/openapi.json`; replace the placeholder server
   URL with the Modal base URL.
2. **Gateway / security:** add the header `X-RapidAPI-Proxy-Secret` = the secret from above
   (RapidAPI's "Secret Headers" / proxy secret setting).
3. **Plans** (RapidAPI enforces call counts and rate; GPU-seconds are bounded by this repo):
   | RapidAPI plan name | Calls | Notes |
   |---|---|---|
   | `BASIC` | hard-limited, small | 10-minute clips; 3x the caller limits |
   | `PRO` | paid | 60-minute clips; 10x |
   | `ULTRA` | paid | 120-minute clips; 30x |
   The plan name reaches the server in `X-RapidAPI-Subscription` and must match one of
   `BASIC`/`PRO`/`ULTRA` (see `PLAN_MAX_SECONDS`); anything else is treated as the free 120 s tier.
   Make the entry plan a **hard** limit, not "pay per overage".
4. **Test through the gateway, not directly:** upload a real clip with RapidAPI's test console and
   confirm it is accepted. RapidAPI may cap request body size; if your clips are refused there,
   the API needs a video-URL input instead of a file upload (not built).
5. Publish only after step 4 passes.

## Operating it
```bash
python -m deploy.spend_guard status
python -m deploy.spend_guard pause "reason"    # stop new jobs now, no redeploy
python -m deploy.spend_guard resume
```
