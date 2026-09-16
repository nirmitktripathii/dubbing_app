#!/usr/bin/env python3
"""Streamlit UI for the dubbing service — the human product + sales demo.

Talks to the FastAPI gateway (async): upload -> submit -> poll -> preview/download. Point it
at your deployed API with DUB_API_URL (and DUB_API_KEY if you front it with RapidAPI or a
direct key). Run locally:  streamlit run deploy/streamlit_app.py
Deploy on Streamlit Community Cloud or Modal (see deploy/README.md).
"""
import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("DUB_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("DUB_API_KEY", "")           # RapidAPI key, or your proxy secret in dev
POLL_EVERY_S = 4

st.set_page_config(page_title="IndicAI Dubbing", page_icon="🎬", layout="centered")
st.title("🎬 IndicAI Dubbing")
st.caption("Upload a video, pick a language, get a studio-grade dub. Voice cloning optional.")


def _headers():
    h = {}
    if API_KEY:
        # In production RapidAPI injects the proxy secret; for a direct deploy send it yourself.
        h["X-RapidAPI-Proxy-Secret"] = API_KEY
    return h


with st.form("dub"):
    up = st.file_uploader("Source video", type=["mp4", "mkv", "mov", "webm", "m4v"])
    col1, col2 = st.columns(2)
    lang = col1.selectbox("Target language", [
        "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Marathi",
        "Bengali", "Gujarati", "Punjabi", "Odia", "Assamese",
    ])
    mode_label = col2.selectbox("Voice", [
        "Basic (native voice) — clean, fast",
        "Premium (clone original speaker) — voice conversion",
    ])
    mode = "vc" if mode_label.startswith("Premium") else "basic"
    submitted = st.form_submit_button("Dub it")

if submitted:
    if not up:
        st.error("Please choose a video first.")
        st.stop()
    with st.spinner("Uploading and queuing…"):
        r = requests.post(
            f"{API_URL}/v1/dub",
            files={"file": (up.name, up.getvalue(), "video/mp4")},
            data={"target_lang": lang, "mode": mode},
            headers=_headers(), timeout=120,
        )
    if r.status_code != 200:
        st.error(f"Submit failed ({r.status_code}): {r.text}")
        st.stop()
    job_id = r.json()["job_id"]
    st.session_state["job_id"] = job_id
    st.success(f"Queued. Job {job_id}")

job_id = st.session_state.get("job_id")
if job_id:
    st.divider()
    st.subheader("Progress")
    box = st.empty()
    bar = st.progress(0)
    STAGE_PCT = {"queued": 5, "starting": 10, "running": 40, "complete": 100, "done": 100}
    terminal = False
    for _ in range(600):  # up to ~40 min of polling
        s = requests.get(f"{API_URL}/v1/dub/{job_id}", headers=_headers(), timeout=30)
        if s.status_code != 200:
            box.error(f"Status check failed ({s.status_code}): {s.text}")
            break
        st_json = s.json()
        status = st_json.get("status", "?")
        bar.progress(min(STAGE_PCT.get(st_json.get("stage", status), STAGE_PCT.get(status, 20)), 100))
        box.write(st_json)
        if status in ("done", "failed"):
            terminal = True
            break
        time.sleep(POLL_EVERY_S)

    if terminal and status == "done":
        st.success(f"Done in {st_json.get('elapsed_s','?')}s · {st_json.get('video_seconds','?')}s of video")
        dl = requests.get(f"{API_URL}/v1/dub/{job_id}/download", headers=_headers(), timeout=300)
        if dl.status_code == 200:
            st.video(dl.content)
            st.download_button("Download dubbed video", dl.content,
                               file_name=f"dubbed_{job_id}.mp4", mime="video/mp4")
        else:
            st.warning(f"Download not ready ({dl.status_code}).")
    elif terminal:
        st.error("Job failed.")
        st.code(st_json.get("log", "")[-2000:] or st_json.get("error", ""))
