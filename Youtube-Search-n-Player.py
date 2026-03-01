"""YouTube Search + Player (Web App)

Runs a small Flask web UI that searches YouTube via the YouTube Data API v3
and embeds a selected video for playback.

Theme/colors are intentionally matched to `web_audio_capture.py`.

Environment:
  - GOOGLE_DEVELOPER_API_KEY: YouTube Data API v3 key
    (can be placed in a `.env` file)
"""

from __future__ import annotations

import os
from typing import Any

import dotenv
import requests
from flask import Flask, redirect, render_template_string, request, url_for


dotenv.load_dotenv()

app = Flask(__name__)


def _theme_css() -> str:
    """CSS copied from `web_audio_capture.py` to keep consistent theme."""
    return """
        :root {
            --bg: #0B0F14;
            --panel: #101823;
            --panel-2: #0E1620;
            --text: #E7EEF6;
            --muted: #9BB0C2;
            --border: #213041;
            --teal: #006A80;
            --orange: #FF8A00;
        }

        * { box-sizing: border-box; }
        html, body { height: 100%; }
        body {
            margin: 0;
            background: radial-gradient(1200px 600px at 20% 10%, rgba(0, 106, 128, 0.22), transparent 55%),
                        radial-gradient(900px 500px at 80% 20%, rgba(255, 138, 0, 0.10), transparent 60%),
                        var(--bg);
            color: var(--text);
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            line-height: 1.4;
        }

        .wrap {
            max-width: 920px;
            margin: 0 auto;
            padding: 40px 16px;
        }

        .card {
            background: linear-gradient(180deg, rgba(16, 24, 35, 0.92), rgba(14, 22, 32, 0.92));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px 18px;
        }

        h2 {
            margin: 0 0 6px;
            font-size: 20px;
            letter-spacing: 0.2px;
        }

        .sub {
            margin: 0 0 18px;
            color: var(--muted);
            font-size: 13px;
        }

        .row {
            display: flex;
            gap: 12px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 14px;
        }

        button, .btn {
            appearance: none;
            border: 1px solid rgba(0, 106, 128, 0.45);
            background: linear-gradient(180deg, rgba(0, 106, 128, 1), rgba(0, 86, 104, 1));
            color: #FFFFFF;
            font-weight: 600;
            border-radius: 10px;
            padding: 10px 14px;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 40px;
        }

        button:hover, .btn:hover {
            border-color: rgba(255, 138, 0, 0.55);
        }

        button:disabled {
            cursor: not-allowed;
            opacity: 0.7;
        }

        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 11px;
            border: 1px solid rgba(0, 106, 128, 0.45);
            color: var(--text);
            background: rgba(0, 106, 128, 0.12);
        }

        form.search {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 8px;
        }

        input[type="text"] {
            width: min(640px, 100%);
            border-radius: 10px;
            border: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.25);
            color: var(--text);
            padding: 10px 12px;
            outline: none;
        }

        input[type="text"]::placeholder { color: rgba(155, 176, 194, 0.85); }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
            margin-top: 18px;
        }

        @media (min-width: 900px) {
            .grid { grid-template-columns: 1fr 1fr; }
        }

        .panel {
            border: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.18);
            border-radius: 14px;
            padding: 14px;
        }

        .results {
            display: grid;
            gap: 10px;
        }

        .result {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
            background: rgba(231, 238, 246, 0.04);
        }

        .result-top {
          display: flex;
          gap: 12px;
          align-items: flex-start;
          flex-wrap: wrap;
        }

        .thumb {
          width: 160px;
          max-width: 100%;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: rgba(0, 0, 0, 0.25);
        }

        .result-body {
          flex: 1;
          min-width: 220px;
        }

        .result-title {
            font-weight: 650;
            margin-bottom: 4px;
        }

        .result-meta {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 10px;
        }

        .result-desc {
          color: var(--muted);
          font-size: 12px;
          margin: 0 0 10px;
          white-space: pre-wrap;
        }

        .result-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }

        .ghost {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
        }

        iframe {
            width: 100%;
            aspect-ratio: 16 / 9;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(0, 0, 0, 0.25);
        }

        .error {
            border: 1px solid rgba(255, 138, 0, 0.55);
            background: rgba(255, 138, 0, 0.10);
            color: var(--text);
            padding: 10px 12px;
            border-radius: 12px;
            margin-top: 12px;
            text-align: left;
            white-space: pre-wrap;
        }
    """


def _youtube_search(api_key: str, query: str, max_results: int = 8) -> dict[str, Any]:
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoEmbeddable": "true",
        "safeSearch": "moderate",
        "maxResults": max(1, min(int(max_results), 25)),
        "key": api_key,
    }

    response = requests.get(url, params=params, timeout=20)
    if not response.ok:
        # Try to preserve the JSON error payload (usually helpful).
        try:
            raise RuntimeError(response.json())
        except ValueError:
            response.raise_for_status()
    return response.json()


@app.route("/", methods=["GET"])
def index():
    api_key = os.getenv("GOOGLE_DEVELOPER_API_KEY")
    query = (request.args.get("q") or "").strip()
    video_id = (request.args.get("v") or "").strip()
    error = None
    results: list[dict[str, Any]] = []

    if query:
        if not api_key:
            error = (
                "Missing GOOGLE_DEVELOPER_API_KEY in environment.\n"
                "Create a .env file with:\n\n"
                "  GOOGLE_DEVELOPER_API_KEY=YOUR_KEY\n"
            )
        else:
            try:
                payload = _youtube_search(api_key=api_key, query=query, max_results=8)
                for item in payload.get("items", []):
                    vid = (item.get("id") or {}).get("videoId")
                    snippet = item.get("snippet") or {}
                    title = snippet.get("title") or "(untitled)"
                  description = snippet.get("description") or ""
                    channel = snippet.get("channelTitle") or ""
                    published = snippet.get("publishedAt") or ""
                  thumbnails = snippet.get("thumbnails") or {}
                  # Prefer medium/high, fall back to default.
                  thumbnail_url = (
                    (thumbnails.get("medium") or {}).get("url")
                    or (thumbnails.get("high") or {}).get("url")
                    or (thumbnails.get("default") or {}).get("url")
                    or ""
                  )
                    if not vid:
                        continue
                    results.append(
                        {
                            "videoId": str(vid),
                            "title": str(title),
                      "description": str(description),
                            "channel": str(channel),
                            "publishedAt": str(published),
                      "thumbnailUrl": str(thumbnail_url),
                        }
                    )
                if not video_id and results:
                    video_id = results[0]["videoId"]
            except Exception as e:
                error = str(e)

    return render_template_string(
        """
        <html>
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>YouTube Search & Player</title>
            <style>{{ css|safe }}</style>
          </head>
          <body>
            <div class="wrap">
              <div class="card">
                <h2>YouTube Search & Player</h2>
                <p class="sub">Search using the YouTube Data API, then play a result inline.</p>

                <form class="search" method="get" action="/">
                  <input type="text" name="q" value="{{ query_value | e }}" placeholder="Search videos (e.g., lofi study mix)" />
                  <button type="submit">Search</button>
                  {% if has_query %}
                    <a class="btn ghost" href="/">Clear</a>
                  {% endif %}
                </form>

                {% if error %}
                  <div class="error">{{ error | e }}</div>
                {% endif %}

                {% if has_query and not error %}
                  <div class="row">
                    <span class="badge">Results: {{ results|length }}</span>
                    {% if video_id %}
                      <span class="badge">Selected: {{ video_id | e }}</span>
                    {% endif %}
                  </div>

                  <div class="grid">
                    <div class="panel">
                      {% if video_id %}
                        <iframe
                          src="https://www.youtube.com/embed/{{ video_id | e }}"
                          title="YouTube video player"
                          frameborder="0"
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                          allowfullscreen
                        ></iframe>
                      {% else %}
                        <div class="sub">No playable video selected.</div>
                      {% endif %}
                    </div>

                    <div class="panel">
                      <div class="results">
                        {% for r in results %}
                          <div class="result">
                            <div class="result-top">
                              {% if r.thumbnailUrl %}
                                <img class="thumb" src="{{ r.thumbnailUrl | e }}" alt="Thumbnail" loading="lazy" />
                              {% endif %}
                              <div class="result-body">
                                <div class="result-title">{{ r.title | e }}</div>
                                <div class="result-meta">
                                  {{ r.channel | e }}{% if r.publishedAt %} • {{ r.publishedAt | e }}{% endif %}
                                </div>
                                {% if r.description %}
                                  <p class="result-desc">{{ r.description | e }}</p>
                                {% endif %}
                                <div class="result-actions">
                                  <a class="btn" href="{{ url_for('index', q=query_url, v=r.videoId) }}">Play</a>
                                  <a class="btn ghost" href="https://www.youtube.com/watch?v={{ r.videoId | e }}" target="_blank" rel="noopener noreferrer">Open on YouTube</a>
                                </div>
                              </div>
                            </div>
                          </div>
                        {% endfor %}
                        {% if results|length == 0 %}
                          <div class="sub">No results found.</div>
                        {% endif %}
                      </div>
                    </div>
                  </div>
                {% endif %}
              </div>
            </div>
          </body>
        </html>
        """,
        css=_theme_css(),
        query_value=query,
        query_url=query,
        has_query=bool(query),
        results=results,
        video_id=video_id,
        error=error,
    )


@app.route("/search", methods=["POST"])
def search_post():
    # Optional endpoint if you want to change the form to POST later.
    q = (request.form.get("q") or "").strip()
    if not q:
        return redirect(url_for("index"))
    return redirect(url_for("index", q=q))


def main() -> None:
    # Flask dev server
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()