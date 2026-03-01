# Dev Log — 2026-03-01 — Journey (YouTube API → Windows Audio Clip App → YouTube Search & Player Web App)

## Scope
This log captures the full path we took in this workspace, from the very first YouTube Data API error, through building a Windows system-audio capture web app ("last 18 seconds" clip), and ending with refactoring a YouTube search/player script into a themed Flask web app.

## High-Level Timeline
1. Fix a YouTube Data API `search` 400 caused by incorrect query encoding.
2. Build a Flask + WASAPI loopback audio app for Windows.
3. Replace live audio streaming with a silent rolling buffer (last 18 seconds in-memory) and on-demand playback.
4. Add capture controls, reliability fixes, a responsive audio-level meter, and a dark teal/orange theme.
5. Turn `Youtube-Search-n-Player.py` into a themed YouTube Search & Player web app.

---

## Phase 1 — YouTube API 400 Debugging (`Youtube-Demo.py`)

### Symptom
- Requests to `https://www.googleapis.com/youtube/v3/search` returned `400 Bad Request`.
- The URL showed `q=single%2Brock%2Bmusic%2Bsong`, which indicated the query contained literal `+` characters that then got URL-encoded as `%2B`.

### Fix
- Do not pre-encode the query with pluses. Use a plain string like `"single rock music song"` and let the HTTP client encode it.
- Improve diagnostics: when the API returns an error, print the error JSON body (instead of only raising a generic exception).
- Add a request timeout so failures don’t hang indefinitely.

---

## Phase 2 — Windows System Audio Capture Web App (`web_audio_capture.py`)

### Starting goal
Capture Windows system audio (speaker output) and play it in the browser.

### Key issues we hit and fixes

#### 1) WASAPI loopback open failures
- Error: `ValueError: Invalid audio channels`.
- Fix: pick channels/sample rate based on the loopback device’s input capabilities (not speaker device input fields), with sane fallbacks (prefer stereo, common sample rates like 48k/44.1k).

#### 2) WAV streaming header overflow
- Error: `struct.error: 'I' format requires 0 <= number <= 4294967295`.
- Root cause: “infinite” WAV header hacks overflow RIFF chunk sizes.
- Fix: stop trying to serve an unbounded WAV; instead generate a correct *finite* WAV header for an actual payload.

#### 3) Feedback loop (capturing output while also playing it)
- Symptom: audio kept looping / re-entering capture.
- Fix: move to a “silent capture” model and disable playback controls while capturing.

### Major redesign: “last 18 seconds in memory”
User request: only keep the latest 18 seconds (in memory), and when the user clicks Play it plays that clip.

Implemented:
- A rolling in-memory buffer of PCM chunks.
- `GET /clip.wav` returns a finite WAV snapshot of the last ~18 seconds.
- `GET /stream.wav` kept only as a backward-compatible alias to `/clip.wav`.
- Capture lifecycle endpoints:
  - `POST /capture/start`
  - `POST /capture/stop`
  - `GET /capture/status`
  - `GET /level` (capture meter)

### Audio level indicator
- Capture mode: server computes RMS level from PCM samples and exposes it via `GET /level`.
- Playback mode: browser uses WebAudio `AnalyserNode` on the `<audio>` element.
- Reduced noisy console logs by filtering `GET /level` request logs.
- Tuned smoothing separately for capture vs playback.

### Latest UX tweak (meter reset)
- When capturing stops, or playback pauses/ends, the level bar now resets to 0 immediately.

---

## Phase 3 — YouTube Search & Player Web App (`Youtube-Search-n-Player.py`)

## Goal
Turn `Youtube-Search-n-Player.py` into a web app and apply the same theme/colors used by `web_audio_capture.py`.

## Starting Point
- `Youtube-Search-n-Player.py` was a CLI script calling `youtube/v3/search` and printing JSON.
- `web_audio_capture.py` already used Flask with an inline HTML template and a defined theme via CSS variables:
  - `--bg`, `--panel`, `--panel-2`, `--text`, `--muted`, `--border`, `--teal`, `--orange`
  - background radial gradients + card/button styling

## What Changed (Chronological)

### 1) Refactor CLI → Flask web app
- Replaced the CLI entrypoint with a Flask app (`GET /`).
- UI behavior:
  - Search box (`q`)
  - Results list from YouTube Data API v3 (`search.list`)
  - Embedded playback via `https://www.youtube.com/embed/<videoId>`

### 2) Apply matching theme
- Reused the theme variables and styling approach from `web_audio_capture.py` (inline `<style>`).
- Kept the same overall look: dark background gradients, card container, consistent buttons and borders.

### 3) URL encoding + escaping hardening
- Adjusted templating so search queries and API-provided text render safely:
  - Use Jinja escaping (`|e`) in the template.
  - Use `url_for()` for Play links to avoid broken query encoding.

### 4) Added thumbnails + descriptions
- Parsed additional YouTube snippet fields:
  - `snippet.description`
  - `snippet.thumbnails.(medium|high|default).url`
- Rendered each result with a thumbnail + title + meta + description (same theme styling).

### 5) Reduce “Video unavailable” embeds
Observed: some search results still fail to embed (embedding disabled, region restrictions, etc.).

Mitigation implemented:
- After `search.list`, call `videos.list` (`part=status,contentDetails`) for returned IDs.
- Filter results to:
  - `status.embeddable == true`
  - `status.privacyStatus == public`
  - pass region restrictions (based on `search.list` response `regionCode` and `contentDetails.regionRestriction`).
- If a user-selected `v=` is filtered out, fall back to the first playable result.

### 6) Layout requested: 1 column with two areas
- Updated layout to always be a single column:
  - player panel
  - results list panel

### 7) Background tiling fix
- Added CSS to avoid the “repeated” gradient look:
  - `background-repeat: no-repeat;`
  - `background-attachment: fixed;`
  - `min-height: 100vh;`

### 8) Attempted “YouTube website iframe” card, then removed
- Tried adding a separate top card embedding `www.youtube.com` in an iframe.
- Expected browser behavior: YouTube blocks full-site iframe embedding (CSP / X-Frame-Options) → “refused to connect”.
- Attempted an embeddable alternative (player/search playlist) but user preferred removal.
- Removed the YouTube embedded website card and related variables.

### 9) Indentation cleanup and stability
- Multiple patches introduced mixed indentation inside `index()`.
- Resolved by rewriting the whole `index()` function with consistent 4-space indentation and verified via `py_compile`.

## Current State (YouTube Search & Player)
- Flask web app that:
  - Searches YouTube by query
  - Shows results with thumbnails + descriptions
  - Embeds a selected video
  - Pre-filters results to reduce non-embeddable selections
  - Uses the same theme/colors as `web_audio_capture.py`
  - Uses a 1-column layout (player then results)

---

## How to Run

### A) Audio Clip App
1. Start:
   - `D:/mse/nguyen_sy_hung_codebases/machine-learning-1/.venv/Scripts/python.exe web_audio_capture.py`
2. Open:
   - `http://127.0.0.1:5000`
3. Use:
   - Click **Capture**, wait 1–2 seconds, click **Stop**.
   - Press Play to hear the most recent 18 seconds.

### B) YouTube Search & Player
1. Ensure `.venv` has dependencies:
   - `flask`, `requests`, `python-dotenv`
2. Set API key in `.env`:
   - `GOOGLE_DEVELOPER_API_KEY=...`
3. Start:
   - `D:/mse/nguyen_sy_hung_codebases/machine-learning-1/.venv/Scripts/python.exe Youtube-Search-n-Player.py`
4. Open:
   - `http://127.0.0.1:5000`

## Notes / Tradeoffs
- Even with `videos.list` filtering, some embeds can still fail due to additional restrictions not fully detectable from these fields.
- The full YouTube website cannot be iframed reliably; only official embed endpoints are intended for that.
- `/clip.wav` is a finite snapshot (not a live stream). `/stream.wav` exists only for backward compatibility.

## Next Checks
- Audio app: confirm capture meter is responsive and resets to 0 on stop/pause/ended.
- YouTube app: confirm search and playback work with your API key + region.
