"""Voice-first, ball-by-ball cricket scorer powered by Sarvam AI."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# Sarvam endpoints and credentials. Set SARVAM_API_KEY in the environment or
# Streamlit secrets. A local .env is read without printing or exposing its value.
SARVAM_API_KEY = "SARVAM_API_KEY"  # environment variable name, never the key itself
SPEECH_TO_TEXT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
CHAT_COMPLETIONS_ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"
ASR_MODEL = "saaras:v4"
CHAT_MODEL = "sarvam-105b-conversations"

REGIONS = (
    "covers",
    "point",
    "third_man",
    "fine_leg",
    "square_leg",
    "mid_wicket",
    "long_on",
    "long_off",
    "none",
)
REGION_POINTS = {
    "covers": (-6.5, 5.5),
    "point": (-8.5, 1.5),
    "third_man": (-7.0, -5.0),
    "fine_leg": (-3.5, -8.0),
    "square_leg": (3.5, -8.0),
    "mid_wicket": (7.0, -5.0),
    "long_on": (7.0, 5.5),
    "long_off": (-7.0, 5.5),
    "none": (0.0, 0.0),
}
VALID_BAT_RUNS = {0, 1, 2, 3, 4, 6}
EXTRA_TYPES = {"none", "wide", "no_ball"}
CORE_STATE_KEYS = (
    "match_state",
    "current_batsmen",
    "current_bowler",
    "batting_stats",
    "bowling_stats",
    "this_over_strip",
    "commentary",
    "wagon_data",
    "batter_roster",
    "dismissed_batters",
    "pending_batter_slot",
    "gully_rules",
)


def _read_local_env() -> None:
    """Load simple KEY=value entries from .env into os.environ, without output."""
    env_path = Path(__file__).with_name(".env")
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and value:
                os.environ.setdefault(name, value)
    except OSError:
        pass


def get_api_key() -> str | None:
    """Read the API key from Streamlit secrets or the process environment."""
    try:
        secret = st.secrets.get(SARVAM_API_KEY)
        if secret:
            return str(secret)
    except Exception:
        pass
    _read_local_env()
    return os.environ.get(SARVAM_API_KEY) or None


def init_state() -> None:
    defaults: dict[str, Any] = {
        "match_state": {
            "total_runs": 0,
            "wickets": 0,
            "legal_balls": 0,
            "balls_in_current_over": 0,
        },
        "current_batsmen": {"striker": "Batter 1", "non_striker": "Batter 2"},
        "current_bowler": {"name": "Bowler 1"},
        "batting_stats": {
            "Batter 1": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "out": False},
            "Batter 2": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "out": False},
        },
        "bowling_stats": {
            "Bowler 1": {"legal_balls": 0, "runs": 0, "wickets": 0, "over_runs": 0},
        },
        "this_over_strip": [],
        "commentary": [],
        "wagon_data": [],
        "batter_roster": ["Batter 1", "Batter 2", "Batter 3", "Batter 4", "Batter 5"],
        "dismissed_batters": [],
        "pending_batter_slot": None,
        "gully_rules": "",
        "undo_snapshot": None,
        "last_audio_hash": None,
        "last_text_hash": None,
        "last_transcript": "",
        "last_event": None,
        "setup_striker": "Batter 1",
        "setup_non_striker": "Batter 2",
        "setup_bowler": "Bowler 1",
        "setup_roster": "Batter 1, Batter 2, Batter 3, Batter 4, Batter 5",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def overs_text(legal_balls: int) -> str:
    return f"{legal_balls // 6}.{legal_balls % 6}"


def safe_json_object(content: str) -> dict[str, Any]:
    """Parse JSON, tolerating a surrounding Markdown code fence."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Sarvam did not return a JSON object.")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The parsed event is not a JSON object.")
    return parsed


def transcribe_audio(audio_bytes: bytes, roster: list[str], api_key: str) -> str:
    """Transcribe one short WAV recording with Saaras v4."""
    form: dict[str, str] = {"model": ASR_MODEL, "mode": "codemix"}
    keyterms = [name for name in roster if name and name not in {"Unknown", "none"}]
    if keyterms:
        form["keyterms"] = json.dumps(keyterms[:50], ensure_ascii=False)
    try:
        response = requests.post(
            SPEECH_TO_TEXT_ENDPOINT,
            headers={"api-subscription-key": api_key},
            data=form,
            files={"file": ("ball.wav", audio_bytes, "audio/wav")},
            timeout=(10, 45),
        )
        response.raise_for_status()
        result = response.json()
        transcript = result.get("transcript", "").strip()
        if not transcript:
            raise ValueError("No speech was detected. Try a short, clear ball update.")
        return transcript
    except requests.Timeout as exc:
        raise RuntimeError("Speech transcription timed out. Please try again.") from exc
    except requests.HTTPError as exc:
        raise RuntimeError(f"Speech transcription failed (HTTP {exc.response.status_code}).") from exc
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("No speech"):
            raise
        raise RuntimeError("Could not transcribe that recording. Please try again.") from exc


EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": ["ball", "bowler_change"]},
        "batsman": {"type": "string"},
        "runs": {"type": "integer", "enum": [0, 1, 2, 3, 4, 6]},
        "extras": {"type": "string", "enum": ["none", "wide", "no_ball"]},
        "is_wicket": {"type": "boolean"},
        "wagon_region": {"type": "string", "enum": list(REGIONS)},
        "auto_commentary": {"type": "string"},
        "new_bowler": {"type": "string"},
    },
    "required": [
        "event_type",
        "batsman",
        "runs",
        "extras",
        "is_wicket",
        "wagon_region",
        "auto_commentary",
        "new_bowler",
    ],
    "additionalProperties": False,
}


def extract_ball_data(transcript: str, rules: str, api_key: str) -> dict[str, Any]:
    """Convert a transcript into a strictly structured ball or bowler event."""
    current = st.session_state.current_batsmen
    bowler = st.session_state.current_bowler.get("name", "Unknown")
    current_over = overs_text(st.session_state.match_state["legal_balls"])
    system_prompt = (
        "You are a careful cricket scorer. Convert the transcript into exactly one event. "
        "Use only the supplied transcript and match context; never invent a player or event. "
        "runs means batter runs, excluding the one-run wide/no-ball penalty. Allowed batter runs: 0, 1, 2, 3, 4, 6. "
        "A wide or no-ball adds one extra team run and does not count as a legal ball. "
        "For a bowler-change announcement, event_type=bowler_change, set new_bowler to the announced name, "
        "and set runs=0, extras=none, is_wicket=false, wagon_region=none. "
        "For a delivery, event_type=ball. Use current striker if batter is not stated. "
        "Map shot direction to one wagon_region. Generate one concise, lively Hinglish commentary sentence. "
        "If details are missing, use safe defaults and do not guess runs beyond what was said."
    )
    user_context = {
        "transcript": transcript,
        "gully_rules": rules.strip() or "Standard cricket rules.",
        "current_striker": current.get("striker") or "Unknown",
        "current_non_striker": current.get("non_striker") or "Unknown",
        "current_bowler": bowler,
        "current_over": current_over,
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "cricket_event", "strict": True, "schema": EVENT_SCHEMA},
        },
        "max_tokens": 350,
        "temperature": 0.1,
        "reasoning_effort": None,
    }
    try:
        response = requests.post(
            CHAT_COMPLETIONS_ENDPOINT,
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 45),
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        event = safe_json_object(content)
        return validate_event(event)
    except requests.Timeout as exc:
        raise RuntimeError("Event extraction timed out. Please try again.") from exc
    except requests.HTTPError as exc:
        raise RuntimeError(f"Event extraction failed (HTTP {exc.response.status_code}).") from exc
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Invalid event"):
            raise
        raise RuntimeError("Sarvam returned an unreadable event. Check the transcript and try again.") from exc


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("event_type", "ball")).lower()
    if event_type not in {"ball", "bowler_change"}:
        raise ValueError("Invalid event type.")
    try:
        runs = int(event.get("runs", 0))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid event: runs must be an integer.") from exc
    if runs not in VALID_BAT_RUNS:
        raise ValueError("Invalid event: runs must be 0, 1, 2, 3, 4, or 6.")
    extras = str(event.get("extras", "none")).lower()
    if extras not in EXTRA_TYPES:
        raise ValueError("Invalid event: unsupported extras type.")
    region = str(event.get("wagon_region", "none")).lower()
    if region not in REGIONS:
        region = "none"
    return {
        "event_type": event_type,
        "batsman": str(event.get("batsman") or "Unknown").strip(),
        "runs": runs,
        "extras": extras,
        "is_wicket": bool(event.get("is_wicket", False)),
        "wagon_region": region,
        "auto_commentary": str(event.get("auto_commentary") or "Ball recorded.").strip(),
        "new_bowler": str(event.get("new_bowler") or "").strip(),
    }


def _ensure_player(name: str) -> None:
    if name not in st.session_state.batting_stats:
        st.session_state.batting_stats[name] = {
            "runs": 0, "balls": 0, "fours": 0, "sixes": 0, "out": False
        }


def _ensure_bowler(name: str) -> None:
    if name not in st.session_state.bowling_stats:
        st.session_state.bowling_stats[name] = {
            "legal_balls": 0, "runs": 0, "wickets": 0, "over_runs": 0
        }


def save_undo_snapshot() -> None:
    st.session_state.undo_snapshot = {
        key: copy.deepcopy(st.session_state[key]) for key in CORE_STATE_KEYS
    }


def apply_event(event: dict[str, Any], transcript: str) -> None:
    """Apply a validated event and update all scorebook state deterministically."""
    if event["event_type"] == "bowler_change":
        new_name = event["new_bowler"] or ""
        if not new_name:
            raise ValueError("The new bowler's name was not clear. Try saying it again.")
        save_undo_snapshot()
        st.session_state.current_bowler["name"] = new_name
        _ensure_bowler(new_name)
        st.session_state.commentary.append(
            {"over": overs_text(st.session_state.match_state["legal_balls"]),
             "description": event["auto_commentary"], "runs": 0, "transcript": transcript}
        )
        st.session_state.last_event = {**event, "summary": f"New bowler: {new_name}"}
        return

    if st.session_state.pending_batter_slot:
        raise ValueError("Select the next batter before recording another ball.")

    save_undo_snapshot()
    match = st.session_state.match_state
    batsmen = st.session_state.current_batsmen
    bowler_name = st.session_state.current_bowler.get("name") or "Bowler 1"
    _ensure_bowler(bowler_name)

    batter_name = event["batsman"]
    if not batter_name or batter_name.lower() == "unknown":
        batter_name = batsmen.get("striker") or "Unknown batter"
    elif batter_name not in {batsmen.get("striker"), batsmen.get("non_striker")}:
        # Keep the scoring state consistent if ASR/LLM returns a known roster name.
        if batter_name in st.session_state.batter_roster:
            if batsmen.get("striker") is None:
                batsmen["striker"] = batter_name
            elif batsmen.get("non_striker") is None:
                batsmen["non_striker"] = batter_name
        else:
            batter_name = batsmen.get("striker") or batter_name
    _ensure_player(batter_name)

    bat_runs = event["runs"]
    extras = event["extras"]
    extra_penalty = 1 if extras in {"wide", "no_ball"} else 0
    team_runs = bat_runs + extra_penalty
    legal_ball = extras == "none"
    old_over = overs_text(match["legal_balls"])

    match["total_runs"] += team_runs
    st.session_state.batting_stats[batter_name]["runs"] += bat_runs
    # A no-ball is faced by the batter; a wide is not.
    if extras != "wide":
        st.session_state.batting_stats[batter_name]["balls"] += 1
    if bat_runs == 4:
        st.session_state.batting_stats[batter_name]["fours"] += 1
    if bat_runs == 6:
        st.session_state.batting_stats[batter_name]["sixes"] += 1

    bowler = st.session_state.bowling_stats[bowler_name]
    bowler["runs"] += team_runs
    bowler["over_runs"] += team_runs
    if legal_ball:
        bowler["legal_balls"] += 1
        match["legal_balls"] += 1
        match["balls_in_current_over"] += 1
    if event["is_wicket"]:
        match["wickets"] += 1
        dismissed = batter_name
        st.session_state.batting_stats[dismissed]["out"] = True
        st.session_state.dismissed_batters.append(dismissed)
        bowler["wickets"] += 1
        slot = "striker" if batsmen.get("striker") == dismissed else "non_striker"
        batsmen[slot] = None
        st.session_state.pending_batter_slot = slot

    # Odd completed batter runs switch ends. The over-end change is applied below.
    if bat_runs % 2 == 1:
        batsmen["striker"], batsmen["non_striker"] = (
            batsmen.get("non_striker"), batsmen.get("striker")
        )
        if st.session_state.pending_batter_slot:
            st.session_state.pending_batter_slot = (
                "non_striker" if st.session_state.pending_batter_slot == "striker" else "striker"
            )

    strip_label = "W" if event["is_wicket"] else (
        "Wd" if extras == "wide" else "Nb" if extras == "no_ball" else str(bat_runs)
    )
    st.session_state.this_over_strip.append(strip_label)
    st.session_state.wagon_data.append(
        {"batsman": batter_name, "bowler": bowler_name, "region": event["wagon_region"],
         "runs": bat_runs, "over": old_over}
    )
    st.session_state.commentary.append(
        {"over": old_over, "description": event["auto_commentary"],
         "runs": team_runs, "transcript": transcript}
    )

    if match["balls_in_current_over"] == 6:
        match["balls_in_current_over"] = 0
        # A maiden is an over with zero runs charged to the bowler.
        if bowler["over_runs"] == 0:
            bowler.setdefault("maidens", 0)
            bowler["maidens"] += 1
        bowler["over_runs"] = 0
        batsmen["striker"], batsmen["non_striker"] = (
            batsmen.get("non_striker"), batsmen.get("striker")
        )
        if st.session_state.pending_batter_slot:
            st.session_state.pending_batter_slot = (
                "non_striker" if st.session_state.pending_batter_slot == "striker" else "striker"
            )
        st.session_state.this_over_strip = []

    st.session_state.last_event = {
        **event,
        "summary": f"{old_over} · {strip_label} · {team_runs} team run(s)",
    }


def undo_last_event() -> None:
    snapshot = st.session_state.undo_snapshot
    if not snapshot:
        return
    for key, value in snapshot.items():
        st.session_state[key] = copy.deepcopy(value)
    st.session_state.undo_snapshot = None
    st.session_state.last_event = None


def render_wagon_wheel() -> None:
    data = st.session_state.wagon_data
    if not data:
        st.info("Wagon wheel points appear after the first scored shot.")
        return
    rows = []
    for item in data:
        x, y = REGION_POINTS.get(item["region"], REGION_POINTS["none"])
        if item["region"] == "none":
            continue
        rows.append({**item, "x": x, "y": y, "label": item["region"].replace("_", " ").title()})
    if not rows:
        st.info("No directional shots recorded yet.")
        return
    frame = pd.DataFrame(rows)
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="runs",
        size="runs",
        hover_name="batsman",
        hover_data={"bowler": True, "label": True, "over": True, "x": False, "y": False},
        color_continuous_scale=["#b8d975", "#7abf54", "#ffb454", "#ff765e"],
        size_max=20,
        labels={"runs": "Batter runs"},
    )
    fig.update_layout(
        height=440,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dfe9df",
        coloraxis_colorbar=dict(title="Runs", tickfont=dict(color="#dfe9df")),
        xaxis=dict(visible=False, range=[-11, 11], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-11, 11]),
        showlegend=False,
        shapes=[
            dict(type="circle", x0=-10, x1=10, y0=-10, y1=10,
                 line=dict(color="#486b52", width=2), fillcolor="rgba(42,87,49,.18)"),
            dict(type="circle", x0=-5, x1=5, y0=-5, y1=5,
                 line=dict(color="#486b52", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)"),
            dict(type="rect", x0=-0.45, x1=0.45, y0=-1.3, y1=1.3,
                 line=dict(color="#d7c29b", width=2), fillcolor="rgba(215,194,155,.28)"),
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Shot direction is estimated from the spoken description. Select a point to inspect batter, bowler, and over.")


def process_audio(audio_bytes: bytes) -> None:
    digest = hashlib.sha256(audio_bytes).hexdigest()
    if digest == st.session_state.last_audio_hash:
        return
    st.session_state.last_audio_hash = digest
    api_key = get_api_key()
    if not api_key:
        st.error("Add SARVAM_API_KEY to the project .env or Streamlit secrets before recording. No API request was made.")
        return
    try:
        with st.status("Processing ball update…", expanded=True) as status:
            status.write("Transcribing with Saaras v4…")
            transcript = transcribe_audio(audio_bytes, st.session_state.batter_roster, api_key)
            st.session_state.last_transcript = transcript
            status.write("Extracting the cricket event with Sarvam-105B…")
            event = extract_ball_data(transcript, st.session_state.gully_rules, api_key)
            apply_event(event, transcript)
            status.update(label="Ball recorded", state="complete", expanded=False)
        st.success(f"{st.session_state.last_event['summary']} — {transcript}")
    except RuntimeError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.warning(str(exc))
    except Exception:
        # Avoid rendering request details or credentials in the UI.
        st.error("The ball could not be recorded. Review the audio and try again.")


st.set_page_config(page_title="Cricket Scorer", page_icon="🏏", layout="centered")
init_state()

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: radial-gradient(ellipse at 15% 0%, #173627 0, #0b1711 50%, #08110d 100%); }
      .block-container { max-width: 880px; padding-top: 1.5rem; padding-bottom: 3rem; }
      h1,h2,h3 { letter-spacing: -.035em; }
      div[data-testid="stMetric"] { background:#13251b; border:1px solid #2a4533; padding:14px 16px; border-radius:14px; }
      div[data-testid="stMetricLabel"] { color:#aab9ac; }
      div[data-testid="stMetricValue"] { color:#f4f6ec; }
      .score-strip { display:flex; gap:7px; flex-wrap:wrap; margin-top:10px; }
      .ball-chip { min-width:33px; text-align:center; padding:6px 9px; border:1px solid #3a5941; background:#1b3022; border-radius:99px; font-weight:700; color:#e6eddf; }
      .ball-wicket { color:#ffb6a4; border-color:#934e3d; background:#48251d; }
      .ball-extra { color:#ffcf87; border-color:#896a3d; background:#3b301c; }
      .player-card { background:linear-gradient(145deg,#14281d,#101d16); border:1px solid #294632; border-radius:13px; padding:13px 15px; min-height:91px; }
      .player-role { color:#a6ba9b; font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
      .player-name { font-size:1.05rem; font-weight:700; margin-top:5px; }
      .player-stat { color:#d6e0d3; font-size:.9rem; margin-top:3px; }
      .stButton>button[kind="primary"] { background:#b8df72; border-color:#b8df72; color:#122014; font-weight:700; }
      .stTabs [data-baseweb="tab-list"] { gap:8px; }
      .small-note { color:#a2b2a4; font-size:.87rem; }
      @media(max-width:600px) { .block-container {padding-left:1rem;padding-right:1rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏏 Cricket Scorer")
st.caption("Speak the ball. Keep the game moving.")

# Match setup is kept compact; player names are also sent as Saaras key terms.
with st.expander("⚙️ Match Settings & Gully Rules", expanded=False):
    with st.form("match_setup"):
        left, right = st.columns(2)
        with left:
            st.text_input("Striker", key="setup_striker")
            st.text_input("Non-striker", key="setup_non_striker")
        with right:
            st.text_input("Current bowler", key="setup_bowler")
            st.text_input("Batting roster (comma-separated)", key="setup_roster")
        rules_input = st.text_area(
            "Gully rules",
            value=st.session_state.gully_rules,
            placeholder="Example: one bounce over the wall is out; six overs per innings.",
            height=85,
            key="rules_input",
        )
        if st.form_submit_button("Save match setup"):
            striker = st.session_state.setup_striker.strip() or "Batter 1"
            non_striker = st.session_state.setup_non_striker.strip() or "Batter 2"
            bowler = st.session_state.setup_bowler.strip() or "Bowler 1"
            roster = [name.strip() for name in st.session_state.setup_roster.split(",") if name.strip()]
            for name in (striker, non_striker):
                if name not in roster:
                    roster.insert(0, name)
            st.session_state.current_batsmen = {"striker": striker, "non_striker": non_striker}
            st.session_state.current_bowler = {"name": bowler}
            st.session_state.batter_roster = list(dict.fromkeys(roster))
            st.session_state.gully_rules = rules_input.strip()
            _ensure_player(striker)
            _ensure_player(non_striker)
            _ensure_bowler(bowler)
            st.success("Match setup saved.")

match = st.session_state.match_state
legal_balls = match["legal_balls"]
overs = overs_text(legal_balls)
run_rate = match["total_runs"] / (legal_balls / 6) if legal_balls else 0.0

with st.container(border=True):
    st.markdown("#### LIVE SCORE")
    score_col, rate_col = st.columns(2)
    score_col.metric("Runs / wickets", f"{match['total_runs']} / {match['wickets']}")
    rate_col.metric("Overs · run rate", f"{overs} · {run_rate:.2f}")
    st.caption("THIS OVER")
    if st.session_state.this_over_strip:
        chips = []
        for ball in st.session_state.this_over_strip:
            cls = "ball-wicket" if ball == "W" else "ball-extra" if ball in {"Wd", "Nb"} else ""
            chips.append(f'<span class="ball-chip {cls}">{ball}</span>')
        st.markdown(f'<div class="score-strip">{"".join(chips)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="score-strip"><span class="ball-chip">·</span></div>', unsafe_allow_html=True)

st.markdown("#### CURRENT ACTION")
striker = st.session_state.current_batsmen.get("striker")
non_striker = st.session_state.current_batsmen.get("non_striker")
bowler_name = st.session_state.current_bowler.get("name", "Bowler 1")
player_cols = st.columns(3)
for col, label, name, active in (
    (player_cols[0], "🏏 STRIKER", striker, True),
    (player_cols[1], "NON-STRIKER", non_striker, False),
):
    if name:
        stats = st.session_state.batting_stats.get(name, {"runs": 0, "balls": 0})
        col.markdown(
            f'<div class="player-card"><div class="player-role">{label}</div>'
            f'<div class="player-name">{name}</div><div class="player-stat">'
            f'{stats["runs"]} ({stats["balls"]}){"  ·  ON STRIKE" if active else ""}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        col.markdown('<div class="player-card"><div class="player-role">NEXT BATTER</div><div class="player-name">Waiting for batter</div></div>', unsafe_allow_html=True)

bowler_stats = st.session_state.bowling_stats.get(
    bowler_name, {"legal_balls": 0, "runs": 0, "wickets": 0, "maidens": 0}
)
player_cols[2].markdown(
    f'<div class="player-card"><div class="player-role">CURRENT BOWLER</div>'
    f'<div class="player-name">{bowler_name}</div><div class="player-stat">'
    f'{overs_text(bowler_stats["legal_balls"])} · {bowler_stats.get("maidens", 0)} · '
    f'{bowler_stats["runs"]} · {bowler_stats["wickets"]}</div></div>',
    unsafe_allow_html=True,
)

if st.session_state.pending_batter_slot:
    slot = st.session_state.pending_batter_slot
    available = [
        name for name in st.session_state.batter_roster
        if name not in st.session_state.dismissed_batters
        and name not in {st.session_state.current_batsmen.get("striker"), st.session_state.current_batsmen.get("non_striker")}
    ]
    st.warning("Wicket recorded. Send the next batter in to continue.")
    choice_col, button_col = st.columns([3, 1])
    next_name = choice_col.selectbox("Next batter", available or ["Add a batter in Match Settings"], key="next_batter_choice")
    if button_col.button("Send in", type="primary", disabled=not available):
        st.session_state.current_batsmen[slot] = next_name
        _ensure_player(next_name)
        st.session_state.pending_batter_slot = None
        st.rerun()

st.markdown("#### VOICE SCORING")
st.markdown('<p class="small-note">Record one short ball update in English or Hinglish. Example: “Ek run fine leg ki taraf.”</p>', unsafe_allow_html=True)
audio_value = st.audio_input("Record what happened on the ball", sample_rate=16000, key="ball_audio")
if audio_value is not None:
    if len(audio_value.getvalue()) > 8_000_000:
        st.error("Recording is too large. Keep ball updates short and under 30 seconds.")
    else:
        process_audio(audio_value.getvalue())

if st.session_state.last_transcript:
    with st.expander("Last recognized update", expanded=False):
        st.write(st.session_state.last_transcript)
        if st.session_state.last_event:
            st.json({k: v for k, v in st.session_state.last_event.items() if k != "summary"})

if st.session_state.undo_snapshot:
    undo_col, last_col = st.columns([1, 3])
    if undo_col.button("↶ Undo last event"):
        undo_last_event()
        st.rerun()
    if st.session_state.last_event:
        last_col.caption(st.session_state.last_event.get("summary", "Last event recorded"))

tab_commentary, tab_scorecard, tab_wagon = st.tabs(
    ["Live Auto-Commentary", "Scorecard", "Wagon Wheel"]
)

with tab_commentary:
    with st.container(height=300, border=True):
        if not st.session_state.commentary:
            st.caption("Ball-by-ball commentary will appear here.")
        for item in reversed(st.session_state.commentary):
            st.markdown(f"**Over {item['over']}**  ")
            st.write(item["description"])
            if item.get("transcript"):
                st.caption(f"Heard: {item['transcript']}")
            st.divider()

with tab_scorecard:
    st.markdown("**Batting**")
    batting_rows = []
    ordered_names = list(dict.fromkeys(
        [name for name in [st.session_state.current_batsmen.get("striker"), st.session_state.current_batsmen.get("non_striker")] if name]
        + st.session_state.batter_roster
    ))
    for name in ordered_names:
        stats = st.session_state.batting_stats.get(name)
        if stats:
            balls = stats["balls"]
            batting_rows.append({
                "Batter": name + (" *" if name in {striker, non_striker} and not stats["out"] else ""),
                "R": stats["runs"], "B": balls, "4s": stats["fours"], "6s": stats["sixes"],
                "SR": round(stats["runs"] * 100 / balls, 1) if balls else 0.0,
                "Status": "Out" if stats["out"] else "Batting" if name in {striker, non_striker} else "Yet to bat",
            })
    if batting_rows:
        st.dataframe(pd.DataFrame(batting_rows), hide_index=True, use_container_width=True)
    st.markdown("**Bowling**")
    bowling_rows = []
    for name, stats in st.session_state.bowling_stats.items():
        balls = stats["legal_balls"]
        bowling_rows.append({
            "Bowler": name,
            "O": overs_text(balls),
            "M": stats.get("maidens", 0),
            "R": stats["runs"],
            "W": stats["wickets"],
            "Econ": round(stats["runs"] / (balls / 6), 2) if balls else 0.0,
        })
    st.dataframe(pd.DataFrame(bowling_rows), hide_index=True, use_container_width=True)

with tab_wagon:
    st.markdown("**Shot direction by batter and bowler**")
    render_wagon_wheel()

st.caption("Sarvam AI · Saaras speech recognition + Sarvam-105B structured event extraction")
