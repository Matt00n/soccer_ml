"""Blinded expert review of creativity measures: clip sampling, clip data and the rating page.

Clips come from four sources, so that ratings test each measure and each
component rather than only the top of one ranking:

* ``top:<measure>``: random draws from every measure's top decisions;
* ``originality_control``: same action type and gender, similar advantage
  (value-matched) but rarity below 0.5, paired with a flagged clip; if
  originality matters, flagged clips should be rated more creative;
* ``value_control``: similar rarity but non-positive advantage, paired with a
  flagged clip; if value matters, flagged clips should be rated more creative;
* ``random``: the base rate.

The rating page shows clips in random order, without scores or selection
reasons; player, teams and time are hidden until the rater reveals them
(after rating the 2D scene), which removes the "famous player" halo from the
first rating.  The key (selection reasons and all scores) goes to a separate
file that is only joined after rating (``scripts/creativity/score_review.py``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from soccer_ml.creativity.measures import MEASURES


def sample_clips(
    table: pd.DataFrame,
    scores: pd.DataFrame,
    per_measure: int = 3,
    pool_fraction: float = 0.01,
    originality_controls: int = 10,
    value_controls: int = 8,
    random_clips: int = 8,
    seed: int = 0,
    exclude_restarts: bool = True,
) -> pd.DataFrame:
    """Selected decisions (``table`` index) with their selection reason and paired clip."""
    rng = np.random.default_rng(seed)
    eligible = table["restart"].fillna("") == "" if exclude_restarts else pd.Series(True, index=table.index)
    chosen: Dict[int, Dict[str, Any]] = {}
    for measure in MEASURES:
        if measure.name not in scores:
            continue
        values = scores[measure.name].where(eligible)
        threshold = values.quantile(1 - pool_fraction)
        pool = values.index[(values >= threshold) & (values > 0)].difference(list(chosen))
        for index in rng.choice(pool, size=min(per_measure, len(pool)), replace=False):
            chosen[int(index)] = {"reason": f"top:{measure.name}", "pair": None}

    baselines = {measure.name for measure in MEASURES if measure.family == "baseline"}
    flagged = [i for i, info in chosen.items() if info["reason"].split(":", 1)[1] not in baselines]
    advantage_pct = table["advantage"].rank(pct=True)
    gender = table["women"].astype(str)

    def control(anchor: int, condition: pd.Series, distance: pd.Series, reason: str) -> None:
        same = (table["action_type"] == table.at[anchor, "action_type"]) & (gender == gender.at[anchor]) & eligible
        candidates = distance[same & condition].drop(list(chosen), errors="ignore").nsmallest(20)
        if len(candidates):
            pick = int(rng.choice(candidates.index))
            chosen[pick] = {"reason": reason, "pair": anchor}

    for anchor in rng.permutation(flagged)[:originality_controls]:
        control(int(anchor), (table["rarity"] < 0.5) & ((advantage_pct - advantage_pct.at[anchor]).abs() <= 0.05),
                (advantage_pct - advantage_pct.at[anchor]).abs(), "originality_control")
    for anchor in rng.permutation(flagged)[:value_controls]:
        control(int(anchor), (table["advantage"] <= 0) & ((table["rarity"] - table.at[anchor, "rarity"]).abs() <= 0.03),
                (table["rarity"] - table.at[anchor, "rarity"]).abs(), "value_control")
    rest = table.index[eligible].difference(list(chosen))
    for index in rng.choice(rest, size=min(random_clips, len(rest)), replace=False):
        chosen[int(index)] = {"reason": "random", "pair": None}

    order = rng.permutation(list(chosen))
    clip_ids = {int(index): f"C{number:03d}" for number, index in enumerate(order, start=1)}
    rows = []
    for index in order:
        info = chosen[int(index)]
        rows.append({"clip_id": clip_ids[int(index)], "row": int(index), "reason": info["reason"],
                     "pair_clip": clip_ids.get(info["pair"], "") if info["pair"] is not None else ""})
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def _match_files(events_path: str, three_sixty_path: str) -> tuple[list, dict]:
    events = sorted(json.loads(Path(events_path).read_text()), key=lambda event: event.get("index", 0))
    frames = {frame["event_uuid"]: frame for frame in json.loads(Path(three_sixty_path).read_text())}
    return events, frames


def _text(value: Any) -> str:
    """CSV cells as display text (NaN -> '')."""
    return "" if value is None or (isinstance(value, float) and value != value) else str(value)


def _flag(value: Any) -> bool:
    return str(value) in ("True", "true", "1", "1.0")


def _mirror(point, flip: bool) -> list[float]:
    x, y = float(point[0]), float(point[1])
    return [120.0 - x, 80.0 - y] if flip else [x, y]


def clip_payload(row: pd.Series, events_path: Path, three_sixty_path: Path, trail_events: int = 5) -> Dict[str, Any]:
    """Freeze frame, visible area, preceding ball path and the action, in the actor's attacking frame."""
    events, frames = _match_files(str(events_path), str(three_sixty_path))
    by_id = {event["id"]: (position, event) for position, event in enumerate(events)}
    position, event = by_id[row["event_uuid"]]
    team = (event.get("team") or {}).get("id")
    frame_id = row["anchor_uuid"] if row["anchor_uuid"] in frames else row["event_uuid"]
    frame = frames.get(frame_id, {})
    frame_team = (by_id.get(frame_id, (0, event))[1].get("team") or {}).get("id")
    flip = frame_team != team  # StatsBomb frames are in the frame event's team perspective
    players = []
    for player in frame.get("freeze_frame", []):
        teammate = bool(player.get("teammate")) != flip
        players.append({"xy": _mirror(player["location"], flip), "teammate": teammate,
                        "actor": bool(player.get("actor")) and not flip, "keeper": bool(player.get("keeper"))})
    polygon = frame.get("visible_area", [])
    visible = [_mirror(point, flip) for point in zip(polygon[0::2], polygon[1::2])]

    trail = []
    for past in reversed(events[max(0, position - 40):position]):
        if past.get("possession") != event.get("possession") or past.get("period") != event.get("period"):
            break
        if isinstance(past.get("location"), list) and (past.get("type") or {}).get("name") not in ("Pressure",):
            same = (past.get("team") or {}).get("id") == team
            trail.append({"xy": _mirror(past["location"], not same), "type": past["type"]["name"], "own": same})
        if len(trail) >= trail_events:
            break
    kind = row["action_type"]
    details = event.get(kind) or {}
    end = details.get("end_location")
    return {
        "players": players,
        "visible": visible,
        "trail": list(reversed(trail)),
        "action": {
            "type": kind, "start": _mirror(event["location"], False),
            "end": _mirror(end, False) if isinstance(end, list) else None,
            "height": _text(row.get("height")), "body_part": _text(row.get("body_part")),
            "technique": _text(row.get("technique")), "outcome": _text(row.get("outcome")),
            "under_pressure": _flag(row.get("under_pressure")), "play_pattern": _text(row.get("play_pattern")),
        },
        "women": _flag(row.get("women")),
        "hidden": {
            "player": _text(row.get("player")), "team": _text(row.get("team")), "opponent": _text(row.get("opponent")),
            "competition": " ".join(filter(None, (_text(row.get(k)) for k in ("competition", "season", "stage")))),
            "date": _text(row.get("match_date")), "period": int(row["period"]),
            "clock": f"{int(row['minute']):02d}:{int(row['second']):02d}",
            "search": f"{_text(row.get('home_team'))} vs {_text(row.get('away_team'))} {_text(row.get('competition'))} "
                      f"{_text(row.get('season'))} highlights",
        },
    }


def review_page(clips: list[Dict[str, Any]], kit_id: str) -> str:
    data = json.dumps({"kit": kit_id, "clips": clips}).replace("</", "<\\/")
    return _PAGE.replace("__DATA__", data)


_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Creativity review</title>
<style>
:root{--ink:#1d2433;--muted:#5d6678;--line:#d9dde5;--bg:#f6f7f9;--card:#fff;--accent:#2457c5;--own:#2457c5;--opp:#c8402f}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:980px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between;margin-bottom:12px}
h1{font-size:18px;margin:0}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:12px}
details summary{cursor:pointer;font-weight:600}
svg{width:100%;height:auto;display:block;background:#3f8f4f;border-radius:6px}
.row{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin:6px 0}
.scale label{display:inline-flex;align-items:center;gap:3px;margin-right:6px}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.q{min-width:170px;font-weight:600}.info{font-size:13px}
textarea{width:100%;min-height:60px;font:12px ui-monospace,monospace}
select,input[type=text]{font:inherit;padding:4px}
.legend span{display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:12px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.done{color:#1f7a3a;font-weight:600}
</style></head><body><main>
<header><h1>Creativity review <span class="muted" id="kit"></span></h1>
<div class="row"><span id="progress" class="muted"></span>
<button id="prev">&larr; Prev</button><select id="jump"></select><button id="next" class="primary">Next &rarr;</button></div></header>
<details class="card"><summary>How to rate (read once)</summary>
<p>Rate the <b>decision</b>: the choice of what to do and where, given the situation in the picture, not the technique and not the outcome.
<b>Creative</b> = original <i>and</i> effective (a surprising choice that was a good idea). <b>Original</b> = how unexpected the choice was in this situation.
<b>Effective</b> = how good the choice was, regardless of whether it worked out.</p>
<p>1. Rate from the 2D scene first. 2. Optionally reveal the match and player, look up the scene (search link), and add a video rating. Flag scenes that look like data problems.</p>
<p class="legend"><span><i class="dot" style="background:var(--own)"></i>team in possession</span><span><i class="dot" style="background:var(--opp)"></i>opponents</span>
<span><i class="dot" style="background:#fff;border:2px solid #111"></i>actor</span><span>light area = camera view</span><span>dotted = ball path before</span><span>arrow = the action</span></p>
</details>
<div class="card"><div id="pitch"></div><div class="info" id="info"></div></div>
<div class="card" id="form"></div>
<div class="card"><div class="row"><button id="export" class="primary">Download ratings CSV</button><button id="show">Show CSV</button>
<span class="muted">Ratings are saved in this browser automatically.</span></div><textarea id="csv" hidden readonly></textarea></div>
</main>
<script>
const DATA = __DATA__;
const KEY = "creativity-review-" + DATA.kit;
let state = {}; try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let current = 0;
const FIELDS = ["creative", "original", "effective", "issue", "revealed", "rated_before_reveal", "video", "creative_video", "notes"];
const ISSUES = ["", "none", "data error (location/player/action wrong)", "target out of camera view", "deflection or mis-hit (not intended)", "set-piece routine", "cannot judge from 2D"];
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function rating(id) { return state[id] || (state[id] = {}); }
const S = 5, W = 120 * S, H = 80 * S;
function p(xy) { return [(xy[0] * S).toFixed(1), (xy[1] * S).toFixed(1)]; }
function pitch(clip) {
  const a = clip.action; let g = [];
  g.push(`<rect x="0" y="0" width="${W}" height="${H}" fill="#3f8f4f"/>`);
  if (clip.visible.length > 2) g.push(`<polygon points="${clip.visible.map(q => p(q).join(",")).join(" ")}" fill="rgba(255,255,255,0.18)" stroke="rgba(255,255,255,0.5)" stroke-dasharray="4 4"/>`);
  const L = 'stroke="rgba(255,255,255,0.85)" stroke-width="2" fill="none"';
  g.push(`<rect x="1" y="1" width="${W - 2}" height="${H - 2}" ${L}/><line x1="${W / 2}" y1="0" x2="${W / 2}" y2="${H}" ${L}/><circle cx="${W / 2}" cy="${H / 2}" r="${10 * S}" ${L}/>`);
  for (const x0 of [0, 120]) { const d = x0 ? -1 : 1;
    g.push(`<rect x="${Math.min(x0, x0 + d * 18) * S}" y="${18 * S}" width="${18 * S}" height="${44 * S}" ${L}/><rect x="${Math.min(x0, x0 + d * 6) * S}" y="${30 * S}" width="${6 * S}" height="${20 * S}" ${L}/>`); }
  g.push(`<text x="${W - 8}" y="${H + 17}" text-anchor="end" fill="#fff" font-size="13">attacking direction →</text>`);
  if (clip.trail.length) { const pts = clip.trail.map(t => p(t.xy).join(",")).concat([p(a.start).join(",")]);
    g.push(`<polyline points="${pts.join(" ")}" fill="none" stroke="#ffe066" stroke-width="2" stroke-dasharray="3 5"/>`);
    clip.trail.forEach(t => { const [x, y] = p(t.xy); g.push(`<circle cx="${x}" cy="${y}" r="3" fill="${t.own ? '#ffe066' : '#ffb3a7'}"><title>${t.type}</title></circle>`); }); }
  for (const pl of clip.players) { const [x, y] = p(pl.xy);
    g.push(`<circle cx="${x}" cy="${y}" r="${pl.actor ? 9 : 7}" fill="${pl.teammate ? 'var(--own)' : 'var(--opp)'}" stroke="${pl.actor ? '#fff' : (pl.keeper ? '#111' : 'rgba(0,0,0,0.35)')}" stroke-width="${pl.actor ? 3 : (pl.keeper ? 3 : 1)}"/>`); }
  const [sx, sy] = p(a.start);
  if (a.end) { const [ex, ey] = p(a.end);
    g.push(`<line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" stroke="#fff" stroke-width="3" ${a.type === 'carry' ? 'stroke-dasharray="8 6"' : ''} marker-end="url(#arrow)"/>`); }
  else g.push(`<circle cx="${sx}" cy="${sy}" r="13" fill="none" stroke="#fff" stroke-width="3"/>`);
  return `<svg viewBox="0 0 ${W} ${H + 24}" role="img" aria-label="pitch"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#fff"/></marker></defs>${g.join("")}</svg>`;
}
function scale(id, field, label, hint) {
  const r = rating(id)[field] || "";
  let s = `<div class="row scale"><span class="q" title="${hint}">${label}</span>`;
  for (let v = 1; v <= 5; v++) s += `<label><input type="radio" name="${field}" value="${v}" ${String(r) === String(v) ? "checked" : ""}>${v}</label>`;
  return s + `<span class="muted">1 = not at all, 5 = very</span></div>`;
}
function render() {
  const clip = DATA.clips[current], id = clip.clip_id, r = rating(id), a = clip.action, h = clip.hidden;
  document.getElementById("kit").textContent = DATA.kit;
  document.getElementById("pitch").innerHTML = pitch(clip);
  const desc = [a.type + (a.height ? ` (${a.height})` : ""), a.body_part, a.technique, a.outcome ? `outcome: ${a.outcome}` : (a.type === "pass" ? "completed" : ""),
    a.under_pressure ? "under pressure" : "", a.play_pattern, clip.women ? "women's football" : "men's football"].filter(Boolean).join(" · ");
  document.getElementById("info").innerHTML = `<div class="row"><b>${id}</b><span>${desc}</span></div>` + (r.revealed ?
    `<div class="row"><span><b>${h.player}</b> (${h.team}) vs ${h.opponent}</span><span>${h.competition}, ${h.date}</span><span>period ${h.period}, ${h.clock}</span>
     <a target="_blank" rel="noopener" href="https://www.youtube.com/results?search_query=${encodeURIComponent(h.search)}">search video</a></div>` : "");
  let f = scale(id, "creative", "Creative", "original AND effective") + scale(id, "original", "Original", "how unexpected") + scale(id, "effective", "Effective", "how good a choice");
  f += `<div class="row"><span class="q">Data issue?</span><select name="issue">${ISSUES.map(v => `<option ${r.issue === v ? "selected" : ""}>${v}</option>`).join("")}</select></div>`;
  f += r.revealed ? `<div class="row"><span class="q">Video</span><select name="video">${["", "found", "not found", "not searched"].map(v => `<option ${r.video === v ? "selected" : ""}>${v}</option>`).join("")}</select></div>` + scale(id, "creative_video", "Creative (after video)", "after watching") :
    `<div class="row"><button id="reveal">Reveal match &amp; player</button><span class="muted">after rating the 2D scene</span></div>`;
  f += `<div class="row"><span class="q">Notes</span><input type="text" name="notes" style="flex:1" value="${(r.notes || "").replace(/"/g, "&quot;")}"></div>`;
  const form = document.getElementById("form"); form.innerHTML = f;
  form.querySelectorAll("input,select").forEach(el => el.addEventListener("change", () => { rating(id)[el.name] = el.value; save(); progress(); }));
  form.querySelectorAll("input[type=text]").forEach(el => el.addEventListener("input", () => { rating(id)[el.name] = el.value; save(); }));
  const rv = document.getElementById("reveal");
  if (rv) rv.onclick = () => { const q = rating(id); q.revealed = "yes"; q.rated_before_reveal = q.creative ? "yes" : "no"; save(); render(); };
  progress();
}
function progress() {
  const done = DATA.clips.filter(c => (state[c.clip_id] || {}).creative).length;
  document.getElementById("progress").innerHTML = `clip ${current + 1} / ${DATA.clips.length} · ` + (done === DATA.clips.length ? `<span class="done">all rated</span>` : `${done} rated`);
  const jump = document.getElementById("jump");
  jump.innerHTML = DATA.clips.map((c, i) => `<option value="${i}" ${i === current ? "selected" : ""}>${c.clip_id}${(state[c.clip_id] || {}).creative ? " ✓" : ""}</option>`).join("");
}
function csv() {
  const esc = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return [["clip_id"].concat(FIELDS).join(",")].concat(DATA.clips.map(c => [c.clip_id].concat(FIELDS.map(k => (state[c.clip_id] || {})[k])).map(esc).join(","))).join("\n");
}
document.getElementById("prev").onclick = () => { current = Math.max(0, current - 1); render(); };
document.getElementById("next").onclick = () => { current = Math.min(DATA.clips.length - 1, current + 1); render(); };
document.getElementById("jump").onchange = e => { current = +e.target.value; render(); };
document.getElementById("export").onclick = () => { const blob = new Blob([csv()], {type: "text/csv"}); const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = `ratings_${DATA.kit}.csv`; document.body.appendChild(link); link.click(); link.remove(); };
document.getElementById("show").onclick = () => { const t = document.getElementById("csv"); t.hidden = !t.hidden; t.value = csv(); };
render();
</script></body></html>
"""


def build_key(table: pd.DataFrame, scores: pd.DataFrame, clips: pd.DataFrame, extra: Optional[list[str]] = None) -> pd.DataFrame:
    """Selection reasons, scores, score percentiles and context of every clip (opened only after rating)."""
    percentiles = scores.rank(pct=True).add_prefix("pct_")
    context = ["model", "match_id", "event_uuid", "player", "team", "position", "action_type", "competition", "season",
               "minute", "second", "rarity", "surprise", "advantage", "td_advantage", "mc_advantage", "q", "v",
               "end_zone_visible", "opponents_bypassed"] + (extra or [])
    context = [c for c in context if c in table]
    rows = clips.set_index("row")
    key = rows.join(table[context]).join(scores.add_prefix("score_")).join(percentiles)
    return key.reset_index(drop=True)
