"""What a campaign is computing: every state's status, every cycle's progress, a terminal
summary and a self-contained HTML page (``<campaign>/status.html``).

Read-only: the run records (attempt.json, run.json), the submission ledger and, unless told
not to, one ``squeue`` listing of your jobs. Nothing is prepared, submitted or changed; only
the page is written. A state is measured against its target stage: ``--through`` when given,
else the stage the refill or its last submission asked for, else 02_final.
"""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from .fileio import atomic_text, read_json, utc_now
from .notation import Notation, plain, site_number
from .vasp import STAGE_FREQ, STAGES, protocol_stages, resolve_through

CATEGORIES = ("finished", "running", "queued", "continue", "attention", "not_started")
LABELS = {
    "finished": "finished",
    "running": "running",
    "queued": "queued",
    "continue": "to continue",
    "attention": "needs you",
    "not_started": "not started",
    "no_structure": "no structure",
}
ICONS = {
    "finished": "✓",
    "running": "▶",
    "queued": "◷",
    "continue": "↻",
    "attention": "!",
    "not_started": "⋅",
    "no_structure": "×",
}
EXPLAIN = {
    "finished": "the target stage is complete",
    "running": "a job is running it",
    "queued": "a job waits in the queue",
    "continue": "ran part of the way; the next submit continues it",
    "attention": "a failed attempt, a used-up budget or an unknown submission waits for you",
    "not_started": "nothing has run yet",
}
# --list choices: what each prints (state IDs, one per line).
LISTS = {
    "finished": {"finished"},
    "running": {"running"},
    "queued": {"queued"},
    "active": {"running", "queued"},
    "continue": {"continue"},
    "attention": {"attention"},
    "not-started": {"not_started"},
    "todo": {"continue", "not_started"},
}
# Best first: the status an intermediate shows when its structures differ.
RANK = {c: k for k, c in enumerate(CATEGORIES)}
UNKNOWN_SUBMISSION = ("submitting", "submission_unknown")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _read(path, default=None):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return default


def _targets(campaign, state_ids, through, latest, config):
    """Target stage folder of every state, and a phrase saying where the targets came from."""
    if through:
        stage = resolve_through(through, config)
        return {sid: stage for sid in state_ids}, f"{stage} (--through {through})"
    refill = (_read(Path(campaign) / "refill.json", {}) or {}).get("targets") or {}
    out, sources = {}, Counter()
    for sid in state_ids:
        goal, source = refill.get(sid), "the refill's targets"
        if goal is None and (latest.get(sid) or {}).get("through"):
            goal, source = latest[sid]["through"], "as last submitted"
        if goal is None:
            goal, source = "final", "02_final where never submitted"
        try:
            out[sid] = resolve_through(goal, config)
        except ValueError:
            out[sid], source = resolve_through("final", config), "02_final where never submitted"
        sources[source] += 1
    origin = "; ".join(
        s for s in ("the refill's targets", "as last submitted", "02_final where never submitted") if sources[s]
    )
    stages = sorted(set(out.values()), key=STAGES.index)
    if len(stages) <= 1:
        return out, f"{stages[0] if stages else 'none'} ({origin})" if origin else "none"
    counts = Counter(out.values())
    return out, ", ".join(f"{s} for {counts[s]}" for s in stages) + f" ({origin}; --through sets one for all)"


def _freq_units(folder):
    """(complete, total) displacement units of 05_freq, or None."""
    units = (_read(folder / "units.json", {}) or {}).get("units") or []
    if not units:
        return None
    done = sum(
        1
        for unit in units
        if (_read(folder / "units" / unit["id"] / "attempt.json", {}) or {}).get("status") == "complete"
    )
    return done, len(units)


def _stage_text(stage, units):
    if stage == STAGE_FREQ and units:
        return f"{stage} {units[0]}/{units[1]} displacements"
    return stage


def _job_text(job):
    text = f"job {job['job_id']} {job['slurm_state'].lower()}"
    if job.get("elapsed") and job["running"]:
        text += f" {job['elapsed']}"
        if job.get("time_left") and job["time_left"] not in ("UNLIMITED", "INVALID", "NOT_SET"):
            text += f" ({job['time_left']} left)"
    elif job.get("reason") and not job["running"]:
        text += f" ({job['reason']})"
    return text


def _classify(state_dir, target, config, stages, jobs, last, slurm_known):
    """One state's record: category, detail, stage statuses, job."""
    from .runner import _archive_counts, submission_check

    sid = state_dir.name
    attempts = {}
    for stage in stages:
        record = _read(state_dir / stage / "attempt.json")
        if record is not None:
            attempts[stage] = record
    status = {stage: (record or {}).get("status", "unreadable") for stage, record in attempts.items()}
    units = _freq_units(state_dir / STAGE_FREQ) if STAGE_FREQ in stages else None
    try:
        check = submission_check(state_dir, target, retry=False, config=config)
    except (OSError, ValueError, KeyError) as exc:
        check = {"action": "skip", "status": "unreadable", "stage": target, "reason": str(exc), "progress": True}
    active = sorted((j for j in jobs if j["active"]), key=lambda j: (not j["running"], j["job_id"]))
    job = active[0] if active else None
    running_stage = next((s for s in reversed(stages) if status.get(s) == "running"), None)
    stage = running_stage or check.get("stage")
    record = {
        "state_id": sid,
        "target": target,
        "stages": status,
        "stage": stage,
        "job": job,
        "other_jobs": [j["job_id"] for j in active[1:]],
        "action": check.get("action"),
    }
    if check["action"] == "complete":
        category, detail = "finished", f"{target} complete"
    elif job is not None:
        category = "running" if job["running"] else "queued"
        detail = f"{_stage_text(stage, units)}, {_job_text(job)}"
    elif last and last.get("status") in UNKNOWN_SUBMISSION:
        category = "attention"
        detail = (
            f"the outcome of an earlier sbatch is unknown ({last['status']}); look for a job named mn_{sid} "
            "with squeue or sacct, then run `mechanet resolve-submission`"
        )
    elif check["action"] == "skip":
        category = "attention"
        detail = check.get("reason") or check.get("status")
        record["skip"] = check.get("status")
    elif not slurm_known and running_stage:
        category = "running"
        detail = f"{_stage_text(running_stage, units)} (from the run record; SLURM was not asked)"
    elif (
        not slurm_known
        and not attempts
        and last
        and last.get("status") == "submitted"
        and last.get("job_id")
        and not (last.get("job_finished_at_utc") or last.get("run_ended_at_utc"))
    ):
        category = "queued"
        detail = f"submitted as job {last['job_id']} (from the ledger; SLURM was not asked)"
    elif check.get("progress") or attempts:
        category = "continue"
        attempt = attempts.get(stage) or {}
        state = attempt.get("status")
        if state == "interrupted":
            reason = attempt.get("reason") or "stopped before the end"
            counts = _archive_counts(state_dir / stage)
            detail = f"{stage} interrupted ({reason}); the next job continues it"
            if counts.get("interrupted"):
                detail += f" (continuation {counts['interrupted'] + 1})"
        elif state == "running":
            detail = f"{stage}: the last run ended without recording its end; the next job checks its outputs"
        elif state == "launcher_error":
            detail = f"{stage}: VASP did not start (launcher error); fix the job environment, then submit again"
        elif stage == STAGE_FREQ and units:
            detail = f"{_stage_text(stage, units)} done; the next job continues"
        else:
            done = [s for s in stages if status.get(s) == "complete"]
            detail = (f"{done[-1]} complete; " if done else "") + f"next: {stage}"
    else:
        category = "not_started"
        detail = "prepared" if (state_dir / "plan.json").is_file() else ""
    record.update(category=category, detail=detail)
    if units:
        record["freq_units"] = {"complete": units[0], "total": units[1]}
    return record


def campaign_status(campaign, through=None, slurm=True, registry=None):
    """Status of every state, intermediate and cycle of ``campaign`` (see the module docstring)."""
    from .plugins import campaign_registry
    from .slurm import campaign_jobs

    campaign = Path(campaign)
    config = read_json(campaign / "config.json")
    network = read_json(campaign / "network.json")
    stages = protocol_stages(config)
    if registry is None:
        try:
            registry = campaign_registry(campaign, config=config)
        except Exception:  # noqa: BLE001 - names fall back to the family IDs
            registry = None
    ledger = _read(campaign / "submissions.json", {}) or {}
    latest = {}
    for entry in ledger.get("entries", []):
        latest[entry.get("state_id")] = entry
    state_dirs = sorted(p.parent for p in (campaign / "states").glob("*/state.json"))
    state_ids = [p.name for p in state_dirs]
    targets, target_text = _targets(campaign, state_ids, through, latest, config)
    jobs, problem = (None, "not asked (--no-slurm)") if not slurm else campaign_jobs(campaign, ledger.get("entries"))
    slurm_known = jobs is not None
    by_state = {}
    for job in jobs or []:
        by_state.setdefault(job["state_id"], []).append(job)
    meta_of = {}
    states = []
    for state_dir in state_dirs:
        sid = state_dir.name
        meta_of[sid] = _read(state_dir / "state.json", {}) or {}
        record = _classify(state_dir, targets[sid], config, stages, by_state.get(sid, []), latest.get(sid), slurm_known)
        record["node_id"] = meta_of[sid].get("node_id")
        states.append(record)
    by_id = {s["state_id"]: s for s in states}
    notation = Notation.from_campaign(campaign, registry=registry, network=network)
    nodes = {}
    for node in network.get("nodes", []):
        owner = node.get("computed_by") or node["id"]
        owner_node = notation.nodes.get(owner) or node
        sids = [s for s in owner_node.get("state_ids", []) if s in by_id]
        nodes[node["id"]] = {
            "label": notation.label(node["id"], _standalone_sites(notation, node["id"])),
            "state_ids": sids,
            "categories": [by_id[s]["category"] for s in sids],
            "computed_by": node.get("computed_by"),
        }
    for state in states:
        node = nodes.get(state.get("node_id") or "")
        state["label"] = node["label"] if node else None
    cycles = []
    for cycle in network.get("cycles", []):
        view = notation.cycle(cycle)
        progress = [nodes.get(n, {}).get("categories", []) for n in view["node_ids"]]
        ready = sum(1 for cats in progress if "finished" in cats)
        view.update(
            progress=progress,
            state_ids=[nodes.get(n, {}).get("state_ids", []) for n in view["node_ids"]],
            n_intermediates=len(progress),
            n_finished=ready,
            complete=bool(progress) and ready == len(progress),
            blocked=any(not cats for cats in progress),
        )
        cycles.append(view)
    counts = Counter(s["category"] for s in states)
    shown = [c for c in cycles if c["symmetry_representative"]]
    families = {}
    for cycle in shown:
        entry = families.setdefault(cycle["family"] or "other", Counter())
        entry["cycles"] += 1
        entry["complete" if cycle["complete"] else "partial" if cycle["n_finished"] else "none"] += 1
    descriptions = {}
    if registry is not None:
        descriptions = {name: fam.description for name, fam in getattr(registry, "families", {}).items()}
    job_rows = sorted(
        (j for j in jobs or [] if j["active"]), key=lambda j: (not j["running"], j["state_id"], j["job_id"])
    )
    for job in job_rows:
        job["label"] = (by_id.get(job["state_id"]) or {}).get("label")
    return {
        "campaign": str(campaign),
        "project": config.get("project") or campaign.name,
        "reaction": network.get("reaction", config.get("reaction")),
        "written_utc": utc_now(),
        "target": target_text,
        "slurm": {
            "asked": slurm,
            "available": slurm_known,
            "note": problem,
            "active_jobs": len(job_rows),
            "running_jobs": sum(1 for j in job_rows if j["running"]),
            "recently_ended": [j for j in jobs or [] if not j["active"]],
        },
        "counts": {c: counts.get(c, 0) for c in CATEGORIES},
        "n_states": len(states),
        "states": states,
        "attention": [s for s in states if s["category"] == "attention"],
        "jobs": job_rows,
        "nodes": nodes,
        "cycles": cycles,
        "families": {
            name: {"description": descriptions.get(name), **dict(entry)} for name, entry in sorted(families.items())
        },
        "summary": {
            "symmetry_unique_cycles": len(shown),
            "cycles": len(cycles),
            "complete": sum(1 for c in shown if c["complete"]),
            "partial": sum(1 for c in shown if c["n_finished"] and not c["complete"]),
            "none": sum(1 for c in shown if not c["n_finished"]),
            "computed_intermediates": sum(1 for n in nodes.values() if not n["computed_by"]),
            "intermediates_finished": sum(
                1 for n in nodes.values() if not n["computed_by"] and "finished" in n["categories"]
            ),
        },
    }


def _standalone_sites(notation, node_id):
    """The sites a node changes, terminal ones first, each kind by site number."""
    sites = notation.changed_sites(node_id)
    return sorted(
        sites, key=lambda s: (notation.symbol(s)[0] != "*", int(site_number(s)) if site_number(s).isdigit() else 0)
    )


def state_list(result, which):
    """State IDs of the --list choice ``which``, in order."""
    wanted = LISTS[which]
    return [s["state_id"] for s in result["states"] if s["category"] in wanted]


# ---------------------------------------------------------------------------
# Terminal text
# ---------------------------------------------------------------------------


def _marks(categories_per_node):
    """One mark per intermediate: the best status among its structures."""
    out = []
    for cats in categories_per_node:
        out.append(ICONS[min(cats, key=RANK.get)] if cats else ICONS["no_structure"])
    return "".join(out)


def status_text(result, show_cycles=False, html_path=None, ascii=False):
    """The terminal summary; ``ascii`` writes it without subscripts, Greek letters or symbols."""
    text = plain if ascii else str
    counts = result["counts"]
    lines = [
        f"{result['project']}: {result['n_states']} states, target {result['target']}",
    ]
    slurm = result["slurm"]
    if slurm["available"]:
        pending = slurm["active_jobs"] - slurm["running_jobs"]
        lines.append(
            f"SLURM: {slurm['active_jobs']} job(s) of this campaign ({slurm['running_jobs']} running, {pending} queued)"
        )
    else:
        lines.append(f"SLURM: {slurm['note']}; job states come from the run records only")
    lines.append("")
    cells = [text(f"{ICONS[c]} {LABELS[c]} {counts[c]}") for c in CATEGORIES]
    lines.append("  " + "   ".join(cells[:3]))
    lines.append("  " + "   ".join(cells[3:]))
    summary = result["summary"]
    lines += [
        "",
        f"Intermediates: {summary['intermediates_finished']} of {summary['computed_intermediates']} have a finished "
        "structure",
        f"Cycles ({summary['symmetry_unique_cycles']} symmetry-unique): {summary['complete']} with every "
        f"intermediate finished, {summary['partial']} partly, {summary['none']} not yet",
    ]
    for name, entry in result["families"].items():
        lines.append(
            f"  {name:<26} {entry.get('cycles', 0):>3} cycles: {entry.get('complete', 0)} complete, "
            f"{entry.get('partial', 0)} partly, {entry.get('none', 0)} not yet"
        )
    if show_cycles:
        lines += ["", "Cycles (one mark per intermediate, release step last):"]
        rows = [c for c in result["cycles"] if c["symmetry_representative"]]
        width = max([0] + [len(text(c["site_key"])) for c in rows])
        short = max([0] + [len(text(c["short"])) for c in rows])
        marks = max([0] + [len(c["progress"]) for c in rows])
        family = None
        for cycle in rows:
            if cycle["family"] != family:
                family = cycle["family"]
                lines.append(f"  {family}")
            lines.append(
                f"    {text(cycle['site_key']).ljust(width)}  {text(cycle['short']).ljust(short)}  "
                f"{text(_marks(cycle['progress'])).ljust(marks)}  {cycle['n_finished']}/{cycle['n_intermediates']}  "
                f"{cycle['cycle_id']}"
            )
    if result["attention"]:
        lines += ["", "Needs you:"]
        for state in result["attention"]:
            lines.append(f"  {state['state_id']}  {state['detail']}")
    if result["jobs"]:
        lines += ["", "Jobs:"]
        for job in result["jobs"][:40]:
            lines.append(
                f"  {job['job_id']:>10}  {job['slurm_state']:<10} {job.get('elapsed') or '':>11}  {job['state_id']}"
                + (f"  {text(job['label'])}" if job.get("label") else "")
            )
        if len(result["jobs"]) > 40:
            lines.append(f"  ... {len(result['jobs']) - 40} more (--json lists all)")
    lines.append("")
    if html_path:
        lines.append(f"Page with every cycle, its steps and its structures: {html_path}")
    lines.append(
        "State lists for --states @FILE: mechanet status --list "
        "{finished,running,queued,active,continue,attention,not-started,todo}"
    )
    if not show_cycles:
        lines.append("Every cycle with its progress: mechanet status --cycles; its steps: mechanet show-cycle ID")
    return text("\n".join(lines))


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

PAGE_CSS = """
.mn{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
  --hair:#e1e0d9;--ring:rgba(11,11,11,.10);--finished:#0ca30c;--running:#2a78d6;--continue:#fab219;
  --attention:#d03b3b;--idle:#c3c2b7;--on-fill:#ffffff;--on-warn:#0b0b0b;--code:#efeee9}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .mn{color-scheme:dark;--page:#0d0d0d;
  --surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;--hair:#2c2c2a;--ring:rgba(255,255,255,.10);
  --running:#3987e5;--idle:#5d5c57;--code:#262624}}
:root[data-theme="dark"] .mn{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;
  --hair:#2c2c2a;--ring:rgba(255,255,255,.10);--running:#3987e5;--idle:#5d5c57;--code:#262624}
html,body{margin:0;background:var(--page)}
.mn{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--page);
  max-width:1240px;margin:0 auto;padding:1.6rem 1.2rem 3rem}
.mn h1{font-size:1.45rem;margin:0 0 .2rem}.mn h2{font-size:1.1rem;margin:2rem 0 .6rem}
.mn h3{font-size:1rem;margin:1.4rem 0 .4rem}.mn .meta{color:var(--ink-2);margin:0}
.mn .desc{color:var(--ink-2);font-weight:400;font-size:.92rem}
.mn code{background:var(--code);padding:0 .3rem;border-radius:4px;font-size:.86em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(8.2rem,1fr));gap:.6rem;margin:1.2rem 0 .4rem}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:.55rem .9rem}
.tile .n{font-size:1.6rem;font-weight:600;line-height:1.2}.tile .l{color:var(--ink-2);font-size:.88rem}
.tile .l .sq{margin-right:.35rem}
.sq{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:3px;
  font-size:10px;line-height:1;font-weight:700;vertical-align:-2px;box-sizing:border-box;margin:1px}
.s-finished{background:var(--finished);color:var(--on-fill)}
.s-running{background:var(--running);color:var(--on-fill)}
.s-queued{background:transparent;color:var(--running);border:2px solid var(--running);font-size:9px}
.s-continue{background:var(--continue);color:var(--on-warn)}
.s-attention{background:var(--attention);color:var(--on-fill)}
.s-not_started{background:transparent;border:1.5px solid var(--idle);color:transparent}
.s-no_structure{background:transparent;border:1.5px dashed var(--muted);color:var(--muted);font-size:9px}
.inter{display:inline-grid;grid-auto-flow:column;grid-template-rows:repeat(var(--rows,1),auto);gap:0;
  margin-right:4px;vertical-align:top}
.grid .sq{width:13px;height:13px;font-size:8.5px;margin:.5px}
.grid .s-queued{font-size:7.5px}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;color:var(--ink-2);font-size:.88rem;margin:.6rem 0}
.legend .item{display:inline-flex;align-items:center;gap:.35rem}
.panel{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:.2rem 1rem .8rem;
  margin:.8rem 0;overflow-x:auto}
table{border-collapse:collapse;font-size:.9rem}
th,td{padding:.35rem .7rem;border-bottom:1px solid var(--hair);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--ink-2)}
.grid th.site{color:var(--ink);font-weight:600;white-space:nowrap}
.grid a.cyc{display:flex;align-items:flex-start;gap:.4rem;color:inherit;text-decoration:none;padding:2px 4px;
  border-radius:6px;white-space:nowrap}
.grid a.cyc .frac{align-self:center}
.grid a.cyc+a.cyc{margin-top:3px}
.grid a.cyc:hover,.grid a.cyc:focus-visible{background:var(--code);outline:none}
.frac{color:var(--ink-2);font-size:.82rem;font-variant-numeric:tabular-nums}
.done .frac{color:var(--ink);font-weight:600}
details.cycle{background:var(--surface);border:1px solid var(--ring);border-radius:10px;margin:.5rem 0;
  scroll-margin-top:1rem}
details.cycle>summary{cursor:pointer;padding:.55rem .9rem;list-style:none;display:flex;flex-wrap:wrap;
  align-items:center;gap:.5rem}
details.cycle>summary::-webkit-details-marker{display:none}
details.cycle>summary::before{content:"▸";color:var(--muted);width:.8rem}
details.cycle[open]>summary::before{content:"▾"}
details.cycle:target{border-color:var(--running)}
.cycle .body{padding:0 1rem .9rem 2.2rem}
.cycle .src,.cycle .sites,.cycle .sym{color:var(--ink-2);font-size:.88rem;margin:.3rem 0}
ol.steps{margin:.5rem 0;padding-left:1.4rem}ol.steps li{margin:.2rem 0}
.eq{font-size:1rem}.kind{color:var(--ink-2);font-size:.82rem;margin-left:.6rem;border:1px solid var(--hair);
  border-radius:999px;padding:0 .5rem}
.kind.chem{border-style:dashed}
.chip{display:inline-flex;align-items:center;gap:.3rem;margin:.1rem .5rem .1rem 0;font-size:.86rem}
.lbl{font-size:1rem;white-space:nowrap}
.filters{display:flex;flex-wrap:wrap;gap:.4rem;margin:.4rem 0}
.filters button{font:inherit;font-size:.86rem;background:var(--surface);color:var(--ink);border:1px solid var(--ring);
  border-radius:999px;padding:.15rem .7rem;cursor:pointer}
.filters button[aria-pressed="true"]{border-color:var(--ink);font-weight:600}
.note{color:var(--ink-2);font-size:.88rem}
"""

PAGE_JS = """
(function(){
  function openTarget(){var id=decodeURIComponent(location.hash.slice(1));if(!id)return;
    var el=document.getElementById(id);if(el&&el.tagName==="DETAILS"){el.open=true;}}
  window.addEventListener("hashchange",openTarget);openTarget();
  var buttons=document.querySelectorAll(".filters button");
  buttons.forEach(function(b){b.addEventListener("click",function(){
    var cat=b.getAttribute("data-cat");
    buttons.forEach(function(x){x.setAttribute("aria-pressed",x===b?"true":"false");});
    document.querySelectorAll("#states tbody tr").forEach(function(r){
      r.hidden=!(cat==="all"||r.getAttribute("data-cat")===cat);});});});
})();
"""


def _e(text):
    return html.escape(str(text if text is not None else ""), quote=True)


def _square(category, title=""):
    return (
        f'<span class="sq s-{category}" role="img" aria-label="{_e(LABELS[category])}" '
        f'title="{_e(title or LABELS[category])}">{ICONS[category] if category != "not_started" else ""}</span>'
    )


def _intermediate_squares(label, state_ids, categories, states):
    """One intermediate: a column of squares, one per structure (up to four per column)."""
    if not state_ids:
        return f'<span class="inter">{_square("no_structure", f"{label}: no starting structure was built")}</span>'
    parts = []
    for sid, category in zip(state_ids, categories):
        detail = (states.get(sid) or {}).get("detail") or ""
        parts.append(_square(category, f"{label} | {sid}: {LABELS[category]}" + (f" ({detail})" if detail else "")))
    rows = min(len(parts), 4)
    return f'<span class="inter" style="--rows:{rows}">{"".join(parts)}</span>'


def _cycle_squares(cycle, states):
    return "".join(
        _intermediate_squares(label, sids, cats, states)
        for label, sids, cats in zip(cycle["labels"], cycle["state_ids"], cycle["progress"])
    )


def _anchor(cycle_id):
    return "c-" + cycle_id


def _grid(result, states):
    """Per family: rows are the site combinations, columns the routes; each cell links to its cycles."""
    out = []
    shown = [c for c in result["cycles"] if c["symmetry_representative"]]
    families = []
    for cycle in shown:
        if cycle["family"] not in families:
            families.append(cycle["family"])
    for family in families:
        rows = [c for c in shown if c["family"] == family]
        routes, sites = [], []
        for cycle in rows:
            if cycle["short"] not in routes:
                routes.append(cycle["short"])
            if cycle["site_key"] not in sites:
                sites.append(cycle["site_key"])
        description = (result["families"].get(family) or {}).get("description") or ""
        out.append(f'<h3>{_e(family)} <span class="desc">{_e(description)}</span></h3>')
        out.append('<div class="panel"><table class="grid"><thead><tr><th>sites</th>')
        out += [f"<th>{_e(r)}</th>" for r in routes]
        out.append("</tr></thead><tbody>")
        for key in sites:
            legend = next((c["site_legend"] for c in rows if c["site_key"] == key), [])
            out.append(f'<tr><th class="site" title="{_e("; ".join(legend))}">{_e(key)}</th>')
            for name in routes:
                cell = [c for c in rows if c["site_key"] == key and c["short"] == name]
                links = []
                for cycle in cell:
                    tip = f"{cycle['title']}" + (f" ({cycle['variant']})" if cycle.get("variant") else "")
                    tip += f": {cycle['n_finished']} of {cycle['n_intermediates']} intermediates finished"
                    cls = "cyc done" if cycle["complete"] else "cyc"
                    links.append(
                        f'<a class="{cls}" href="#{_e(_anchor(cycle["cycle_id"]))}" title="{_e(tip)}">'
                        f"{_cycle_squares(cycle, states)}"
                        f'<span class="frac">{cycle["n_finished"]}/{cycle["n_intermediates"]}</span></a>'
                    )
                out.append(f"<td>{''.join(links)}</td>")
            out.append("</tr>")
        out.append("</tbody></table></div>")
    return "\n".join(out)


def _cycle_details(cycle, states):
    sid_rows = []
    for label, node_id, sids, cats in zip(cycle["labels"], cycle["node_ids"], cycle["state_ids"], cycle["progress"]):
        chips = (
            "".join(
                f'<span class="chip">{_square(cat)}<code>{_e(sid)}</code> '
                f"{_e(LABELS[cat])}{(': ' + _e(states[sid]['detail'])) if states.get(sid, {}).get('detail') else ''}"
                "</span><br>"
                for sid, cat in zip(sids, cats)
            )
            if sids
            else f'<span class="chip">{_square("no_structure")} no starting structure</span>'
        )
        sid_rows.append(
            f'<tr><td><span class="lbl">{_e(label)}</span><br><code>{_e(node_id)}</code></td><td>{chips}</td></tr>'
        )
    steps = "".join(
        f'<li><span class="eq">{_e(step["equation"])}</span>'
        f'<span class="kind{" chem" if step["kind"] == "chemical" else ""}">{_e(step["kind"])}</span></li>'
        for step in cycle["steps"]
    )
    title = cycle["title"] + (f" ({cycle['variant']})" if cycle.get("variant") else "")
    sym = ""
    if cycle.get("symmetry_equivalent_cycles"):
        sym = (
            '<p class="sym">Also stands for the symmetry-equivalent '
            + ", ".join(f"<code>{_e(c)}</code>" for c in cycle["symmetry_equivalent_cycles"])
            + ", computed through this cycle's structures.</p>"
        )
    return (
        f'<details class="cycle" id="{_e(_anchor(cycle["cycle_id"]))}"><summary>'
        f"<span>{_cycle_squares(cycle, states)}</span><b>{_e(title)}</b>"
        f'<span class="lbl">{_e(cycle["site_key"])}</span><code>{_e(cycle["cycle_id"])}</code>'
        f'<span class="frac">{cycle["n_finished"]}/{cycle["n_intermediates"]} intermediates finished</span>'
        f'</summary><div class="body">'
        f'<p class="src">{_e(cycle["sources"])}</p>'
        f'<p class="sites">Sites: {_e("; ".join(cycle["site_legend"]))}</p>'
        f'<ol class="steps">{steps}</ol>'
        f"<table><thead><tr><th>intermediate</th><th>structures</th></tr></thead><tbody>{''.join(sid_rows)}"
        f"</tbody></table>{sym}</div></details>"
    )


def status_page(result):
    """The status page as one HTML document (no external files, scripts or fonts)."""
    states = {s["state_id"]: s for s in result["states"]}
    counts = result["counts"]
    slurm = result["slurm"]
    if slurm["available"]:
        slurm_text = (
            f"SLURM: {slurm['active_jobs']} job(s) of this campaign, {slurm['running_jobs']} running, "
            f"{slurm['active_jobs'] - slurm['running_jobs']} queued"
        )
    else:
        slurm_text = f"SLURM: {slurm['note']}; job states come from the run records only"
    tiles = "".join(
        f'<div class="tile" title="{_e(EXPLAIN[c])}"><div class="n">{counts[c]}</div>'
        f'<div class="l">{_square(c)}{_e(LABELS[c])}</div></div>'
        for c in CATEGORIES
    )
    legend = "".join(f'<span class="item">{_square(c)} {_e(LABELS[c])}: {_e(EXPLAIN[c])}</span>' for c in CATEGORIES)
    legend += f'<span class="item">{_square("no_structure")} no structure: no starting structure was built</span>'
    summary = result["summary"]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(result['project'])}: campaign status</title>",
        f"<style>{PAGE_CSS}</style></head><body>",
        '<main class="mn">',
        f"<h1>{_e(result['project'])}: campaign status</h1>",
        f'<p class="meta">{_e(result["campaign"])}; written {_e(result["written_utc"][:16].replace("T", " "))} UTC; '
        f"{result['n_states']} states, target {_e(result['target'])}</p>",
        f'<p class="meta">{_e(slurm_text)}</p>',
        f'<div class="tiles">{tiles}</div>',
        f'<p class="note">{summary["intermediates_finished"]} of {summary["computed_intermediates"]} computed '
        f"intermediates have a finished structure. Of {summary['symmetry_unique_cycles']} symmetry-unique cycles, "
        f"{summary['complete']} have every intermediate finished, {summary['partial']} partly, "
        f"{summary['none']} not yet.</p>",
    ]
    if result["attention"]:
        rows = "".join(
            f"<tr><td><code>{_e(s['state_id'])}</code></td><td>{_e(s.get('label') or '')}</td>"
            f"<td>{_e(s['detail'])}</td></tr>"
            for s in result["attention"]
        )
        parts += [
            "<h2>Needs you</h2>",
            '<div class="panel"><table><thead><tr><th>state</th><th>intermediate</th><th>why</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>",
            '<p class="note">A failed attempt reruns with <code>mechanet submit --states ID --retry --execute</code> '
            "(same settings, once more); look at the stage's OUTCAR first.</p></div>",
        ]
    parts += [
        "<h2>Mechanisms</h2>",
        '<p class="note">One column per intermediate, in the order of the cycle (the step that releases the '
        "product last), with one square per structure. Select a cell for the cycle's steps and structures.</p>",
        f'<div class="legend">{legend}</div>',
        _grid(result, states),
        "<h2>Cycles</h2>",
    ]
    family = None
    for cycle in result["cycles"]:
        if not cycle["symmetry_representative"]:
            continue
        if cycle["family"] != family:
            family = cycle["family"]
            parts.append(f"<h3>{_e(family)}</h3>")
        parts.append(_cycle_details(cycle, states))
    if result["jobs"]:
        rows = "".join(
            f"<tr><td>{_e(j['job_id'])}</td><td>{_square('running' if j['running'] else 'queued')} "
            f"{_e(j['slurm_state'].lower())}</td><td>{_e(j.get('elapsed') or '')}</td>"
            f"<td>{_e(j.get('time_left') or '')}</td><td><code>{_e(j['state_id'])}</code></td>"
            f'<td class="lbl">{_e(j.get("label") or "")}</td></tr>'
            for j in result["jobs"]
        )
        parts += [
            "<h2>Jobs</h2>",
            '<div class="panel"><table><thead><tr><th>job</th><th>state</th><th>elapsed</th><th>left</th>'
            f"<th>campaign state</th><th>intermediate</th></tr></thead><tbody>{rows}</tbody></table></div>",
        ]
    buttons = '<button type="button" data-cat="all" aria-pressed="true">all</button>' + "".join(
        f'<button type="button" data-cat="{c}" aria-pressed="false">{_e(LABELS[c])} ({counts[c]})</button>'
        for c in CATEGORIES
        if counts[c]
    )
    rows = "".join(
        f'<tr data-cat="{s["category"]}"><td><code>{_e(s["state_id"])}</code></td>'
        f'<td class="lbl">{_e(s.get("label") or "")}</td><td>{_square(s["category"])} {_e(LABELS[s["category"]])}'
        f"</td><td>{_e(s['detail'])}</td></tr>"
        for s in result["states"]
    )
    parts += [
        "<h2>States</h2>",
        f'<div class="filters">{buttons}</div>',
        '<div class="panel"><table id="states"><thead><tr><th>state</th><th>intermediate</th><th>status</th>'
        f"<th>detail</th></tr></thead><tbody>{rows}</tbody></table></div>",
        '<p class="note">Written by <code>mechanet status</code> from the run records'
        + (" and squeue" if slurm["available"] else "")
        + "; run it again to refresh. Notation: * terminal site, μ bridging site, □ vacancy, subscript = site "
        "number (a0065 → 65).</p>",
        f"</main><script>{PAGE_JS}</script></body></html>",
    ]
    return "\n".join(parts) + "\n"


def write_status_page(result, path):
    path = Path(path)
    atomic_text(path, status_page(result))
    return path
