"""Prepare a deterministic blinded A/B listening set for expressive Pocket TTS."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


_RATING_COLUMNS = [
    "id",
    "preferred",
    "emotional_fit_a_1_5",
    "emotional_fit_b_1_5",
    "naturalness_a_1_5",
    "naturalness_b_1_5",
    "speaker_consistency_a_1_5",
    "speaker_consistency_b_1_5",
    "transition_smoothness_a_1_5",
    "transition_smoothness_b_1_5",
    "notes",
]


def _build_listening_html(trials: list[dict[str, object]]) -> str:
    """Build a dependency-free local blind-listening UI."""

    trials_json = json.dumps(trials, ensure_ascii=False).replace("</", "<\\/")
    columns_json = json.dumps(_RATING_COLUMNS)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expressive TTS Blind A/B</title>
<style>
body { font-family: system-ui,sans-serif; max-width:900px; margin:2rem auto; padding:0 1rem; }
article { border:1px solid #ccc; border-radius:12px; padding:1rem; margin:1rem 0; }
audio { width:100%; margin:.35rem 0 .8rem; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
label { display:block; margin:.45rem 0; }
select,textarea,button { font:inherit; }
textarea { width:100%; min-height:4rem; }
button { padding:.65rem 1rem; cursor:pointer; }
@media (max-width:700px) { .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<h1>Expressive TTS Blind A/B</h1>
<p>Rate A and B before opening <code>answer-key.json</code>.</p>
<div id="trials"></div>
<button id="download">Download ratings.csv</button>
<script>
const trials = __TRIALS__;
const columns = __COLUMNS__;
const fields = ["emotional_fit","naturalness","speaker_consistency","transition_smoothness"];
const root = document.getElementById("trials");
function makeSelect(id, values) {
  const s=document.createElement("select"); s.id=id;
  s.innerHTML="<option value=\"\">—</option>";
  for (const v of values) {
    const o=document.createElement("option");
    o.value=v;
    o.textContent=v;
    s.append(o);
  }
  return s;
}
for (const trial of trials) {
  const card=document.createElement("article");
  const h=document.createElement("h2"); h.textContent=trial.id; card.append(h);
  const p=document.createElement("p"); p.textContent=trial.text || ""; card.append(p);
  const grid=document.createElement("div"); grid.className="grid";
  for (const label of ["A","B"]) {
    const side=document.createElement("div");
    const strong=document.createElement("strong"); strong.textContent=label; side.append(strong);
    const audio=document.createElement("audio"); audio.controls=true; audio.preload="metadata";
    audio.src=label==="A" ? trial.audio_a : trial.audio_b; side.append(audio);
    for (const field of fields) {
      const lab=document.createElement("label"); lab.append(field.replaceAll("_"," ")+" ");
      lab.append(makeSelect(trial.id+"_"+field+"_"+label.toLowerCase(),[1,2,3,4,5]));
      side.append(lab);
    }
    grid.append(side);
  }
  card.append(grid);
  const pref=document.createElement("label"); pref.append("Preferred ");
  pref.append(makeSelect(trial.id+"_preferred",["A","B","Tie"])); card.append(pref);
  const notes=document.createElement("textarea");
  notes.id=trial.id+"_notes";
  notes.placeholder="Optional notes";
  card.append(notes); root.append(card);
}
function esc(value) { const t=String(value || ""); return "\"" + t.replaceAll("\"","\"\"") + "\""; }
document.getElementById("download").addEventListener("click", function() {
  const rows=[columns];
  for (const trial of trials) {
    rows.push([
      trial.id,
      document.getElementById(trial.id+"_preferred").value,
      document.getElementById(trial.id+"_emotional_fit_a").value,
      document.getElementById(trial.id+"_emotional_fit_b").value,
      document.getElementById(trial.id+"_naturalness_a").value,
      document.getElementById(trial.id+"_naturalness_b").value,
      document.getElementById(trial.id+"_speaker_consistency_a").value,
      document.getElementById(trial.id+"_speaker_consistency_b").value,
      document.getElementById(trial.id+"_transition_smoothness_a").value,
      document.getElementById(trial.id+"_transition_smoothness_b").value,
      document.getElementById(trial.id+"_notes").value
    ]);
  }
  const csv=rows.map(function(row){ return row.map(esc).join(","); }).join("\\r\\n")+"\\r\\n";
  const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
  const url=URL.createObjectURL(blob); const a=document.createElement("a");
  a.href=url; a.download="ratings.csv"; a.click(); URL.revokeObjectURL(url);
});
</script>
</body>
</html>
""".replace("__TRIALS__", trials_json).replace("__COLUMNS__", columns_json)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Randomize baseline-vs-expressive WAV order and create a human-rating worksheet."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory produced by scripts/generate_expressive_eval.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Blind set directory (default: <run_dir>/blind)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261006,
        help="Deterministic A/B randomization seed",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = report.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{report_path} contains no evaluation items")

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "blind"
    )
    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    trials: list[dict[str, object]] = []
    answer_key: list[dict[str, str]] = []

    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise ValueError("report items must be objects")
        item_id = str(raw_item["id"])
        baseline = run_dir / str(raw_item["baseline_wav"])
        expressive = run_dir / str(raw_item["expressive_wav"])
        if not baseline.is_file() or not expressive.is_file():
            raise FileNotFoundError(
                f"missing A/B audio for {item_id}: {baseline} / {expressive}"
            )

        expressive_is_a = bool(rng.getrandbits(1))
        sources = {
            "A": expressive if expressive_is_a else baseline,
            "B": baseline if expressive_is_a else expressive,
        }
        conditions = {
            "A": "expressive" if expressive_is_a else "baseline-neutral",
            "B": "baseline-neutral" if expressive_is_a else "expressive",
        }

        trial_audio: dict[str, str] = {}
        for label, source in sources.items():
            destination = audio_dir / f"{item_id}_{label}.wav"
            shutil.copy2(source, destination)
            trial_audio[label] = str(destination.relative_to(output_root))

        trials.append(
            {
                "id": item_id,
                "purpose": raw_item.get("purpose"),
                "text": raw_item.get("text"),
                "audio_a": trial_audio["A"],
                "audio_b": trial_audio["B"],
            }
        )
        answer_key.append(
            {
                "id": item_id,
                "A": conditions["A"],
                "B": conditions["B"],
            }
        )

    trials_path = output_root / "trials.json"
    trials_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": args.seed,
                "instructions": (
                    "Rate every trial before opening answer-key.json. "
                    "A and B are randomized per item."
                ),
                "trials": trials,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    answer_key_path = output_root / "answer-key.json"
    answer_key_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": args.seed,
                "answers": answer_key,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    ratings_path = output_root / "ratings.csv"
    with ratings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_RATING_COLUMNS)
        for trial in trials:
            writer.writerow([trial["id"], "", "", "", "", "", "", "", "", "", ""])

    listening_path = output_root / "index.html"
    listening_path.write_text(_build_listening_html(trials), encoding="utf-8")

    print(f"Blind trials: {trials_path}")
    print(f"Listening page: {listening_path}")
    print(f"Ratings sheet: {ratings_path}")
    print("Do not open answer-key.json until ratings are complete.")


if __name__ == "__main__":
    main()
