"""
Autonomous SAE autoresearch runner.

Reads ideas from ideas.json, applies each one, trains, evaluates,
and records results. Runs for a specified number of hours.

Usage:
  python experiments/autoresearch/run_auto.py --agent-id 1 --hours 12
  python experiments/autoresearch/run_auto.py --agent-id 2 --hours 12 --start-from 10

Two agents can run concurrently — they read different ideas by offset.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_SCRIPT = os.path.join(REPO_DIR, "experiments/autoresearch/train.py")
IDEAS_FILE = os.path.join(REPO_DIR, "experiments/autoresearch/ideas.json")
DATA_DIR = os.path.join(REPO_DIR, "experiments/autoresearch/.data/train")


def read_original_train():
    with open(TRAIN_SCRIPT) as f:
        return f.read()


def apply_idea(original_code, idea):
    """Apply an idea's patches to train.py and return modified code."""
    code = original_code
    patch = idea.get("patch", {})

    # Patch custom_loss
    if "custom_loss" in patch:
        new_body = patch["custom_loss"]
        # Replace the body of custom_loss function
        code = re.sub(
            r'(def custom_loss\(out, x, sae, step, fire_ema=None\):\n    """[\s\S]*?"""\n)([\s\S]*?)(\n\ndef custom_init)',
            r'\1    ' + new_body.replace('\n', '\n    ') + r'\3',
            code
        )

    # Patch on_step_end
    if "on_step_end" in patch:
        new_body = patch["on_step_end"]
        code = re.sub(
            r'(def on_step_end\(sae, step, fire_counts, total_samples\):\n    """[\s\S]*?"""\n)    pass',
            r'\1    ' + new_body.replace('\n', '\n    '),
            code
        )

    # Patch train config values
    if "train_config" in patch:
        for key, value in patch["train_config"].items():
            if isinstance(value, float):
                # Replace existing key or add it
                if re.search(rf'{key}=[\d.e-]+', code):
                    code = re.sub(rf'{key}=[\d.e-]+', f'{key}={value}', code)
                else:
                    code = code.replace(
                        'dead_feature_threshold=50000,',
                        f'dead_feature_threshold=50000,\n    {key}={value},'
                    )
            elif isinstance(value, int):
                if re.search(rf'{key}=\d+', code):
                    code = re.sub(rf'{key}=\d+', f'{key}={value}', code)

    # Patch N_SAMPLES
    if "n_samples" in patch:
        code = re.sub(r'N_SAMPLES = \d+', f'N_SAMPLES = {patch["n_samples"]}', code)

    return code


def run_experiment(code, agent_id, idea_id):
    """Write modified train.py, run it, return results dict."""
    # Write modified code
    with open(TRAIN_SCRIPT, 'w') as f:
        f.write(code)

    env = os.environ.copy()
    env["HF_HOME"] = "/tmp/hf_cache"
    env["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/hf_cache/st"

    log_dir = os.path.join(REPO_DIR, "experiments/autoresearch/logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"agent{agent_id}_{idea_id}_{timestamp}.log")

    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT,
             "--data-dir", DATA_DIR,
             "--d-in", "384",
             "--device", "mps"],
            capture_output=True, text=True, timeout=600, env=env,
            cwd=REPO_DIR
        )

        # Save full output
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        if result.returncode != 0:
            return {"status": "crash", "error": result.stderr[-200:] if result.stderr else "unknown"}

        # Parse results
        metrics = {}
        for line in result.stdout.split('\n'):
            if ':' in line and not line.startswith('#'):
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val

        if "composite_score" not in metrics:
            return {"status": "no_results", "error": "composite_score not found"}

        metrics["status"] = "ok"
        metrics["log_file"] = log_file
        return metrics

    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", type=int, default=1)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N ideas")
    args = parser.parse_args()

    with open(IDEAS_FILE) as f:
        ideas = json.load(f)

    original_code = read_original_train()
    results_file = os.path.join(REPO_DIR, f"experiments/autoresearch/results_agent{args.agent_id}.tsv")

    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("timestamp\tidea_id\tcomposite\tclinc150\tscifact\tmmcs\tfvu\tactive\tstatus\tdescription\n")

    end_time = time.time() + args.hours * 3600
    exp_num = 0

    # Agent 1 runs odd-indexed ideas, Agent 2 runs even-indexed
    # This way they don't collide
    if args.agent_id == 1:
        my_ideas = ideas[0::2]  # 0, 2, 4, ...
    else:
        my_ideas = ideas[1::2]  # 1, 3, 5, ...

    my_ideas = my_ideas[args.start_from:]

    print(f"[Agent {args.agent_id}] Starting with {len(my_ideas)} ideas for {args.hours} hours")
    print(f"[Agent {args.agent_id}] End time: {time.strftime('%H:%M', time.localtime(end_time))}")

    best_composite = 0.0

    # Phase 1: Run all individual ideas
    for idea in my_ideas:
        if time.time() >= end_time:
            break

        exp_num += 1
        idea_id = idea["id"]
        desc = idea["description"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*60}")
        print(f"[Agent {args.agent_id}] Exp {exp_num}: {idea_id}")
        print(f"  {desc}")
        print(f"{'='*60}")

        modified_code = apply_idea(original_code, idea)
        metrics = run_experiment(modified_code, args.agent_id, idea_id)

        # Restore original
        with open(TRAIN_SCRIPT, 'w') as f:
            f.write(original_code)

        # Log results
        composite = metrics.get("composite_score", 0)
        clinc = metrics.get("clinc150_sparse", 0)
        scifact = metrics.get("scifact_sparse", 0)
        mmcs = metrics.get("mmcs", 0)
        fvu = metrics.get("fvu", 0)
        active = metrics.get("active_features", 0)
        status = metrics.get("status", "unknown")

        with open(results_file, 'a') as f:
            f.write(f"{timestamp}\t{idea_id}\t{composite}\t{clinc}\t{scifact}\t{mmcs}\t{fvu}\t{active}\t{status}\t{desc}\n")

        if status == "ok":
            marker = "★" if composite > best_composite else " "
            print(f"  {marker} composite={composite:.4f} clinc={clinc:.4f} scifact={scifact:.4f} mmcs={mmcs:.4f}")
            if composite > best_composite:
                best_composite = composite
                print(f"  ★ NEW BEST! (was {best_composite:.4f})")
        else:
            print(f"  FAILED: {status} — {metrics.get('error', '')[:100]}")

    # Phase 2: If time remains, combine the top ideas
    if time.time() < end_time:
        print(f"\n{'='*60}")
        print(f"[Agent {args.agent_id}] Phase 1 complete. Reading results for combinations...")
        print(f"{'='*60}")

        # Read results and find top ideas
        with open(results_file) as f:
            lines = f.readlines()[1:]  # skip header

        scored = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 8 and parts[7] == "ok":
                scored.append((float(parts[2]), parts[1]))  # (composite, idea_id)

        scored.sort(reverse=True)
        print(f"  Top ideas: {[(s, i) for s, i in scored[:5]]}")

        # Try combining top 2 ideas (if they have compatible patches)
        if len(scored) >= 2 and time.time() < end_time:
            top_ids = [s[1] for s in scored[:3]]
            idea_map = {i["id"]: i for i in ideas}

            for i, id1 in enumerate(top_ids):
                for id2 in top_ids[i+1:]:
                    if time.time() >= end_time:
                        break

                    idea1 = idea_map.get(id1, {})
                    idea2 = idea_map.get(id2, {})
                    combo_id = f"{id1}+{id2}"

                    print(f"\n[Agent {args.agent_id}] Combo: {combo_id}")
                    code = original_code
                    code = apply_idea(code, idea1)
                    code = apply_idea(code, idea2)

                    metrics = run_experiment(code, args.agent_id, combo_id)

                    with open(TRAIN_SCRIPT, 'w') as f:
                        f.write(original_code)

                    composite = metrics.get("composite_score", 0)
                    status = metrics.get("status", "unknown")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    with open(results_file, 'a') as f:
                        f.write(f"{timestamp}\t{combo_id}\t{composite}\t{metrics.get('clinc150_sparse',0)}\t{metrics.get('scifact_sparse',0)}\t{metrics.get('mmcs',0)}\t{metrics.get('fvu',0)}\t{metrics.get('active_features',0)}\t{status}\tcombination\n")

                    if status == "ok":
                        print(f"  composite={composite:.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"[Agent {args.agent_id}] DONE. {exp_num} experiments in {args.hours} hours.")
    print(f"  Best composite: {best_composite:.4f}")
    print(f"  Results: {results_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
