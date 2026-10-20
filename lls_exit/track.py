"""Exit-dump results -> the training run's own wandb run (summary keys + exits.pt artifact) and exit_summary.json."""
import json
import logging
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

ART_TYPE = "exits"
SPLITS = ("test", "val")


def art_name(run: str) -> str:
    return f"exits-{run}"


def exit_summary(ex: dict[str, Any]) -> dict[str, Any]:
    """Flat, filterable numbers from a dump: per-exit accuracy, MAC shares, coherence, oracle accuracy."""
    m, c, h = ex["meta"], ex["costs"], ex["heads"]
    full = c["full"]
    s: dict[str, Any] = {"exit/epochs": m["epochs_done"], "exit/n_heads": len(h["dim"]), "exit/names": m["names"],
                         "exit/heads_mac_frac": c["head"][:-1].sum().item() / full}
    for k, (b, d, co) in enumerate(zip(c["backbone"][:-1].tolist(), h["dim"], h["coherence"])):
        s |= {f"exit/{k}/mac_frac": b / full, f"exit/{k}/dim": d, f"exit/{k}/coherence": co}
    for sp in SPLITS:
        if sp not in ex["splits"]:
            continue
        r = ex["splits"][sp]
        y = r["y"]
        hit = r["scores"].argmax(2) == y[:, None]  # (N, K)
        cls = r["logits"].argmax(1) == y
        s[f"exit/{sp}_cls_acc"] = cls.float().mean().item()
        s[f"exit/{sp}_oracle_acc"] = (hit.any(1) | cls).float().mean().item()
        for k, a in enumerate(hit.float().mean(0).tolist()):
            s[f"exit/{k}/{sp}_acc"] = a
    log.debug("exit summary: %d keys, test cls %.4f", len(s), s.get("exit/test_cls_acc", float("nan")))
    return s


def save_local(rd: Path, s: dict[str, Any]) -> None:
    (rd / "exit_summary.json").write_text(json.dumps(s, indent=2))


def publish(rd: Path, ex_path: Path, s: dict[str, Any], project: str, entity: str | None) -> bool:
    """Reopen the training run by its stored id, add the summary and the exits.pt artifact. Never raises."""
    idf = rd / "wandb_id"
    if not idf.exists():
        log.warning("%s: no wandb_id (trained with --no-wandb?); results kept in local files only", rd.name)
        return False
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; results kept in local files only")
        return False
    try:
        run = wandb.init(project=project, entity=entity, id=idf.read_text().strip(), resume="must", dir=str(rd))
        run.summary.update(s)
        top = {k: v for k, v in s.items() if k.count("/") == 1}  # run-level keys only, not per-exit ones
        art = wandb.Artifact(art_name(rd.name), type=ART_TYPE, metadata={"run": rd.name, **top})
        art.add_file(str(ex_path))
        run.log_artifact(art)
        run.finish()
    except Exception as e:  # a wandb problem must not lose a finished dump
        log.warning("%s: wandb upload failed (%s); results kept in local files only", rd.name, e)
        return False
    log.info("%s: summary + artifact %s uploaded (%.1f MB)", rd.name, art_name(rd.name),
             ex_path.stat().st_size / 2**20)
    return True


def load_exits(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
