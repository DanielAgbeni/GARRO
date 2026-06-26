#!/usr/bin/env python3
"""
GARRO Checkpoint Inspector
==========================

An interactive terminal tool to scan the project for .pt checkpoint files,
inspect their internal structure, and export human-readable CSV and image
plots — all without requiring the full training stack to be running.

Usage
-----
    source garro_env/bin/activate
    python inspect_checkpoint.py

Outputs (written alongside the chosen .pt file, or in a user-selected dir)
-----------
    <name>_inspection.csv     — Layer-by-layer parameter table
    <name>_inspection.png     — Multi-panel visual summary

The plot header always shows the .pt filename being examined.
"""

from __future__ import annotations

import os
import re
import sys
import math
import textwrap
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# ── Matplotlib: non-interactive backend first ─────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
WHITE   = "\033[97m"

def clr(text, *codes):
    return "".join(codes) + str(text) + RESET

def banner(title, width=68):
    top    = "╔" + "═" * (width - 2) + "╗"
    mid    = "║" + title.center(width - 2) + "║"
    bottom = "╚" + "═" * (width - 2) + "╝"
    return clr(top + "\n" + mid + "\n" + bottom, BOLD, CYAN)

def section(title, width=68):
    line = "─" * width
    return clr(f"\n{line}\n  {title}\n{line}", BOLD, BLUE)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Scan project for .pt files
# ─────────────────────────────────────────────────────────────────────────────

def find_pt_files(root):
    """Recursively find all .pt files under root, sorted by modification time desc."""
    pts = sorted(root.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def print_file_table(pts, root):
    print(section("Found Checkpoint Files"))
    print()
    header = f"  {'#':>3}  {'Relative path':<55}  {'Size':>8}  {'Modified'}"
    print(clr(header, BOLD, WHITE))
    print(clr("  " + "─" * 82, DIM))
    for i, pt in enumerate(pts, 1):
        try:
            rel = str(pt.relative_to(root))
        except ValueError:
            rel = str(pt)
        size  = format_bytes(pt.stat().st_size)
        mtime = datetime.fromtimestamp(pt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        row   = f"  {i:>3}  {rel:<55}  {size:>8}  {mtime}"
        colour = CYAN if i % 2 == 0 else WHITE
        print(clr(row, colour))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load and introspect the checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(pt_path):
    """Load a .pt file safely, mapping all tensors to CPU."""
    print(clr(f"\n  Loading: {pt_path}", DIM))
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return ckpt


def _tensor_stats(t):
    f = t.float()
    return {
        "min":  float(f.min()),
        "max":  float(f.max()),
        "mean": float(f.mean()),
        "std":  float(f.std()) if t.numel() > 1 else 0.0,
        "norm": float(f.norm()),
    }


def introspect_checkpoint(ckpt, pt_path):
    """
    Walk the checkpoint dict and extract per-layer statistics.
    Returns a structured dict ready for CSV/plot.
    """
    info = {
        "file":       str(pt_path),
        "filename":   pt_path.name,
        "stem":       pt_path.stem,
        "file_size":  pt_path.stat().st_size,
        "top_keys":   list(ckpt.keys()),
        "components": OrderedDict(),
    }

    for comp_name, val in ckpt.items():
        if not isinstance(val, dict):
            continue

        layers = []
        for layer_name, tensor in val.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            s = _tensor_stats(tensor)
            layers.append({
                "component": comp_name,
                "layer":     layer_name,
                "shape":     str(list(tensor.shape)),
                "dtype":     str(tensor.dtype).replace("torch.", ""),
                "params":    tensor.numel(),
                "min":       round(s["min"],  6),
                "max":       round(s["max"],  6),
                "mean":      round(s["mean"], 6),
                "std":       round(s["std"],  6),
                "l2_norm":   round(s["norm"], 6),
            })

        if layers:
            info["components"][comp_name] = {
                "layers":       layers,
                "total_params": sum(l["params"] for l in layers),
            }

    # Optimizer group info
    opt_info = {}
    for k, v in ckpt.items():
        if isinstance(v, dict) and "param_groups" in v:
            groups = v["param_groups"]
            opt_info[k] = [
                {gk: gv for gk, gv in g.items() if gk != "params"}
                for g in groups
            ]
    info["optimizer_info"] = opt_info
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Print human-readable terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(info):
    print(section(f"Checkpoint: {info['filename']}"))
    print(f"  Path      : {clr(info['file'], DIM)}")
    print(f"  File size : {clr(format_bytes(info['file_size']), GREEN)}")
    print(f"  Top-level keys: {clr(', '.join(info['top_keys']), YELLOW)}")
    print()

    grand_total = 0
    for comp, cdata in info["components"].items():
        n = cdata["total_params"]
        grand_total += n
        print(clr(f"  ┌─ {comp}  ({n:,} params)", BOLD, MAGENTA))

        col_layer = min(max(len(l["layer"]) for l in cdata["layers"]), 52)
        header = (f"  │  {'Layer':<{col_layer}}  {'Shape':<22}  {'dtype':<10}"
                  f"  {'Params':>10}  {'Mean':>10}  {'Std':>10}  {'L2 Norm':>10}")
        print(clr(header, BOLD, WHITE))
        print(clr("  │  " + "─" * (col_layer + 80), DIM))

        for row in cdata["layers"]:
            lname = row["layer"][:col_layer]
            line  = (f"  │  {lname:<{col_layer}}  {row['shape']:<22}  {row['dtype']:<10}"
                     f"  {row['params']:>10,}  {row['mean']:>10.5f}  {row['std']:>10.5f}"
                     f"  {row['l2_norm']:>10.4f}")
            print(line)

        print(clr(f"  └─ Subtotal: {n:,} parameters", BOLD, GREEN))
        print()

    print(clr(f"  ★ Grand total parameters: {grand_total:,}", BOLD, GREEN))

    if info["optimizer_info"]:
        print(section("Optimizer Hyperparameters"))
        for opt_name, groups in info["optimizer_info"].items():
            print(clr(f"\n  {opt_name}:", BOLD, YELLOW))
            for i, g in enumerate(groups):
                print(f"    Group {i}: " + "  ".join(f"{k}={v}" for k, v in g.items()))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build the CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_csv(df, out_path):
    df.to_csv(str(out_path), index=False)
    print(clr(f"  ✔  CSV  → {out_path}", GREEN, BOLD))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build the multi-panel image
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#4C9BE8", "#E84C4C", "#4CE87A", "#E8C84C",
    "#C84CE8", "#4CE8E8", "#E8874C", "#8CE84C",
]

DARK_BG  = "#161B22"
AXIS_COL = "#8B949E"
TEXT_COL = "#E6EDF3"
ACCENT   = "#4C9BE8"
TITLE_FONT = {"fontfamily": "DejaVu Sans", "fontweight": "bold"}


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.tick_params(colors=AXIS_COL, labelsize=7.5)
    ax.xaxis.label.set_color(AXIS_COL)
    ax.yaxis.label.set_color(AXIS_COL)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=9, pad=6, **TITLE_FONT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color="#21262D", linewidth=0.5, alpha=0.8)


def build_image(info, df, out_path):
    fig = plt.figure(figsize=(22, 14), facecolor="#0F1117")
    fig.patch.set_facecolor("#0F1117")

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        hspace=0.42, top=0.91, bottom=0.05, left=0.04, right=0.97,
    )
    row0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.38)
    row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.38)

    ax_header = fig.add_subplot(row0[0])
    axes_r1   = [fig.add_subplot(row1[i]) for i in range(4)]
    axes_r2   = [fig.add_subplot(row2[i]) for i in range(4)]

    # ── Header card ───────────────────────────────────────────────────────
    ax_header.set_facecolor("#0D1117")
    ax_header.axis("off")
    rect = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=ACCENT,
        facecolor="#161B22",
        transform=ax_header.transAxes, zorder=0,
    )
    ax_header.add_patch(rect)

    grand_total = sum(c["total_params"] for c in info["components"].values())
    n_layers    = len(df)
    n_comps     = len(info["components"])
    dtype_str   = "  |  ".join(
        f"{v}× {k}" for k, v in df["dtype"].value_counts().items()
    ) if "dtype" in df.columns else "–"

    ax_header.text(
        0.5, 0.72,
        f"GARRO Checkpoint Inspector  ·  {info['filename']}",
        ha="center", va="center",
        fontsize=16, color=TEXT_COL, fontweight="bold",
        transform=ax_header.transAxes,
        path_effects=[pe.withStroke(linewidth=3, foreground="#0D1117")],
    )
    meta_parts = [
        f"File: {info['file']}",
        f"Size: {format_bytes(info['file_size'])}",
        f"Components: {n_comps}",
        f"Total Layers: {n_layers}",
        f"Total Parameters: {grand_total:,}",
        f"DTypes: {dtype_str}",
    ]
    ax_header.text(
        0.5, 0.28, "   ·   ".join(meta_parts),
        ha="center", va="center",
        fontsize=8.5, color=AXIS_COL,
        transform=ax_header.transAxes,
    )

    comp_names  = list(info["components"].keys())
    comp_params = [info["components"][c]["total_params"] for c in comp_names]
    comp_color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(comp_names)}

    # ── Panel 1: Param count per component ───────────────────────────────
    ax = axes_r1[0]
    style_ax(ax, "Parameters per Component", "Component", "Count")
    short_names = [c.replace("_", "\n") for c in comp_names]
    colours = [comp_color_map[c] for c in comp_names]
    bars = ax.bar(short_names, comp_params, color=colours,
                  edgecolor="#21262D", linewidth=0.8, alpha=0.9)
    for bar, val in zip(bars, comp_params):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:,}", ha="center", va="bottom", fontsize=7, color=TEXT_COL,
        )
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(v):,}" for v in yticks], fontsize=7)

    # ── Panel 2: Top-K layers horizontal bar ─────────────────────────────
    ax = axes_r1[1]
    TOP_K = 15
    style_ax(ax, f"Top-{TOP_K} Layers by Parameter Count", "Parameters", "")
    top_df = df.nlargest(TOP_K, "params")[["layer", "params", "component"]].copy()
    top_df["label"] = top_df["layer"].apply(lambda s: ".".join(s.split(".")[-2:]))
    bar_colours = [comp_color_map.get(r["component"], ACCENT) for _, r in top_df.iterrows()]
    y_pos = list(range(len(top_df)))
    ax.barh(y_pos, top_df["params"].values, color=bar_colours,
            edgecolor="#21262D", linewidth=0.5, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["label"].values, fontsize=6.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(v):,}" for v in xticks], fontsize=7)

    # ── Panel 3: L2 norm per component scatter ────────────────────────────
    ax = axes_r1[2]
    style_ax(ax, "L2 Norm Distribution per Component", "L2 Norm", "")
    for i, (comp, cdata) in enumerate(info["components"].items()):
        norms = [l["l2_norm"] for l in cdata["layers"]]
        if not norms:
            continue
        col = _PALETTE[i % len(_PALETTE)]
        ax.scatter(norms, [comp] * len(norms), alpha=0.7, s=30, color=col, edgecolors="none")
        ax.scatter([np.mean(norms)], [comp], s=80, color=col,
                   marker="D", zorder=5, edgecolors="white", linewidths=0.5)
    ax.tick_params(axis="y", labelsize=7)

    # ── Panel 4: Mean vs Std scatter ──────────────────────────────────────
    ax = axes_r1[3]
    style_ax(ax, "Weight Mean vs Std (all layers)", "Mean", "Std Dev")
    for i, (comp, cdata) in enumerate(info["components"].items()):
        means = [l["mean"] for l in cdata["layers"]]
        stds  = [l["std"]  for l in cdata["layers"]]
        ax.scatter(means, stds, alpha=0.75, s=20, color=_PALETTE[i % len(_PALETTE)],
                   label=comp, edgecolors="none")
    ax.axhline(0, color="#30363D", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#30363D", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=6.5, facecolor=DARK_BG, labelcolor=TEXT_COL,
              edgecolor="#30363D", loc="best")

    # ── Panel 5: Cumulative parameter profile ─────────────────────────────
    ax = axes_r2[0]
    style_ax(ax, "Cumulative Parameter Profile", "Layer Index", "Cumulative Params")
    cumsum = np.cumsum(df["params"].values)
    ax.fill_between(range(len(cumsum)), cumsum, alpha=0.25, color=ACCENT)
    ax.plot(cumsum, color=ACCENT, linewidth=1.5)
    idx = 0
    for i, (comp, cdata) in enumerate(info["components"].items()):
        end = idx + len(cdata["layers"])
        ax.axvline(end - 1, color=_PALETTE[i % len(_PALETTE)],
                   linewidth=0.8, linestyle=":")
        mid = (idx + end - 1) // 2
        if mid < len(cumsum):
            ax.text(mid, cumsum[mid] * 0.5, comp.split("_")[0],
                    ha="center", fontsize=6, color=_PALETTE[i % len(_PALETTE)],
                    rotation=90, va="bottom")
        idx = end
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    yticks2 = ax.get_yticks()
    ax.set_yticks(yticks2)
    ax.set_yticklabels([f"{int(v):,}" for v in yticks2], fontsize=7)

    # ── Panel 6: DType pie ────────────────────────────────────────────────
    ax = axes_r2[1]
    ax.set_facecolor(DARK_BG)
    ax.axis("off")
    ax.set_title("Parameter DType Composition", color=TEXT_COL,
                 fontsize=9, pad=6, **TITLE_FONT)
    if "dtype" in df.columns:
        dtype_counts = df.groupby("dtype")["params"].sum()
        ax.pie(
            dtype_counts.values,
            labels=dtype_counts.index,
            autopct="%1.1f%%",
            colors=[_PALETTE[i] for i in range(len(dtype_counts))],
            wedgeprops={"edgecolor": "#0F1117", "linewidth": 1.5},
            textprops={"color": TEXT_COL, "fontsize": 8},
            pctdistance=0.75,
        )

    # ── Panel 7: L2 norm line chart ───────────────────────────────────────
    ax = axes_r2[2]
    style_ax(ax, "L2 Norm per Layer (all layers)", "Layer Index", "L2 Norm")
    start = 0
    for i, (comp, cdata) in enumerate(info["components"].items()):
        norms = [l["l2_norm"] for l in cdata["layers"]]
        xs = list(range(start, start + len(norms)))
        col = _PALETTE[i % len(_PALETTE)]
        ax.plot(xs, norms, color=col, linewidth=1.2, label=comp)
        ax.fill_between(xs, norms, alpha=0.1, color=col)
        start += len(norms)
    ax.legend(fontsize=6.5, facecolor=DARK_BG, labelcolor=TEXT_COL,
              edgecolor="#30363D", loc="best")

    # ── Panel 8: Summary stats table ──────────────────────────────────────
    ax = axes_r2[3]
    ax.set_facecolor(DARK_BG)
    ax.axis("off")
    ax.set_title("Component Summary Statistics", color=TEXT_COL,
                 fontsize=9, pad=6, **TITLE_FONT)
    col_labels = ["Component", "Layers", "Params", "Avg L2", "Max L2"]
    rows_data = []
    for comp, cdata in info["components"].items():
        norms = [l["l2_norm"] for l in cdata["layers"]]
        rows_data.append([
            comp[:18],
            str(len(cdata["layers"])),
            f"{cdata['total_params']:,}",
            f"{np.mean(norms):.4f}" if norms else "–",
            f"{np.max(norms):.4f}"  if norms else "–",
        ])
    if rows_data:
        tbl = ax.table(
            cellText=rows_data, colLabels=col_labels,
            cellLoc="center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#21262D" if r == 0 else DARK_BG)
            cell.set_edgecolor("#30363D")
            cell.set_text_props(
                color=ACCENT if r == 0 else TEXT_COL,
                fontweight="bold" if r == 0 else "normal",
            )

    # ── Super title ───────────────────────────────────────────────────────
    fig.suptitle(
        f"GARRO Checkpoint Inspection  ·  {info['filename']}",
        fontsize=15, fontweight="bold", color=TEXT_COL, y=0.96,
    )

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(clr(f"  ✔  Plot → {out_path}", GREEN, BOLD))


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────────────────────

def pick_pt_file(pts, root):
    while True:
        raw = input(clr("\n  Enter file number (or 'q' to quit): ", BOLD, YELLOW)).strip()
        if raw.lower() in ("q", "quit", "exit"):
            print(clr("\n  Goodbye! 👋\n", CYAN))
            sys.exit(0)
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(pts):
                chosen = pts[idx]
                print(clr(f"\n  Selected: {chosen}", GREEN))
                return chosen
            else:
                print(clr(f"  ✖  Please enter a number between 1 and {len(pts)}.", RED))
        except ValueError:
            print(clr("  ✖  Invalid input — enter a number or 'q'.", RED))


def pick_output_dir(pt_path):
    default = pt_path.parent
    raw = input(
        clr(f"\n  Output directory [{default}] (press Enter for default): ", BOLD, YELLOW)
    ).strip()
    if not raw:
        return default
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def confirm(prompt):
    ans = input(clr(f"\n  {prompt} [Y/n]: ", BOLD, YELLOW)).strip().lower()
    return ans in ("", "y", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    root       = script_dir

    print()
    print(banner("  GARRO — Checkpoint Inspector  v1.0  "))
    print()
    print(clr(f"  Scanning for .pt files under: {root}", DIM))
    print()

    pts = find_pt_files(root)
    if not pts:
        print(clr("  ✖  No .pt files found in the project!", RED))
        sys.exit(1)

    print_file_table(pts, root)

    while True:
        chosen_pt  = pick_pt_file(pts, root)
        output_dir = pick_output_dir(chosen_pt)
        stem       = chosen_pt.stem

        csv_path = output_dir / f"{stem}_inspection.csv"
        img_path = output_dir / f"{stem}_inspection.png"

        try:
            ckpt = load_checkpoint(chosen_pt)
        except Exception as exc:
            print(clr(f"\n  ✖  Failed to load checkpoint: {exc}", RED))
            if not confirm("Try a different file?"):
                sys.exit(1)
            continue

        info = introspect_checkpoint(ckpt, chosen_pt)
        print_summary(info)

        all_rows = []
        for cdata in info["components"].values():
            all_rows.extend(cdata["layers"])
        df = pd.DataFrame(all_rows)

        if df.empty:
            print(clr("  ⚠  No tensor layers found in this checkpoint.", YELLOW))
        else:
            print(section("Exporting CSV"))
            build_csv(df, csv_path)

            print(section("Generating Plot"))
            try:
                build_image(info, df, img_path)
            except Exception as exc:
                print(clr(f"  ⚠  Plot generation failed: {exc}", YELLOW))
                import traceback; traceback.print_exc()

        if not confirm("Inspect another checkpoint?"):
            break

    print(clr("\n  All done! ✓\n", BOLD, GREEN))


if __name__ == "__main__":
    main()
