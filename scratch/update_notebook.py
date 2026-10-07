import json
import os

notebooks = [
    r"c:\Users\Daniel\Documents\SchProject\GARRO\GARRO_Kaggle_Dual_T4_Training.ipynb",
    r"c:\Users\Daniel\Documents\SchProject\GARRO\train.ipynb"
]

new_markdown = [
    "## 8. Multi-Checkpoint Benchmark Evaluation & Automatic Model Selection\n",
    "Benchmarks ALL saved curriculum checkpoints in `/kaggle/working/checkpoints/` against OSPF, strict equal-cost ECMP, and dynamic Hedera to automatically detect and rank the 🏆 optimal model (since the final epoch might not always be the best policy due to late-stage exploration)."
]

new_code = [
    "import os\n",
    "\n",
    "ckpt_dir = \"/kaggle/working/checkpoints\" if os.path.exists(\"/kaggle/working/checkpoints\") else \"checkpoints\"\n",
    "\n",
    "if os.path.exists(ckpt_dir) and any(f.endswith('.pt') for f in os.listdir(ckpt_dir)):\n",
    "    print(f\"[Eval] Benchmarking ALL checkpoints found in '{ckpt_dir}'...\")\n",
    "    print(\"[Eval] Evaluates curriculum progression, compares against baselines, and crowns the 🏆 BEST model!\")\n",
    "    !python evaluate.py \\\n",
    "        --topology fat_tree \\\n",
    "        --checkpoint {ckpt_dir} \\\n",
    "        --episodes 500 \\\n",
    "        --compare-all\n",
    "else:\n",
    "    print(f\"No .pt checkpoints found in '{ckpt_dir}'. Complete training step first.\")"
]

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        continue
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if "Comparative Benchmark" in source or "Multi-Checkpoint" in source:
                cell["source"] = new_markdown
        elif cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "final_model =" in source or "--checkpoint {ckpt_dir}" in source or "garro_fat_tree_final.pt" in source:
                cell["source"] = new_code

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Updated: {nb_path}")
