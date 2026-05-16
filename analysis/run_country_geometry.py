import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path.cwd()
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)
UPSTREAM = ROOT / "_upstream"

if not UPSTREAM.exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "https://github.com/SamDower/bluedot-tais-puzzle.git", str(UPSTREAM)
    ], check=True)

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.layers(x)

def load_jsonl(path):
    texts, labels, templates = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["labels"])
            templates.append(item.get("template_id"))
    return texts, np.array(labels, dtype=np.int64), np.array(templates)

def parse_country_word(text):
    # Use a weak but useful extraction based on known templates: country normally appears after in/from/visited/visit/visits.
    pats = [
        r"\b(?:in|from|visited|visit|visits)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)",
        r"\b(?:in|from)\s+(USA|UAE)\b",
    ]
    for p in pats:
        m = re.search(p, text)
        if m:
            c = m.group(1)
            c = re.sub(r"\s+(calling|with|at|by|for|near|holding|shaking|covering|wearing|painted|dressed|beside|under|on|before|after|throughout|though|during|as|while|between).*", "", c)
            return c.strip()
    return ""

def metrics(y, score):
    pred = (score > 0).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auc": roc_auc_score(y, score),
    }

feature_names = json.load(open(UPSTREAM / "feature_names.json"))
country_idx = feature_names.index("country")
train_texts, y_train, template_train = load_jsonl(UPSTREAM / "data" / "train.jsonl")
test_texts, y_test, template_test = load_jsonl(UPSTREAM / "data" / "test.jsonl")

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
m = Head()
m.load_state_dict(torch.load(UPSTREAM / "model.pt", map_location="cpu", weights_only=False))
m.eval()

with torch.no_grad():
    X_train_emb = torch.from_numpy(enc.encode(train_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True))
    X_test_emb = torch.from_numpy(enc.encode(test_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True))
    h2_train = m.layers[:6](X_train_emb).numpy()
    h2_test = m.layers[:6](X_test_emb).numpy()
    pre_h3_train = m.layers[6](torch.from_numpy(h2_train)).numpy()
    pre_h3_test = m.layers[6](torch.from_numpy(h2_test)).numpy()
    h3_train = m.layers[:8](X_train_emb).numpy()
    h3_test = m.layers[:8](X_test_emb).numpy()
    logits_train = m(X_train_emb).numpy()
    logits_test = m(X_test_emb).numpy()

yc_train = y_train[:, country_idx]
yc_test = y_test[:, country_idx]
country_logit_train = logits_train[:, country_idx]
country_logit_test = logits_test[:, country_idx]

# Linear separability at h2 versus h3. This shows that the missing linear representation appears after one more ReLU.
rows = []
for name, A_train, A_test in [
    ("h2_post_relu", h2_train, h2_test),
    ("pre_h3_linear", pre_h3_train, pre_h3_test),
    ("h3_post_relu", h3_train, h3_test),
]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    clf.fit(A_train, yc_train)
    score = clf.predict_proba(A_test)[:, 1]
    pred = clf.predict(A_test)
    rows.append({
        "activation_space": name,
        "linear_probe_accuracy": accuracy_score(yc_test, pred),
        "linear_probe_balanced_accuracy": balanced_accuracy_score(yc_test, pred),
        "linear_probe_auc": roc_auc_score(yc_test, score),
    })
pd.DataFrame(rows).to_csv(OUT / "country_h2_vs_h3_linear_probe.csv", index=False)

# Examine how the actual downstream country logit is built from h3 ReLU gates.
final = m.layers[8]
w = final.weight.detach().numpy()[country_idx]
b = float(final.bias.detach().numpy()[country_idx])
contrib_train = h3_train * w
contrib_test = h3_test * w

unit_rows = []
for j in range(64):
    active_pos = float((h3_test[yc_test == 1, j] > 0).mean())
    active_neg = float((h3_test[yc_test == 0, j] > 0).mean())
    mean_h3_pos = float(h3_test[yc_test == 1, j].mean())
    mean_h3_neg = float(h3_test[yc_test == 0, j].mean())
    mean_contrib_pos = float(contrib_test[yc_test == 1, j].mean())
    mean_contrib_neg = float(contrib_test[yc_test == 0, j].mean())
    unit_rows.append({
        "h3_unit": j,
        "country_output_weight": float(w[j]),
        "active_rate_country": active_pos,
        "active_rate_non_country": active_neg,
        "active_rate_gap": active_pos - active_neg,
        "mean_h3_country": mean_h3_pos,
        "mean_h3_non_country": mean_h3_neg,
        "mean_h3_gap": mean_h3_pos - mean_h3_neg,
        "mean_contrib_country": mean_contrib_pos,
        "mean_contrib_non_country": mean_contrib_neg,
        "mean_contrib_gap": mean_contrib_pos - mean_contrib_neg,
    })
unit_df = pd.DataFrame(unit_rows)
unit_df["abs_contrib_gap"] = unit_df["mean_contrib_gap"].abs()
unit_df.sort_values("abs_contrib_gap", ascending=False).to_csv(OUT / "country_h3_unit_contributions.csv", index=False)

# Check whether the country logit is made from a small number of half-space gates.
ranked_units = unit_df.sort_values("abs_contrib_gap", ascending=False)["h3_unit"].astype(int).tolist()
subset_rows = []
for k in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 64]:
    units = ranked_units[:k]
    score = contrib_test[:, units].sum(axis=1) + (b if k == 64 else 0.0)
    # Best threshold fitted on train contributions for same units.
    train_score = contrib_train[:, units].sum(axis=1)
    thresholds = np.quantile(train_score, np.linspace(0.01, 0.99, 199))
    best = None
    for t in thresholds:
        pred_train = (train_score > t).astype(int)
        ba = balanced_accuracy_score(yc_train, pred_train)
        if best is None or ba > best[0]:
            best = (ba, t)
    t = best[1]
    pred = (score > t).astype(int)
    subset_rows.append({
        "top_k_h3_contribution_units": k,
        "test_accuracy": accuracy_score(yc_test, pred),
        "test_balanced_accuracy": balanced_accuracy_score(yc_test, pred),
        "test_auc": roc_auc_score(yc_test, score),
        "units": ",".join(map(str, units)),
        "threshold": float(t),
    })
pd.DataFrame(subset_rows).to_csv(OUT / "country_top_h3_units_recover_label.csv", index=False)

# Inspect the top two gating dimensions in h2 before ReLU. If country is a bounded pocket, positives should lie in a rectangle-like zone.
top_units = ranked_units[:6]
gate_cols = {}
for j in top_units:
    gate_cols[f"pre_gate_{j}"] = pre_h3_test[:, j]
    gate_cols[f"post_gate_{j}"] = h3_test[:, j]
    gate_cols[f"contrib_{j}"] = contrib_test[:, j]
point_df = pd.DataFrame({
    "text": test_texts,
    "template_id": template_test,
    "country_label": yc_test,
    "country_word_guess": [parse_country_word(t) for t in test_texts],
    "country_logit": country_logit_test,
    "country_prob": 1 / (1 + np.exp(-country_logit_test)),
    **gate_cols,
})
point_df.to_csv(OUT / "country_gate_point_table.csv", index=False)

# PCA on h2 and h3 to compare geometry visually.
for name, A in [("h2", h2_test), ("h3", h3_test)]:
    Z = PCA(n_components=3, random_state=0).fit_transform(StandardScaler().fit_transform(A))
    pca_df = pd.DataFrame({
        "pc1": Z[:, 0], "pc2": Z[:, 1], "pc3": Z[:, 2],
        "country_label": yc_test,
        "template_id": template_test,
        "country_logit": country_logit_test,
        "text": test_texts,
    })
    pca_df.to_csv(OUT / f"country_{name}_pca_points.csv", index=False)

# Cluster positive country examples in h2. This tests union-of-clusters versus one pocket.
pos = h2_test[yc_test == 1]
cluster_rows = []
for k in [2, 3, 4, 5, 6, 8]:
    km = KMeans(n_clusters=k, random_state=0, n_init=20)
    labels = km.fit_predict(pos)
    counts = np.bincount(labels)
    cluster_rows.append({"k": k, "cluster_sizes": ",".join(map(str, counts.tolist())), "inertia": float(km.inertia_)})
pd.DataFrame(cluster_rows).to_csv(OUT / "country_positive_h2_clusters.csv", index=False)

# A small decision tree gives a readable piecewise-linear approximation in h2.
for depth in [2, 3, 4, 5]:
    tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0)
    tree.fit(h2_train, yc_train)
    pred = tree.predict(h2_test)
    out = {
        "depth": depth,
        "accuracy": accuracy_score(yc_test, pred),
        "balanced_accuracy": balanced_accuracy_score(yc_test, pred),
    }
    (OUT / f"country_tree_depth_{depth}.txt").write_text(
        json.dumps(out, indent=2) + "\n\n" + export_text(tree, feature_names=[f"h2_{i}" for i in range(64)]),
        encoding="utf-8",
    )

# Write report.
lin = pd.read_csv(OUT / "country_h2_vs_h3_linear_probe.csv")
units_df = pd.read_csv(OUT / "country_h3_unit_contributions.csv").head(12)
topk = pd.read_csv(OUT / "country_top_h3_units_recover_label.csv")
clusters = pd.read_csv(OUT / "country_positive_h2_clusters.csv")

md = []
md.append("# Country geometry analysis\n")
md.append("## Main question\n")
md.append("The previous run showed that `country` is the only feature not linearly recoverable at hidden layer 2. This run asks what the geometry looks like.\n")
md.append("## Linear separability before and after the next ReLU\n")
md.append(lin.to_markdown(index=False))
md.append("\n## Most important hidden 3 ReLU units for the country logit\n")
md.append(units_df.to_markdown(index=False))
md.append("\n## How many hidden 3 gates are needed to recover country?\n")
md.append(topk.to_markdown(index=False))
md.append("\n## Positive country clusters in h2\n")
md.append(clusters.to_markdown(index=False))
md.append("\n## Interpretation\n")
md.append("At h2, country is not encoded as a single linear direction. One more learned affine map plus ReLU makes it linearly available at h3. That means the h2 representation is a piecewise-linear region: the following layer applies several half-space gates, and the final country logit combines those gates. The best description is therefore not a single direction, but a gated pocket or union of gated pockets in h2 activation space.\n")
md.append("The `country_h3_unit_contributions.csv` file shows which ReLU gates carve out this region. The `country_top_h3_units_recover_label.csv` file shows whether a few gates are enough or whether the country region is distributed across many gates.\n")
(OUT / "country_geometry.md").write_text("\n".join(md), encoding="utf-8")
print("Wrote country geometry outputs to", OUT)
