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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC
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

def metrics_from_scores(y, pred, score):
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred),
    }
    try:
        out["auc"] = roc_auc_score(y, score)
    except Exception:
        out["auc"] = np.nan
    return out

feature_names = json.load(open(UPSTREAM / "feature_names.json"))
train_texts, y_train, template_train = load_jsonl(UPSTREAM / "data" / "train.jsonl")
test_texts, y_test, template_test = load_jsonl(UPSTREAM / "data" / "test.jsonl")

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
m = Head()
m.load_state_dict(torch.load(UPSTREAM / "model.pt", map_location="cpu", weights_only=False))
m.eval()

with torch.no_grad():
    X_train_emb = torch.from_numpy(enc.encode(train_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True))
    X_test_emb = torch.from_numpy(enc.encode(test_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True))
    logits_train = m(X_train_emb)
    logits_test = m(X_test_emb)
    probs_train = torch.sigmoid(logits_train).numpy()
    probs_test = torch.sigmoid(logits_test).numpy()
    pred_train = (probs_train > 0.5).astype(int)
    pred_test = (probs_test > 0.5).astype(int)
    X_train = m.layers[:6](X_train_emb).numpy()
    X_test = m.layers[:6](X_test_emb).numpy()

model_rows = []
for i, name in enumerate(feature_names):
    row = {"feature": name}
    row.update(metrics_from_scores(y_test[:, i], pred_test[:, i], probs_test[:, i]))
    model_rows.append(row)
pd.DataFrame(model_rows).to_csv(OUT / "model_final_accuracy.csv", index=False)

linear_rows = []
linear_models = {
    "logreg_l2": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
    "linear_svc": lambda: make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=20000, dual="auto")),
    "ridge": lambda: make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced")),
}
for model_name, maker in linear_models.items():
    for i, name in enumerate(feature_names):
        clf = maker()
        clf.fit(X_train, y_train[:, i])
        pred = clf.predict(X_test)
        if hasattr(clf[-1], "predict_proba"):
            score = clf.predict_proba(X_test)[:, 1]
        else:
            score = clf.decision_function(X_test)
        row = {"probe": model_name, "feature": name}
        row.update(metrics_from_scores(y_test[:, i], pred, score))
        linear_rows.append(row)
linear_df = pd.DataFrame(linear_rows)
linear_df.to_csv(OUT / "linear_probes.csv", index=False)

# A small set of nonlinear checks. The RBF SVM is trained only once per feature and is manageable at this dataset size.
nonlinear_rows = []
nonlinear_models = {
    "mlp_32": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0, early_stopping=True)),
    "tree_depth3": lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0),
    "forest_200": lambda: RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", random_state=0, n_jobs=-1),
    "poly2_logreg": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
    "rbf_svm": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=False, gamma="scale")),
}
for model_name, maker in nonlinear_models.items():
    for i, name in enumerate(feature_names):
        clf = maker()
        clf.fit(X_train, y_train[:, i])
        pred = clf.predict(X_test)
        if hasattr(clf[-1] if hasattr(clf, "__getitem__") else clf, "predict_proba"):
            score = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf[-1] if hasattr(clf, "__getitem__") else clf, "decision_function"):
            score = clf.decision_function(X_test)
        else:
            score = pred
        row = {"probe": model_name, "feature": name}
        row.update(metrics_from_scores(y_test[:, i], pred, score))
        nonlinear_rows.append(row)
nonlinear_df = pd.DataFrame(nonlinear_rows)
nonlinear_df.to_csv(OUT / "nonlinear_probes.csv", index=False)

# Single-neuron evidence and top neurons.
neuron_rows = []
for i, name in enumerate(feature_names):
    y = y_train[:, i]
    for j in range(X_train.shape[1]):
        x = X_train[:, j]
        if np.std(x) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        neuron_rows.append({"feature": name, "neuron": j, "corr": corr, "abs_corr": abs(corr)})
neuron_df = pd.DataFrame(neuron_rows)
neuron_df.to_csv(OUT / "neuron_label_correlations.csv", index=False)

# Choose candidate based on lowest mean linear AUC, then inspect it with a shallow tree and PCA.
mean_linear = linear_df.groupby("feature").agg(mean_linear_acc=("accuracy", "mean"), mean_linear_auc=("auc", "mean")).reset_index()
mean_nonlinear = nonlinear_df.groupby("feature").agg(best_nonlinear_acc=("accuracy", "max"), best_nonlinear_auc=("auc", "max")).reset_index()
summary = mean_linear.merge(mean_nonlinear, on="feature")
summary["acc_gap"] = summary["best_nonlinear_acc"] - summary["mean_linear_acc"]
summary["auc_gap"] = summary["best_nonlinear_auc"] - summary["mean_linear_auc"]
summary = summary.sort_values(["mean_linear_auc", "mean_linear_acc"])
summary.to_csv(OUT / "feature_summary.csv", index=False)
candidate = summary.iloc[0]["feature"]
cand_idx = feature_names.index(candidate)

# Train a depth-3 tree for the candidate to get a human-readable nonlinear rule.
tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
tree.fit(X_train, y_train[:, cand_idx])
tree_text = export_text(tree, feature_names=[f"h2_{i}" for i in range(64)])

# PCA coordinates for candidate, plus optional number subtype labels.
pca = PCA(n_components=2, random_state=0)
Z_test = pca.fit_transform(StandardScaler().fit_transform(X_test))

def number_subtype(text):
    words = set(re.findall(r"[A-Za-z]+|\d+", text.lower()))
    digit = bool(re.search(r"\b\d+\b", text))
    num_words = {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "hundred"
    }
    if digit:
        return "digit"
    if words & num_words:
        return "written_number"
    return "other_or_none"

pca_df = pd.DataFrame({
    "text": test_texts,
    "template_id": template_test,
    "pc1": Z_test[:, 0],
    "pc2": Z_test[:, 1],
    "candidate_label": y_test[:, cand_idx],
    "candidate_final_prob": probs_test[:, cand_idx],
    "number_subtype": [number_subtype(t) for t in test_texts],
})
pca_df.to_csv(OUT / "candidate_pca_points.csv", index=False)

# Template-wise performance for the candidate. This helps detect whether the weird feature is tied to sentence templates.
template_rows = []
for tid in sorted(set(template_test.tolist())):
    mask = template_test == tid
    if mask.sum() == 0:
        continue
    # Use the best linear probe, logistic regression, for template breakdown.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    clf.fit(X_train, y_train[:, cand_idx])
    pred = clf.predict(X_test[mask])
    score = clf.predict_proba(X_test[mask])[:, 1]
    row = {"template_id": int(tid), "n": int(mask.sum())}
    row.update(metrics_from_scores(y_test[mask, cand_idx], pred, score))
    template_rows.append(row)
pd.DataFrame(template_rows).to_csv(OUT / "candidate_template_breakdown.csv", index=False)

# Markdown report.
final_acc = pd.DataFrame(model_rows).sort_values("feature")
lin_pivot = linear_df.pivot(index="feature", columns="probe", values="accuracy").reset_index()
auc_pivot = linear_df.pivot(index="feature", columns="probe", values="auc").reset_index()
nonlin_pivot = nonlinear_df.pivot(index="feature", columns="probe", values="accuracy").reset_index()

md = []
md.append("# Puzzle run results\n")
md.append("## Loaded data\n")
md.append(f"Train examples: {len(train_texts)}\n")
md.append(f"Test examples: {len(test_texts)}\n")
md.append(f"Layer 2 activation shape: train {X_train.shape}, test {X_test.shape}\n")
md.append("\n## Final model held-out accuracy\n")
md.append(final_acc.to_markdown(index=False))
md.append("\n\n## Linear probe accuracies\n")
md.append(lin_pivot.to_markdown(index=False))
md.append("\n\n## Linear probe AUCs\n")
md.append(auc_pivot.to_markdown(index=False))
md.append("\n\n## Nonlinear probe accuracies\n")
md.append(nonlin_pivot.to_markdown(index=False))
md.append("\n\n## Feature summary\n")
md.append(summary.to_markdown(index=False))
md.append(f"\n\n## Candidate feature\n\nThe automatic candidate is `{candidate}`.\n")
md.append("\n## Depth 3 decision tree for candidate\n")
md.append("```\n" + tree_text + "\n```\n")
md.append("\n## Top correlated hidden layer 2 neurons\n")
for name in feature_names:
    top = neuron_df[neuron_df.feature == name].sort_values("abs_corr", ascending=False).head(8)
    md.append(f"\n### {name}\n")
    md.append(top[["neuron", "corr", "abs_corr"]].to_markdown(index=False))

(OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
print("Candidate:", candidate)
print(summary.to_markdown(index=False))
print("Wrote outputs to", OUT)
