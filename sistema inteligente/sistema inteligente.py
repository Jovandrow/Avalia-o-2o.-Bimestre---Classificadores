import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, confusion_matrix, ConfusionMatrixDisplay)


df = pd.read_csv("default_of_credit_card_clients.csv", sep=";")
df.rename(columns={"default payment next month": "DEFAULT"}, inplace=True)
df.drop(columns=["ID"], inplace=True)

# Codifica variáveis categóricas
from sklearn.preprocessing import LabelEncoder
for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    df[col] = df[col].astype(str).str.strip()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df.dropna(inplace=True)

X = df.drop(columns=["DEFAULT"])
y = df["DEFAULT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


candidates = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=150, max_depth=10,
                                       random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                            max_depth=5, random_state=42))
    ]),
}

# (ROC-AUC via CV) 

print("=" * 55)
print("  BANCO — SISTEMA DE RISCO DE INADIMPLÊNCIA")
print("=" * 55)
print("\n[1] Avaliação por validação cruzada (ROC-AUC, 5-fold):\n")

cv_scores = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring="roc_auc", n_jobs=-1)
    cv_scores[name] = scores
    print(f"  {name:<25}  AUC = {scores.mean():.4f}  (±{scores.std():.4f})")

best_name = max(cv_scores, key=lambda k: cv_scores[k].mean())
best_pipe  = candidates[best_name]
print(f"\n  ✔ Melhor modelo: {best_name}")


best_pipe.fit(X_train, y_train)

#AVALIAÇÃO NO CONJUNTO DE TESTE

y_pred      = best_pipe.predict(X_test)
y_proba     = best_pipe.predict_proba(X_test)[:, 1]
auc_test    = roc_auc_score(y_test, y_proba)

print(f"\n[2] Resultado no conjunto de teste:\n")
print(f"  ROC-AUC  : {auc_test:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Adimplente','Inadimplente'])}")

#MÓDULO DE INFERÊNCIA 

def avaliar_cliente(dados: dict) -> dict:
    """
    Recebe um dicionário com as features do cliente e retorna:
      - risco      : 'ALTO' | 'MÉDIO' | 'BAIXO'
      - score      : probabilidade de inadimplência (0–1)
      - distribuicao: dict com P(adimplente) e P(inadimplente)
    """
    entrada = pd.DataFrame([dados])[X.columns]
    score   = best_pipe.predict_proba(entrada)[0, 1]

    if score >= 0.60:
        risco = "ALTO"
    elif score >= 0.35:
        risco = "MÉDIO"
    else:
        risco = "BAIXO"

    return {
        "risco"       : risco,
        "score"       : round(score, 4),
        "distribuicao": {
            "P(adimplente)"  : round(1 - score, 4),
            "P(inadimplente)": round(score, 4),
        }
    }

exemplo_cliente = {
    "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 1, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 45000, "BILL_AMT2": 43000, "BILL_AMT3": 40000,
    "BILL_AMT4": 38000, "BILL_AMT5": 35000, "BILL_AMT6": 30000,
    "PAY_AMT1": 500,  "PAY_AMT2": 500,  "PAY_AMT3": 500,
    "PAY_AMT4": 500,  "PAY_AMT5": 500,  "PAY_AMT6": 500,
}

resultado = avaliar_cliente(exemplo_cliente)
print("[3] Inferência — cliente de exemplo:")
print(f"  Risco       : {resultado['risco']}")
print(f"  Score       : {resultado['score']}")
print(f"  Distribuição: {resultado['distribuicao']}")

#saida
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Banco AgoraVai — {best_name}  |  AUC={auc_test:.4f}",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

#ROC
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_test:.4f}")
ax1.plot([0, 1], [0, 1], "k--", lw=1)
ax1.set_title("Curva ROC")
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.legend()

#Matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Adim.", "Inadim."]).plot(ax=ax2, colorbar=False)
ax2.set_title("Matriz de Confusão")

#score
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(y_proba[y_test == 0], bins=40, alpha=0.6, color="green",  label="Adimplente")
ax3.hist(y_proba[y_test == 1], bins=40, alpha=0.6, color="crimson", label="Inadimplente")
ax3.axvline(resultado["score"], color="black", linestyle="--", label=f"Cliente ex. ({resultado['score']})")
ax3.set_title("Distribuição de Probabilidade")
ax3.set_xlabel("Score de Inadimplência"); ax3.set_ylabel("Frequência")
ax3.legend(fontsize=8)

#  Comparação AUC dos modelos
ax4 = fig.add_subplot(gs[1, 0])
names  = list(cv_scores.keys())
means  = [cv_scores[n].mean() for n in names]
stds   = [cv_scores[n].std()  for n in names]
colors = ["gold" if n == best_name else "steelblue" for n in names]
bars = ax4.barh(names, means, xerr=stds, color=colors, edgecolor="black", height=0.5)
ax4.set_xlim(0.5, 0.9)
ax4.set_title("AUC por Modelo (CV)")
ax4.set_xlabel("ROC-AUC")
for bar, m in zip(bars, means):
    ax4.text(m + 0.002, bar.get_y() + bar.get_height()/2,
             f"{m:.4f}", va="center", fontsize=9)

# Feature Importance
ax5 = fig.add_subplot(gs[1, 1:])
clf_step = best_pipe.named_steps.get("clf")
if hasattr(clf_step, "feature_importances_"):
    importances = clf_step.feature_importances_
    indices = np.argsort(importances)[-15:]
    ax5.barh(X.columns[indices], importances[indices], color="steelblue")
    ax5.set_title("Feature Importance (Top 15)")
    ax5.set_xlabel("Importância")
elif hasattr(clf_step, "coef_"):
    coef = np.abs(clf_step.coef_[0])
    indices = np.argsort(coef)[-15:]
    ax5.barh(X.columns[indices], coef[indices], color="steelblue")
    ax5.set_title("Coeficientes Absolutos (Top 15)")
    ax5.set_xlabel("|Coeficiente|")
else:
    ax5.text(0.5, 0.5, "Feature importance não disponível",
             ha="center", va="center", transform=ax5.transAxes)

plt.savefig("resultado_inadimplencia.png", dpi=150, bbox_inches="tight")
print("\n[4] Gráfico salvo em: resultado_inadimplencia.png")
print("\nPipeline concluído com sucesso.")
