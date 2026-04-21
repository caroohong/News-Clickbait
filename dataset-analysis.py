"""
TAREA 2 - NLP & XAI: Análisis Exploratorio del Dataset (EDA)
  python dataset-analysis.py --csv dataset_output/dataset_etiquetado_v3.csv
"""
import argparse
import os
import re
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# CONFIGURACIÓN ESTÉTICA
PALETTE = {
    "informativo": "#2E86AB",
    "clickbait":   "#E84855",
    "fake_news":   "#F4A261",
}
ACCENT   = "#E84855"
NEUTRAL  = "#2E86AB"
LIGHT_BG = "#F8F9FA"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor":  LIGHT_BG,
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DDDDDD",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelcolor":   "#333333",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "grid.color":        "#EEEEEE",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})

STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "los", "del", "las", "un", "por",
    "con", "una", "para", "es", "se", "al", "lo", "que", "su", "le",
    "más", "pero", "sus", "como", "o", "si", "no", "fue", "ha", "ya",
    "este", "esto", "esta", "ese", "esa", "son", "está", "hay",
    "ser", "han", "dos", "nueva", "nuevo", "tras", "sobre", "entre",
    "ante", "desde", "hasta", "chile", "año", "años", "tras",
    "así", "he", "me", "mi", "te", "tu", "yo", "ni", "aunque", "donde",
    "cuando", "como", "quien", "cual", "todo", "todos", "estos", "estas",
    "https", "http", "www", "xml", "sitemaps", "com", "net", "html"
}

def save_fig(fig, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → Guardado: {path}")

def short_portal(name: str, max_len: int = 22) -> str:
    name = str(name)
    return name if len(name) <= max_len else name[:max_len - 1] + "…"

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    # Quedarse solo con las clases de interés
    df = df[df["etiqueta_final"].isin(["informativo", "clickbait", "fake_news"])].copy()

    if "fecha_publicacion" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce", utc=True)
        df["fecha_dt"] = df["fecha_dt"].dt.tz_localize(None)
    
    df["n_chars"]  = df["titulo"].astype(str).str.len()
    df["n_words"]  = df["titulo"].astype(str).str.split().str.len()
    df["has_excl"] = df["titulo"].astype(str).str.contains(r"!", regex=False).astype(int)
    df["has_qmark"]= df["titulo"].astype(str).str.contains(r"\?", regex=True).astype(int)
    df["es_clickbait"] = (df["etiqueta_final"] == "clickbait").astype(int)

    # Limpiar columnas de heurística si existen
    numeric_cols = ["cb_heuristic", "cb_brecha", "cb_exageracion", "cb_emocion", "cb_ambiguedad"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "cb_eje_dominante" in df.columns:
        df["cb_eje_dominante"] = df["cb_eje_dominante"].astype(str).str.strip().str.lower()

    print(f"[✓] Dataset cargado: {len(df):,} titulares")
    return df

def print_executive_summary(df: pd.DataFrame):
    total = len(df)
    print("\n" + "=" * 70)
    print("RESUMEN EJECUTIVO DEL DATASET")
    print("=" * 70)
    print(f"Total de titulares: {total:,}")

    print("\nDistribución de clases:")
    dist = df["etiqueta_final"].value_counts()
    for label, count in dist.items():
        print(f"  - {label:<12}: {count:>5,} ({count / total * 100:5.2f}%)")

    if "origen" in df.columns:
        print("\nDistribución por origen:")
        origen = df["origen"].value_counts()
        for label, count in origen.items():
            print(f"  - {label:<12}: {count:>5,} ({count / total * 100:5.2f}%)")
    print("=" * 70)

# 1. Ranking de Sensacionalismo
def analisis_1_ranking(df: pd.DataFrame, out: str):
    MIN_TITULARES = 25
    stats = (
        df.groupby(["portal", "origen"])
        .agg(total=("titulo", "count"), clickbait=("es_clickbait", "sum"))
        .query(f"total >= {MIN_TITULARES}")
        .assign(pct=lambda x: x["clickbait"] / x["total"] * 100)
        .sort_values("pct", ascending=False)
        .head(25)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {"nacional": "#E84855", "internacional": "#2E86AB"}
    
    sns.barplot(data=stats, x="pct", y="portal", hue="origen", 
                palette=colors, dodge=False, ax=ax, edgecolor="#333", linewidth=0.5)
    
    ax.set_title("Uso de Clickbait por Medio de Comunicación\n(Diferenciación por origen geográfico)")
    ax.set_xlabel("Porcentaje de titulares detectados (%)")
    ax.set_ylabel("")
    ax.set_yticklabels([short_portal(p) for p in stats["portal"]])
    
    for i, v in enumerate(stats["pct"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5)

    ax.legend(title="Origen del medio", loc="lower right")
    save_fig(fig, os.path.join(out, "1_ranking_sensacionalismo.png"))

# 2. Concentración por Medio
def analisis_2_concentracion(df: pd.DataFrame, out: str):
    cb_df = df[df["etiqueta_final"] == "clickbait"]
    stats = cb_df["portal"].value_counts().head(12)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=sns.color_palette("Reds_r", n_colors=12),
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.set_ylabel("")
    ax.set_title("Origen del Clickbait Analizado\n¿Qué medios aportan el mayor volumen de titulares sospechosos?")
    save_fig(fig, os.path.join(out, "2_concentracion_medios.png"))

# 3. Estilo Chile vs Internacional
def analisis_3_geografico(df: pd.DataFrame, out: str):
    pivot = pd.crosstab(df["origen"], df["etiqueta_final"], normalize="index") * 100
    
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[c] for c in pivot.columns], 
               edgecolor="white", width=0.6)
    ax.set_title("Estilo Editorial: Prensa Chilena vs. Internacional")
    ax.set_ylabel("Distribución de contenidos (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(["Internacional", "Nacional"], rotation=0)
    ax.legend(title="Tipo de noticia", bbox_to_anchor=(1, 1))
    save_fig(fig, os.path.join(out, "3_estilo_geografico.png"))

# 4. Tendencia Temporal
def analisis_4_evolucion(df: pd.DataFrame, out: str):
    df_f = df.dropna(subset=["fecha_dt"]).copy()
    df_f = df_f[df_f["fecha_dt"].dt.year >= 2024]
    if len(df_f) < 30: return

    df_f["mes"] = df_f["fecha_dt"].dt.to_period("M").dt.to_timestamp()
    trend = df_f.groupby("mes")["es_clickbait"].mean() * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend.index, trend.values, marker='o', color=ACCENT, linewidth=2)
    ax.fill_between(trend.index, trend.values, alpha=0.1, color=ACCENT)
    ax.set_title("Evolución en el Uso de Tácticas de Enganche\n(Promedio mensual de titulares clickbait)")
    ax.set_ylabel("% Detectado")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save_fig(fig, os.path.join(out, "4_tendencia_temporal.png"))

# 5. Extensión
def analisis_5_extension(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="etiqueta_final", y="n_chars", palette=PALETTE, ax=ax, showfliers=False)
    ax.set_title("Anatomía del Titular: ¿Son más largos los engañosos?")
    ax.set_ylabel("Longitud (caracteres)")
    ax.set_xlabel("")
    ax.set_xticklabels(["Informativo", "Clickbait", "Fake News"])
    save_fig(fig, os.path.join(out, "5_anatomia_extension.png"))

# 6. Ganchos de Puntuación
def analisis_6_puntuacion(df: pd.DataFrame, out: str):
    stats = df.groupby("etiqueta_final").agg(
        Exclamación=("has_excl", "mean"),
        Interrogación=("has_qmark", "mean")
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    stats.plot(kind="bar", ax=ax, color=["#FF6B6B", "#4D96FF"], edgecolor="white")
    ax.set_title("Uso de Ganchos Visuales\n% de titulares que usan signos para forzar el interés")
    ax.set_ylabel("% de titulares")
    ax.set_xlabel("")
    ax.set_xticklabels(["Informativo", "Clickbait", "Fake News"], rotation=0)
    save_fig(fig, os.path.join(out, "6_ganchos_puntuacion.png"))

# 7. Diccionario del Clickbait
def analisis_7_diccionario(df: pd.DataFrame, out: str):
    def get_top_words(text_series, top_n=20):
        words = []
        for t in text_series.dropna():
            tokens = re.findall(r"\b[a-záéíóúüñ]{4,}\b", str(t).lower())
            words.extend([w for w in tokens if w not in STOPWORDS_ES])
        return pd.DataFrame(Counter(words).most_common(top_n), columns=["Palabra", "Frecuencia"])

    cb_words = get_top_words(df[df["etiqueta_final"] == "clickbait"]["titulo"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cb_words, x="Frecuencia", y="Palabra", palette="Oranges_r", ax=ax)
    ax.set_title("El Diccionario del Sensacionalismo\nPalabras con mayor presencia en titulares clickbait")
    save_fig(fig, os.path.join(out, "7_diccionario_clickbait.png"))

# 8. Términos que Delatan
def analisis_8_asociacion(df: pd.DataFrame, out: str):
    df_bin = df[df["etiqueta_final"].isin(["clickbait", "informativo"])].copy()
    vectorizer = CountVectorizer(stop_words=list(STOPWORDS_ES), min_df=5, token_pattern=r"\b[a-z]{4,}\b")
    X = vectorizer.fit_transform(df_bin["titulo"].str.lower())
    vocab = vectorizer.get_feature_names_out()
    
    mask_cb = (df_bin["etiqueta_final"] == "clickbait").values
    mask_info = (df_bin["etiqueta_final"] == "informativo").values
    counts_cb = X[mask_cb].sum(axis=0).A1 + 1
    counts_info = X[mask_info].sum(axis=0).A1 + 1
    
    ratio = (counts_cb / counts_cb.sum()) / (counts_info / counts_info.sum())
    delatadores = pd.DataFrame({"Palabra": vocab, "Fuerza": ratio}).sort_values("Fuerza", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=delatadores, x="Fuerza", y="Palabra", palette="magma", ax=ax)
    ax.set_title("Términos que Delatan el Clickbait\n(Fuerza de asociación: Probabilidad de ser clickbait si la palabra aparece)")
    ax.set_xlabel("Fuerza de Asociación")
    save_fig(fig, os.path.join(out, "8_palabras_delatadoras.png"))

# 9. Táctica Favorita por Portal (Nacional e Internacional por separado)
def analisis_9_estilo_por_portal(df: pd.DataFrame, out: str):
    if "cb_eje_dominante" not in df.columns: return
    subset_all = df[(df["etiqueta_final"] == "clickbait") & (df["cb_eje_dominante"] != "ninguno")].copy()
    if subset_all.empty: return

    cmap = {"brecha": "#F4A261", "exageracion": "#E84855", "emocion": "#2A9D8F", "ambiguedad": "#457B9D"}
    
    for orig in ["nacional", "internacional"]:
        subset = subset_all[subset_all["origen"] == orig].copy()
        if subset.empty: continue

        stats = pd.crosstab(subset["portal"], subset["cb_eje_dominante"])
        stats = stats[stats.sum(axis=1) >= 3] # Volumen mínimo por origen
        if stats.empty: continue
        
        top_style = stats.idxmax(axis=1)
        intensity = (stats.max(axis=1) / stats.sum(axis=1)) * 100
        res = pd.DataFrame({"Estilo": top_style, "Intensidad": intensity}).sort_values("Intensidad", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(11, 7))
        colors = [cmap.get(s, "#999") for s in res["Estilo"]]

        ax.barh([short_portal(p, 20) for p in res.index[::-1]], res["Intensidad"].values[::-1], 
                color=colors[::-1], edgecolor="#333", linewidth=0.5)
        
        for i, (p, row) in enumerate(res.iloc[::-1].iterrows()):
            ax.text(row["Intensidad"] + 1, i, f"{row['Estilo'].capitalize()} ({row['Intensidad']:.1f}%)", 
                    va="center", fontsize=8.5, fontweight="bold")

        title_orig = "Prensa Chilena" if orig == "nacional" else "Prensa Internacional"
        ax.set_title(f"Especialización del Engaño: {title_orig}\n¿Cuál es la táctica dominante de cada medio?")
        ax.set_xlabel("% del clickbait del portal explicado por su táctica principal")
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=c, lw=6, label=t.capitalize()) 
                           for t, c in cmap.items() if t in res["Estilo"].unique()]
        ax.legend(handles=legend_elements, title="Táctica Dominante", loc="lower right", fontsize=9)
        
        suffix = "nacional" if orig == "nacional" else "internacional"
        save_fig(fig, os.path.join(out, f"9_{suffix}_tactica.png"))

# 10. Anatomía del Engaño (Ponderación de tácticas)
def analisis_10_anatomia_engano(df: pd.DataFrame, out: str):
    comps = {"cb_brecha": "Ocultar información", "cb_exageracion": "Exageración", 
             "cb_emocion": "Apelación emocional", "cb_ambiguedad": "Ambigüedad"}
    avail = [c for c in comps.keys() if c in df.columns]
    if not avail: return

    means = df[df["etiqueta_final"] == "clickbait"][avail].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=means.values, y=[comps[c] for c in means.index], palette="Reds_r", ax=ax)
    ax.set_title("Anatomía del Engaño\n¿Qué recursos se usan con mayor intensidad para atraer clics?")
    ax.set_xlabel("Intensidad de uso (promedio)")
    save_fig(fig, os.path.join(out, "10_tacticas_comunes.png"))

# 11. Dispersión del Sensacionalismo
def analisis_11_dispersion(df: pd.DataFrame, out: str):
    stats = (df.groupby(["portal", "origen"]).agg(total=("titulo", "count"), cb=("es_clickbait", "sum"))
             .query("total >= 15").assign(pct=lambda x: x["cb"]/x["total"]*100).reset_index())

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"nacional": "#E84855", "internacional": "#2E86AB"}
    sns.boxplot(data=stats, x="origen", y="pct", palette=colors, ax=ax, width=0.4, showfliers=False)
    sns.stripplot(data=stats, x="origen", y="pct", color="#333", alpha=0.4, size=6, ax=ax, jitter=True)
    ax.set_title("Dispersión del Sensacionalismo\n¿Es una práctica generalizada o concentrada en pocos medios?")
    ax.set_ylabel("% de Clickbait por medio individual")
    ax.set_xticklabels(["Internacional", "Nacional"])
    save_fig(fig, os.path.join(out, "11_dispersion_sensacionalismo.png"))

# 12. Monopolio del Clickbait
def analisis_12_monopolio(df: pd.DataFrame, out: str):
    cb_df = df[df["etiqueta_final"] == "clickbait"].copy()
    top_list = []
    for orig in ["nacional", "internacional"]:
        sub = cb_df[cb_df["origen"] == orig]
        if sub.empty: continue
        counts = sub["portal"].value_counts().head(5)
        for portal, count in counts.items():
            top_list.append({"Portal": short_portal(portal, 18), "Origen": orig.capitalize(), 
                             "Share": (count/len(sub))*100, "Cant": count})

    stats_df = pd.DataFrame(top_list).sort_values(["Origen", "Share"], ascending=[False, False])
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=stats_df, x="Share", y="Portal", hue="Origen", palette={"Nacional": "#E84855", "Internacional": "#2E86AB"}, 
                dodge=False, ax=ax, edgecolor="#333", linewidth=0.5)
    ax.set_title("Monopolio del Clickbait: Los 5 Principales Infractores por Origen\n(% del clickbait total de cada categoría aportado por cada medio)")
    ax.set_xlabel("% del clickbait total de su categoría")
    for i, p in enumerate(ax.patches):
        if p.get_width() > 0:
            ax.text(p.get_width() + 0.5, p.get_y() + p.get_height()/2, f"{p.get_width():.1f}%", va="center", fontsize=9)
    save_fig(fig, os.path.join(out, "12_monopolio_clickbait.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset_output/dataset_etiquetado_v3.csv")
    parser.add_argument("--out", default="eda_output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_dataset(args.csv)
    print_executive_summary(df)

    print("\nGenerando 12 visualizaciones narrativas...")
    analisis_1_ranking(df, args.out)
    analisis_2_concentracion(df, args.out)
    analisis_3_geografico(df, args.out)
    analisis_4_evolucion(df, args.out)
    analisis_5_extension(df, args.out)
    analisis_6_puntuacion(df, args.out)
    analisis_7_diccionario(df, args.out)
    analisis_8_asociacion(df, args.out)
    analisis_9_estilo_por_portal(df, args.out)
    analisis_10_anatomia_engano(df, args.out)
    analisis_11_dispersion(df, args.out)
    analisis_12_monopolio(df, args.out)

    print(f"\n Análisis completado. Gráficos en: {args.out}/")

if __name__ == "__main__":
    main()
