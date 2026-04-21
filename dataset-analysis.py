"""
TAREA 2 - NLP & XAI: Análisis Exploratorio del Dataset (EDA)
Versión Narrativa PRO — Foco en el comportamiento del Clickbait

Genera 10 visualizaciones clave para entender el fenómeno:
  1. Ranking de Sensacionalismo por Medio (% clickbait por portal)
  2. Distribución por Medio (Quién aporta más al total)
  3. Estilo Editorial: Chile vs El Mundo (Comparativa geográfica)
  4. Tendencia en el Tiempo (Evolución de tácticas)
  5. Anatomía del Titular: Extensión (Largo de textos)
  6. Anatomía del Titular: Ganchos Visuales (Signos ! y ?)
  7. El Diccionario del Clickbait (Palabras más frecuentes)
  8. Términos que Delatan el Clickbait (Asociación palabra vs clase)
  9. ¿El volumen afecta al estilo? (Relación cantidad vs sensacionalismo)
  10. Tácticas más comunes (Brecha, Exageración, Emoción, Ambigüedad)

ANÁLISIS EXTRA AGREGADOS:
  11. Robustez del ranking: intervalos de confianza por portal
  12. Test estadístico Chile vs Internacional
  13. Test estadístico de longitud por clase
  14. Eje dominante del clickbait
  15. Sesgo de recolección: etiqueta_base vs etiqueta_final
  16. Sesgo de recolección: metodo_obtencion vs etiqueta_final

USO:
  python dataset-analysis.py --csv dataset_output/dataset_etiquetado_v4.csv
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

from scipy.stats import chi2_contingency, kruskal
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
    "ante", "desde", "hasta", "chile", "año", "años",
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

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - margin, center + margin

# CARGA Y LIMPIEZA
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    df = df[df["etiqueta_final"].isin(["informativo", "clickbait", "fake_news"])].copy()

    if "fecha_publicacion" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce", utc=True)
        df["fecha_dt"] = df["fecha_dt"].dt.tz_localize(None)

    df["n_chars"] = df["titulo"].astype(str).str.len()
    df["n_words"] = df["titulo"].astype(str).str.split().str.len()
    df["has_excl"] = df["titulo"].astype(str).str.contains(r"!", regex=False).astype(int)
    df["has_qmark"] = df["titulo"].astype(str).str.contains(r"\?", regex=True).astype(int)
    df["es_clickbait"] = (df["etiqueta_final"] == "clickbait").astype(int)

    numeric_cols = ["cb_heuristic", "cb_brecha", "cb_exageracion", "cb_emocion", "cb_ambiguedad"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "cb_eje_dominante" in df.columns:
        df["cb_eje_dominante"] = df["cb_eje_dominante"].astype(str).str.strip().str.lower()

    print(f"[✓] Dataset cargado: {len(df):,} titulares")
    print("\nDistribución de clases:")
    print(df["etiqueta_final"].value_counts().to_string())
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

    if "fecha_dt" in df.columns and df["fecha_dt"].notna().any():
        print("\nCobertura temporal:")
        print(f"  - Desde: {df['fecha_dt'].min().date()}")
        print(f"  - Hasta: {df['fecha_dt'].max().date()}")

    print("\nPromedios generales:")
    print(f"  - Largo medio (caracteres): {df['n_chars'].mean():.2f}")
    print(f"  - Largo medio (palabras):   {df['n_words'].mean():.2f}")
    print("=" * 70)

# 1
def analisis_1_ranking(df: pd.DataFrame, out: str):
    MIN_TITULARES = 25
    stats = (
        df.groupby("portal")
        .agg(total=("titulo", "count"), clickbait=("es_clickbait", "sum"))
        .query(f"total >= {MIN_TITULARES}")
        .assign(pct=lambda x: x["clickbait"] / x["total"] * 100)
        .sort_values("pct", ascending=False)
        .head(20)
    )

    print("\n[1] Ranking de sensacionalismo por medio")
    print(stats.round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=stats["pct"],
        y=[short_portal(p) for p in stats.index],
        palette="Reds_r",
        ax=ax,
        edgecolor="#333",
        linewidth=0.5
    )
    ax.set_title("Uso de Clickbait por Medio de Comunicación\n(% de contenido sensacionalista en su muestra)")
    ax.set_xlabel("Porcentaje de titulares detectados (%)")
    ax.set_ylabel("")
    for i, v in enumerate(stats["pct"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    save_fig(fig, os.path.join(out, "1_ranking_sensacionalismo.png"))
    return stats

# 2
def analisis_2_concentracion(df: pd.DataFrame, out: str):
    cb_df = df[df["etiqueta_final"] == "clickbait"]
    stats = cb_df["portal"].value_counts().head(12)

    print("\n[2] Concentración del clickbait total")
    print(stats.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    stats.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax,
        colors=sns.color_palette("Reds_r", n_colors=12),
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax.set_ylabel("")
    ax.set_title("Origen del Clickbait Analizado\n¿Qué medios aportan el mayor volumen de titulares sospechosos?")
    save_fig(fig, os.path.join(out, "2_concentracion_medios.png"))
    return stats

# 3
def analisis_3_geografico(df: pd.DataFrame, out: str):
    pivot = pd.crosstab(df["origen"], df["etiqueta_final"], normalize="index") * 100

    print("\n[3] Estilo editorial: Chile vs Internacional")
    print(pivot.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[PALETTE[c] for c in pivot.columns],
        edgecolor="white",
        width=0.6
    )
    ax.set_title("Estilo Editorial: Prensa Chilena vs. Internacional")
    ax.set_ylabel("Distribución de contenidos (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(["Internacional", "Nacional"], rotation=0)
    ax.legend(title="Tipo de noticia", bbox_to_anchor=(1, 1))
    save_fig(fig, os.path.join(out, "3_estilo_geografico.png"))
    return pivot

# 4
def analisis_4_evolucion(df: pd.DataFrame, out: str):
    df_f = df.dropna(subset=["fecha_dt"]).copy()
    df_f = df_f[df_f["fecha_dt"].dt.year >= 2024]
    if len(df_f) < 30:
        print("\n[4] Tendencia temporal omitida: pocos datos.")
        return None

    df_f["mes"] = df_f["fecha_dt"].dt.to_period("M").dt.to_timestamp()
    trend = df_f.groupby("mes")["es_clickbait"].mean() * 100

    print("\n[4] Tendencia temporal del clickbait")
    print(trend.tail(12).round(2).to_string())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend.index, trend.values, marker="o", color=ACCENT, linewidth=2)
    ax.fill_between(trend.index, trend.values, alpha=0.1, color=ACCENT)
    ax.set_title("Evolución en el Uso de Tácticas de Enganche\n(Promedio mensual de titulares clickbait)")
    ax.set_ylabel("% Detectado")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save_fig(fig, os.path.join(out, "4_tendencia_temporal.png"))
    return trend

# 5
def analisis_5_extension(df: pd.DataFrame, out: str):
    stats = df.groupby("etiqueta_final")[["n_chars", "n_words"]].mean()

    print("\n[5] Anatomía del titular: extensión")
    print(stats.round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="etiqueta_final", y="n_chars", palette=PALETTE, ax=ax, showfliers=False)
    ax.set_title("Anatomía del Titular: ¿Son más largos los engañosos?")
    ax.set_ylabel("Longitud (caracteres)")
    ax.set_xlabel("")
    ax.set_xticklabels(["Informativo", "Clickbait", "Fake News"])
    save_fig(fig, os.path.join(out, "5_anatomia_extension.png"))
    return stats

# 6
def analisis_6_puntuacion(df: pd.DataFrame, out: str):
    stats = df.groupby("etiqueta_final").agg(
        Exclamación=("has_excl", "mean"),
        Interrogación=("has_qmark", "mean")
    ) * 100

    print("\n[6] Ganchos visuales")
    print(stats.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    stats.plot(kind="bar", ax=ax, color=["#FF6B6B", "#4D96FF"], edgecolor="white")
    ax.set_title("Uso de Ganchos Visuales\n% de noticias que usan signos para forzar el interés")
    ax.set_ylabel("% de titulares")
    ax.set_xlabel("")
    ax.set_xticklabels(["Informativo", "Clickbait", "Fake News"], rotation=0)
    save_fig(fig, os.path.join(out, "6_ganchos_puntuacion.png"))
    return stats

# 7
def analisis_7_diccionario(df: pd.DataFrame, out: str):
    def get_top_words(text_series, top_n=20):
        words = []
        for t in text_series.dropna():
            tokens = re.findall(r"\b[a-záéíóúüñ]{4,}\b", str(t).lower())
            words.extend([w for w in tokens if w not in STOPWORDS_ES])
        return pd.DataFrame(Counter(words).most_common(top_n), columns=["Palabra", "Frecuencia"])

    cb_words = get_top_words(df[df["etiqueta_final"] == "clickbait"]["titulo"])

    print("\n[7] Diccionario del clickbait")
    print(cb_words.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cb_words, x="Frecuencia", y="Palabra", palette="Oranges_r", ax=ax)
    ax.set_title("El Diccionario del Sensacionalismo\nPalabras con mayor presencia en titulares clickbait")
    save_fig(fig, os.path.join(out, "7_diccionario_clickbait.png"))
    return cb_words

# 8
def analisis_8_asociacion(df: pd.DataFrame, out: str):
    df_bin = df[df["etiqueta_final"].isin(["clickbait", "informativo"])].copy()
    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS_ES),
        min_df=5,
        token_pattern=r"(?u)\b[a-záéíóúüñ]{4,}\b"
    )
    X = vectorizer.fit_transform(df_bin["titulo"].str.lower())
    vocab = vectorizer.get_feature_names_out()

    mask_cb = (df_bin["etiqueta_final"] == "clickbait").values
    mask_info = (df_bin["etiqueta_final"] == "informativo").values

    counts_cb = X[mask_cb].sum(axis=0).A1 + 1
    counts_info = X[mask_info].sum(axis=0).A1 + 1

    ratio = (counts_cb / counts_cb.sum()) / (counts_info / counts_info.sum())
    delatadores = pd.DataFrame({"Palabra": vocab, "Fuerza": ratio}).sort_values("Fuerza", ascending=False).head(15)

    print("\n[8] Términos que delatan el clickbait")
    print(delatadores.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=delatadores, x="Fuerza", y="Palabra", palette="magma", ax=ax)
    ax.set_title("Términos que Delatan el Clickbait\n(Veces más probable que un titular sea sensacionalista si contiene la palabra)")
    ax.set_xlabel("Fuerza de Asociación (probabilidad relativa)")
    save_fig(fig, os.path.join(out, "8_palabras_delatadoras.png"))
    return delatadores

# 9
def analisis_9_tipo_clickbait_por_portal(df: pd.DataFrame, out: str):
    if "cb_eje_dominante" not in df.columns:
        print("\n[9] Análisis omitido: no existe columna cb_eje_dominante.")
        return None

    subset = df[df["etiqueta_final"] == "clickbait"].copy()
    subset = subset[subset["cb_eje_dominante"].notna()]
    subset = subset[subset["cb_eje_dominante"] != ""]

    if len(subset) == 0:
        print("\n[9] Análisis omitido: no hay titulares clickbait con eje dominante.")
        return None

    MIN_CLICKBAIT = 5

    conteo = pd.crosstab(subset["portal"], subset["cb_eje_dominante"])
    conteo["total_clickbait"] = conteo.sum(axis=1)
    conteo = conteo[conteo["total_clickbait"] >= MIN_CLICKBAIT].copy()

    if len(conteo) == 0:
        print("\n[9] Análisis omitido: ningún portal tiene suficientes casos de clickbait.")
        return None

    # tipo dominante por portal
    tipos = [c for c in conteo.columns if c != "total_clickbait"]
    conteo["tipo_mayoritario"] = conteo[tipos].idxmax(axis=1)
    conteo["freq_tipo_mayoritario"] = conteo[tipos].max(axis=1)
    conteo["pct_tipo_mayoritario"] = conteo["freq_tipo_mayoritario"] / conteo["total_clickbait"] * 100

    resumen = conteo[["total_clickbait", "tipo_mayoritario", "freq_tipo_mayoritario", "pct_tipo_mayoritario"]]
    resumen = resumen.sort_values(["pct_tipo_mayoritario", "total_clickbait"], ascending=[False, False]).head(15)

    print("\n[9] Tipo de clickbait mayoritario por portal")
    print(resumen.round(2).to_string())

    # Gráfico: portal vs tipo mayoritario
    fig, ax = plt.subplots(figsize=(11, 6))

    color_map = {
        "exageracion": "#E84855",
        "brecha": "#F4A261",
        "emocion": "#2A9D8F",
        "ambiguedad": "#457B9D",
        "ninguno": "#9E9E9E"
    }

    colors = [color_map.get(t, "#999999") for t in resumen["tipo_mayoritario"]]

    ax.barh(
        [short_portal(p) for p in resumen.index[::-1]],
        resumen["pct_tipo_mayoritario"].values[::-1],
        color=colors[::-1],
        edgecolor="white"
    )

    for i, (_, row) in enumerate(resumen.iloc[::-1].iterrows()):
        ax.text(
            row["pct_tipo_mayoritario"] + 1,
            i,
            f"{row['tipo_mayoritario']} ({row['pct_tipo_mayoritario']:.1f}%)",
            va="center",
            fontsize=8
        )

    ax.set_title("Tipo de Clickbait Mayoritario por Portal\n(portales con al menos 5 titulares clickbait)")
    ax.set_xlabel("% del clickbait del portal explicado por su táctica principal")
    ax.set_ylabel("")
    save_fig(fig, os.path.join(out, "9_tipo_clickbait_por_portal.png"))

    return resumen

# 10
def analisis_10_tacticas(df: pd.DataFrame, out: str):
    components = {
        "cb_brecha": "Ocultar información",
        "cb_exageracion": "Exageración / Hipérbole",
        "cb_emocion": "Apelación emocional",
        "cb_ambiguedad": "Ambigüedad"
    }
    available = [c for c in components.keys() if c in df.columns]
    if not available:
        print("\n[10] Tácticas omitidas: columnas no disponibles.")
        return None

    subset = df[df["etiqueta_final"] == "clickbait"].copy()
    means = subset[available].mean().sort_values(ascending=False)

    print("\n[10] Tácticas más comunes")
    print(means.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=means.values, y=[components[c] for c in means.index], palette="Reds_r", ax=ax)
    ax.set_title("Anatomía del Engaño\n¿Qué tácticas se usan con mayor intensidad en el clickbait?")
    ax.set_xlabel("Intensidad de uso (promedio)")
    ax.set_ylabel("")
    save_fig(fig, os.path.join(out, "10_tacticas_comunes.png"))
    return means

# 11
def analisis_11_intervalos_confianza(df: pd.DataFrame, out: str):
    MIN_TITULARES = 25
    stats = (
        df.groupby("portal")
        .agg(total=("titulo", "count"), clickbait=("es_clickbait", "sum"))
        .query(f"total >= {MIN_TITULARES}")
        .copy()
    )
    stats["pct"] = stats["clickbait"] / stats["total"] * 100
    cis = stats.apply(lambda row: wilson_ci(int(row["clickbait"]), int(row["total"])), axis=1)
    stats["ci_low"] = [c[0] * 100 for c in cis]
    stats["ci_high"] = [c[1] * 100 for c in cis]
    stats = stats.sort_values("pct", ascending=False).head(15)

    print("\n[11] Robustez del ranking: intervalos de confianza")
    print(stats[["total", "clickbait", "pct", "ci_low", "ci_high"]].round(2).to_string())

    stats_plot = stats.reset_index().iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        [short_portal(p) for p in stats_plot["portal"]],
        stats_plot["pct"],
        color=ACCENT,
        alpha=0.85,
        edgecolor="white"
    )

    ax.errorbar(
        stats_plot["pct"],
        np.arange(len(stats_plot)),
        xerr=[
            stats_plot["pct"] - stats_plot["ci_low"],
            stats_plot["ci_high"] - stats_plot["pct"]
        ],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1
    )

    ax.set_title("Robustez del Ranking por Medio\n% clickbait con intervalo de confianza 95%")
    ax.set_xlabel("Porcentaje de clickbait (%)")
    ax.set_ylabel("")
    save_fig(fig, os.path.join(out, "11_intervalos_confianza.png"))

# 12
def analisis_12_test_geografico(df: pd.DataFrame):
    df_bin = df[df["etiqueta_final"].isin(["clickbait", "informativo"])].copy()
    tabla = pd.crosstab(df_bin["origen"], df_bin["etiqueta_final"])
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        print("\n[12] Test geográfico omitido.")
        return

    chi2, p, dof, _ = chi2_contingency(tabla)
    n = tabla.values.sum()
    cramer_v = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))

    print("\n[12] Test estadístico Chile vs Internacional")
    print(tabla.to_string())
    print(f"\nChi2 = {chi2:.4f}")
    print(f"p-value = {p:.6g}")
    print(f"grados de libertad = {dof}")
    print(f"Cramér's V = {cramer_v:.4f}")

    if p < 0.05:
        print("Interpretación: existe asociación estadísticamente significativa entre origen y condición clickbait.")
    else:
        print("Interpretación: no se observa asociación estadísticamente significativa entre origen y condición clickbait.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset_output/dataset_etiquetado_v3.csv")
    parser.add_argument("--out", default="eda_output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_dataset(args.csv)
    print_executive_summary(df)

    print("\n[→] Generando visualizaciones narrativas para el informe...")
    analisis_1_ranking(df, args.out)
    analisis_2_concentracion(df, args.out)
    analisis_3_geografico(df, args.out)
    analisis_4_evolucion(df, args.out)
    analisis_5_extension(df, args.out)
    analisis_6_puntuacion(df, args.out)
    analisis_7_diccionario(df, args.out)
    analisis_8_asociacion(df, args.out)
    analisis_9_tipo_clickbait_por_portal(df, args.out)
    analisis_10_tacticas(df, args.out)
    analisis_11_intervalos_confianza(df, args.out)
    analisis_12_test_geografico(df)

    print(f"\n[✓] Análisis completado. Gráficos generados en: {args.out}/")

if __name__ == "__main__":
    main()