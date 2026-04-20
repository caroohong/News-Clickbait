"""
TAREA 2 - NLP & XAI: Análisis Exploratorio del Dataset (EDA)
Versión Simplificada y Profesional — Enfoque en Hallazgos de Negocio/Educativos

Genera 9 análisis clave sobre el fenómeno del clickbait:
  1. Ranking de Sensacionalismo por Medio (% de clickbait por portal)
  2. Concentración del Fenómeno (Quiénes aportan más clickbait al total)
  3. Comparativa Geográfica (Chile vs Extranjero)
  4. Tendencia del Clickbait en el Tiempo (Evolución mensual)
  5. Anatomía del Titular: Extensión (Largo de textos por clase)
  6. Anatomía del Titular: Uso de Ganchos (Signos ! y ?)
  7. Diccionario del Clickbait (Palabras más frecuentes)
  8. Términos que Delatan el Clickbait (Asociación diferencial de palabras)
  9. Relación Volumen vs Estilo (Producción total vs % de sensacionalismo)

USO:
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
    "cuando", "como", "quien", "cual", "todo", "todos", "estos", "estas"
}

# CARGA Y LIMPIEZA
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    # Quedarse solo con las clases de interés
    df = df[df["etiqueta_final"].isin(["informativo", "clickbait", "fake_news"])].copy()

    if "fecha_publicacion" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce", utc=True)
        df["fecha_dt"] = df["fecha_dt"].dt.tz_localize(None)
    
    df["n_chars"]  = df["titulo"].astype(str).str.len()
    df["has_excl"] = df["titulo"].astype(str).str.contains(r"!", regex=False).astype(int)
    df["has_qmark"]= df["titulo"].astype(str).str.contains(r"\?", regex=True).astype(int)
    df["es_clickbait"] = (df["etiqueta_final"] == "clickbait").astype(int)

    print(f"[✓] Dataset cargado: {len(df):,} titulares")
    return df

def save_fig(fig, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → Guardado: {path}")

def short_portal(name: str, max_len: int = 22) -> str:
    name = str(name)
    return name if len(name) <= max_len else name[:max_len - 1] + "…"

# ANÁLISIS 1: Ranking de Portales
def analisis_1_ranking_sensacionalismo(df: pd.DataFrame, out: str):
    MIN_TITULARES = 25
    stats = (
        df.groupby("portal")
        .agg(total=("titulo", "count"), clickbait=("es_clickbait", "sum"))
        .query(f"total >= {MIN_TITULARES}")
        .assign(pct=lambda x: x["clickbait"] / x["total"] * 100)
        .sort_values("pct", ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=stats["pct"], y=[short_portal(p) for p in stats.index], 
                palette="Reds_r", ax=ax, edgecolor="#333", linewidth=0.5)
    ax.set_title("Ranking de Sensacionalismo por Medio\n(% de su contenido identificado como clickbait)")
    ax.set_xlabel("Porcentaje de titulares clickbait (%)")
    ax.set_ylabel("")
    
    for i, v in enumerate(stats["pct"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    save_fig(fig, os.path.join(out, "1_ranking_portales.png"))

# ANÁLISIS 2: Concentración
def analisis_2_concentracion(df: pd.DataFrame, out: str):
    cb_df = df[df["etiqueta_final"] == "clickbait"]
    stats = cb_df["portal"].value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=sns.color_palette("muted"),
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.set_ylabel("")
    ax.set_title("Distribución del Clickbait Total\n¿Qué medios generan el mayor volumen de este contenido?")
    save_fig(fig, os.path.join(out, "2_concentracion_total.png"))

# ANÁLISIS 3: Chile vs Internacional
def analisis_3_comparativa_geografica(df: pd.DataFrame, out: str):
    pivot = pd.crosstab(df["origen"], df["etiqueta_final"], normalize="index") * 100
    
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[c] for c in pivot.columns], 
               edgecolor="white", width=0.6)
    ax.set_title("Estilo Editorial: Prensa Chilena vs. Internacional")
    ax.set_ylabel("Distribución de contenidos (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(["Internacional", "Nacional"], rotation=0)
    ax.legend(title="Tipo de contenido", bbox_to_anchor=(1, 1))
    
    save_fig(fig, os.path.join(out, "3_comparativa_origen.png"))

# ANÁLISIS 4: Evolución Temporal
def analisis_4_evolucion(df: pd.DataFrame, out: str):
    df_f = df.dropna(subset=["fecha_dt"]).copy()
    df_f = df_f[df_f["fecha_dt"].dt.year >= 2023]
    if len(df_f) < 50: return

    df_f["mes"] = df_f["fecha_dt"].dt.to_period("M").dt.to_timestamp()
    trend = df_f.groupby("mes")["es_clickbait"].mean() * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend.index, trend.values, marker='o', color=ACCENT, linewidth=2, markersize=4)
    ax.fill_between(trend.index, trend.values, alpha=0.1, color=ACCENT)
    ax.set_title("Evolución del Clickbait en el Tiempo\n(Promedio mensual de uso de tácticas de enganche)")
    ax.set_ylabel("% de Clickbait")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save_fig(fig, os.path.join(out, "4_evolucion_temporal.png"))

# ANÁLISIS 5: Extensión
def analisis_5_extension(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="etiqueta_final", y="n_chars", palette=PALETTE, ax=ax, showfliers=False)
    ax.set_title("Anatomía del Titular: Extensión del Texto")
    ax.set_ylabel("Cantidad de caracteres")
    ax.set_xlabel("")
    ax.set_xticklabels(["Informativo", "Clickbait", "Fake News"])
    save_fig(fig, os.path.join(out, "5_extension_titulares.png"))

# ANÁLISIS 6: Ganchos de Puntuación
def analisis_6_puntuacion(df: pd.DataFrame, out: str):
    stats = df.groupby("etiqueta_final").agg(
        Exclamación=("has_excl", "mean"),
        Interrogación=("has_qmark", "mean")
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    stats.plot(kind="bar", ax=ax, color=["#FF6B6B", "#4D96FF"], edgecolor="white")
    ax.set_title("Uso de Ganchos Visuales\n% de titulares que usan signos de puntuación para atraer")
    ax.set_ylabel("% de titulares")
    ax.set_xlabel("")
    ax.set_xticklabels(["Informativo", "Clickbait", "Fake News"], rotation=0)
    save_fig(fig, os.path.join(out, "6_uso_puntuacion.png"))

# ANÁLISIS 7: Diccionario del Clickbait
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
    ax.set_title("Diccionario del Clickbait\nPalabras que aparecen con mayor frecuencia en titulares sensacionalistas")
    save_fig(fig, os.path.join(out, "7_diccionario_clickbait.png"))

# ANÁLISIS 8: Términos que Delatan
def analisis_8_terminos_delatadores(df: pd.DataFrame, out: str):
    df_bin = df[df["etiqueta_final"].isin(["clickbait", "informativo"])].copy()
    vectorizer = CountVectorizer(stop_words=list(STOPWORDS_ES), min_df=5, token_pattern=r"\b[a-z]{4,}\b")
    X = vectorizer.fit_transform(df_bin["titulo"].str.lower())
    vocab = vectorizer.get_feature_names_out()
    
    # Convertir a numpy array para evitar problemas con sparse indexing de scipy
    mask_cb = (df_bin["etiqueta_final"] == "clickbait").values
    mask_info = (df_bin["etiqueta_final"] == "informativo").values

    counts_cb = X[mask_cb].sum(axis=0).A1 + 1
    counts_info = X[mask_info].sum(axis=0).A1 + 1
    
    # Ratio de probabilidad (qué tan más probable es que sea clickbait si aparece la palabra)
    ratio = (counts_cb / counts_cb.sum()) / (counts_info / counts_info.sum())
    delatadores = pd.DataFrame({"Palabra": vocab, "Ratio": ratio}).sort_values("Ratio", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=delatadores, x="Ratio", y="Palabra", palette="magma", ax=ax)
    ax.set_title("Términos que Delatan el Clickbait\nPalabras con mayor probabilidad de pertenecer a un titular sensacionalista")
    ax.set_xlabel("Fuerza de Asociación (veces más frecuente en clickbait vs informativo)")
    save_fig(fig, os.path.join(out, "8_terminos_delatadores.png"))

# ANÁLISIS 9: Volumen vs Estilo
def analisis_9_volumen_estilo(df: pd.DataFrame, out: str):
    stats = df.groupby("portal").agg(
        Total=("titulo", "count"),
        Sensacionalismo=("es_clickbait", "mean")
    ).query("Total >= 20")
    stats["Sensacionalismo"] *= 100

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=stats, x="Total", y="Sensacionalismo", scatter_kws={'alpha':0.5, 's':stats['Total']*2}, 
                line_kws={'color':ACCENT}, ax=ax)
    ax.set_title("Relación entre Volumen de Noticias y Estilo Sensacionalista")
    ax.set_xlabel("Cantidad total de noticias analizadas")
    ax.set_ylabel("% de contenido clickbait")
    
    for p in stats.sort_values("Sensacionalismo", ascending=False).head(5).index:
        ax.annotate(short_portal(p), (stats.loc[p, "Total"], stats.loc[p, "Sensacionalismo"]), fontsize=8)

    save_fig(fig, os.path.join(out, "9_volumen_vs_estilo.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset_output/dataset_etiquetado_v3.csv")
    parser.add_argument("--out", default="eda_output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_dataset(args.csv)

    print("\n[→] Generando visualizaciones narrativas...")
    analisis_1_ranking_sensacionalismo(df, args.out)
    analisis_2_concentracion(df, args.out)
    analisis_3_comparativa_geografica(df, args.out)
    analisis_4_evolucion(df, args.out)
    analisis_5_extension(df, args.out)
    analisis_6_puntuacion(df, args.out)
    analisis_7_diccionario(df, args.out)
    analisis_8_terminos_delatadores(df, args.out)
    analisis_9_volumen_estilo(df, args.out)

    print(f"\n[✓] EDA Completado. Los 9 gráficos están en: {args.out}/")

if __name__ == "__main__":
    main()
