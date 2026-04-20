"""
TAREA 2 - NLP & XAI: Análisis Exploratorio del Dataset (EDA)
Genera 10 análisis sobre el dataset de clickbait:
  1.  Índice de clickbait por portal (% de clickbait por portal)
  2.  Comparación Chile vs Internacional
  3.  Relación volumen de noticias vs % clickbait por portal
  4.  Evolución del clickbait en el tiempo (fecha_publicacion)
  5.  cb_heuristic vs etiqueta_final (aciertos y errores)
  6.  Clickbait según metodo_obtencion
  7.  Longitud de titulares por clase
  8.  Uso de signos "!" y "?" por clase
  9.  Palabras más frecuentes en titulares clickbait
  10. Portales con mayor concentración de clickbait

USO:
  python dataset_analysis.py                          # usa dataset_etiquetado_v2.csv por defecto
  python dataset_analysis.py --csv mi_dataset.csv     # ruta personalizada
  python dataset_analysis.py --out resultados/        # carpeta de salida personalizada

DEPENDENCIAS:
  pip install pandas matplotlib seaborn wordcloud scikit-learn
"""

import argparse
import os
import re
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")   # backend sin pantalla (compatible con servidores/Colab)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# Intentar importar WordCloud (opcional)
try:
    from wordcloud import WordCloud, STOPWORDS
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("[AVISO] wordcloud no está instalado. El análisis 9 usará barplot en su lugar.")
    print("        Para instalarlo: pip install wordcloud")

#  CONFIGURACIÓN ESTÉTICA
PALETTE = {
    "informativo": "#2E86AB",
    "clickbait":   "#E84855",
    "fake_news":   "#F4A261",
    "posible_clickbait": "#A8DADC",
}
ACCENT   = "#E84855"
NEUTRAL  = "#2E86AB"
DARK_BG  = "#0F1117"
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
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})

STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "los", "del", "las", "un", "por",
    "con", "una", "para", "es", "se", "al", "lo", "que", "su", "le",
    "más", "pero", "sus", "como", "o", "si", "no", "fue", "ha", "ya",
    "este", "esto", "esta", "ese", "esa", "son", "está", "son", "hay",
    "ser", "han", "dos", "nueva", "nuevo", "tras", "sobre", "entre",
    "ante", "desde", "hasta", "Chile", "chile", "año", "años",
    "tras", "así", "he", "me", "mi", "te", "tu", "yo", "ni", "aunque",
}

#  CARGA Y LIMPIEZA
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Normalizar nombres de columnas
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"titulo", "portal", "origen", "etiqueta_final", "cb_heuristic"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")

    # Limpiar etiquetas
    df["etiqueta_final"] = df["etiqueta_final"].str.strip().str.lower()
    df["origen"]         = df["origen"].str.strip().str.lower()
    df["portal"]         = df["portal"].str.strip()
    df["cb_heuristic"]   = pd.to_numeric(df["cb_heuristic"], errors="coerce").fillna(0)

    # Fecha
    if "fecha_publicacion" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce", utc=True)
        df["fecha_dt"] = df["fecha_dt"].dt.tz_localize(None)
        df["mes"]      = df["fecha_dt"].dt.to_period("M")
    else:
        df["fecha_dt"] = pd.NaT
        df["mes"]      = pd.NaT

    # Longitud del titular
    df["n_chars"]  = df["titulo"].str.len()
    df["n_words"]  = df["titulo"].str.split().str.len()
    df["has_excl"] = df["titulo"].str.contains(r"!", regex=False).astype(int)
    df["has_qmark"]= df["titulo"].str.contains(r"?", regex=False).astype(int)

    # Columna binaria clickbait (incluye posible_clickbait para análisis)
    df["es_clickbait"] = df["etiqueta_final"].isin(["clickbait", "posible_clickbait"]).astype(int)

    print(f"[✓] Dataset cargado: {len(df):,} titulares")
    print(df["etiqueta_final"].value_counts().to_string())
    return df

def save_fig(fig, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → Guardado: {path}")


def color_bars(ax, palette: dict, labels=None):
    """Aplica colores de la paleta a las barras de un barplot."""
    if labels is None:
        labels = [p.get_text() for p in ax.get_xticklabels()]
    for bar, lbl in zip(ax.patches, labels):
        bar.set_facecolor(palette.get(lbl, NEUTRAL))


def short_portal(name: str, max_len: int = 25) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"

#  ANÁLISIS 1: Índice de clickbait por portal
def analisis_1_clickbait_por_portal(df: pd.DataFrame, out: str):
    """
    Calcula el porcentaje de titulares clickbait dentro de cada portal.
    Solo muestra portales con >= 20 titulares para evitar ruido estadístico.
    """
    MIN_TITULARES = 20

    stats = (
        df.groupby("portal")
        .agg(total=("titulo", "count"), clickbait=("es_clickbait", "sum"))
        .query(f"total >= {MIN_TITULARES}")
        .assign(pct_clickbait=lambda x: x["clickbait"] / x["total"] * 100)
        .sort_values("pct_clickbait", ascending=False)
        .head(25)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(
        [short_portal(p) for p in stats.index[::-1]],
        stats["pct_clickbait"].values[::-1],
        color=[ACCENT if v >= 50 else NEUTRAL for v in stats["pct_clickbait"].values[::-1]],
        edgecolor="white", linewidth=0.5,
    )
    ax.axvline(50, color="#888", linestyle="--", linewidth=1, alpha=0.7, label="50%")
    ax.set_xlabel("% de titulares clasificados como clickbait")
    ax.set_title("Índice de clickbait por portal\n(portales con ≥ 20 titulares, top 25)")
    ax.legend(fontsize=9)

    # Etiquetas de valor
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", ha="left", fontsize=8.5, color="#333")

    fig.tight_layout()
    save_fig(fig, os.path.join(out, "1_clickbait_por_portal.png"))
    return stats

#  ANÁLISIS 2: Chile vs Internacional
def analisis_2_origen(df: pd.DataFrame, out: str):
    """
    Compara la distribución de etiquetas entre origen nacional e internacional.
    """
    etiquetas = ["informativo", "clickbait", "fake_news", "posible_clickbait"]
    etiquetas = [e for e in etiquetas if e in df["etiqueta_final"].unique()]

    pivot = (
        df.groupby(["origen", "etiqueta_final"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=etiquetas, fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: barras apiladas (%)
    colors = [PALETTE.get(e, "#AAAAAA") for e in pivot_pct.columns]
    pivot_pct.plot(kind="bar", stacked=True, ax=axes[0], color=colors,
                   edgecolor="white", linewidth=0.5, width=0.55)
    axes[0].set_title("Distribución de etiquetas por origen (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Porcentaje (%)")
    axes[0].set_xticklabels(pivot_pct.index, rotation=0, fontsize=11)
    axes[0].legend(title="Etiqueta", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    for i, (_, row) in enumerate(pivot_pct.iterrows()):
        cumulative = 0
        for j, v in enumerate(row):
            if v > 5:
                axes[0].text(i, cumulative + v / 2, f"{v:.1f}%",
                             ha="center", va="center", fontsize=8.5,
                             color="white", fontweight="bold")
            cumulative += v

    # Subplot 2: % clickbait puro como dot-plot
    cb_pct = pivot_pct.get("clickbait", pd.Series(dtype=float)).reset_index()
    cb_pct.columns = ["origen", "pct_cb"]
    sc_colors = [PALETTE["clickbait"] if o == "nacional" else PALETTE["informativo"]
                 for o in cb_pct["origen"]]
    axes[1].barh(cb_pct["origen"], cb_pct["pct_cb"], color=sc_colors,
                 height=0.45, edgecolor="white")
    axes[1].set_title("% clickbait: Chile vs Internacional")
    axes[1].set_xlabel("% de titulares clickbait")
    for i, v in enumerate(cb_pct["pct_cb"]):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10.5)

    fig.suptitle("Comparación Chile vs Internacional", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "2_origen_clickbait.png"))
    return pivot_pct

#  ANÁLISIS 3: Volumen vs % clickbait por portal
def analisis_3_volumen_vs_clickbait(df: pd.DataFrame, out: str):
    """
    Scatter: eje X = total de titulares, eje Y = % clickbait.
    Tamaño del punto = volumen. Color = origen.
    """
    MIN_TITULARES = 15
    stats = (
        df.groupby(["portal", "origen"])
        .agg(total=("titulo", "count"), clickbait=("es_clickbait", "sum"))
        .reset_index()
        .query(f"total >= {MIN_TITULARES}")
        .assign(pct_cb=lambda x: x["clickbait"] / x["total"] * 100)
    )

    color_map = {"nacional": ACCENT, "internacional": NEUTRAL}
    fig, ax = plt.subplots(figsize=(12, 7))

    for origen, grp in stats.groupby("origen"):
        ax.scatter(
            grp["total"], grp["pct_cb"],
            s=grp["total"] * 1.5,
            c=color_map.get(origen, "#999"),
            alpha=0.7, edgecolors="white", linewidth=0.8,
            label=origen.capitalize(),
        )
        # Etiquetar portales con > 50% clickbait o > 100 titulares
        for _, row in grp.iterrows():
            if row["pct_cb"] > 50 or row["total"] > 100:
                ax.annotate(
                    short_portal(row["portal"], 18),
                    xy=(row["total"], row["pct_cb"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=7.5, color="#333",
                )

    ax.axhline(50, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Número de titulares en el dataset")
    ax.set_ylabel("% clickbait")
    ax.set_title("Volumen de titulares vs Índice de clickbait por portal")
    ax.legend(title="Origen", fontsize=10)
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "3_volumen_vs_clickbait.png"))
    return stats

#  ANÁLISIS 4: Evolución temporal del clickbait
def analisis_4_evolucion_temporal(df: pd.DataFrame, out: str):
    """
    Evolución mensual del % de clickbait sobre el total de artículos con fecha.
    """
    df_fecha = df.dropna(subset=["fecha_dt"]).copy()
    
    # Filtrar fechas anteriores a 2012 (basado en auditoría de datos)
    df_fecha = df_fecha[df_fecha["fecha_dt"].dt.year >= 2012]

    if len(df_fecha) < 30:
        print("  [!] Análisis 4 omitido: menos de 30 titulares con fecha válida.")
        return None

    df_fecha["mes"] = df_fecha["fecha_dt"].dt.to_period("M")

    monthly = (
        df_fecha.groupby("mes")
        .agg(total=("titulo", "count"), clickbait=("es_clickbait", "sum"))
        .reset_index()
        .assign(pct_cb=lambda x: x["clickbait"] / x["total"] * 100)
    )
    monthly["mes_dt"] = monthly["mes"].dt.to_timestamp()
    monthly = monthly.sort_values("mes_dt")

    # Suavizado con media móvil de 3 meses
    monthly["pct_cb_ma3"] = monthly["pct_cb"].rolling(3, center=True, min_periods=1).mean()

    # Calcular intervalo de ticks según rango de fechas
    n_meses = len(monthly)
    if n_meses <= 24:
        tick_interval  = 1          # cada mes
        date_fmt       = "%b %Y"
        rotation       = 45
    elif n_meses <= 60:
        tick_interval  = 3          # cada trimestre
        date_fmt       = "%b %Y"
        rotation       = 45
    elif n_meses <= 120:
        tick_interval  = 6          # cada semestre
        date_fmt       = "%b\n%Y"
        rotation       = 0
    else:
        tick_interval  = 12         # cada año
        date_fmt       = "%Y"
        rotation       = 0

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.fill_between(monthly["mes_dt"], monthly["pct_cb"],
                     alpha=0.2, color=ACCENT, label="% clickbait (mensual)")
    ax1.plot(monthly["mes_dt"], monthly["pct_cb_ma3"],
             color=ACCENT, linewidth=2.2, label="Media móvil 3 meses")
    ax1.set_ylabel("% clickbait", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax1.set_ylim(0, 105)

    ax2.bar(monthly["mes_dt"], monthly["total"],
            width=20, alpha=0.25, color=NEUTRAL, label="Nº titulares")
    ax2.set_ylabel("Nº titulares con fecha", color=NEUTRAL)
    ax2.tick_params(axis="y", labelcolor=NEUTRAL)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

    plt.setp(ax1.get_xticklabels(),
             rotation=rotation,
             ha="right" if rotation > 0 else "center",
             fontsize=9)

    # Ajustar límites X para no dejar espacio muerto
    x_min = monthly["mes_dt"].min() - pd.Timedelta(days=15)
    x_max = monthly["mes_dt"].max() + pd.Timedelta(days=15)
    ax1.set_xlim(x_min, x_max)

    ax1.set_xlabel("Año", labelpad=10)
    ax1.set_title("Evolución mensual del clickbait en el tiempo", pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    save_fig(fig, os.path.join(out, "4_evolucion_temporal.png"))
    return monthly

#  ANÁLISIS 5: cb_heuristic vs etiqueta_final (aciertos y errores)
def analisis_5_heuristic_vs_etiqueta(df: pd.DataFrame, out: str):
    """
    Evalúa qué tan bien el puntaje heurístico se alinea con la etiqueta final.
    Genera: distribución de scores por clase + matriz de confusión binarizada.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: distribución del score heurístico por clase
    etiquetas_ord = [e for e in ["informativo", "clickbait", "fake_news", "posible_clickbait"]
                     if e in df["etiqueta_final"].unique()]
    for etq in etiquetas_ord:
        subset = df[df["etiqueta_final"] == etq]["cb_heuristic"]
        subset.plot.kde(ax=axes[0], label=etq.capitalize(),
                        color=PALETTE.get(etq, "#999"), linewidth=2)

    axes[0].axvline(0.25, color="#888", linestyle="--", linewidth=1,
                    label="Umbral 0.25")
    axes[0].set_xlabel("Score heurístico de clickbait")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title("Distribución del score heurístico\npor clase (KDE)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-0.05, 1.05)

    # Subplot 2: matriz confusión binaria
    UMBRAL = 0.25
    df_eval = df[df["etiqueta_final"].isin(["informativo", "clickbait"])].copy()
    df_eval["pred_cb"]  = (df_eval["cb_heuristic"] >= UMBRAL).astype(int)
    df_eval["real_cb"]  = (df_eval["etiqueta_final"] == "clickbait").astype(int)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(df_eval["real_cb"], df_eval["pred_cb"])
    labels_cm = ["Informativo\n(Real)", "Clickbait\n(Real)"]
    col_labels = ["Informativo\n(Pred)", "Clickbait\n(Pred)"]

    total = cm.sum()
    cm_pct = cm / total * 100
    annot  = np.array([[f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)])

    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="RdBu_r",
                xticklabels=col_labels, yticklabels=labels_cm,
                ax=axes[1], linewidths=1, linecolor="white",
                vmin=0, vmax=cm_pct.max(),
                cbar_kws={"label": "% del total"})
    axes[1].set_title(f"Heurístico vs Etiqueta real\n(umbral = {UMBRAL})")

    report = classification_report(df_eval["real_cb"], df_eval["pred_cb"],
                                   target_names=["informativo", "clickbait"],
                                   output_dict=True)
    axes[1].set_xlabel(
        f"Precisión: {report['clickbait']['precision']:.2f} | "
        f"Recall: {report['clickbait']['recall']:.2f} | "
        f"F1: {report['clickbait']['f1-score']:.2f}",
        fontsize=9
    )

    fig.suptitle("Análisis heurístico: ¿Detecta bien el clickbait?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "5_heuristic_vs_etiqueta.png"))
    return report

#  ANÁLISIS 6: Clickbait según método de obtención
def analisis_6_metodo_obtencion(df: pd.DataFrame, out: str):
    """
    Compara el % de clickbait y la distribución de etiquetas
    según el método de obtención del titular.
    """
    if "metodo_obtencion" not in df.columns:
        print("  [!] Análisis 6 omitido: columna 'metodo_obtencion' no encontrada.")
        return None

    stats = (
        df.groupby(["metodo_obtencion", "etiqueta_final"])
        .size()
        .unstack(fill_value=0)
    )
    stats_pct = stats.div(stats.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Barras apiladas
    etiquetas = [e for e in ["informativo", "clickbait", "fake_news", "posible_clickbait"]
                 if e in stats_pct.columns]
    colors = [PALETTE.get(e, "#999") for e in etiquetas]
    stats_pct[etiquetas].plot(kind="bar", stacked=True, ax=axes[0],
                               color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    axes[0].set_title("Distribución de etiquetas\npor método de obtención (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Porcentaje (%)")
    axes[0].set_xticklabels(stats_pct.index, rotation=20, ha="right")
    axes[0].legend(title="Etiqueta", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    # Score heurístico promedio por método
    heuristic_mean = df.groupby("metodo_obtencion")["cb_heuristic"].mean().sort_values(ascending=False)
    axes[1].bar(heuristic_mean.index, heuristic_mean.values, color=ACCENT, alpha=0.85,
                edgecolor="white")
    axes[1].set_title("Score heurístico promedio\npor método de obtención")
    axes[1].set_ylabel("cb_heuristic (promedio)")
    axes[1].set_xticklabels(heuristic_mean.index, rotation=20, ha="right")
    for i, v in enumerate(heuristic_mean.values):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Clickbait según método de obtención del dato",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "6_metodo_obtencion.png"))
    return stats_pct


#  ANÁLISIS 7: Longitud de titulares por clase
def analisis_7_longitud_titulares(df: pd.DataFrame, out: str):
    """
    Boxplot + violinplot de largo (palabras y caracteres) por etiqueta.
    """
    etiquetas = [e for e in ["informativo", "clickbait", "fake_news"]
                 if e in df["etiqueta_final"].unique()]
    colors = [PALETTE.get(e, "#999") for e in etiquetas]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Boxplot caracteres
    data_chars = [df[df["etiqueta_final"] == e]["n_chars"].dropna() for e in etiquetas]
    bp = axes[0, 0].boxplot(data_chars, patch_artist=True,
                             medianprops=dict(color="white", linewidth=2),
                             whiskerprops=dict(linewidth=1.2),
                             capprops=dict(linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0, 0].set_xticklabels([e.capitalize() for e in etiquetas])
    axes[0, 0].set_title("Longitud en caracteres por clase")
    axes[0, 0].set_ylabel("Número de caracteres")

    # 2. Boxplot palabras
    data_words = [df[df["etiqueta_final"] == e]["n_words"].dropna() for e in etiquetas]
    bp2 = axes[0, 1].boxplot(data_words, patch_artist=True,
                              medianprops=dict(color="white", linewidth=2),
                              whiskerprops=dict(linewidth=1.2),
                              capprops=dict(linewidth=1.2))
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0, 1].set_xticklabels([e.capitalize() for e in etiquetas])
    axes[0, 1].set_title("Longitud en palabras por clase")
    axes[0, 1].set_ylabel("Número de palabras")

    # 3. Violinplot caracteres
    df_plot = df[df["etiqueta_final"].isin(etiquetas)].copy()
    sns.violinplot(data=df_plot, x="etiqueta_final", y="n_chars",
                   order=etiquetas, palette=PALETTE,
                   ax=axes[1, 0], inner="quartile", linewidth=1)
    axes[1, 0].set_title("Distribución caracteres (violinplot)")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Número de caracteres")
    axes[1, 0].set_xticklabels([e.capitalize() for e in etiquetas])

    # 4. Histograma solapado palabras
    for etq, col in zip(etiquetas, colors):
        subset = df[df["etiqueta_final"] == etq]["n_words"].dropna()
        axes[1, 1].hist(subset, bins=30, alpha=0.55, color=col,
                        label=etq.capitalize(), edgecolor="white", linewidth=0.3)
    axes[1, 1].set_title("Distribución palabras (histograma)")
    axes[1, 1].set_xlabel("Número de palabras")
    axes[1, 1].set_ylabel("Frecuencia")
    axes[1, 1].legend(fontsize=9)

    # Estadísticas en consola
    print("\n  Longitud media por clase:")
    for etq in etiquetas:
        sub = df[df["etiqueta_final"] == etq]
        print(f"    {etq:<15} chars: {sub['n_chars'].mean():.1f} | "
              f"words: {sub['n_words'].mean():.1f}")

    fig.suptitle("Longitud de los titulares por clase", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "7_longitud_titulares.png"))

#  ANÁLISIS 8: Uso de "!" y "?" por clase
def analisis_8_signos_puntuacion(df: pd.DataFrame, out: str):
    """
    % de titulares con "!" y "?" por clase + intensidad (recuento promedio).
    """
    etiquetas = [e for e in ["informativo", "clickbait", "fake_news"]
                 if e in df["etiqueta_final"].unique()]

    # Recuento de signos
    df = df.copy()
    df["n_excl"]  = df["titulo"].str.count(r"!")
    df["n_qmark"] = df["titulo"].str.count(r"\?")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. % titulares con al menos un "!"
    pct_excl = [df[df["etiqueta_final"] == e]["has_excl"].mean() * 100 for e in etiquetas]
    axes[0, 0].bar([e.capitalize() for e in etiquetas], pct_excl,
                   color=[PALETTE.get(e, "#999") for e in etiquetas],
                   edgecolor="white", alpha=0.85)
    axes[0, 0].set_title('% de titulares con "!"')
    axes[0, 0].set_ylabel("%")
    for i, v in enumerate(pct_excl):
        axes[0, 0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

    # 2. % titulares con al menos un "?"
    pct_qmark = [df[df["etiqueta_final"] == e]["has_qmark"].mean() * 100 for e in etiquetas]
    axes[0, 1].bar([e.capitalize() for e in etiquetas], pct_qmark,
                   color=[PALETTE.get(e, "#999") for e in etiquetas],
                   edgecolor="white", alpha=0.85)
    axes[0, 1].set_title('% de titulares con "?"')
    axes[0, 1].set_ylabel("%")
    for i, v in enumerate(pct_qmark):
        axes[0, 1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

    # 3. Recuento promedio de "!" por titular
    avg_excl = [df[df["etiqueta_final"] == e]["n_excl"].mean() for e in etiquetas]
    axes[1, 0].bar([e.capitalize() for e in etiquetas], avg_excl,
                   color=[PALETTE.get(e, "#999") for e in etiquetas],
                   edgecolor="white", alpha=0.85)
    axes[1, 0].set_title('Promedio de "!" por titular')
    axes[1, 0].set_ylabel("Recuento promedio")
    for i, v in enumerate(avg_excl):
        axes[1, 0].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)

    # 4. Recuento promedio de "?" por titular
    avg_qmark = [df[df["etiqueta_final"] == e]["n_qmark"].mean() for e in etiquetas]
    axes[1, 1].bar([e.capitalize() for e in etiquetas], avg_qmark,
                   color=[PALETTE.get(e, "#999") for e in etiquetas],
                   edgecolor="white", alpha=0.85)
    axes[1, 1].set_title('Promedio de "?" por titular')
    axes[1, 1].set_ylabel("Recuento promedio")
    for i, v in enumerate(avg_qmark):
        axes[1, 1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle('Uso de signos "!" y "?" en los titulares',
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "8_signos_puntuacion.png"))

#  ANÁLISIS 9: Palabras más frecuentes en clickbait
def analisis_9_palabras_frecuentes(df: pd.DataFrame, out: str):
    """
    Top-30 palabras más frecuentes en titulares clickbait vs informativos.
    Si wordcloud está instalado, también genera una nube de palabras.
    """
    def tokenize(texts: pd.Series) -> Counter:
        words = []
        for t in texts.dropna():
            tokens = re.findall(r"\b[a-záéíóúüñ]{3,}\b", t.lower())
            words.extend([w for w in tokens if w not in STOPWORDS_ES])
        return Counter(words)

    cb_texts   = df[df["etiqueta_final"] == "clickbait"]["titulo"]
    info_texts = df[df["etiqueta_final"] == "informativo"]["titulo"]
    cb_freq    = tokenize(cb_texts)
    info_freq  = tokenize(info_texts)

    top_cb   = pd.DataFrame(cb_freq.most_common(30),   columns=["palabra", "frecuencia"])
    top_info = pd.DataFrame(info_freq.most_common(30), columns=["palabra", "frecuencia"])

    n_plots = 3 if HAS_WORDCLOUD else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7))

    # Barplot clickbait
    axes[0].barh(top_cb["palabra"][::-1], top_cb["frecuencia"][::-1],
                 color=ACCENT, alpha=0.85, edgecolor="white")
    axes[0].set_title("Top 30 palabras en titulares\nClickbait")
    axes[0].set_xlabel("Frecuencia")

    # Barplot informativo
    axes[1].barh(top_info["palabra"][::-1], top_info["frecuencia"][::-1],
                 color=NEUTRAL, alpha=0.85, edgecolor="white")
    axes[1].set_title("Top 30 palabras en titulares\nInformativo")
    axes[1].set_xlabel("Frecuencia")

    # WordCloud si está disponible
    if HAS_WORDCLOUD and n_plots == 3:
        wc_text = " ".join(cb_texts.dropna())
        wc = WordCloud(
            width=900, height=600,
            background_color="white",
            colormap="Reds",
            stopwords=STOPWORDS_ES,
            max_words=120,
            prefer_horizontal=0.9,
        ).generate(wc_text)
        axes[2].imshow(wc, interpolation="bilinear")
        axes[2].axis("off")
        axes[2].set_title("Nube de palabras — Clickbait")

    fig.suptitle("Palabras más frecuentes según clase",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "9_palabras_frecuentes.png"))
    return top_cb, top_info

#  ANÁLISIS 10: Portales con mayor concentración de clickbait
def analisis_10_concentracion_clickbait(df: pd.DataFrame, out: str):
    """
    Responde: ¿qué portales concentran la mayor PROPORCIÓN del total de
    titulares clickbait en toda la muestra? (no solo % dentro del portal)
    Incluye comparativa nacional vs internacional.
    """
    cb_df = df[df["etiqueta_final"] == "clickbait"]
    total_cb = len(cb_df)

    conc = (
        cb_df.groupby(["portal", "origen"])
        .size()
        .reset_index(name="n_cb")
        .assign(pct_del_total=lambda x: x["n_cb"] / total_cb * 100)
        .sort_values("pct_del_total", ascending=False)
        .head(20)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Barras por portal
    col_map = {"nacional": ACCENT, "internacional": NEUTRAL}
    colors = [col_map.get(o, "#999") for o in conc["origen"]]
    bars = axes[0].barh(
        [short_portal(p) for p in conc["portal"]][::-1],
        conc["pct_del_total"].values[::-1],
        color=colors[::-1], edgecolor="white", alpha=0.9,
    )
    axes[0].set_title(f"Portales con mayor concentración de clickbait\n"
                      f"(top 20, total clickbait = {total_cb:,})")
    axes[0].set_xlabel("% del total de titulares clickbait")

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ACCENT, label="Nacional"),
                       Patch(facecolor=NEUTRAL, label="Internacional")]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=9)

    for bar in bars:
        w = bar.get_width()
        axes[0].text(w + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{w:.1f}%", va="center", ha="left", fontsize=8)

    # Pie chart nacional vs internacional dentro del clickbait
    origen_cb = cb_df["origen"].value_counts()
    axes[1].pie(
        origen_cb.values,
        labels=[f"{o.capitalize()}\n({v:,} titulares)"
                for o, v in zip(origen_cb.index, origen_cb.values)],
        colors=[col_map.get(o, "#999") for o in origen_cb.index],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    axes[1].set_title("Distribución del clickbait\nNacional vs Internacional")

    fig.suptitle("Concentración del clickbait en la muestra",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(out, "10_concentracion_clickbait.png"))
    return conc

def print_summary(df: pd.DataFrame):
    print("  RESUMEN EJECUTIVO DEL DATASET")
    print(f"  Total titulares     : {len(df):,}")
    print(f"  Columnas            : {list(df.columns)}")
    print(f"\n  Distribución clases :")
    for etq, cnt in df["etiqueta_final"].value_counts().items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {etq:<20} {cnt:>6,}  ({pct:.1f}%)  {bar}")

    print(f"\n  Origen :")
    for orig, cnt in df["origen"].value_counts().items():
        print(f"    {orig:<20} {cnt:>6,}")

    n_con_fecha = df["fecha_dt"].notna().sum()
    print(f"\n  Titulares con fecha : {n_con_fecha:,} ({n_con_fecha/len(df)*100:.1f}%)")
    print(f"  Longitud media (chars): {df['n_chars'].mean():.1f}")
    print(f"  Longitud media (words): {df['n_words'].mean():.1f}")

def main():
    parser = argparse.ArgumentParser(description="EDA dataset clickbait tarea 2")
    parser.add_argument("--csv", default="dataset_output/dataset_etiquetado_v2.csv",
                        help="Ruta al CSV del dataset etiquetado")
    parser.add_argument("--out", default="eda_output",
                        help="Carpeta donde guardar los gráficos")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"  EDA — Dataset Clickbait Tarea 2")
    print(f"  CSV : {args.csv}")
    print(f"  OUT : {args.out}/")

    df = load_dataset(args.csv)
    print_summary(df)

    analyses = [
        ("1  Índice clickbait por portal",          lambda: analisis_1_clickbait_por_portal(df, args.out)),
        ("2  Chile vs Internacional",               lambda: analisis_2_origen(df, args.out)),
        ("3  Volumen vs % clickbait",               lambda: analisis_3_volumen_vs_clickbait(df, args.out)),
        ("4  Evolución temporal",                   lambda: analisis_4_evolucion_temporal(df, args.out)),
        ("5  Heurístico vs etiqueta",               lambda: analisis_5_heuristic_vs_etiqueta(df, args.out)),
        ("6  Clickbait por método obtención",       lambda: analisis_6_metodo_obtencion(df, args.out)),
        ("7  Longitud de titulares",                lambda: analisis_7_longitud_titulares(df, args.out)),
        ('8  Uso de "!" y "?"',                     lambda: analisis_8_signos_puntuacion(df, args.out)),
        ("9  Palabras más frecuentes",              lambda: analisis_9_palabras_frecuentes(df, args.out)),
        ("10 Concentración del clickbait",          lambda: analisis_10_concentracion_clickbait(df, args.out)),
    ]

    for name, fn in analyses:
        print(f"\n[→] Análisis {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  [✗] Error: {e}")

    print(f"    EDA completado. {len(analyses)} gráficos guardados en: {args.out}/")
    print("  Archivos generados:")
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".png"):
            print(f"     {f}")


if __name__ == "__main__":
    main()