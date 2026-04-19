"""
NLP & XAI: Detección de Clickbait en Prensa Chilena
"""
import time, random, logging, re, os
from datetime import datetime
from typing import Optional
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.8",
}
REQUEST_DELAY = (1.5, 3.0)
MAX_RETRIES   = 3
TIMEOUT       = 20
OUTPUT_DIR    = "dataset_output"
RAW_CSV       = os.path.join(OUTPUT_DIR, "dataset_raw_v2.csv")
FINAL_CSV     = os.path.join(OUTPUT_DIR, "dataset_etiquetado_v2.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PER_CLASS = 1100
CLICKBAIT_PATTERNS = [
    r"\bno (te|vas a|podrás)\b.*\bcreer\b",
    r"\b(sorprende(rá|rás|nte)|impresionante|increíble|brutal|viral|impactante)\b",
    r"\besto es lo que\b",
    r"\bnadie (te|lo|les) (dijo|contó|esperaba)\b",
    r"\b(te|tus|tu)\b.{0,20}\b(debes|tienes que|necesitas)\b",
    r"\blo que (necesitas|debes|tienes que) saber\b",
    r"\b\d+\s+(razones|cosas|formas|tips|secretos|trucos|fotos|imágenes)\b",
    r"\?$",
    r"\b¿(sabías|sabes|conoces|adivinas)\b",
    r"\b(el mejor|el peor|el más|la más|jamás|nunca antes|histórico)\b",
    r"\b(reaccionó|respondió|explotó|lloró|confesó|reveló|admitió)\b",
    r"\b(así|de esta manera|de este modo)\b.{0,30}\b(sorprend|impresion|viral)\b",
    r"!\s*$",
    r"\b(mira|descubre|entérate|conoce)\s+(cómo|qué|quién|cuándo|dónde)\b",
    r"\b(esto|lo que|lo que pasó|lo ocurrido)\b.{0,20}\b(dejó|dejará)\b",
]
CLICKBAIT_RE = [re.compile(p, re.IGNORECASE) for p in CLICKBAIT_PATTERNS]
def clickbait_score(title: str) -> float:
    if not title:
        return 0.0
    hits = sum(1 for p in CLICKBAIT_RE if p.search(title))
    return round(min(hits / 4.0, 1.0), 3)

#  ESTRATEGIA: GOOGLE NEWS RSS
#  Google News actúa como proxy: devuelve titulares de cualquier
#  portal sin requerir acceso directo al sitio. Permite filtrar
#  por site:, por tema y por idioma. No requiere API key.
def gnews_url(query: str, lang: str = "es-419", country: str = "CL") -> str:
    """Construye URL de Google News RSS para una búsqueda."""
    q = requests.utils.quote(query)
    ceid = f"{country}:{lang.split('-')[0]}"
    return (
        f"https://news.google.com/rss/search"
        f"?q={q}&hl={lang}&gl={country}&ceid={ceid}"
    )
def gnews_topic_url(topic: str) -> str:
    """URLs de secciones temáticas de Google News (mayor volumen)."""
    topics = {
        "headlines_cl":   "https://news.google.com/rss?hl=es-419&gl=CL&ceid=CL:es-419",
        "headlines_es":   "https://news.google.com/rss?hl=es&gl=ES&ceid=ES:es",
        "headlines_ar":   "https://news.google.com/rss?hl=es-419&gl=AR&ceid=AR:es-419",
        "headlines_mx":   "https://news.google.com/rss?hl=es-419&gl=MX&ceid=MX:es-419",
        "headlines_us_es":"https://news.google.com/rss?hl=es-419&gl=US&ceid=US:es-419",
    }
    return topics.get(topic, "")

# NACIONALES INFORMATIVOS — portales serios chilenos
NATIONAL_INFORMATIVE_QUERIES = [
    'site:latercera.com',
    'site:emol.com',
    'site:cooperativa.cl',
    'site:biobiochile.cl',
    'site:24horas.cl',
    'site:cnnchile.com',
    'site:elmostrador.cl',
    'site:radioagricultura.cl',
    'site:df.cl', # Diario Financiero
    'site:eldesconcierto.cl',
    'site:chilevision.cl',
    'economia Chile presupuesto',
    'politica Chile gobierno ministerio',
    'Chile salud Minsal',
    'Chile educacion Mineduc',
    'Chile tribunal justicia sentencia',
    'Chile congreso proyecto ley',
    'Chile Banco Central inflacion',
    'terremoto Chile sismo',
    'Chile cancilleria relaciones exteriores',
]

# NACIONALES CLICKBAIT — portales de entretenimiento/sensacionalismo
NATIONAL_CLICKBAIT_QUERIES = [
    'site:publimetro.cl',
    'site:meganoticias.cl',
    'site:lun.com',
    'site:redgol.cl',
    'site:eldinamo.cl',
    'site:ahoranoticias.cl',
    'site:soychile.cl',
    'site:fotech.cl',
    'site:glamorama.cl',        # suplemento lifestyle de La Tercera
    'site:t13.cl entretenimiento',
    # queries que atraen contenido clickbait chileno
    'viral impactante Chile famoso',
    'Chile famoso sorprendió impactante reacción',
    'Chile tiktoker youtuber viral',
    'Chile farándula impactó sorprendió',
    'Chile deportes gol increíble',
    'Chile revelación confesó lloró',
    'Chile misterio insólito curioso',
    'Chile terror pánico susto viral',
    'Chile descuento oferta ahorro truco',
    'Chile receta truco secreto increíble',
]

# INTERNACIONALES INFORMATIVOS
INTERNATIONAL_INFORMATIVE_QUERIES = [
    'site:bbc.com/mundo',
    'site:france24.com/es',
    'site:dw.com/es',
    'site:reuters.com español',
    'site:apnews.com',
    'site:elpais.com',
    'site:lavanguardia.com',
    'site:efe.com',
    'site:euronews.com/es',
    'site:nytimes.com español',
    'economia global banco mundial FMI',
    'conflicto internacional ONU',
    'cambio climatico COP acuerdo',
    'elecciones democracia internacional',
    'ciencia descubrimiento estudio investigacion',
    'tecnologia inteligencia artificial innovacion',
    'salud OMS pandemia vacuna',
    'derechos humanos amnistia',
]

INTERNATIONAL_CLICKBAIT_QUERIES = [
    'site:infobae.com',
    'site:20minutos.es',
    'site:marca.com',
    'site:muyinteresante.es',
    'site:elconfidencial.com',
    'site:sport.es',
    'site:as.com',
    'site:antena3.com',
    'site:lasexta.com',
    'viral famoso sorprendió impactante reacción',
    'increíble descubrimiento insólito viral mundo',
    'nunca antes visto sorprendente impresionante',
    'reveló confesó secreto impactante famoso',
    'viral tiktoker youtuber challenge récord',
    'fotos impactantes increíbles famoso estrella',
    'razones secretos trucos impresionante vida',
    'misterio inexplicable increíble descubren',
]

def scrape_gnews_queries(
    queries: list[str],
    label: str,
    origin: str,
    target: int,
    lang: str = "es-419",
    country: str = "CL",
) -> list[dict]:
    """
    Itera sobre múltiples queries de Google News RSS hasta alcanzar `target`.
    """
    records: list[dict] = []
    seen_titles: set[str] = set()

    for query in queries:
        if len(records) >= target:
            break

        url = gnews_url(query, lang=lang, country=country)
        try:
            feed = feedparser.parse(url)
            entries = feed.entries
        except Exception as e:
            log.warning(f"[GNews] Error en query '{query}': {e}")
            time.sleep(random.uniform(*REQUEST_DELAY))
            continue

        if not entries:
            log.warning(f"[GNews] Sin resultados para: {query}")
            time.sleep(random.uniform(*REQUEST_DELAY))
            continue

        new_count = 0
        for entry in entries:
            title = entry.get("title", "").strip()
            # Google News a veces incluye " - Portal" al final; limpiarlo
            title = re.sub(r'\s+[-–]\s+[\w\s\.]+$', '', title).strip()
            # Extraer portal desde la fuente del feed
            source = entry.get("source", {}).get("title", "")
            link   = entry.get("link", "")
            pub    = entry.get("published", "")

            if not title or len(title) < 12:
                continue
            title_key = title.lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            records.append({
                "titulo":            title,
                "url":               link,
                "fecha_publicacion": pub,
                "portal":            source or f"GNews:{query[:30]}",
                "origen":            origin,
                "etiqueta_base":     label,
                "cb_heuristic":      clickbait_score(title),
                "etiqueta_final":    label,
                "metodo_obtencion":  "gnews_rss",
            })
            new_count += 1

        log.info(f"[GNews] '{query[:45]}' → {new_count} nuevos (total: {len(records)})")
        time.sleep(random.uniform(*REQUEST_DELAY))

    # Topic feeds como complemento si faltan
    if len(records) < target:
        log.info(f"[GNews] Complementando con topic feeds ({len(records)}/{target})...")
        topic_map = {
            "nacional":       ["headlines_cl"],
            "internacional":  ["headlines_es", "headlines_ar", "headlines_mx", "headlines_us_es"],
        }
        for topic in topic_map.get(origin, []):
            if len(records) >= target:
                break
            url = gnews_topic_url(topic)
            if not url:
                continue
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if len(records) >= target:
                        break
                    title = re.sub(r'\s+[-–]\s+[\w\s\.]+$', '',
                                   entry.get("title", "").strip()).strip()
                    if not title or len(title) < 12:
                        continue
                    if title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())
                    source = entry.get("source", {}).get("title", "")
                    records.append({
                        "titulo":            title,
                        "url":               entry.get("link", ""),
                        "fecha_publicacion": entry.get("published", ""),
                        "portal":            source or topic,
                        "origen":            origin,
                        "etiqueta_base":     label,
                        "cb_heuristic":      clickbait_score(title),
                        "etiqueta_final":    label,
                        "metodo_obtencion":  "gnews_topic",
                    })
                log.info(f"[GNews topic:{topic}] total acumulado: {len(records)}")
                time.sleep(random.uniform(*REQUEST_DELAY))
            except Exception as e:
                log.warning(f"[GNews topic] Error: {e}")

    log.info(f"[GNews] Clase '{label}/{origin}': {len(records)} titulares recolectados.")
    return records[:target]


def load_fakenews(target: int = TARGET_PER_CLASS) -> list[dict]:
    """
    Combina 3 fuentes de fake news en español:
      1. HuggingFace: mariagrandury/fake_news_corpus_spanish (split "test", ~570 entradas)
      2. HuggingFace: GonzaloA/fake_news (en inglés — se usa el título como proxy
         para el bonus; puede traducirse o usarse directamente con etiqueta "fake_news_en")
      3. Google News RSS queries de portales de desinformación / fact-checking
    """
    records: list[dict] = []
    seen: set[str] = set()

    # Fuente 1: mariagrandury/fake_news_corpus_spanish
    try:
        from datasets import load_dataset
        log.info("[FakeNews] Cargando mariagrandury/fake_news_corpus_spanish...")
        # El dataset tiene splits: 'test' únicamente
        ds = load_dataset("mariagrandury/fake_news_corpus_spanish")
        # Iterar todos los splits disponibles
        for split_name in ds.keys():
            split = ds[split_name]
            for row in split:
                if len(records) >= target:
                    break
                # label 0 = fake en este dataset
                if row.get("label") not in (0, "0", "fake", False):
                    continue
                title = str(row.get("title") or row.get("headline") or "").strip()
                if not title or len(title) < 10 or title.lower() in seen:
                    continue
                seen.add(title.lower())
                records.append({
                    "titulo":            title,
                    "url":               str(row.get("url") or ""),
                    "fecha_publicacion": str(row.get("date") or ""),
                    "portal":            "fake_news_corpus_spanish (HF)",
                    "origen":            "internacional",
                    "etiqueta_base":     "fake_news",
                    "cb_heuristic":      clickbait_score(title),
                    "etiqueta_final":    "fake_news",
                    "metodo_obtencion":  "huggingface",
                })
        log.info(f"[FakeNews HF-1] {len(records)} titulares cargados.")
    except Exception as e:
        log.warning(f"[FakeNews HF-1] Error: {e}")

    # Fuente 2: liar dataset en español / FakeDeS via HF
    if len(records) < target:
        try:
            from datasets import load_dataset
            log.info("[FakeNews] Cargando GonzaloA/fake_news (inglés, como proxy)...")
            ds2 = load_dataset("GonzaloA/fake_news")
            for split_name in ds2.keys():
                split = ds2[split_name]
                for row in split:
                    if len(records) >= target:
                        break
                    # label 0 = fake
                    if row.get("label") not in (0, "0", "fake"):
                        continue
                    title = str(row.get("title") or "").strip()
                    if not title or len(title) < 10 or title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    records.append({
                        "titulo":            title,
                        "url":               "",
                        "fecha_publicacion": "",
                        "portal":            "GonzaloA/fake_news (HF-EN)",
                        "origen":            "internacional",
                        "etiqueta_base":     "fake_news",
                        "cb_heuristic":      clickbait_score(title),
                        "etiqueta_final":    "fake_news",
                        "metodo_obtencion":  "huggingface",
                    })
            log.info(f"[FakeNews HF-2] total acumulado: {len(records)}")
        except Exception as e:
            log.warning(f"[FakeNews HF-2] Error: {e}")

    # Fuente 3: Google News RSS — portales fact-checking y desinfo
    if len(records) < target:
        log.info("[FakeNews] Complementando con Google News RSS...")
        fakenews_queries = [
            'site:maldita.es',
            'site:newtral.es',
            'site:elordenmundial.com bulo falso',
            'site:chequeado.com falso',
            'site:factual.news falso',
            'bulo falso desinformacion viral España',
            'fake news desinformacion viral America Latina',
            'hoax falso verificado mentira viral',
            'conspiración teoría falsa viral rumor',
            'deepfake manipulado falso viral noticias',
            'infodemia desinformación salud mentira',
            'electoral fraude falso desinformación',
            'estafa viral fraude mentira falso',
        ]
        extra = scrape_gnews_queries(
            fakenews_queries, "fake_news", "internacional",
            target=target - len(records),
        )
        for rec in extra:
            t = rec["titulo"].lower()
            if t not in seen:
                seen.add(t)
                records.append(rec)
        log.info(f"[FakeNews GNews] total acumulado: {len(records)}")

    log.info(f"[FakeNews] TOTAL fake_news: {len(records)}")
    return records[:target]

def run_scraping(target: int = TARGET_PER_CLASS, include_fake_news: bool = True) -> pd.DataFrame:
    all_records: list[dict] = []
    print(" Prensa Nacional Informativa")
    all_records += scrape_gnews_queries(
        NATIONAL_INFORMATIVE_QUERIES, "informativo", "nacional", target
    )
    print(" Prensa Nacional Clickbait")
    all_records += scrape_gnews_queries(
        NATIONAL_CLICKBAIT_QUERIES, "clickbait", "nacional", target
    )
    print(" Prensa Internacional Informativa")
    all_records += scrape_gnews_queries(
        INTERNATIONAL_INFORMATIVE_QUERIES, "informativo", "internacional",
        target, lang="es-419", country="US",
    )
    print(" Prensa Internacional Clickbait")
    all_records += scrape_gnews_queries(
        INTERNATIONAL_CLICKBAIT_QUERIES, "clickbait", "internacional",
        target, lang="es-419", country="US",
    )

    if include_fake_news:
        print(" Fake News")
        all_records += load_fakenews(target)

    # ── Consolidar ────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["titulo"]).reset_index(drop=True)
    df = df[df["titulo"].str.len() >= 15].reset_index(drop=True)

    # Refinamiento heurístico: informativo con alto score → revisión
    mask = (df["cb_heuristic"] >= 0.5) & (df["etiqueta_base"] == "informativo")
    df.loc[mask, "etiqueta_final"] = "posible_clickbait"

    # Resumen
    print("\n" + "="*65)
    log.info(f"TOTAL titulares únicos: {len(df):,}")
    log.info("\n" + df["etiqueta_final"].value_counts().to_string())
    log.info(f"Nacionales     : {(df['origen']=='nacional').sum():,}")
    log.info(f"Internacionales: {(df['origen']=='internacional').sum():,}")
    print("="*65)
    return df


# ════════════════════════════════════════════════════════════════════
#  GUARDADO Y EDA RÁPIDO
# ════════════════════════════════════════════════════════════════════

def save_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Dataset raw → {RAW_CSV}")

    df_final = df[df["etiqueta_final"] != "posible_clickbait"].copy()
    df_final.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Dataset final → {FINAL_CSV}")
    log.info("\n" + df_final["etiqueta_final"].value_counts().to_string())
    return df_final


def quick_eda(df: pd.DataFrame):
    print("\n" + "="*65)
    print("EDA RÁPIDO")
    print("="*65)
    print(f"\nTotal titulares: {len(df):,}")
    print(f"\nDistribución de clases:\n{df['etiqueta_final'].value_counts().to_string()}")
    print(f"\nOrigen:\n{df['origen'].value_counts().to_string()}")

    # Top portales con más titulares
    print(f"\nTop 15 portales por volumen:")
    print(df['portal'].value_counts().head(15).to_string())

    # Portales con mayor score heurístico de clickbait
    cb = (df.groupby("portal")["cb_heuristic"].mean()
            .sort_values(ascending=False).head(10))
    print(f"\nTop 10 portales con mayor score heurístico clickbait (promedio):")
    print(cb.to_string())

    # Longitud promedio
    df = df.copy()
    df["len"] = df["titulo"].str.len()
    print(f"\nLongitud promedio titular por clase:")
    print(df.groupby("etiqueta_final")["len"].mean().round(1).to_string())


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Tarea 2 IAA — Dataset Scraper v2                           ║
║  Estrategia: Google News RSS (proxy anti-bloqueo)           ║
║  Clases: Informativo / Clickbait / Fake News (bonus)        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    df_raw   = run_scraping(target=TARGET_PER_CLASS, include_fake_news=True)
    df_final = save_dataset(df_raw)
    quick_eda(df_final)

    print(f"""
✅  Scraping completado.
    Raw    → {RAW_CSV}
    Final  → {FINAL_CSV}

PRÓXIMOS PASOS:
  1. Revisar manualmente los "posible_clickbait" del raw CSV
  2. Balancear clases si hay diferencia > 20% entre ellas
  3. Anotar muestra de verificación (~200 por clase, Cohen's Kappa)
  4. Ejecutar notebook de EDA avanzado (frecuencias, portales, autores)
""")