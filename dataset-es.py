"""
TAREA 2 - NLP & XAI: Detección de Clickbait en Prensa Chilena
Módulo de Obtención del Dataset — v2 (corregido)

CAMBIOS RESPECTO A v1:
  - RSS chilenos reemplazados por Google News RSS (proxy que evita bloqueos)
  - Paginación multi-query para alcanzar cuota de 1000 por clase
  - Fix HuggingFace: usa split correcto + múltiples datasets de FakeNews
  - URLs corregidas (Chequeado, Reuters, AP)
  - Estrategia de expansión: múltiples queries temáticas por portal

INSTALACIÓN (ejecutar una vez en Colab):
  !pip install requests feedparser beautifulsoup4 lxml datasets pandas tqdm
"""
import time, random, logging, re, os
from datetime import datetime
from typing import Optional

import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Configuración global
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

TARGET_PER_CLASS = 1100   # objetivo ligeramente mayor para compensar duplicados

#  HEURÍSTICO DE CLICKBAIT (Refinado para reducir falsos positivos y captar brechas estructurales)
CLICKBAIT_PATTERNS = [
    # 1. Brecha de Curiosidad (Curiosity Gap) - Ocultar el sujeto o el resultado
    r"^(Este|Esta|Esto|Estos|Estas)\s+(es\s+l[ao]s?|son\s+l[ao]s?)\b", # Este es el..., Esta es la...
    r"^(Este|Esta|Esto|Estos|Estas)\s+\w+.*\bque\b",                 # Esta innovación que..., Esta fruta que te...
    r"^(El|La|Lo|Los|Las)\s+.*\s+que\s+(no\s+sabías|debes\s+conocer|sorprende|te\s+cambiará)\b",
    r"\b(es lo que|es la|es el)\b",
    r"\b(lo que (pasó|ocurrió|sucedi|dejó))\b",
    r"\b(mira|descubre|entérate|conoce)\s+(cómo|qué|quién|cuándo|dónde)\b",
    r"\b(así|de esta manera)\b.*\b(quedó|reaccionó|luce|está)\b",
    r"^Por qué\b", # Por qué Chile tiene que...

    # 2. Sensacionalismo Vacío (Empty Sensationalism)
    r"\b(sorprende(rá|rás|nte)|impresionante|increíble|brutal|viral|impactante|insólito)\b",
    r"\b(no (te|vas a|podrás)\b.*\bcreer)\b",
    r"\b(quedarás (en shock|helado|sorprendido))\b",
    r"\b(el video que|la foto que)\b",

    # 3. Listicles (Estructura de lista con promesa de valor)
    r"^\d+\s+(razones|cosas|formas|tips|secretos|trucos|fotos|imágenes|pasos|claves)\b",

    # 4. Preguntas Retóricas o de Enganche
    r"¿\s*(sabías|sabes|conoces|adivinas|te imaginas|buscas|quieres)\b",
    r"¿\s*(quién|qué|cuál|cómo)\s+(es|será|pasará)\s*\?$",

    # 5. Personalización Forzada (Forced Personalization)
    r"\b(tú|te|tus|tu)\b.{0,20}\b(debes|tienes que|necesitas|podrás)\b",
    r"\b(nadie (te|lo|les) (dijo|contó|esperaba))\b",

    # 6. Reacciones Exageradas (Exaggerated Reactions)
    r"\b(reaccionó|explotó|lloró|confesó|reveló|admitió|se sinceró)\b",
    r"\b(estallan las redes|en llamas|causa furor)\b",
]
CLICKBAIT_RE = [re.compile(p, re.IGNORECASE) for p in CLICKBAIT_PATTERNS]

def clickbait_score(title: str) -> float:
    if not title:
        return 0.0
    hits = sum(1 for p in CLICKBAIT_RE if p.search(title))
    # Umbral más sensible: con 2 hits ya es sospechoso (0.66), con 3 es seguro (1.0)
    return round(min(hits / 3.0, 1.0), 3)

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

def extract_date_from_html(url: str) -> str:
    """
    Fallback: extrae la fecha de publicación directamente del HTML del artículo
    cuando el RSS no la provee. Busca en orden:
      1. Meta tags estándar (article:published_time, datePublished, etc.)
      2. JSON-LD structured data (schema.org/NewsArticle)
      3. Atributo datetime en tag <time>
    Retorna string ISO o "" si no encuentra nada.
    """
    if not url or not url.startswith("http"):
        return ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")

        # 1. Meta tags — cubren la mayoría de CMS de noticias
        meta_candidates = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name":     "article:published_time"}),
            ("meta", {"property": "og:article:published_time"}),
            ("meta", {"itemprop": "datePublished"}),
            ("meta", {"name":     "pubdate"}),
            ("meta", {"name":     "date"}),
            ("meta", {"name":     "DC.date.issued"}),
        ]
        for tag, attrs in meta_candidates:
            el = soup.find(tag, attrs)
            if el and el.get("content"):
                return el["content"].strip()

        # 2. JSON-LD (schema.org/NewsArticle) — estándar moderno
        import json
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                data = json.loads(script.string or "")
                items = data if isinstance(data, list) else [data]
                for item in items:
                    for key in ("datePublished", "dateCreated", "uploadDate"):
                        if item.get(key):
                            return str(item[key]).strip()
            except Exception:
                continue

        # 3. Tag <time> con atributo datetime
        time_tag = soup.find("time", {"datetime": True})
        if time_tag:
            return time_tag["datetime"].strip()

    except Exception:
        pass
    return ""

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
    'site:df.cl',              # Diario Financiero
    'site:eldesconcierto.cl',
    'site:chilevision.cl',
    # queries temáticos que traen prensa seria chilena
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

# INTERNACIONALES CLICKBAIT
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
            pub    = entry.get("published", "").strip()

            if not title or len(title) < 12:
                continue
            title_key = title.lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            # Fallback: si el RSS no trae fecha, intentar extraerla del HTML
            if not pub and link:
                pub = extract_date_from_html(link)

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
                    entry_link = entry.get("link", "")
                    entry_pub  = entry.get("published", "").strip()
                    if not entry_pub and entry_link:
                        entry_pub = extract_date_from_html(entry_link)
                    records.append({
                        "titulo":            title,
                        "url":               entry_link,
                        "fecha_publicacion": entry_pub,
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

#  BONUS: CLASE FAKE NEWS
#  Estrategia multi-fuente para garantizar >= 1000 titulares
def load_fakenews(target: int = TARGET_PER_CLASS) -> list[dict]:
    """
    Obtiene >= 1000 titulares de fake news **en español**, con representación
    tanto nacional (Chile) como internacional.

    Fuentes en orden de prioridad:
      1. HuggingFace: mariagrandury/fake_news_corpus_spanish (~570, español México)
      2. Google News RSS — portales de fact-checking hispanohablantes:
           • Nacionales CL: FigaroNewsCL, Puroperiodismo, LaBot.cl verificación
           • Internacionales ES: Maldita.es, Newtral.es, Chequeado.com
      3. Google News RSS — queries temáticos de desinformación en español

    NOTA: GonzaloA/fake_news (inglés) se EXCLUYE deliberadamente porque
    mezclar idiomas perjudica el fine-tuning de modelos en español.
    """
    records: list[dict] = []
    seen: set[str] = set()

    def add(title, url, portal, origen, fuente, fecha=""):
        k = title.lower().strip()
        if k in seen or len(title) < 10:
            return False
        seen.add(k)
        # Si no hay fecha y tenemos URL, intentar extraerla del HTML
        if not fecha and url and fuente == "gnews_rss":
            fecha = extract_date_from_html(url)
        records.append({
            "titulo":            title.strip(),
            "url":               url,
            "fecha_publicacion": fecha,
            "portal":            portal,
            "origen":            origen,
            "etiqueta_base":     "fake_news",
            "cb_heuristic":      clickbait_score(title),
            "etiqueta_final":    "fake_news",
            "metodo_obtencion":  fuente,
        })
        return True

    # Fuente 1: mariagrandury/fake_news_corpus_spanish (español)
    try:
        from datasets import load_dataset
        log.info("[FakeNews] Cargando mariagrandury/fake_news_corpus_spanish...")
        ds = load_dataset("mariagrandury/fake_news_corpus_spanish")
        for split_name in ds.keys():
            for row in ds[split_name]:
                if len(records) >= target:
                    break
                if row.get("label") not in (0, "0", "fake", False):
                    continue
                title = str(row.get("title") or row.get("headline") or "").strip()
                add(title, str(row.get("url") or ""),
                    "fake_news_corpus_spanish (HF)", "internacional", "huggingface",
                    fecha=str(row.get("date") or ""))
        log.info(f"[FakeNews HF] {len(records)} titulares en español cargados.")
    except Exception as e:
        log.warning(f"[FakeNews HF] Error: {e}")

    # Fuente 2: Google News RSS — fact-checking y desinformación
    # Organizados por origen para garantizar cobertura nacional e internacional
    if len(records) < target:
        log.info("[FakeNews] Complementando con Google News RSS (español)...")

        # Queries para fake news NACIONALES (Chile)
        fn_national_queries = [
            'site:labot.cl',                             # fact-checker chileno
            'site:puroperiodismo.cl verificacion',
            'desinformación Chile falso verificado',
            'Chile bulo viral falso noticias',
            'Chile rumor fake news desmentido',
            'Chile electoral fraude falso desinformacion',
            'Chile salud falso bulo viral desmentido',
            'Chile estafa viral fraude engaño',
            'Chile conspiracion falso teoria rumor',
            'Chile Covid desinformacion vacuna falso',
        ]
        extra_nat = scrape_gnews_queries(
            fn_national_queries, "fake_news", "nacional",
            target=min(400, target - len(records)),
            lang="es-419", country="CL",
        )
        for rec in extra_nat:
            add(rec["titulo"], rec["url"], rec["portal"], "nacional", "gnews_rss",
                fecha=rec.get("fecha_publicacion", ""))

        log.info(f"[FakeNews GNews-CL] total acumulado: {len(records)}")

        # Queries para fake news INTERNACIONALES (España, Latinoamérica)
        fn_intl_queries = [
            'site:maldita.es',
            'site:newtral.es',
            'site:chequeado.com falso',
            'site:colombiacheck.com falso',
            'site:afpfactual.com falso español',
            'site:pagina12.com.ar falso desmentido',
            'bulo falso desinformacion viral España',
            'fake news desinformacion America Latina español',
            'hoax falso verificado mentira viral español',
            'deepfake manipulado falso noticias español',
            'infodemia desinformación salud mentira español',
            'fraude electoral falso desinformación latinoamerica',
            'estafa viral fraude mentira falso español',
            'conspiración teoría falsa viral español',
            'desinformacion pandemia vacuna mentira español',
        ]
        extra_intl = scrape_gnews_queries(
            fn_intl_queries, "fake_news", "internacional",
            target=target - len(records),
            lang="es-419", country="US",
        )
        for rec in extra_intl:
            add(rec["titulo"], rec["url"], rec["portal"], "internacional", "gnews_rss",
                fecha=rec.get("fecha_publicacion", ""))

        log.info(f"[FakeNews GNews-INTL] total acumulado: {len(records)}")

    log.info(f"[FakeNews] TOTAL: {len(records)} (español únicamente)")
    return records[:target]

# ==============================================================================
# RÚBRICA DE ETIQUETADO — CRITERIOS DE VERDAD DE TERRENO
# ==============================================================================
# CLICKBAIT se etiqueta si:
#   1. Brecha de Curiosidad: Oculta deliberadamente el sujeto o el desenlace.
#   2. Lenguaje Emocional: Usa adjetivos extremos o promesas hiperbólicas.
#   3. Apelación Directa: Usa el "tú" o imperativos para forzar la acción.
#   4. Dependencia de Clic: El titular no se explica por sí solo.
#
# INFORMATIVO se etiqueta si:
#   1. Autocontenido: Entrega el hecho principal (Actor + Acción + Contexto).
#   2. Tono Neutro: Evita juicios de valor o lenguaje sensacionalista.
#   3. Especificidad: Incluye datos, nombres o lugares concretos.
# ==============================================================================

def apply_labeling_rubric(row: pd.Series) -> str:
    """
    Decide la etiqueta final basada en la rúbrica lingüística.
    Prioriza el contenido detectado (cb_heuristic) sobre la fuente (etiqueta_base).
    """
    title = str(row["titulo"]).lower()
    score = row["cb_heuristic"]
    base  = row["etiqueta_base"]

    if base == "fake_news":
        return "fake_news"

    # RESCATE DE NOTICIAS DE SERVICIO O CRÍTICAS (Hard News / Service News)
    # Palabras que suelen indicar contenido informativo serio aunque usen lenguaje llamativo
    hard_news_signals = [
        "fallece", "muere", "tragedia", "accidente", "homicidio", "detenido", "carabineros",
        "gobierno", "fiscalía", "presupuesto", "inflación", "censos", "clases", "escolar",
        "bono", "subsidio", "beneficio", "pago", "calendario", "postular", "requisitos",
        "oficial", "confirmado", "sentencia", "tribunal", "decreto", "ley", "clásico",
        "tiroteo", "amenaza", "robo", "delincuencia", "policía", "colegio", "universidad",
        "estudiante", "fallecido", "muertos", "heridos", "incendio", "sismo", "terremoto",
        "triunfo", "derrota", "partido", "gol", "fútbol", "tenis", "atletismo"
    ]
    # Frases de servicio directo
    service_phrases = ["cómo postular", "cómo obtener", "cuándo pagan", "fecha de pago", "dónde ver", "revisa el"]
    
    is_hard_or_service = any(word in title for word in hard_news_signals) or \
                         any(phrase in title for phrase in service_phrases)

    # Regla de rescate inmediata para noticias de servicio/tragedias con score bajo/medio
    if is_hard_or_service and score <= 0.4:
        return "informativo"

    # REGLA 1: Si el contenido es puramente informativo (score 0), 
    # lo movemos a informativo sin importar de qué query venga.
    if score == 0:
        return "informativo"

    # REGLA 2: Si tiene fuerte evidencia lingüística de clickbait (score >= 0.6)
    # lo movemos a clickbait sin importar el portal.
    if score >= 0.6:
        return "clickbait"

    # REGLA 3: Zona de conflicto (Portal dice una cosa, Heurístico otra)
    # Si viene de portal serio pero tiene algo de clickbait, o viceversa.
    if base == "informativo" and score > 0.3:
        if is_hard_or_service: return "informativo"
        return "posible_clickbait"  # Para revisión manual o descarte
    
    if base == "clickbait" and score < 0.3:
        if is_hard_or_service: return "informativo"
        return "posible_informativo" # Contenido neutro en portal popular

    return base

#  ORQUESTADOR PRINCIPAL
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

    # Consolidar
    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["titulo"]).reset_index(drop=True)
    df = df[df["titulo"].str.len() >= 15].reset_index(drop=True)

    # APLICAR RÚBRICA DE ETIQUETADO
    log.info("Aplicando rúbrica de etiquetado lingüístico...")
    df["etiqueta_final"] = df.apply(apply_labeling_rubric, axis=1)

    # Resumen
    log.info(f"TOTAL titulares únicos: {len(df):,}")
    log.info("\nDistribución tras aplicar rúbrica:\n" + df["etiqueta_final"].value_counts().to_string())
    log.info(f"Nacionales     : {(df['origen']=='nacional').sum():,}")
    log.info(f"Internacionales: {(df['origen']=='internacional').sum():,}")
    print("="*65)
    return df

def save_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Dataset raw → {RAW_CSV}")

    # Para el dataset final, solo nos quedamos con las clases limpias
    clean_classes = ["informativo", "clickbait", "fake_news"]
    df_final = df[df["etiqueta_final"].isin(clean_classes)].copy()
    
    df_final.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Dataset final (limpio por rúbrica) → {FINAL_CSV}")
    log.info("\n" + df_final["etiqueta_final"].value_counts().to_string())
    return df_final

def quick_eda(df: pd.DataFrame):
    print("EDA RÁPIDO")
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

if __name__ == "__main__":
    print("""
    Tarea 2 IAA — Dataset Scraper v2
    Estrategia: Google News RSS (proxy anti-bloqueo)
    Clases: Informativo / Clickbait / Fake News (bonus)
    """)

    df_raw   = run_scraping(target=TARGET_PER_CLASS, include_fake_news=True)
    df_final = save_dataset(df_raw)
    quick_eda(df_final)

    print(f"""
  Scraping completado.
    Raw    → {RAW_CSV}
    Final  → {FINAL_CSV}

PRÓXIMOS PASOS:
  1. Revisar manualmente los "posible_clickbait" del raw CSV
  2. Balancear clases si hay diferencia > 20% entre ellas
  3. Anotar muestra de verificación (~200 por clase, Cohen's Kappa)
  4. Ejecutar notebook de EDA avanzado (frecuencias, portales, autores)
""")