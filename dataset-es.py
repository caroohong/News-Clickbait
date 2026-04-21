"""
TAREA 2 - NLP & XAI: Detección de Clickbait en Prensa Chilena
  pip install requests feedparser beautifulsoup4 lxml datasets pandas tqdm
"""

import time, random, logging, re, os, argparse
from dataclasses import dataclass, field
from typing import Optional
import requests
import feedparser
import pandas as pd
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_dataset
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
REQUEST_DELAY    = (1.5, 3.0)
MAX_RETRIES      = 3
TIMEOUT          = 20
OUTPUT_DIR       = "dataset_output"
RAW_CSV          = os.path.join(OUTPUT_DIR, "dataset_raw_v3.csv")
FINAL_CSV        = os.path.join(OUTPUT_DIR, "dataset_etiquetado_v3.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PER_CLASS = 1100

#  4 EJES TEÓRICOS
#  Fundamentación académica:
#    Eje 1 (Brecha): Loewenstein (1994) — Information Gap Theory
#    Eje 2 (Exageración): Bazaco et al. (2019) — Clickbait como hipérbole
#    Eje 3 (Emoción): Hamby et al. (2018) — Emotional Appeals in News
#    Eje 4 (Ambigüedad): Blom & Hansen (2015) — Forward-reference as lure
#  Cada eje aporta hasta 1.0 punto. El score final es el promedio ponderado de los 4 ejes. Los pesos reflejan la
#  relevancia empírica de cada dimensión para la clasificación de clickbait

@dataclass
class ClickbaitScore:
    """Resultado detallado del análisis heurístico de un titular."""
    brecha:      float = 0.0   # Eje 1: retención estratégica de información
    exageracion: float = 0.0   # Eje 2: hipérbole y sensacionalismo vacío
    emocion:     float = 0.0   # Eje 3: apelación emocional y reacciones
    ambiguedad:  float = 0.0   # Eje 4: vaguedad deliberada del referente
    matches:     dict  = field(default_factory=dict)

    @property
    def total(self) -> float:
        """
        Score ponderado final [0, 1].
        Pesos: Brecha=0.35, Exageración=0.25, Emoción=0.25, Ambigüedad=0.15
        Refleja que la brecha informativa es el mecanismo más definitorio del clickbait.
        """
        score = (
            self.brecha      * 0.35 +
            self.exageracion * 0.25 +
            self.emocion     * 0.25 +
            self.ambiguedad  * 0.15
        )
        return round(min(score, 1.0), 4)

    def dominant_axis(self) -> str:
        """Retorna el eje con mayor activación (útil para XAI)."""
        axes = {
            "brecha":      self.brecha,
            "exageracion": self.exageracion,
            "emocion":     self.emocion,
            "ambiguedad":  self.ambiguedad,
        }
        return max(axes, key=axes.get) if any(axes.values()) else "ninguno"

    def explain(self) -> str:
        """Explicación legible de los patrones activados."""
        if not any(self.matches.values()):
            return "Sin señales de clickbait detectadas."
        lines = [f"Score total: {self.total:.3f} | Eje dominante: {self.dominant_axis()}"]
        labels = {
            "brecha":      "Eje 1 — Brecha de información",
            "exageracion": "Eje 2 — Exageración / Hipérbole",
            "emocion":     "Eje 3 — Apelación emocional",
            "ambiguedad":  "Eje 4 — Ambigüedad deliberada",
        }
        for eje, patterns in self.matches.items():
            if patterns:
                lines.append(f"  {labels[eje]} ({getattr(self, eje):.2f}): {patterns}")
        return "\n".join(lines)


# EJE 1: BRECHA DE INFORMACIÓN / RETENCIÓN ESTRATÉGICA
# El titular promete información pero omite deliberadamente el dato central, forzando al usuario a clicar para resolver
# la curiosidad. Mecanismo: forward-reference, pronombres vagos, preguntas sin respuesta.

EJE1_BRECHA = [
    # Pronombre inicial que oculta el sujeto
    (r"^(Este|Esta|Estos|Estas|Esto)\b.{0,60}\b(que|y|pero)\b",
     "Pronombre demostrativo + oración incompleta"),

    # "Lo que X no quiere que sepas"
    (r"\b(lo que|aquello que)\b.{0,40}\b(no quiere|oculta|esconde|calla|te niega)\b",
     "Retención conspirativa"),

    # Promesa de revelación futura sin dato
    (r"\b(el motivo|la razón|el secreto|la verdad|el misterio)\b.{0,40}\b(detrás|que hay|que nadie)\b",
     "Promesa de revelación oculta"),

    # Preguntas que no se responden en el titular
    (r"¿\s*(qué|quién|cómo|cuándo|dónde|por qué|cuál|cuánto|de qué)\s+(es|será|pasará|trata)\s*\?$",
     "Pregunta final sin respuesta (ej: ¿de qué se trata?)"),
    (r"¿\s*(qué|quién|cómo|cuándo|dónde|por qué|cuál|cuánto)\b.{5,80}\?$",
     "Pregunta sin respuesta en titular"),

    # "Esto es lo que / Así es como"
    (r"^(esto es (lo que|cómo)|así (es|fue|quedó|luce|reaccionó))\b",
     "Referencia deíctica sin antecedente"),

    # "Descubre / Entérate / Mira lo que"
    (r"\b(descubre|entérate|mira|conoce|averigua)\s+(qué|cómo|quién|cuándo|dónde|lo que)\b",
     "Imperativo + interrogativo indirecto"),

    # "Lo que pasó después / nadie esperaba"
    (r"\b(lo que (pasó|ocurrió|sucedió|dijo|hizo))\s+(después|luego|entonces|a continuación)\b",
     "Suspenso narrativo post-evento"),

    (r"\bnadie (esperaba|sabía|imaginaba|podía creer) (lo que|que)\b",
     "Sorpresa anunciada sin revelar"),

    # "Por qué X decidió / anunció"  (retiene la causa)
    (r"^por qué\s+.{3,50}\s+(decidió|anunció|renunció|abandonó|dejó)\b",
     "Retención de causa principal"),

    # Estructura "X hizo Y — y lo que pasó después..."
    (r"\b(y lo que (pasó|ocurrió|dijo|hizo) (después|luego))\b",
     "Cliffhanger mid-sentence"),

    # "Quién es [Sujeto]" sin signos de interrogación (retiene biografía)
    (r"^(quién|qué)\s+es\b(?!.*[\?,])",
     "Quién/Qué es [Sujeto] sin respuesta concreta"),

    # Metáfora de atención (se robará miradas)
    (r"\b(se (robará|robó|llevará|llevó)|robó)\s+las\s+miradas\b",
     "Metáfora de atracción de atención"),
]

# EJE 2: EXAGERACIÓN / HIPÉRBOLE
# Uso de lenguaje superlativo, adjetivos extremos o afirmaciones grandiosas que elevan artificialmente la percepción
# de importancia. No necesariamente es falso, pero distorsiona la magnitud del hecho.
EJE2_EXAGERACION = [
    # Adjetivos extremos de impacto
    (r"\b(increíble|impresionante|impactante|brutal|épico|monumental|histórico|sin precedentes|sorprendente|insólito|extraordinario|asombroso|pánico|caos|urgente)\b",
     "Adjetivo de impacto extremo o pánico"),

    # Viral / fenómeno de redes
    (r"\b(viral|arrasó|explotó en redes|se volvió viral|causa furor|estallan las redes|en llamas|causa impacto)\b",
     "Indicador de viralidad"),

    # Superlativo absoluto con énfasis
    (r"\b(el|la|los|las)\b.*\b(mejor|peor|más|mayor|menor)\b.{0,40}\b(de (la|el|los|las) historia|de todos los tiempos|jamás (visto|hecho|dicho|marcado|convertido))\b",
     "Superlativo histórico"),

    # "Nunca antes" / "por primera vez en"
    (r"\b(nunca antes|por primera vez en|inédito|sin igual|sin precedente)\b",
     "Afirmación de unicidad extrema"),

    # Cuantificadores hiperbólicos
    (r"\b(millones de|miles de)\s+(personas|usuarios|fans)\s+(quedaron|reaccionaron|lloraron|enloquecieron)\b",
     "Hipérbole de audiencia"),

    # Urgencia fabricada
    (r"\b(antes de que (sea tarde|lo borren|desaparezca)|no (te lo puedes perder|te lo pierdas))\b",
     "Urgencia artificial (FOMO)"),

    # "¡Atención!" / "¡Alerta!" como gancho vacío
    (r"^(¡|!)?\s*(atención|alerta|urgente|importante|exclusivo|pánico|terror|caos)\s*[:!]",
     "Alerta de atención vacía o sensacionalista"),

    # Promesas de transformación absoluta
    (r"\b(cambiará tu vida|lo cambia todo|nada volverá a ser igual|transforma(rá|rás))\b",
     "Promesa de transformación absoluta"),

    # Comparaciones superlativas sin base
    (r"\b(más (grande|importante|grave|increíble|impresionante) (que|de)\s+(lo que|lo imaginado|lo esperado))\b",
     "Comparación superlativa vaga"),

    # Etiquetas de contenido multimedia llamativo
    (r"\[\s*(FOTOS|VIDEO|FOTOGALERÍA|MAPA|MINUTO A MINUTO)\s*\]",
     "Etiqueta de contenido multimedia como gancho"),
]

# EJE 3: APELACIÓN EMOCIONAL
# Activación deliberada de emociones (miedo, ira, ternura, indignación, admiración) para motivar el clic
# independientemente del valor informativo. Incluye reacciones emocionales de terceros usadas como gancho.
EJE3_EMOCION = [
    # Verbos de reacción emocional intensa (de terceros)
    (r"\b(lloró|lloraron|se quebró|se emocionó|estalló|explotó|enloqueció|enloquecieron)\b",
     "Reacción emocional extrema de tercero"),

    # Verbos de revelación íntima o confesión
    (r"\b(confesó|reveló|admitió|se sinceró|se abrió|habló por primera vez|revela el motivo oculto)\b",
     "Confesión o revelación íntima"),

    # Apelación al miedo cotidiano — detecta en cualquier orden gramatical
    (r"\b(peligro(so)?|tóxico|mortal|letal|cancerígeno|dañino|venenoso|nocivo)\b.{0,60}\b(que (consumes|usas|tienes|haces|comes|bebes|tocas|respiras))\b",
     "Apelación al miedo cotidiano — adjetivo precede al nexo"),
    (r"\b(que (consumes|usas|tienes|haces|comes|bebes|tocas|respiras))\b.{0,60}\b(peligro(so)?|tóxico|mortal|letal|cancerígeno|dañino|venenoso|nocivo)\b",
     "Apelación al miedo cotidiano — nexo precede al adjetivo"),
    #Sin nexo relativo explícito — "todos los días es mortal"
    (r"\b(todos los días|cada día|a diario|habitualmente|normalmente)\b.{0,40}\b(peligro(so)?|tóxico|mortal|letal|cancerígeno|dañino|nocivo)\b",
     "Apelación al miedo cotidiano — hábito diario es peligroso"),
    (r"\b(peligro(so)?|tóxico|mortal|letal|cancerígeno|dañino|nocivo)\b.{0,40}\b(todos los días|cada día|a diario|sin saberlo|sin que lo sepas)\b",
     "Apelación al miedo cotidiano — peligro en rutina diaria"),

    # Apelación a la ira / indignación
    (r"\b(indignante|vergonzoso|escándalo|repudiable|inaceptable|polémico)\b",
     "Apelación a indignación moral"),

    # Apelación a ternura / nostalgia (baby/pet content clickbait)
    (r"\b(adorable|tierno|ternura|enternece|derritió (corazones|las redes))\b",
     "Apelación a ternura"),

    # Sorpresa como mecanismo emocional
    (r"\b(sorprendió a todos|dejó a todos (sin palabras|boquiabiertos|helados|sorprendidos))\b",
     "Sorpresa colectiva"),

    # Meta-reacciones (cómo te sentirás)
    (r"\b(te costará (reconocer|creer|entender)|tal y como lo conocemos)\b",
     "Predicción de reacción o impacto personal"),

    # Apelación a culpa / responsabilidad del lector
    (r"\b(si (eres|tienes|haces|usas|comes|bebes).{0,30}(debes|tienes que|necesitas))\b",
     "Apelación a responsabilidad del lector"),

    # Emociones negativas extremas como gancho
    (r"\b(trágico|devastador|desgarrador|estremecedor|escalofriante|perturbador)\b",
     "Adjetivo emocional negativo extremo"),

    # Segunda persona como vector emocional directo
    (r"\b(te (va a|hará|dejará|sorprenderá|impactará|emocionará|angustiará))\b",
     "Predicción emocional dirigida al lector"),

    # "Nadie puede creerlo" / Incredulidad colectiva
    (r"\b(nadie (puede|pudo|podía) creer(lo)?|todos quedaron (impactados|sorprendidos|sin palabras))\b",
     "Incredulidad colectiva como gancho"),
]

# EJE 4: AMBIGÜEDAD DELIBERADA
# El titular usa referencias vagas, pronombres indefinidos o estructuras gramaticales incompletas para crear una
# sensación de misterio que solo se resuelve haciendo clic. Diferente a la brecha informativa: la vaguedad es el
# mecanismo, no la omisión de datos concretos.
EJE4_AMBIGUEDAD = [
    # Pronombres vagos como sujeto principal
    (r"^(un|una|unos|unas)\s+(hombre|mujer|niño|niña|joven|anciano|sujeto|individuo|personaje)\b.{0,50}\b(y (lo que|lo que pasó|su reacción))\b",
     "Sujeto indefinido + consecuencia velada"),

    # "Algo" / "alguien" como referente principal
    (r"^(algo|alguien|algún)\b.{0,60}\b(que (nadie|todos|jamás))\b",
     "Referente indefinido con alcance absoluto"),

    # "Esto" / "aquello" sin antecedente claro
    (r"^(esto|aquello|eso)\s+(que (está|están|hace|hacen|dijo|dijeron))\b",
     "Deíctico sin antecedente"),

    # Estructura "X famoso" sin revelar nombre
    (r"\b(un (famoso|conocido|popular|reconocido)|una (famosa|conocida|popular))\s+(actor|actriz|cantante|deportista|político|chef|influencer)\b",
     "Personaje famoso sin identificar"),

    # "La razón real" / "el verdadero motivo" (implica que hay una versión oculta)
    (r"\b(la (razón|verdad|historia|versión) (real|verdadera|oculta|detrás))\b",
     "Verdad alternativa implícita"),

    # "Nadie lo sabe pero..." / "Pocas personas saben que..."
    (r"\b(nadie (lo )?sabe (pero|que)|pocas personas saben|solo el \d+%)\b",
     "Conocimiento exclusivo/secreto"),

    # Elipsis deliberada al final del titular  "..."
    (r"\.\.\.\s*$",
     "Elipsis de suspenso al final"),

    # "Lo que realmente pasó con X"
    (r"\b(lo que realmente (pasó|ocurrió|sucedió|hay detrás|significa))\b",
     "Versión 'real' implícita vs. versión oficial"),

    # Titulares con "así" sin completar la acción
    (r"^así\s+(fue|es|quedó|luce|reaccionó|respondió|lo hizo)\b",
     "Adverbio de modo sin acción completada"),

    # "Mira cómo X" sin dar el resultado
    (r"^(mira|observa|ve)\s+(cómo|cuándo|dónde|qué)\s+.{3,60}$",
     "Imperativo visual sin resolución"),
]

#  MOTOR DE PUNTUACIÓN
def _score_axis(title: str, patterns: list[tuple]) -> tuple[float, list[str]]:
    """
    Evalúa un eje de clickbait sobre el titular.
    Retorna (score_normalizado, lista_de_descripciones_activadas).
    El score por eje se normaliza: 1 hit = 0.5, 2+ hits = 1.0 para evitar doble penalización por patrones semánticamente similares.
    """
    hits = []
    for regex, description in patterns:
        if re.search(regex, title, re.IGNORECASE):
            hits.append(description)
    # Normalización suave: primer hit vale 0.5, segundo 0.35, resto 0.15 c/u
    if len(hits) == 0:
        return 0.0, []
    elif len(hits) == 1:
        return 0.5, hits
    elif len(hits) == 2:
        return 0.85, hits
    else:
        return 1.0, hits

def analyze_clickbait(title: str) -> ClickbaitScore:
    """
    Análisis completo de un titular con los 4 ejes.
    Retorna un objeto ClickbaitScore con scores por eje y patrones activados.
    """
    if not title or len(title.strip()) < 5:
        return ClickbaitScore()

    s1, m1 = _score_axis(title, EJE1_BRECHA)
    s2, m2 = _score_axis(title, EJE2_EXAGERACION)
    s3, m3 = _score_axis(title, EJE3_EMOCION)
    s4, m4 = _score_axis(title, EJE4_AMBIGUEDAD)

    return ClickbaitScore(
        brecha      = s1,
        exageracion = s2,
        emocion     = s3,
        ambiguedad  = s4,
        matches     = {
            "brecha":      m1,
            "exageracion": m2,
            "emocion":     m3,
            "ambiguedad":  m4,
        }
    )

def clickbait_score(title: str) -> float:
    """retorna solo el score total [0,1]."""
    return analyze_clickbait(title).total

def explain_score(title: str) -> str:
    """
    Explicación completa del score de un titular.
    Ejemplo:
        explain_score("El famoso chileno que nadie esperaba confesó todo")
    """
    result = analyze_clickbait(title)
    print(f'\nTitular: "{title}"')
    print(result.explain())
    return result.explain()

#  SEÑALES DE HARD NEWS (rescate anti-falsos-positivos)
#  Palabras y frases que indican contenido informativo serio,
#  aunque el titular contenga algo de lenguaje emocional.
HARD_NEWS_WORDS = {
    # Tragedias y sucesos verificables
    "fallece", "muere", "murió", "fallecieron", "tragedia", "accidente",
    "homicidio", "femicidio", "detenido", "imputado", "formalizado",
    "carabineros", "pdi", "fiscalía", "condenado", "sentenciado",
    # Instituciones y política
    "gobierno", "ministerio", "minsal", "mineduc", "cámara", "senado",
    "congreso", "diputados", "senadores", "presidente", "ministro",
    "intendente", "alcalde", "municipio", "decreto", "ley", "proyecto",
    "banco central", "inflación", "pib", "presupuesto", "hacienda",
    "cancillería", "tribunal", "corte", "recurso", "amparo",
    # Organismos de control y auditoría
    "contraloría", "cntv", "fiscalización", "auditoría", "seremi",
    "anfp", "caf", "fifa", "uefa", "conmebol",
    # Beneficios y servicios (noticias útiles)
    "bono", "subsidio", "beneficio", "pago", "postular", "requisito",
    "fecha de pago", "calendario", "trámite", "registro",
    # Emergencias naturales
    "sismo", "terremoto", "tsunami", "incendio", "alerta", "evacuación",
    "volcán", "tormenta", "inundación", "emergencia", "desastre", "temblor",
    # Deportes — RESULTADOS CONCRETOS (score, marcador, clasificación)
    # Solo palabras que implican un resultado verificable, no lenguaje hiperbólico
    "triunfo", "derrota", "empate", "clasificó", "eliminó", "campeonato",
    "torneo", "copa", "mundial", "olimpiadas", "ascenso", "descenso",
    "medalla", "oro", "plata", "bronce", "podio", "récord",
    "gana", "ganan", "venció", "vencieron", "cayó", "cayeron",
    "anotó", "marcó", "convirtió",   # goles con sujeto concreto
    # Salud pública
    "vacuna", "medicamento", "tratamiento", "diagnóstico", "síntoma",
    "pandemia", "epidemia", "brote", "contagio", "oms",
    # Educación
    "prueba", "psu", "paes", "universidad", "colegio", "matrícula",
    # Economía y finanzas
    "tasa", "dólar", "uf", "crédito", "deuda", "cae", "pensión", "afp",
    "isapre", "contrato", "despido", "huelga", "sindicato",
    # Resultado electoral / judicial concreto
    "aprobó", "rechazó", "aprueba", "rechaza", "promulgó", "firmó",
    "renunció", "renuncia", "destituido", "nombrado", "designado",
}

HARD_NEWS_PHRASES = [
    # Servicio y datos prácticos
    "cómo postular", "cómo obtener", "cuándo pagan", "fecha de pago",
    "dónde ver", "revisa el", "quiénes pueden", "cuáles son los requisitos",
    "así funciona", "qué es el", "qué es la",
    # Preguntas educativas/explicativas — Eje1 activa, pero son periodismo serio
    # Patrón: ¿Por qué X [tiene/es/hace]? con tema geopolítico, científico o institucional
    "por qué argentina", "por qué chile", "por qué españa", "por qué eeuu",
    "por qué el golfo", "por qué la ue", "por qué la onu", "por qué europa",
    "por qué brasil", "por qué méxico", "por qué colombia",
    "qué es la cop", "qué es el fmi", "qué es la otan", "qué es el g7",
    "qué es la agenda", "qué es el acuerdo", "qué es la ley",
    "cómo funciona", "cómo se calcula", "cómo afecta",
    "cuáles son las claves", "cuáles son los factores",
    # Preguntas de programación deportiva/cultural — servicio puro
    "cuándo se jugará", "cuándo juega", "dónde juega", "a qué hora",
    "dónde se celebra", "cuándo se celebra",
    "dónde ver en vivo", "cómo ver en vivo",
    # Preguntas electorales y políticas explicativas
    "qué pasa si gana", "qué pasa si pierde", "qué pasa si empata",
    "qué significa", "qué implica", "qué cambia",
]

def is_hard_news(title: str) -> bool:
    """True si el titular contiene señales de noticia informativa seria."""
    t_lower = title.lower()
    return (
        any(w in t_lower for w in HARD_NEWS_WORDS) or
        any(ph in t_lower for ph in HARD_NEWS_PHRASES)
    )

# Contexto deportivo concreto
# "Increíble gol de Alexis Sánchez ante Perú" es periodismo deportivo, no clickbait: tiene sujeto nombrado + acción + contexto.
# En contraste, "Increíble lo que hizo este jugador" SÍ es clickbait. Contexto: nombre propio + verbo de resultado.
_SPORT_CONCRETE_RE = re.compile(
    # Nombre propio (al menos dos tokens con mayúscula) cerca de verbo deportivo
    r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+'  # nombre propio
    r'.{0,60}'
    r'\b(anotó|marcó|convirtió|falló|erró|pateó|cabeceó|atajó|'
    r'venció|venció|derrotó|goleó|empató|clasificó|eliminó|'
    r'ascendió|descendió|ganó|perdió|logró|consiguió)\b',
    re.IGNORECASE
)

_SPORT_SCORE_RE = re.compile(
    # Marcador numérico en el titular (2-1, 3-0, etc.)
    r'\b\d+\s*[-–]\s*\d+\b'
)

def is_concrete_sport_news(title: str) -> bool:
    """
    True si el titular deportivo tiene contexto concreto: nombre propio + verbo de resultado, O marcador numérico.
    Esto protege 'increíble gol de X' de ser clasificado como clickbait.
    """
    return bool(_SPORT_CONCRETE_RE.search(title)) or bool(_SPORT_SCORE_RE.search(title))

def apply_labeling_rubric(row: pd.Series) -> str:
    """
    Rúbrica de etiquetado: se aplicó pq inicialmente clasificaba mal

    Árbol de decisión (en orden de ejecución):
    ─ fake_news base → fake_news (siempre)
    ─ R0: Pregunta de brecha (¿?) sin ser servicio → clickbait
    ─ R1: Hard news + score < 0.25 → informativo
    ─ R1b: Deporte concreto (marcador/nombre+verbo) + score < 0.45 → informativo
    ─ R2: Score total >= 0.55 → clickbait
    ─ R3: Un eje >= 0.85 → clickbait (con excepciones hard news)
    ─ R3b: 'Quién es X' sin dato en el titular → clickbait
    ─ R3c: 'viral como tema' con sujeto anónimo/vago → clickbait
    ─ R4: Hard news + score < 0.45 → informativo
    ─ R5: Score == 0 → informativo
    ─ R6: Base=informativo + score 0.25-0.54 + no hard news → posible_clickbait
    ─ R7: Base=clickbait + score < 0.25 + hard news → informativo
    ─ R8: Default → etiqueta_base
    """
    title   = str(row.get("titulo", ""))
    base    = str(row.get("etiqueta_base", "informativo"))
    t_lower = title.lower()

    if base == "fake_news":
        return "fake_news"

    result      = analyze_clickbait(title)
    total_score = result.total
    hard        = is_hard_news(title)
    sport_ok    = is_concrete_sport_news(title)
    max_eje     = max(result.brecha, result.exageracion, result.emocion, result.ambiguedad)

    # Pre-calcular banderas de estructura
    _superlativo_hist_re = re.compile(
        r'\b(de (la|el|los|las) historia|de todos los tiempos|jamás (visto|hecho|marcado|convertido))\b',
        re.IGNORECASE
    )
    is_superlativo_historico = bool(_superlativo_hist_re.search(title))

    # R0: Pregunta de brecha con signo de interrogación
    # Solo si NO es pregunta de servicio/educativa (cubiertas en HARD_NEWS_PHRASES)
    if (result.brecha >= 0.5
            and result.dominant_axis() == "brecha"
            and re.search(r"[¿?]", title)):
        is_service = any(ph in t_lower for ph in HARD_NEWS_PHRASES)
        # Excepción adicional: "¿Cuándo/Dónde/Cómo + verbo + evento concreto"
        # Refinado: permite cualquier caracter antes del signo de apertura
        is_scheduling = bool(re.search(
            r'(cuándo|dónde|a qué hora|en qué canal|cómo)\s+(se\s+(jugará|celebrará|estrenará|realizará)|juega|emite|transmite|jugará|ver|postular)',
            title, re.IGNORECASE
        ))
        if not is_service and not is_scheduling:
            return "clickbait"

    # R0b: Superlativo histórico fuerte es clickbait siempre
    if result.exageracion >= 0.5 and is_superlativo_historico:
        return "clickbait"

    # R0c: Metáforas de atención puras no se rescatan
    if "robará las miradas" in t_lower or "se robó las miradas" in t_lower:
        return "clickbait"

    # R0d: Casos Titanic/NASA/Misterios sin info concreta (SE MUEVE AL INICIO)
    if ("sorprendente" in t_lower or "insólito" in t_lower) and \
       any(w in t_lower for w in ["tesoro", "hallazgo", "secreto", "misterio", "descubren"]):
        return "clickbait"

    # R1: Hard news con score muy bajo → informativo sin dudas
    if hard and total_score < 0.25 and result.brecha < 0.3:
        return "informativo"

    # R1b: Deporte concreto con adjetivo hiperbólico
    if sport_ok and total_score < 0.45 and result.brecha == 0 and not is_superlativo_historico:
        return "informativo"

    # R2: Evidencia fuerte multi-eje
    if total_score >= 0.55:
        return "clickbait"

    # R3: Evidencia fuerte en UN solo eje
    if max_eje >= 0.85:
        dominant = result.dominant_axis()
        if hard and dominant in ("exageracion", "emocion"):
            if result.brecha > 0.3 or result.ambiguedad > 0.3:
                return "clickbait"
            return "informativo"
        return "clickbait"

    # R3b: "Quién es X" / "Qué es X" sin dato en el titular
    quien_re = re.compile(r'^(quién es|qué es)\s+\w', re.IGNORECASE)
    if quien_re.match(title):
        is_vague = bool(re.search(r'\bque\b.*\b(encantó|viral|sorprendió|impactó|enamoró)\b', t_lower))
        has_concrete_data = any(w in t_lower for w in ["ministro", "actor", "director", "científico", "político", "médico"])
        if is_vague and not has_concrete_data:
            return "clickbait"

        has_apposition = bool(re.search(r',\s+[a-záéíóúñ]', title, re.IGNORECASE))
        if not has_apposition:
            return "clickbait"

    # R3d: Casos Titanic/NASA/Misterios sin info concreta
    if ("sorprendente" in t_lower or "insólito" in t_lower) and \
       any(w in t_lower for w in ["tesoro", "hallazgo", "secreto", "misterio", "descubren"]):
        return "clickbait"

    # R4: Hard news con score medio
    if hard and total_score < 0.45 and result.brecha < 0.4:
        return "informativo"

    # R5: Sin señal
    if total_score == 0.0:
        # Elipsis de suspenso aunque no tenga otras señales
        if "..." in title:
            return "clickbait"
        return "informativo"

    # R6: Zona gris — portal serio con señales leves
    if base == "informativo" and 0.25 <= total_score < 0.55:
        return "posible_clickbait"

    # R7: Portal clickbait pero titular duro
    if base == "clickbait" and total_score < 0.25 and hard:
        return "informativo"

    # R8: Default
    return base

def extract_date_from_html(url: str) -> str:
    if not url or not url.startswith("http"):
        return ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
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
        time_tag = soup.find("time", {"datetime": True})
        if time_tag:
            return time_tag["datetime"].strip()
    except Exception:
        pass
    return ""

def gnews_url(query: str, lang: str = "es-419", country: str = "CL") -> str: #google news
    q = requests.utils.quote(query)
    ceid = f"{country}:{lang.split('-')[0]}"
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={ceid}"

def gnews_topic_url(topic: str) -> str:
    topics = {
        "headlines_cl":    "https://news.google.com/rss?hl=es-419&gl=CL&ceid=CL:es-419",
        "headlines_es":    "https://news.google.com/rss?hl=es&gl=ES&ceid=ES:es",
        "headlines_ar":    "https://news.google.com/rss?hl=es-419&gl=AR&ceid=AR:es-419",
        "headlines_mx":    "https://news.google.com/rss?hl=es-419&gl=MX&ceid=MX:es-419",
        "headlines_us_es": "https://news.google.com/rss?hl=es-419&gl=US&ceid=US:es-419",
    }
    return topics.get(topic, "")

NATIONAL_INFORMATIVE_QUERIES = [
    'site:latercera.com', 'site:emol.com', 'site:cooperativa.cl',
    'site:biobiochile.cl', 'site:24horas.cl', 'site:cnnchile.com',
    'site:elmostrador.cl', 'site:radioagricultura.cl', 'site:df.cl',
    'site:eldesconcierto.cl', 'site:chilevision.cl',
    'economia Chile presupuesto', 'politica Chile gobierno ministerio',
    'Chile salud Minsal', 'Chile educacion Mineduc',
    'Chile tribunal justicia sentencia', 'Chile congreso proyecto ley',
    'Chile Banco Central inflacion', 'terremoto Chile sismo',
    'Chile cancilleria relaciones exteriores',
]
NATIONAL_CLICKBAIT_QUERIES = [
    'site:publimetro.cl', 'site:meganoticias.cl', 'site:lun.com',
    'site:redgol.cl', 'site:eldinamo.cl', 'site:ahoranoticias.cl',
    'site:soychile.cl', 'site:fotech.cl', 'site:glamorama.cl',
    'site:t13.cl entretenimiento',
    'viral impactante Chile famoso', 'Chile famoso sorprendió impactante reacción',
    'Chile tiktoker youtuber viral', 'Chile farándula impactó sorprendió',
    'Chile deportes gol increíble', 'Chile revelación confesó lloró',
    'Chile misterio insólito curioso', 'Chile terror pánico susto viral',
    'Chile descuento oferta ahorro truco', 'Chile receta truco secreto increíble',
]
INTERNATIONAL_INFORMATIVE_QUERIES = [
    'site:bbc.com/mundo', 'site:france24.com/es', 'site:dw.com/es',
    'site:reuters.com español', 'site:apnews.com', 'site:elpais.com',
    'site:lavanguardia.com', 'site:efe.com', 'site:euronews.com/es',
    'site:nytimes.com español',
    'economia global banco mundial FMI', 'conflicto internacional ONU',
    'cambio climatico COP acuerdo', 'elecciones democracia internacional',
    'ciencia descubrimiento estudio investigacion',
    'tecnologia inteligencia artificial innovacion',
    'salud OMS pandemia vacuna', 'derechos humanos amnistia',
]
INTERNATIONAL_CLICKBAIT_QUERIES = [
    'site:infobae.com', 'site:20minutos.es', 'site:marca.com',
    'site:muyinteresante.es', 'site:elconfidencial.com', 'site:sport.es',
    'site:as.com', 'site:antena3.com', 'site:lasexta.com',
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
    queries, label, origin, target,
    lang="es-419", country="CL",
) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()

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
            time.sleep(random.uniform(*REQUEST_DELAY))
            continue

        new_count = 0
        for entry in entries:
            title  = re.sub(r'\s+[-–]\s+[\w\s\.]+$', '', entry.get("title", "").strip()).strip()
            source = entry.get("source", {}).get("title", "")
            link   = entry.get("link", "")
            pub    = entry.get("published", "").strip()
            if not title or len(title) < 12:
                continue
            if title.lower() in seen:
                continue
            seen.add(title.lower())
            if not pub and link:
                pub = extract_date_from_html(link)

            result = analyze_clickbait(title)
            records.append({
                "titulo":            title,
                "url":               link,
                "fecha_publicacion": pub,
                "portal":            source or f"GNews:{query[:30]}",
                "origen":            origin,
                "etiqueta_base":     label,
                "cb_heuristic":      result.total,
                "cb_brecha":         result.brecha,       # nuevo: score por eje
                "cb_exageracion":    result.exageracion,
                "cb_emocion":        result.emocion,
                "cb_ambiguedad":     result.ambiguedad,
                "cb_eje_dominante":  result.dominant_axis(),
                "etiqueta_final":    label,
                "metodo_obtencion":  "gnews_rss",
            })
            new_count += 1

        log.info(f"[GNews] '{query[:45]}' → {new_count} nuevos (total: {len(records)})")
        time.sleep(random.uniform(*REQUEST_DELAY))

    # Complemento con topic feeds
    if len(records) < target:
        topic_map = {
            "nacional":      ["headlines_cl"],
            "internacional": ["headlines_es", "headlines_ar", "headlines_mx", "headlines_us_es"],
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
                    if not title or len(title) < 12 or title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    entry_link = entry.get("link", "")
                    entry_pub  = entry.get("published", "").strip()
                    if not entry_pub and entry_link:
                        entry_pub = extract_date_from_html(entry_link)
                    result = analyze_clickbait(title)
                    records.append({
                        "titulo":            title,
                        "url":               entry_link,
                        "fecha_publicacion": entry_pub,
                        "portal":            entry.get("source", {}).get("title", "") or topic,
                        "origen":            origin,
                        "etiqueta_base":     label,
                        "cb_heuristic":      result.total,
                        "cb_brecha":         result.brecha,
                        "cb_exageracion":    result.exageracion,
                        "cb_emocion":        result.emocion,
                        "cb_ambiguedad":     result.ambiguedad,
                        "cb_eje_dominante":  result.dominant_axis(),
                        "etiqueta_final":    label,
                        "metodo_obtencion":  "gnews_topic",
                    })
                time.sleep(random.uniform(*REQUEST_DELAY))
            except Exception as e:
                log.warning(f"[GNews topic] Error: {e}")

    log.info(f"[GNews] Clase '{label}/{origin}': {len(records)} titulares recolectados.")
    return records[:target]

def load_fakenews(target: int = TARGET_PER_CLASS) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()

    def add(title, url, portal, origen, fuente, fecha=""):
        k = title.lower().strip()
        if k in seen or len(title) < 10:
            return False
        seen.add(k)
        if not fecha and url and fuente == "gnews_rss":
            fecha = extract_date_from_html(url)
        result = analyze_clickbait(title)
        records.append({
            "titulo":            title.strip(),
            "url":               url,
            "fecha_publicacion": fecha,
            "portal":            portal,
            "origen":            origen,
            "etiqueta_base":     "fake_news",
            "cb_heuristic":      result.total,
            "cb_brecha":         result.brecha,
            "cb_exageracion":    result.exageracion,
            "cb_emocion":        result.emocion,
            "cb_ambiguedad":     result.ambiguedad,
            "cb_eje_dominante":  result.dominant_axis(),
            "etiqueta_final":    "fake_news",
            "metodo_obtencion":  fuente,
        })
        return True

    try:
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
        log.info(f"[FakeNews HF] {len(records)} titulares cargados.")
    except Exception as e:
        log.warning(f"[FakeNews HF] Error: {e}")

    if len(records) < target:
        fn_national = [
            'site:labot.cl', 'site:puroperiodismo.cl verificacion',
            'desinformación Chile falso verificado', 'Chile bulo viral falso noticias',
            'Chile rumor fake news desmentido', 'Chile electoral fraude falso desinformacion',
            'Chile salud falso bulo viral desmentido', 'Chile estafa viral fraude engaño',
            'Chile conspiracion falso teoria rumor', 'Chile Covid desinformacion vacuna falso',
        ]
        extra_nat = scrape_gnews_queries(fn_national, "fake_news", "nacional",
                                          target=min(400, target - len(records)),
                                          lang="es-419", country="CL")
        for rec in extra_nat:
            add(rec["titulo"], rec["url"], rec["portal"], "nacional", "gnews_rss",
                fecha=rec.get("fecha_publicacion", ""))

    if len(records) < target:
        fn_intl = [
            'site:maldita.es', 'site:newtral.es', 'site:chequeado.com falso',
            'site:colombiacheck.com falso', 'site:afpfactual.com falso español',
            'bulo falso desinformacion viral España',
            'fake news desinformacion America Latina español',
            'hoax falso verificado mentira viral español',
            'infodemia desinformación salud mentira español',
            'conspiración teoría falsa viral español',
        ]
        extra_intl = scrape_gnews_queries(fn_intl, "fake_news", "internacional",
                                           target=target - len(records),
                                           lang="es-419", country="US")
        for rec in extra_intl:
            add(rec["titulo"], rec["url"], rec["portal"], "internacional", "gnews_rss",
                fecha=rec.get("fecha_publicacion", ""))

    log.info(f"[FakeNews] TOTAL: {len(records)}")
    return records[:target]

def run_scraping(target: int = TARGET_PER_CLASS, include_fake_news: bool = True) -> pd.DataFrame:
    all_records: list[dict] = []

    print("\n Prensa Nacional Informativa")
    all_records += scrape_gnews_queries(NATIONAL_INFORMATIVE_QUERIES, "informativo", "nacional", target)
    print("\n Prensa Nacional Clickbait")
    all_records += scrape_gnews_queries(NATIONAL_CLICKBAIT_QUERIES, "clickbait", "nacional", target)
    print("\n Prensa Internacional Informativa")
    all_records += scrape_gnews_queries(INTERNATIONAL_INFORMATIVE_QUERIES, "informativo", "internacional",
                                         target, lang="es-419", country="US")
    print("\n Prensa Internacional Clickbait")
    all_records += scrape_gnews_queries(INTERNATIONAL_CLICKBAIT_QUERIES, "clickbait", "internacional",
                                         target, lang="es-419", country="US")
    if include_fake_news:
        print("\n Fake News")
        all_records += load_fakenews(target)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["titulo"]).reset_index(drop=True)
    df = df[df["titulo"].str.len() >= 15].reset_index(drop=True)

    log.info("Aplicando rúbrica de etiquetado (4 ejes)...")
    df["etiqueta_final"] = df.apply(apply_labeling_rubric, axis=1)

    log.info(f"TOTAL titulares únicos: {len(df):,}")
    log.info("\n" + df["etiqueta_final"].value_counts().to_string())
    log.info(f"Nacionales     : {(df['origen']=='nacional').sum():,}")
    log.info(f"Internacionales: {(df['origen']=='internacional').sum():,}")
    return df

def save_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Dataset raw → {RAW_CSV}")
    df_final = df[df["etiqueta_final"].isin(["informativo", "clickbait", "fake_news"])].copy()
    df_final.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Dataset final → {FINAL_CSV}")
    log.info("\n" + df_final["etiqueta_final"].value_counts().to_string())
    return df_final

def run_tests():
    """
    Casos de prueba para validar que el sistema detecta correctamente los 4 ejes (clasificados manualmente y de acuerdo a eso se creó la rubrica de arriba).
    python dataset-es.py --test
    """
    test_cases = [
        # (titular, etiqueta_esperada, descripcion)
        # EJE 1: BRECHA
        ("¿Por qué el Presidente decidió renunciar a este cargo inesperado?",
         "clickbait", "Eje1: pregunta sin respuesta + retención de causa"),
        ("Nadie esperaba lo que pasó después del partido de anoche",
         "clickbait", "Eje1: suspenso narrativo post-evento"),
        ("Descubre cómo este vegetal puede salvar tu vida",
         "clickbait", "Eje1: imperativo + interrogativo indirecto"),

        # EJE 2: EXAGERACIÓN
        ("El gol más increíble de la historia del fútbol chileno se marcó ayer",
         "clickbait", "Eje2: superlativo histórico"),
        ("La decisión de Apple que cambiará tu vida para siempre",
         "clickbait", "Eje2: promesa de transformación absoluta"),
        ("¡Atención! Lo que debes saber antes de que sea tarde",
         "clickbait", "Eje2: urgencia artificial + alerta vacía"),

        # EJE 3: EMOCIÓN
        ("La actriz lloró en vivo al recordar su pasado y dejó a todos sin palabras",
         "clickbait", "Eje3: reacción emocional + sorpresa colectiva"),
        ("Confesó todo: el futbolista habló por primera vez de su adicción",
         "clickbait", "Eje3: revelación íntima"),
        ("El alimento que consumes todos los días es mortal y no lo sabías",
         "clickbait", "Eje3: apelación al miedo cotidiano"),

        # EJE 4: AMBIGÜEDAD
        ("Un famoso cantante sorprendió a todos con su radical cambio de imagen...",
         "clickbait", "Eje4: personaje sin identificar + elipsis"),
        ("La verdad real detrás del escándalo que sacudió al país",
         "clickbait", "Eje4: verdad alternativa implícita"),
        ("Así quedó la conversación entre los dos políticos",
         "clickbait", "Eje4: adverbio de modo sin acción completada"),

        # HARD NEWS — NO deben ser clickbait
        ("Banco Central sube la tasa de interés al 5,5% en reunión de política monetaria",
         "informativo", "Hard news económica sin señales CB"),
        ("Carabineros detiene a imputado por homicidio en La Florida",
         "informativo", "Hard news policial"),
        ("Minsal confirma nuevo brote de hepatitis A en región Metropolitana",
         "informativo", "Hard news sanitaria"),
        ("Chile clasifica al Mundial 2026 tras vencer 2-1 a Venezuela",
         "informativo", "Hard news deportiva con marcador concreto"),
        ("Senado aprueba proyecto de ley de 40 horas laborales con 25 votos a favor",
         "informativo", "Hard news legislativa con datos"),

        # Fix R1b: deportes concretos con adjetivo hiperbólico = informativo
        ("El increíble gol que se perdió Iván Morales en el empate de Argentinos Juniors ante Tigre",
         "informativo", "Fix R1b: increíble + nombre propio + resultado deportivo concreto"),
        ("Maxloren Castro falló increíble gol tras perfecto pase de Joao Grimaldo en Perú vs Chile",
         "informativo", "Fix R1b: increíble + jugadores nombrados + partido concreto"),
        ("Perú vs Chile: Paolo Guerrero falló increíble gol solo",
         "informativo", "Fix R1b: increíble + nombre propio + partido concreto"),

        # Fix R3b: 'Quién es X' sin dato = clickbait
        ("Quién es Raquel Castillo, la mujer que se volvió viral en Chile",
         "clickbait", "Fix R3b: Quién es sin apósición explicativa"),
        ("Quién es Ángela Mármol, la influencer que encantó a Tom Cruise",
         "clickbait", "Fix R3b: Quién es sin dato concreto después"),

        # Quién es X CON apósición = informativo
        ("Quién es Péter Magyar, el exaliado de Viktor Orbán que se convirtió en su crítico",
         "informativo", "Fix R3b excepción: Quién es + coma + apósición explicativa"),

        # Fix R3c: Tiktoker/influencer viral con sujeto anónimo = clickbait
        ("Tiktoker chileno se hace viral al comparar pasos de cebra en Chile y Argentina",
         "clickbait", "Fix R3c: tiktoker anónimo + viral como tema"),
        ("Joven francesa se vuelve viral al dar ocho razones por las que Chile es mejor que Francia",
         "clickbait", "Fix R3c: joven anónima + viral como tema"),

        # Fix: preguntas educativas/geopolíticas = informativo (cubiertas por HARD_NEWS_PHRASES)
        ("¿Por qué Argentina y Bolivia pagarán en yuanes sus importaciones chinas?",
         "informativo", "Pregunta educativa geopolítica con países concretos"),
        ("¿Por qué el golfo Pérsico tiene más petróleo que cualquier otro lugar?",
         "informativo", "Pregunta educativa geopolítica/científica"),

        # Fix: preguntas de servicio deportivo = informativo
        ("EN VIVO por TV y ONLINE: ¿Dónde ver Chile vs Uruguay en la Liga de Naciones?",
         "informativo", "Pregunta de servicio deportivo (dónde ver)"),
        ("Chile ya está clasificado: ¿Cuándo y dónde se jugará el Mundial Sub 17 de la FIFA 2026?",
         "informativo", "Pregunta de servicio deportivo (cuándo y dónde)"),

        # EXTRA (análisis de fallos previos)
        ("‘No make up': ¡Las 'celebrities' sin maquillaje que te costará reconocer!",
         "clickbait", "Fallo previo: predicción impacto personal + hipérbole"),
        ("Hicieron un scaneo 3D del Titanic y encontraron un SORPRENDENTE tesoro oculto",
         "clickbait", "Fallo previo: adjetivo de impacto + misterio"),
        ("El sorprendente hallazgo que hizo la Nasa sobre la superficie de Marte, ¿de qué se trata?",
         "clickbait", "Fallo previo: brecha final ¿de qué se trata?"),
        ("Google Maps: halla ‘conejo rosa gigante’, hace zoom y descubre insólito secreto [FOTOS]",
         "clickbait", "Fallo previo: tag [FOTOS] + imperativo + adjetivo"),
        ("Mine revela el motivo oculto por el que jamás traicionará a Cihan ante Alya",
         "clickbait", "Fallo previo: revelación íntima + motivo oculto"),
        ("Pánico en el Arsenal",
         "clickbait", "Fallo previo: alerta de pánico vacía"),
        ("La economía global resiste... de momento",
         "clickbait", "Fallo previo: elipsis de suspenso al final"),
        ("Este es el fin de internet tal y como lo conocemos",
         "clickbait", "Fallo previo: deíctico sin antecedente + impacto personal"),
        ("Hacienda: la comisión que se robará las miradas esta semana",
         "clickbait", "Fallo previo: metáfora de atención (se robará miradas)"),
        ("“¡Aguante las cabras!”: Blue Mary y Flor de Rap lanzan videoclip de su canción viral en redes",
         "clickbait", "Fallo previo: indicador de viralidad como gancho"),
    ]

    print("  TEST UNITARIO — Sistema Heurístico 4 Ejes")
    passed = 0
    failed = 0
    for title, expected_label, description in test_cases:
        result   = analyze_clickbait(title)
        # Aplicar la misma lógica de la rúbrica manualmente
        row = pd.Series({
            "titulo":       title,
            "etiqueta_base": "clickbait" if expected_label == "clickbait" else "informativo",
        })
        predicted = apply_labeling_rubric(row)
        # Para test: "posible_clickbait" cuenta como incorrecto si se esperaba informativo
        ok = (predicted == expected_label) or \
             (expected_label == "clickbait" and predicted in ("clickbait", "posible_clickbait"))

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} [{description}]")
        print(f"   Titular  : {title[:80]}")
        print(f"   Esperado : {expected_label:<15} | Predicho: {predicted}")
        print(f"   Scores   : total={result.total:.3f} | "
              f"brecha={result.brecha:.2f} exag={result.exageracion:.2f} "
              f"emoc={result.emocion:.2f} ambig={result.ambiguedad:.2f} "
              f"| eje_dom={result.dominant_axis()}")

    print(f"  Resultado: {passed}/{len(test_cases)} tests pasados "
          f"({'%.0f' % (passed/len(test_cases)*100)}%)")

def quick_eda(df: pd.DataFrame):
    print("EDA")
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

def relabel_existing_dataset():
    """Carga el raw CSV y aplica la rúbrica actualizada sin hacer nuevos requests."""
    if not os.path.exists(RAW_CSV):
        log.error(f"No se encontró el archivo base {RAW_CSV}. Debes ejecutar el scrape primero.")
        return

    log.info(f"Relabeling: Cargando {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    
    log.info("Re-calculando heurísticas y aplicando nueva rúbrica...")
    # Actualizar scores por eje
    tqdm.pandas(desc="Analizando titulares")
    
    def update_row(row):
        res = analyze_clickbait(str(row["titulo"]))
        row["cb_heuristic"] = res.total
        row["cb_brecha"] = res.brecha
        row["cb_exageracion"] = res.exageracion
        row["cb_emocion"] = res.emocion
        row["cb_ambiguedad"] = res.ambiguedad
        row["cb_eje_dominante"] = res.dominant_axis()
        row["etiqueta_final"] = apply_labeling_rubric(row)
        return row

    df = df.progress_apply(update_row, axis=1)
    
    save_dataset(df)
    quick_eda(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper Dataset Clickbait v3")
    parser.add_argument("--test",   action="store_true", help="Ejecutar tests unitarios y salir")
    parser.add_argument("--relabel", action="store_true", help="Re-etiquetar el CSV existente sin scrapear")
    parser.add_argument("--explain", type=str, default=None, help="Explicar el score de un titular")
    parser.add_argument("--no-fake", action="store_true", help="Omitir clase fake news")
    args = parser.parse_args()
    if args.test:
        run_tests()
        exit(0)
    if args.relabel:
        relabel_existing_dataset()
        exit(0)
    if args.explain:
        explain_score(args.explain)
        exit(0)

    print("""
  Tarea 2 IAA — Dataset Scraper v3
  Heurístico: 4 ejes ponderados (Brecha/Exag/Emoción/Ambig) 
  Clases: Informativo / Clickbait / Fake News (bonus)
    """)

    df_raw   = run_scraping(target=TARGET_PER_CLASS, include_fake_news=not args.no_fake)
    df_final = save_dataset(df_raw)

    print(f"\n Scraping completado.")
    print(f"   Raw    → {RAW_CSV}")
    print(f"   Final  → {FINAL_CSV}")
    print(f"\nModo explicación:")
    print(f'  python dataset-es.py --explain "Titular a analizar"')
    print(f"\nPara tests:")
    print(f'  python dataset-es.py --test')