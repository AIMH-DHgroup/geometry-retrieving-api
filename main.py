# ======= Import libraries =======  

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import RedirectResponse
from urllib.parse import urlparse, unquote
import urllib.error
from fastapi import UploadFile, File
import xml.etree.ElementTree as ET
from typing import Optional
from langdetect import detect
from uuid import uuid4
#from flair.data import Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter
from sentence_transformers import SentenceTransformer, util
from SPARQLWrapper import SPARQLWrapper, JSON
#from geopy.distance import geodesic
from pydantic import BaseModel
from geoparser import Geoparser
from rapidfuzz import process, fuzz
import spacy
import requests
import time
import pandas as pd
import json
import os
import re
import logging
import sys
import traceback
import tempfile
import zipfile
from typing import Dict, List, Optional, Tuple
 
# ======= Logger =======

# Logger config
logger = logging.getLogger("warnings_logger")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("warnings.txt", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# add handlers to logger
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# ======= Init =======

class WikipediaRateLimitException(Exception):
    pass



# tags_metadata = [
    # {
        # "name": "GeoLinks",
        # "description": "These endpoints are the most performant and should be used by users and for demonstrations.",
    # },
    # {
        # "name": "Test",
        # "description": "These endpoints are under development and have been used for testing and experiments.",
    # },
    # {
        # "name": "Old GeoLinks",
        # "description": "These endpoints use an older version of the algorithm: the entity linking simply involves searching Wikidata for the corresponding entity, always taking the first result. The limitations are obvious.",
    # },
# ]

tags_metadata = [
    {
        "name": "GeoLinks API Functionality",
        "description": "",
    }
]

app = FastAPI(
    title="GeoLinks API",     # docs title
    # description="""GeoLink is a powerful API designed to analyze input text and extract detailed information about geographical or other domain-specific entities.
 
# ### Features
 
# - **Entity Detection**: Identifies entities embedded within arbitrary text inputs.
# - **Rich Contextual Data**: Returns descriptive metadata for each recognized entity, including classification, geolocation (when applicable), and standardized identifiers.
# - **Flexible Usage**: Supports both batch processing and real-time requests.
 
# ### Use Cases
 
# 1. Annotating place names and linking to knowledge bases.
# 2. Enriching text with geo-context for mapping or GIS applications.
# 3. Enabling advanced search by entity attributes within natural language content.""",
description="""**GeoLinks** is a [multilingual](https://spacy.io/usage/models) API that processes either a **text** or a **GeoNames IRI** to enrich geographical entities with geospatial information, including an associated **Wikidata IRI** for each entity. For input text, the API first extracts the geographical entities and then returns their corresponding coordinates (latitude and longitude), polygon geometry, and the associated Wikidata IRI. For entities provided as GeoNames IRIs, coordinates are not retrieved since this information is already available in GeoNames; however, the API retrieves the corresponding polygon geometry and the related Wikidata IRIs. The geographic data are automatically obtained from **Wikidata** and **OpenStreetMap**, and the results are provided in JSON-LD format.
 
In conclusion, the API produces a **GeoSPARQL** graph of interconnected geographical entities enriched with spatial information.

The API code is open-source and available at the following link: [https://github.com/AIMH-DHgroup/geometry-retrieving-api](https://github.com/AIMH-DHgroup/geometry-retrieving-api)

 ### Input: Text

When provided with a text input, GeoLink offers two endpoints for Named Entity Recognition (NER):

- **SpaCy 3.8-based  endpoint**: <code>https://gel.isti.cnr.it/spacy</code> — uses [SpaCy](https://spacy.io/) for NER.

- **SpaCy 3.8+Flair endpoint**: <code>https://gel.isti.cnr.it/spacy-flair</code> — uses a hybrid SpaCy + [Flair](https://github.com/flairNLP/flair) model for enhanced entity recognition.

**Usage Instructions** 

To use the API, send a POST request to one of the URLs above, including the text you want to analyze in the request body.

For example, to analyze the text <code>Paris is the capital city of France</code>, the body of the POST request should contain the following JSON object:

<code>{ "text": "Paris is the capital city of France." }</code>

**Results** 

The results are returned in a JSON-LD file, having the following structure:

<div style="display:inline-block; background-color:#f6f8fa; padding:10px; border-radius:6px;">
<pre><code>{
  "@context": {
    "geo": "https://www.opengis.net/ont/geosparql#",
    "schema": "https://schema.org/"
  },
  "@graph": [
    {
      "@id": "https://wikidata.org/entity/Q90",
      "@type": "geo:Feature",
      "schema:name": "Paris",
      "geo:hasGeometry": [
        {
          "@id": "wd:Q90-geom-point",
          "@type": "geo:Geometry",
          "asWKT": "SRID=4326;POINT (2.352222222 48.856666666)"
        },
        {
          "@id": "wd:Q90-geom-polygon",
          "@type": "geo:Geometry",
          "geo:asWKT": "SRID=4326;MULTIPOLYGON (((2.3198901 48.9004581, ... )))
        }
      ]
    },
    {
      "@id": "https://wikidata.org/entity/Q142",
      "@type": "geo:Feature", 
      "schema:name": "France",
      "geo:hasGeometry": [
        {
          "@id": "wd:Q142-geom-point",
          "@type": "geo:Geometry",
          "asWKT": "SRID=4326;POINT (2.0 47.0)"
        },
        {
          "@id": "wd:Q142-geom-polygon",
          "@type": "geo:Geometry",
          "geo:asWKT": "SRID=4326;MULTIPOLYGON (((6.8700721 45.8284379, ... )))
        }
      ]
    }
  ]
}</code></pre>
    </div>

### Input: GeoNames IRI

For GeoNames IRIs, GeoLink provides the following endpoint:

<code>https://gel.isti.cnr.it/iri</code>

**Usage Instructions** 

To use the API, send a GET request by appending the GeoNames IRI of the entity you want to query to the endpoint URL.

For example, to retrieve information for Paris (GeoNames ID 2988507), whose GeoNames page is <code>https://www.geonames.org/2988507/</code>,
the URL to load is:

<code>https://gel.isti.cnr.it/iri?iri=https://www.geonames.org/2988507/</code>

**Results**

The results are returned in a JSON-LD file, having the following structure:

<div style="display:inline-block; background-color:#f6f8fa; padding:10px; border-radius:6px;">
<pre><code>{
      "@context": {
          "geo": "https://www.opengis.net/ont/geosparql#",
          "schema": "https://schema.org/"
      },
      "@graph": [
        {
          "@id": "https://www.wikidata.org/entity/Q90",
          "@type": "geo:Feature",
          "schema:name": "Paris",
          "geo:hasGeometry": [
             {
               "@id": "wd:Q90-geom-point",
               "@type": "geo:Geometry",
               "asWKT": "SRID=4326;POINT (2.352222222 48.856666666)"
             },
             {
               "@id": "wd:Q90-geom-polygon",
               "@type": "geo:Geometry",
               "geo:asWKT": "SRID=4326;MULTIPOLYGON (((2.3198901 48.9004581, ... )))"
             } 
          ]          
        }
     ]
}</code></pre>
    </div>
""",
    version="1.0.0",
    docs_url="/docs",         # URL Swagger
    redoc_url="/redoc",       # URL ReDoc
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1  # 👈 nasconde completamente la sezione "Schemas"
    }
)

    
    
# ======= FastAPI endpoints =======

# redirect from root to /docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


     
        
        

###SPACY 


QUERY_COUNTER = {
    "wikidata_sparql": 0,
    "wikidata_api": 0,
    "qlever_osm": 0
}

TIME_STATS = {
    "spacy_ner_total": [],
    "tokenize_text": [],
    "extract_geo_entity": [],
    "wikidata_sparql": [],
    "wikidata_api": [],
    "qlever_osm": [],
    "choose_best": [],
    "entity_total": []
}

def add_timing(key: str, elapsed: float):
    if key not in TIME_STATS:
        TIME_STATS[key] = []
    TIME_STATS[key].append(elapsed)

def print_query_stats():
    total = sum(QUERY_COUNTER.values())
    print("\n============================")
    print("API QUERY STATS")
    print("============================")
    print(f"Wikidata SPARQL queries : {QUERY_COUNTER['wikidata_sparql']}")
    print(f"Wikidata API queries    : {QUERY_COUNTER['wikidata_api']}")
    print(f"QLever OSM queries      : {QUERY_COUNTER['qlever_osm']}")
    print(f"TOTAL HTTP queries      : {total}")
    print("============================\n")

def print_timing_stats():
    print("\n============================")
    print("API TIMING STATS")
    print("============================")

    for key, values in TIME_STATS.items():
        if values:
            total = sum(values)
            avg = total / len(values)
            min_v = min(values)
            max_v = max(values)
            print(
                f"{key:20} "
                f"count={len(values):3d}  "
                f"total={total:8.3f}s  "
                f"avg={avg:7.3f}s  "
                f"min={min_v:7.3f}s  "
                f"max={max_v:7.3f}s"
            )

    print("============================\n")



SPACY_MODELS = {
    "en": "en_core_web_trf",
    "it": "it_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "ru": "ru_core_news_sm",
    "pl": "pl_core_news_sm",
    "el": "el_core_news_sm",
    "xx": "xx_ent_wiki_sm"
}

loaded_models: Dict[str, any] = {}

GEOSPARQL_CONTEXT = {
    "@context": {
        "geo": "https://www.opengis.net/ont/geosparql#",
        "schema": "https://schema.org/",
    }
}

sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HTTP_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "GeoEntityLinker/1.0 (aimhdhgroup@gmail.com)"
}


def get_spacy_model(lang: str = "en"):
    if lang not in SPACY_MODELS:
        print(f"⚠️ Unsupported language '{lang}'. Falling back to English.")
        lang = "en"

    model_name = SPACY_MODELS[lang]

    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = spacy.load(model_name)
        except OSError:
            print(f"⚠️ spaCy model '{model_name}' not found. Falling back to English.")
            fallback_model = SPACY_MODELS["en"]
            if fallback_model not in loaded_models:
                loaded_models[fallback_model] = spacy.load(fallback_model)
            return loaded_models[fallback_model]

    return loaded_models[model_name]


def tokenize_text(text: str, lang: str = "en"):
    nlp = get_spacy_model(lang)
    doc = nlp(text)
    return doc, nlp


def extract_geo_entity(doc, context=None):
    geo_labels = {"LOC", "GPE", "FAC"}
    if context is None:
        return [ent.text for ent in doc.ents if ent.label_ in geo_labels]

    return [
        {"text": ent.text, "context": context}
        for ent in doc.ents
        if ent.label_ in geo_labels
    ]


def parse_wikidata_point(coord_value: str) -> Optional[Tuple[float, float]]:
    """
    Converte una stringa WKT tipo 'Point(12.5 41.9)' in (lat, lon).
    """
    if not coord_value or not coord_value.startswith("Point("):
        return None

    try:
        raw = coord_value.replace("Point(", "").replace(")", "").strip()
        lon, lat = map(float, raw.split())
        return lat, lon
    except Exception:
        return None


def escape_sparql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

QLEVER_OSM_URL = "https://qlever.dev/api/osm-planet"

def get_wkt_from_qlever(qid: str, attempt: int = 1) -> Optional[str]:
    
    start_query = time.perf_counter()
    
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX osm: <https://www.openstreetmap.org/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX osm2rdfkey: <https://osm2rdf.cs.uni-freiburg.de/rdf/key#>

    SELECT ?osm_id ?wkt WHERE {{
      ?osm_id osm2rdfkey:wikidata wd:{qid} .
      ?osm_id rdf:type osm:relation .
      ?osm_id geo:hasGeometry/geo:asWKT ?wkt .
      FILTER(
        STRSTARTS(STR(?wkt), "POLYGON")
        || STRSTARTS(STR(?wkt), "MULTIPOLYGON")
      )
    }}
    LIMIT 1
    """

    try:
        QUERY_COUNTER["qlever_osm"] += 1
        print(f"[QLEVER OSM QUERY #{QUERY_COUNTER['qlever_osm']}] qid={qid}")

        response = requests.post(
            QLEVER_OSM_URL,
            data={"query": query},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": HTTP_HEADERS["User-Agent"]
            },
            timeout=60
        )
        response.raise_for_status()
        
        elapsed = time.perf_counter() - start_query
        add_timing("qlever_osm", elapsed)
        print(f"[QLEVER TIME] {qid} -> {elapsed:.3f}s")
        
        data = response.json()

        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["wkt"]["value"]

        return None

    except requests.RequestException as e:
        if attempt < 3:
            wait = [2, 5, 10][attempt - 1]
            print(f"[QLEVER OSM] Error for '{qid}': {e}. Retrying in {wait}s...")
            time.sleep(wait)
            return get_wkt_from_qlever(qid, attempt + 1)

        print(f"[QLEVER OSM] Final error for '{qid}': {e}")
        return None

def query_candidates_full(
    sparql: SPARQLWrapper,
    label: str,
    lang: str = "en",
    attempt: int = 1
) -> List[dict]:
    """
    Una sola query SPARQL per entità.
    Mantiene la tua struttura con match sulla Wikipedia locale.
    Ritorna già: qid, label, description, coord, osm_id
    e filtra gli item geografici direttamente in SPARQL.
    """
    safe_label = escape_sparql_string(label)
    
    start_query = time.perf_counter()
    
    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?description ?coord ?osmId WHERE {{
      ?article schema:about ?item ;
               schema:isPartOf <https://{lang}.wikipedia.org/> ;
               schema:name "{safe_label}"@{lang} .

      ?item wdt:P31 ?type .
      ?type wdt:P279* wd:Q618123 .

      OPTIONAL {{
        ?item schema:description ?description .
        FILTER(LANG(?description) = "{lang}")
      }}

      OPTIONAL {{ ?item wdt:P625 ?coord }}
      OPTIONAL {{ ?item wdt:P402 ?osmId }}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{lang},en"
      }}
    }}
    LIMIT 10
    """

    try:
        sparql.setQuery(query)
        
        QUERY_COUNTER["wikidata_sparql"] += 1
        print(f"[SPARQL QUERY #{QUERY_COUNTER['wikidata_sparql']}] label='{label}' lang='{lang}'")
        
        results = sparql.query().convert()["results"]["bindings"]
        
        elapsed = time.perf_counter() - start_query
        add_timing("wikidata_sparql", elapsed)
        print(f"[SPARQL TIME] '{label}' -> {elapsed:.3f}s")

        out = []
        for r in results:
            qid = r["item"]["value"].split("/")[-1]
            coord_raw = r.get("coord", {}).get("value", "")
            latlon = parse_wikidata_point(coord_raw)

            out.append({
                "qid": qid,
                "label": r.get("itemLabel", {}).get("value", label),
                "description": r.get("description", {}).get("value", ""),
                "coord": latlon,
                "osm_id": r.get("osmId", {}).get("value")
            })

        print(f"\n[WIKIDATA] Candidates found for '{label}' ({lang}):")
        print(out)
        return out

    except Exception as e:
        if attempt < 3:
            wait = [2, 5, 10][attempt - 1]
            print(f"[WIKIDATA] Error for '{label}': {e}. Retrying in {wait}s...")
            time.sleep(wait)
            return query_candidates_full(sparql, label, lang, attempt + 1)

        print(f"[WIKIDATA] Final error for '{label}': {e}")
        return []


def get_wikidata_coord(qid: str, attempt: int = 1) -> Optional[Tuple[float, float]]:

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": qid,
        "props": "claims"
    }

    try:
        start_query = time.perf_counter()

        QUERY_COUNTER["wikidata_api"] += 1
        print(f"[WIKIDATA API #{QUERY_COUNTER['wikidata_api']}] get coord for '{qid}'")

        response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=20)
        response.raise_for_status()

        elapsed = time.perf_counter() - start_query
        add_timing("wikidata_api", elapsed)
        print(f"[WIKIDATA API TIME] coord '{qid}' -> {elapsed:.3f}s")

        data = response.json()
        entity = data.get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})
        p625 = claims.get("P625", [])

        if not p625:
            return None

        mainsnak = p625[0].get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        value = datavalue.get("value", {})

        lat = value.get("latitude")
        lon = value.get("longitude")

        if lat is None or lon is None:
            return None

        return (lat, lon)

    except requests.RequestException as e:
        if attempt < 3:
            wait = [2, 5, 10][attempt - 1]
            print(f"[WIKIDATA API] Error getting coord for '{qid}': {e}. Retrying in {wait}s...")
            time.sleep(wait)
            return get_wikidata_coord(qid, attempt + 1)

        print(f"[WIKIDATA API] Final error getting coord for '{qid}': {e}")
        return None
        
def search_wikidata_entity(query: str, language: str = "en", attempt: int = 1) -> Optional[dict]:
    """
    Fallback solo se la query SPARQL con match Wikipedia non trova nulla.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": query,
        "language": language,
        "limit": 10
    }

    try:
        start_query = time.perf_counter()
        
        QUERY_COUNTER["wikidata_api"] += 1
        print(f"[WIKIDATA API #{QUERY_COUNTER['wikidata_api']}] search='{query}'")
        response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=20)
        response.raise_for_status()
        
        elapsed = time.perf_counter() - start_query
        add_timing("wikidata_api", elapsed)
        print(f"[WIKIDATA API TIME] '{query}' -> {elapsed:.3f}s")
        
        results = response.json().get("search", [])

        print(f"\n[WIKIDATA API] Search fallback for '{query}':")
        print(results)

        if not results:
            return None

        # qui prendiamo il primo risultato come fallback debole.
        result = results[0]
        return {
            "id": result["id"],
            "label": result.get("label", query),
            "description": result.get("description", ""),
            "coord": None,
            "osm_id": None
        }

    except requests.RequestException as e:
        if attempt < 3:
            wait = [2, 5, 10][attempt - 1]
            print(f"[WIKIDATA API] Error for '{query}': {e}. Retrying in {wait}s...")
            time.sleep(wait)
            return search_wikidata_entity(query, language, attempt + 1)

        print(f"[WIKIDATA API] Final error for '{query}': {e}")
        return None


def choose_best(model, candidates: List[dict], context: str) -> Optional[dict]:
    if not candidates:
        return None

    context_embedding = model.encode(context, convert_to_tensor=True)
    scored = []

    for c in candidates:
        description = c.get("description") or c.get("label") or ""
        desc_embedding = model.encode(description, convert_to_tensor=True)
        sim = util.cos_sim(context_embedding, desc_embedding).item()
        scored.append((sim, c))

    best_score, best_candidate = max(scored, key=lambda x: x[0])
    best_candidate["cosine"] = best_score
    return best_candidate


def get_geometry_from_osm(osm_id: str, attempt: int = 1) -> List[List[Tuple[float, float]]]:
    """
    Recupera la geometria della relation OSM da Overpass.
    Ritorna una lista di linee/poligoni, ciascuno come lista di tuple (lon, lat).
    """
    query = f"""
    [out:json][timeout:60];
    relation({osm_id});
    out geom;
    """

    try:
        QUERY_COUNTER["osm_overpass"] += 1
        print(f"[OSM QUERY #{QUERY_COUNTER['osm_overpass']}] relation={osm_id}")
        response = requests.get(
            OVERPASS_URL,
            params={"data": query},
            headers={"User-Agent": HTTP_HEADERS["User-Agent"]},
            timeout=90
        )
        response.raise_for_status()
        data = response.json()

        coordinates = []

        for element in data.get("elements", []):
            for member in element.get("members", []):
                geom = member.get("geometry")
                if geom:
                    coords = [(pt["lon"], pt["lat"]) for pt in geom]
                    if coords:
                        coordinates.append(coords)

        return coordinates

    except requests.RequestException as e:
        if attempt < 3:
            wait = [2, 5, 10][attempt - 1]
            print(f"[OSM] Error for relation '{osm_id}': {e}. Retrying in {wait}s...")
            time.sleep(wait)
            return get_geometry_from_osm(osm_id, attempt + 1)

        print(f"[OSM] Final error for relation '{osm_id}': {e}")
        return []


def ring_is_closed(coords: List[Tuple[float, float]]) -> bool:
    return len(coords) >= 4 and coords[0] == coords[-1]


def convert_to_wkt(geometries: List[List[Tuple[float, float]]]) -> Optional[str]:
    """
    Converte l'output di Overpass in WKT.
    Heuristica semplice:
    - 1 geometria chiusa -> POLYGON
    - più geometrie chiuse -> MULTIPOLYGON
    - altrimenti MULTILINESTRING
    """
    if not geometries:
        return None

    closed_geometries = [g for g in geometries if ring_is_closed(g)]
    open_geometries = [g for g in geometries if not ring_is_closed(g)]

    def format_coords(coords: List[Tuple[float, float]]) -> str:
        return ", ".join(f"{lon} {lat}" for lon, lat in coords)

    if closed_geometries and not open_geometries:
        if len(closed_geometries) == 1:
            return f"POLYGON (({format_coords(closed_geometries[0])}))"

        parts = []
        for poly in closed_geometries:
            parts.append(f"(({format_coords(poly)}))")
        return f"MULTIPOLYGON ({', '.join(parts)})"

    multiline_parts = [f"({format_coords(line)})" for line in geometries if len(line) >= 2]
    if multiline_parts:
        if len(multiline_parts) == 1:
            return f"LINESTRING {multiline_parts[0]}"
        return f"MULTILINESTRING ({', '.join(multiline_parts)})"

    return None


def append_feature(row: dict, label: str, qid: str):
    row["entities"].append({
        "text_label": label,
        "Wikidata_ID": qid
    })


def do_geosparql_from_candidate(candidate: dict, fts: List[dict]):
    qid = candidate["qid"]
    label = candidate["label"]
    coord = candidate.get("coord")

    wkt_geom = get_wkt_from_qlever(qid)

    # fallback ulteriore: se non ho poligono e non ho punto, provo a prendere P625 da Wikidata API
    if not wkt_geom and not coord:
        coord = get_wikidata_coord(qid)

    wkt_point = None

    if coord:
        lat, lon = coord
        wkt_point = f"POINT ({lon} {lat})"

    geometries = []

    if wkt_point:
        geometries.append({
            "@id": f"wd:{qid}-geom-point",
            "@type": "geo:Geometry",
            "geo:asWKT": f"SRID=4326;{wkt_point}"
        })

    if wkt_geom:
        geometries.append({
            "@id": f"wd:{qid}-geom-main",
            "@type": "geo:Geometry",
            "geo:asWKT": wkt_geom if wkt_geom.startswith("SRID=") else f"SRID=4326;{wkt_geom}"
        })

    if geometries:
        feature_geo = {
            "@id": f"https://wikidata.org/entity/{qid}",
            "@type": "geo:Feature",
            "schema:name": label,
            "geo:hasGeometry": geometries if len(geometries) > 1 else geometries[0]
        }
        fts.append(feature_geo)


def download_features(geosparql_doc: dict):
    """
    Placeholder semplice.
    Se hai già una tua funzione custom, puoi rimettere la tua.
    """
    return {
        "filename": "features.json",
        "content": geosparql_doc
    }

















###ENDPOINT


class SpacyInput(BaseModel):
    text: str
    lang: str = "en"
    download: bool = False
    
@app.post("/spacy", tags=["GeoLinks API Functionality"], summary=" ", operation_id="")
async def spacy_ner(payload: SpacyInput):
    try:
        start_time = time.time()
        text = payload.text
        lang = payload.lang
        download = payload.download

        features = []
        features_geosparql = []
        
        start_total = time.perf_counter()
        t0 = time.perf_counter()
        
        doc, _ = tokenize_text(text, lang=lang)
        
        add_timing("tokenize_text", time.perf_counter() - t0)
        print("All entities found:")
        print(doc.ents)
        
        
        t0 = time.perf_counter()
        entities_spacy = extract_geo_entity(doc, text)
        add_timing("extract_geo_entity", time.perf_counter() - t0)
        print("Geo entities with context:")
        print(entities_spacy)

        row = {"row": 1, "entities": []}

        sparql = SPARQLWrapper(WIKIDATA_SPARQL_URL)
        sparql.setReturnFormat(JSON)

        processed_qids = set()

        for ent in entities_spacy:
            entity_start = time.perf_counter()

            label = ent["text"]
            context = ent["context"]

            candidates = query_candidates_full(sparql, label, lang)

            if candidates:
                t0 = time.perf_counter()
                best = choose_best(sentence_model, candidates, context)
                elapsed_best = time.perf_counter() - t0
                add_timing("choose_best", elapsed_best)
                print(f"[RANKING TIME] '{label}' -> {elapsed_best:.3f}s")

                if best:
                    print("\nBest candidate selected:")
                    print(best)

                    if best["qid"] not in processed_qids:
                        append_feature(row, label, best["qid"])
                        do_geosparql_from_candidate(best, features_geosparql)
                        processed_qids.add(best["qid"])

            else:
                fallback_entity = search_wikidata_entity(label, lang)
                if fallback_entity and fallback_entity["id"] not in processed_qids:
                    append_feature(row, label, fallback_entity["id"])

                    do_geosparql_from_candidate(
                        {
                            "qid": fallback_entity["id"],
                            "label": fallback_entity.get("label", label),
                            "description": fallback_entity.get("description", ""),
                            "coord": fallback_entity.get("coord"),
                            "osm_id": fallback_entity.get("osm_id")
                        },
                        features_geosparql
                    )

                    processed_qids.add(fallback_entity["id"])

            entity_elapsed = time.perf_counter() - entity_start
            add_timing("entity_total", entity_elapsed)
            print(f"[ENTITY TOTAL TIME] '{label}' -> {entity_elapsed:.3f}s")

        features.append(row)

        geosparql_doc = {
            **GEOSPARQL_CONTEXT,
            "@graph": features_geosparql
        }
        
        total_elapsed = time.perf_counter() - start_total
        add_timing("spacy_ner_total", total_elapsed)
        print(f"[TOTAL API TIME] {total_elapsed:.3f}s")

        print_query_stats()
        print_timing_stats()
        
        return geosparql_doc if not download else download_features(geosparql_doc)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text_last = tb[-1]
        error_message = f'{str(e)} (File "{filename}", line {lineno}, in {func}: {text_last})'
        raise HTTPException(status_code=500, detail=error_message)
        
        
        
        
class SpacyFlairPayload(BaseModel):
    text: str
    download: bool = False


@app.post("/spacy-flair", tags=["GeoLinks API Functionality"], summary=" ", operation_id="")
async def ner_spacy_flair(payload: SpacyFlairPayload):
    try:
        text = payload.text
        download = payload.download
        lang = "en"

        content = [text]

        features = []
        features_geosparql = []

        sparql = SPARQLWrapper(WIKIDATA_SPARQL_URL)
        sparql.setReturnFormat(JSON)

        tagger = SequenceTagger.load("ner")
        splitter = SegtokSentenceSplitter()

        processed_qids = set()

        for i, event in enumerate(content):
            row = {"row": i + 1, "entities": []}

            # spaCy
            doc, _ = tokenize_text(event, lang=lang)
            entities_spacy = extract_geo_entity(doc, event)

            # Flair
            sentences = splitter.split(event)
            tagger.predict(sentences)

            existing_texts = {item["text"] for item in entities_spacy}

            for sentence in sentences:
                for entity in sentence.get_spans("ner"):
                    if entity.get_label("ner").value == "LOC":
                        if entity.text not in existing_texts:
                            entities_spacy.append({
                                "text": entity.text,
                                "context": sentence.to_original_text()
                            })
                            existing_texts.add(entity.text)

            print("Geo entities with context (spaCy + Flair):")
            print(entities_spacy)

            for ent in entities_spacy:
                label = ent["text"]
                context = ent["context"]

                candidates = query_candidates_full(sparql, label, lang)

                if candidates:
                    t0 = time.perf_counter()
                    best = choose_best(sentence_model, candidates, context)
                    elapsed_best = time.perf_counter() - t0
                    add_timing("choose_best", elapsed_best)
                    print(f"[RANKING TIME] '{label}' -> {elapsed_best:.3f}s")

                    if best:
                        print("\nBest candidate selected:")
                        print(best)

                        if best["qid"] not in processed_qids:
                            append_feature(row, label, best["qid"])
                            do_geosparql_from_candidate(best, features_geosparql)
                            processed_qids.add(best["qid"])

                else:
                    fallback_entity = search_wikidata_entity(label, lang)

                    if fallback_entity and fallback_entity["id"] not in processed_qids:
                        append_feature(row, label, fallback_entity["id"])

                        do_geosparql_from_candidate(
                            {
                                "qid": fallback_entity["id"],
                                "label": fallback_entity.get("label", label),
                                "description": fallback_entity.get("description", ""),
                                "coord": fallback_entity.get("coord"),
                                "osm_id": fallback_entity.get("osm_id")
                            },
                            features_geosparql
                        )

                        processed_qids.add(fallback_entity["id"])

            features.append(row)

        geosparql_doc = {
            **GEOSPARQL_CONTEXT,
            "@graph": features_geosparql
        }

        if not download:
            return JSONResponse(content=geosparql_doc, media_type="application/json")

        return download_features(geosparql_doc)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text_last = tb[-1]
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text_last})"
        raise HTTPException(status_code=500, detail=error_message)      
        
        
@app.get("/iri", tags=["GeoLinks API Functionality"], summary=" ")
async def read_iri_geonames(
    iri: str = Query(..., description="GeoNames IRI (e.g. https://www.geonames.org/2988507/)"),
    download: bool = Query(False, description="If true, returns downloadable JSON-LD")
):
    try:
        lang = "en"

        match = re.search(r"/(\d+)/?", iri)
        if not match:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid GeoNames IRI format."}
            )

        geonames_id = match.group(1)
        print(f"[GEONAMES ID] {geonames_id}")

        sparql = SPARQLWrapper(WIKIDATA_SPARQL_URL)
        sparql.setReturnFormat(JSON)

        start_query = time.perf_counter()

        query = f"""
        SELECT ?item ?itemLabel ?description ?coord ?osmId WHERE {{
          ?item wdt:P1566 "{geonames_id}" .

          OPTIONAL {{
            ?item schema:description ?description .
            FILTER(LANG(?description) = "{lang}")
          }}

          OPTIONAL {{ ?item wdt:P625 ?coord }}
          OPTIONAL {{ ?item wdt:P402 ?osmId }}

          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "{lang},en"
          }}
        }}
        LIMIT 1
        """

        QUERY_COUNTER["wikidata_sparql"] += 1
        print(f"[SPARQL QUERY #{QUERY_COUNTER['wikidata_sparql']}] geonames_id='{geonames_id}'")

        sparql.setQuery(query)
        results = sparql.query().convert()["results"]["bindings"]

        elapsed = time.perf_counter() - start_query
        add_timing("wikidata_sparql", elapsed)
        print(f"[SPARQL TIME] geonames_id='{geonames_id}' -> {elapsed:.3f}s")

        if not results:
            return JSONResponse(
                status_code=404,
                content={"error": f"No Wikidata entity found for GeoNames ID {geonames_id}."}
            )

        r = results[0]
        qid = r["item"]["value"].split("/")[-1]
        label = r.get("itemLabel", {}).get("value", "")
        description = r.get("description", {}).get("value", "")
        coord_raw = r.get("coord", {}).get("value", "")
        coord = parse_wikidata_point(coord_raw)
        osm_id = r.get("osmId", {}).get("value")

        candidate = {
            "qid": qid,
            "label": label,
            "description": description,
            "coord": coord,
            "osm_id": osm_id
        }

        features_geosparql = []
        do_geosparql_from_candidate(candidate, features_geosparql)

        geosparql_doc = {
            **GEOSPARQL_CONTEXT,
            "@graph": features_geosparql
        }

        if not download:
            return JSONResponse(
                content=geosparql_doc,
                media_type="application/ld+json"
            )

        return download_features(geosparql_doc)

    except Exception as e:
        full_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\nTraceback:\n{full_trace}")