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

GEOSPARQL_CONTEXT = {
    "@context": {
        "geo":        "https://www.opengis.net/ont/geosparql#",
        "schema":     "https://schema.org/",
        "xsd":        "http://www.w3.org/2001/XMLSchema#",
        "label":      "schema:name",
        "description":"schema:description",
        "qid":        "schema:identifier",
        "wikidata":   "schema:sameAs",
        "osm_id":     "schema:identifier",
        "Feature":    "geo:Feature",
        "Geometry":   "geo:Geometry",
        "hasGeometry":"geo:hasGeometry",
        "asWKT": {
            "@id": "geo:asWKT",
            "@type": "geo:wktLiteral"
        }
    }
}

SUPPORTED_LANGUAGES = ["en", "it", "de", "fr", "es", "pt", "nl", "ru", "pl", "xx"]  # official Wikifier supported languages

WIKIFIER_API_KEY = "xlwepdphbtmqmnyjysnyeubopqovgm"

not_supported_message = "Language not supported. Please insert one value among \'en\' (English), \'it\' (Italian), \'fr\' (French), \'de\' (Deutsch), \'ru\' (Russian), \'pt\' (Portuguese), \'es\' (Spanish), \'nl\' (Dutch) , \'pl\' (Polish) or \'xx\' (for multi language texts)."

tags_metadata = [
    {
        "name": "GeoLinks",
        "description": "These endpoints are the most performant and should be used by users and for demonstrations.",
    },
    {
        "name": "Test",
        "description": "These endpoints are under development and have been used for testing and experiments.",
    },
    {
        "name": "Old GeoLinks",
        "description": "These endpoints use an old version of the algorithm: the entity linking simply involves searching Wikidata for the entity provided as input, always taking the first result. The limitations are obvious.",
    },
]

app = FastAPI(
    title="GeoLinks API",     # docs title
    description="""GeoLink is a powerful API designed to analyze input text and extract detailed information about geographical or other domain-specific entities.
 
### Features
 
- **Entity Detection**: Identifies entities embedded within arbitrary text inputs.
- **Rich Contextual Data**: Returns descriptive metadata for each recognized entity, including classification, geolocation (when applicable), and standardized identifiers.
- **Flexible Usage**: Supports both batch processing and real-time requests.
 
### Use Cases
 
1. Annotating place names and linking to knowledge bases.
2. Enriching text with geo-context for mapping or GIS applications.
3. Enabling advanced search by entity attributes within natural language content.""",
    version="1.0.0",
    docs_url="/docs",         # URL Swagger
    redoc_url="/redoc",       # URL ReDoc
    openapi_tags=tags_metadata
)

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
    "xx": "xx_ent_wiki_sm" # multilanguage
}

loaded_models = {}

#WIKIFIER_API_KEY = os.getenv("WIKIFIER_API_KEY")
#if not WIKIFIER_API_KEY:
#    raise EnvironmentError("WIKIFIER_API_KEY not defined in environment.")


# ======= Pydantic model =======
class TextInput(BaseModel):
    text: str
    lang: Optional[str] = "en"


# ======= Utility functions =======

def get_spacy_model(lang="en"):
    model_name = SPACY_MODELS.get(lang, "en_core_web_trf")
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = spacy.load(model_name)
        except OSError:
            print(f"⚠️ spaCy model '{model_name}' not found. Use fallback 'en_core_web_trf'.")
            model_name = "en_core_web_trf"
            loaded_models[model_name] = spacy.load(model_name)
    return loaded_models[model_name]

def tokenize_text(text, lang="en"):
    nlp = get_spacy_model(lang)
    doc = nlp(text)
    return doc, nlp

def extract_geo_entity(doc, context=None):
    if context is None:
        return [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE", "NOUN", "PROPN"]]
    else:
        return [{"text": ent.text, "context": context} for ent in doc.ents if ent.label_ in ["LOC", "GPE", "NOUN", "PROPN"]]

def disambiguation_with_wikifier(text, lang="en"):
    url = "https://www.wikifier.org/annotate-article"
    data = {
        "text": text,
        "lang": lang,
        "userKey": WIKIFIER_API_KEY,
        "support": "true",
        "pageRankSqThreshold": "0.8",
        "applyFilters": "true",
        "filterCategories": "true",
        "threshold": "0.8",
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json().get("annotations", [])

def call_wikifier(text: str, lang: str = "en", threshold: float = 0.8):
    url = "https://www.wikifier.org/annotate-article"
    data = {
        'text': text,
        'lang': lang,
        'userKey': WIKIFIER_API_KEY,
        'support': 1,
        'pageRankSqThreshold': threshold,
        'applyPageRankSqThreshold': True,
        'nTopDfValuesToIgnore': 200,
        'fastMode': False
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json().get("annotations", [])

def is_geographic_entity(qid, attempt=1):
    try:
        query = f"""
        ASK {{
          wd:{qid} wdt:P31 ?type .
          ?type wdt:P279* wd:Q618123 .
        }}
        """
        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/sparql-results+json"}
        response = requests.get(url, params={"query": query}, headers=headers)
        response.raise_for_status()
        return response.json()['boolean']
    except urllib.error.HTTPError as e:
        if e.code == 504:
            print(f"[WIKIDATA] Timeout for entity '{qid}'")
        elif e.code == 429:
            attempt += 1
            wait = 2
            if attempt == 2:
                wait = 5
            elif attempt == 3:
                wait = 10
            if attempt <= 3:
                print(f"[WIKIDATA] Too many requests for entity '{qid}'. Retrying attempt {attempt}...")
                time.sleep(wait)
                is_geographic_entity(qid, attempt)
            else:
                print(f"[WIKIDATA] Too many requests for entity '{qid}'. Skipping...")
        else:
            print(f"[WIKIDATA] Error: {e} for entity '{qid}'")
            return []

def get_osm_relation_id(qid):
    query = f"""
    SELECT ?osmId WHERE {{
      wd:{qid} wdt:P402 ?osmId .
    }}
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={"query": query}, headers=headers)
    response.raise_for_status()
    bindings = response.json()["results"]["bindings"]
    if bindings:
        return bindings[0]["osmId"]["value"]
    return None

def get_geometry_from_osm(osm_id):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    relation({osm_id});
    out geom;
    """
    response = requests.get(overpass_url, params={"data": query})
    response.raise_for_status()
    data = response.json()
    coordinates = []
    for element in data["elements"]:
        for el in element:
            if el == "members":
                for e in element[el]:
                    for prop in e:
                        if prop == "geometry":
                            coords = [(pt["lon"], pt["lat"]) for pt in e[prop]]
                            if coords:
                                coordinates.append(coords)
    return coordinates

def convert_to_vkt(coordinates):
    from shapely.geometry import Polygon, MultiPolygon
    polygons = [Polygon(coords) for coords in coordinates if len(coords) >= 3]
    if not polygons:
        return None
    multi = MultiPolygon(polygons)
    return multi.wkt

#def save_geojson(file, filename="output.geojson"):
#    features = []
#
#    for res in file:
#        vkt_value = res.get("vkt")
#        if not vkt_value:
#            continue
#        try:
#            shape = wkt.loads(vkt_value)
#            geojson_geom = geojson.Feature(
#                geometry=geojson.loads(geojson.dumps(shape.__geo_interface__)),
#                properties={
#                    "label": res["label"],
#                    "qid": res["qid"],
#                    "description": res.get("description"),
#                    "wikidata_url": res["wikidata_url"],
#                    "osm_id": res.get("osm_id")
#                }
#            )
#            features.append(geojson_geom)
#        except Exception as e:
#            print(f"❌ Error converting GeoJSON to {res['label']}: {e}")
#
#    feature_collection = geojson.FeatureCollection(features)
#    with open(filename, "w", encoding="utf-8") as f:
#        geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)
#    print(f"\n✅ GeoJSON saved in: {filename}")

def find_similar_string(target, candidates, threshold=0.7):
    result = process.extractOne(target, candidates, scorer=fuzz.token_sort_ratio)

    if result and result[1] / 100 >= threshold:
        return result[0]
    return None

def get_coordinates_from_wikidata(qid):
    query = f"""
    SELECT ?coord WHERE {{
      wd:{qid} wdt:P625 ?coord .
    }}
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={"query": query}, headers=headers)
    response.raise_for_status()
    coors_bindings = response.json()["results"]["bindings"]
    if coors_bindings:
        coord_str = coors_bindings[0]["coord"]["value"]
        if coord_str.startswith("Point("):  # WKT
            parts = coord_str[6:-1].split()
            lon, lat = float(parts[0]), float(parts[1])
            return lat, lon
    return None

def fallback_wikidata_search(entity_text, lang="en"):
    """
    Search for an entity on Wikidata using the search bar (wbsearchentities API),
    similar to the website behavior.
    Returns the first result if available.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "language": lang,
        "format": "json",
        "limit": 1
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if data.get("search"):
        result = data["search"][0]
        return {
            "wikiDataItemId": result["id"],
            "title": result.get("label", entity_text),
            "description": result.get("description", "")
        }

    return None

def segment_by_language(text, nlp):
    segments = []
    current_lang = None
    current_block = []

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc = nlp(text)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        try:
            lang = detect(sent_text)
        except:
            lang = "en"  # fallback

        if lang != current_lang:
            if current_block:
                segments.append({
                    "lang": current_lang,
                    "text": " ".join(current_block)
                })
            current_block = [sent_text]
            current_lang = lang
        else:
            current_block.append(sent_text)

    if current_block:
        segments.append({
            "lang": current_lang,
            "text": " ".join(current_block)
        })

    return segments

def retrieve_geometry(annotation, label, qid, entities, processed_qids, only_geometry, allow_duplicates):
    try:
        if not only_geometry:
            if annotation.get("cosine", 1.0) < 0.5 or not is_geographic_entity(qid):
                return
        elif allow_duplicates:
            if qid in processed_qids:
                print(f"\n⚠️ Skipping {qid}, already processed.")
                return
        print(f"\n🔍 Entity check: {label} ({qid})...")
        osm_id = get_osm_relation_id(qid)
        print(f"✔️ It is geographic - OSM ID: {osm_id}")
        vkt = None
        if osm_id:
            coords = get_geometry_from_osm(osm_id)
            if coords:
                vkt = convert_to_vkt(coords)
            else:
                print("⚠️ No OSM geometry found. Trying with coordinates...")
        if not vkt:
            coords_point = get_coordinates_from_wikidata(qid)
            if coords_point:
                lat, lon = coords_point
                vkt = f"POINT ({lon} {lat})"
                print(f"📍 Coordinates found: {lat}, {lon}")
                print(f"📍 VKT: {vkt[:80]}..." if vkt else "⚠️ No valid geometry.")
        else:
            geom_type = vkt.split()[0]
            print(f"📐 Geometry type: {geom_type}")

        if not only_geometry:
            description = annotation.get("description")
        else:
            description = ""

        entities.append({
            "label": label,
            "qid": qid,
            "description": description,
            "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
            "osm_id": osm_id,
            "vkt": vkt,
            "wkt": f"SRID=4326;{vkt}"  # compliant with geo:wktLiteral
        })

        if not only_geometry and not allow_duplicates:
            processed_qids.add(qid)

        time.sleep(4)  # Avoid rate limit

        if only_geometry:
            return entities

    except Exception as e:
        print(f"❌ Error with {label}: {e}")
        print("Retrying...")
        time.sleep(10)
        retrieve_geometry(annotation, label, qid, entities, processed_qids, only_geometry, allow_duplicates)

def process_annotation(annotation, processed_qids, entities):
    try:
        qid = annotation["wikiDataItemId"]
        label = annotation["title"]
    except KeyError as e:
        print(f"\n⚠️ Warning. The key {e} is missing from the annotation '{annotation['title']}'.")
        return

    if qid in processed_qids:
        return

    retrieve_geometry(annotation, label, qid, entities, processed_qids, False, False)

def analyze(annotation_text, entities, processed_qids):
    for ann in annotation_text:
        process_annotation(ann, processed_qids, entities)

def detect_spacy_and_fallback(entities_spacy, processed_qids, entities, lg, to_detect):
    for ent_text in entities_spacy:

        if to_detect:
            try:
                lg = detect(ent_text)
            except:
                lg = "en"  # fallback

        ent_annotations = disambiguation_with_wikifier(ent_text, lg)
        if not ent_annotations:
            print(f"\n⚠️ No annotations from Wikifier for: '({lg}) {ent_text}', trying fallback...")
            fallback_result = fallback_wikidata_search(ent_text, lg)
            if fallback_result:
                process_annotation(fallback_result, processed_qids, entities)

        else:
            for ann in ent_annotations:
                process_annotation(ann, processed_qids, entities)

def analyze_text(text, lang="en"):
    doc, nlp = tokenize_text(text, lang=lang)
    entities_spacy = extract_geo_entity(doc)
    print(f"\nEntities found by spaCy: {', '.join(entities_spacy)}")

    entities = []
    processed_qids = set()

    # workflow: Wikifier disambiguation of the entities found by spaCy and then repeat the disambiguation of all the text by Wikifier
    # the difference between mixed language and a single one is that in the first case we need to detect the language of each phrase
    if lang == "xx":

        detect_spacy_and_fallback(entities_spacy, processed_qids, entities, lang, to_detect=True)

        # then try again and leave to Wikifier all the tasks
        multilingual_segments = segment_by_language(text, nlp)

        for segment in multilingual_segments:
            entities_temp = []
            annotations = disambiguation_with_wikifier(segment['text'], lang=segment['lang'])
            analyze(annotations, entities_temp, processed_qids)
            entities.extend(entities_temp)

    else:
        detect_spacy_and_fallback(entities_spacy, processed_qids, entities, lang, to_detect=False)
        annotations = disambiguation_with_wikifier(text, lang)
        analyze(annotations, entities, processed_qids)

    return entities

def perform_sparql_query(query: str):
    endpoint = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json"
    }
    response = requests.get(endpoint, params={"query": query}, headers=headers)
    if response.status_code == 200:
        return response.json().get("results", {}).get("bindings", [])
    else:
        return []


def get_wikipedia_article_from_geonames(geonames_iri):
    rdf_url = geonames_iri.rstrip('/') + '/about.rdf'
    response = requests.get(rdf_url)

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ns = {
            'gn': 'https://www.geonames.org/ontology#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }

        for wiki_elem in root.findall('.//gn:wikipediaArticle', ns):
            url = wiki_elem.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            if url and 'en.wikipedia.org' in url:
                return url

    return None


def get_wikidata_entity_from_geonames(geonames_iri):
    rdf_url = geonames_iri.rstrip('/') + '/about.rdf'
    response = requests.get(rdf_url)
    wikipedia_url = ''

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ns = {
            'gn': 'https://www.geonames.org/ontology#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }

        for wiki_elem in root.findall('.//gn:wikipediaArticle', ns):
            url = wiki_elem.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            if url and 'en.wikipedia.org' in url:
                wikipedia_url = url
                break

    if wikipedia_url and 'en.wikipedia.org' in wikipedia_url:
        parsed_url = urlparse(wikipedia_url)
        title = unquote(parsed_url.path.split("/wiki/")[-1])

        wiki_api_url = f"https://{parsed_url.hostname}/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageprops",
            "format": "json"
        }

        headers = {
            "User-Agent": "MyPythonScript/1.0 (claudio.demartino@isti.cnr.it)"
        }

        try:
            response = requests.get(wiki_api_url, params=params, headers=headers)

            if response.status_code == 429:
                raise WikipediaRateLimitException("Rate limit exceeded (HTTP 429). Try again later.")

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                if data["error"].get("code") == "ratelimited":
                    raise WikipediaRateLimitException("Rate limit exceeded (API error 'ratelimited'). Try again later.")
                else:
                    raise Exception(f"API returned an error: {data['error']}")

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                wikidata_id = page.get("pageprops", {}).get("wikibase_item")
                if wikidata_id:
                    wikidata_api_url = "https://www.wikidata.org/w/api.php"
                    label_params = {
                        "action": "wbgetentities",
                        "ids": wikidata_id,
                        "format": "json",
                        "props": "labels",
                        "languages": "en"
                    }
                    wd_response = requests.get(wikidata_api_url, params=label_params, headers=headers)
                    wd_response.raise_for_status()
                    wd_data = wd_response.json()

                    label = wd_data.get("entities", {}).get(wikidata_id, {}).get("labels", {}).get("en", {}).get("value", "")
                    return {"id": wikidata_id, "label": label}

            return {}

        except requests.RequestException as e:
            raise Exception(f"HTTP request failed: {e}")


def get_geonames_label(geonames_id):
    #rdf_url = geonames_iri.rstrip('/') + '/about.rdf'
    rdf_url = f"https://www.geonames.org/{geonames_id}/about.rdf"
    response = requests.get(rdf_url)

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ns = {
            'gn': 'https://www.geonames.org/ontology#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
        }

        label_elem = root.find('.//gn:name', ns)
        if label_elem is not None and label_elem.text:
            return label_elem.text

    return None


def get_wikidata_entity_from_wikipedia_url(wikipedia_url: str, language: str = "en") -> dict:
    parsed_url = urlparse(wikipedia_url)
    title = unquote(parsed_url.path.split("/wiki/")[-1])

    wiki_api_url = f"https://{parsed_url.hostname}/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json"
    }

    headers = {
        "User-Agent": "MyPythonScript/1.0 (claudio.demartino@isti.cnr.it)"
    }

    try:
        response = requests.get(wiki_api_url, params=params, headers=headers)

        if response.status_code == 429:
            raise WikipediaRateLimitException("Rate limit exceeded (HTTP 429). Try again later.")

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            if data["error"].get("code") == "ratelimited":
                raise WikipediaRateLimitException("Rate limit exceeded (API error 'ratelimited'). Try again later.")
            else:
                raise Exception(f"API returned an error: {data['error']}")

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            wikidata_id = page.get("pageprops", {}).get("wikibase_item")
            if wikidata_id:
                wikidata_api_url = "https://www.wikidata.org/w/api.php"
                label_params = {
                    "action": "wbgetentities",
                    "ids": wikidata_id,
                    "format": "json",
                    "props": "labels",
                    "languages": language
                }
                wd_response = requests.get(wikidata_api_url, params=label_params, headers=headers)
                wd_response.raise_for_status()
                wd_data = wd_response.json()

                label = wd_data.get("entities", {}).get(wikidata_id, {}).get("labels", {}).get(language, {}).get("value", "")
                return {"id": wikidata_id, "label": label}

        return {}

    except requests.RequestException as e:
        raise Exception(f"HTTP request failed: {e}")


def search_wikidata_entity(query, language='en', attempt=1):
    url = "https://www.wikidata.org/w/api.php"
    headers = {
        "User-Agent": "GeoEntityLinker/1.0 (aimhdhgroup@gmail.com)"
    }
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": query,
        "language": language,
        "limit": 10
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json().get("search", [])

        for result in results:
            entity_id = result["id"]
            if is_geographic_entity(entity_id):
                return {
                    "id": result["id"],
                    "label": result.get("label"),
                    "description": result.get("description")
                }

    except requests.RequestException as e:
        attempt += 1
        wait = 2
        if attempt == 2:
            wait = 5
        elif attempt == 3:
            wait = 10
        if attempt <= 3:
            print(f"\n⚠️ Wikidata query error : {e}. Retrying attempt {attempt}...")
            time.sleep(wait)
            search_wikidata_entity(query, language, attempt)
        else:
            print(f"\n⚠️ Wikidata query error : {e}. Skipping...")

    return None


async def parse_excel_xml(file):
    content = await file.read()
    tree = ET.ElementTree(ET.fromstring(content))
    root = tree.getroot()

    ns = {
        'ss': 'urn:schemas-microsoft-com:office:spreadsheet',
        'html': 'http://www.w3.org/TR/REC-html40'
    }

    data_elements = root.findall(".//ss:Data[@ss:Type='String']", namespaces=ns)

    extracted_texts = []
    for elem in data_elements:
        full_text = ''.join(elem.itertext()).strip()
        extracted_texts.append(full_text)

    return extracted_texts

def query_wikidata(entity_label, lang="en"):
    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item rdfs:label "{entity_label}"@{lang} .
      ?item wdt:P31/wdt:P279* wd:Q618123 .  # instance of (or subclass of) geographical entity
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}" }}
    }}
    LIMIT 3
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={'query': query}, headers=headers)
    results = response.json().get("results", {}).get("bindings", [])
    return [{"label": r["itemLabel"]["value"], "id": r["item"]["value"].split("/")[-1]} for r in results]

def filter_by_mentions(wikifier_result, mentions):
    entities = []
    for ann in wikifier_result.get("annotations", []):
        if ann["title"] in mentions:
            entities.append({
                "title": ann["title"],
                "wikiDataId": ann.get("wikiDataId")
            })
    return entities

def choose_best(model, candidates, context):    #, other_coords=[]
    if not candidates:
        return None

    context_embedding = model.encode(context, convert_to_tensor=True)
    scored = []
    for c in candidates:
        desc_embedding = model.encode(c['description'], convert_to_tensor=True)
        sim = util.cos_sim(context_embedding, desc_embedding).item()

        #dist_penalty = 0
        #if c['coord'] and other_coords:
        #    distances = [geodesic(c['coord'], other).km for other in other_coords if other]
        #    dist_penalty = sum(distances) / len(distances)

        score = sim # sim - (dist_penalty / 10000)
        scored.append((score, c))

    best = max(scored, key=lambda x: x[0])[1]
    return best

def query_candidates(sparql, label, lang="en", attempt=1):

    try:
        sparql.setQuery(f"""
            SELECT ?item ?itemLabel ?description ?coord WHERE {{
              ?article schema:about ?item ;
                       schema:isPartOf <https://{lang}.wikipedia.org/> ;
                       schema:name "{label}"@{lang} .
              OPTIONAL {{ ?item schema:description ?description FILTER (lang(?description) = "{lang}") }}
              OPTIONAL {{ ?item wdt:P625 ?coord }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}" }}
            }}
            LIMIT 3
            """)
        results = sparql.query().convert()['results']['bindings']
        out = []
        for r in results:
            qid = r["item"]["value"].split("/")[-1]
            desc = r.get("description", {}).get("value", "")
            coord = r.get("coord", {}).get("value", "")
            latlon = None
            if coord:
                raw = coord.replace("Point(", "").replace(")", "")
                lon, lat = map(float, raw.split())
                latlon = (lat, lon)
            if is_geographic_entity(qid):
                out.append({
                    "qid": qid,
                    "label": r["itemLabel"]["value"],
                    "description": desc,
                    "coord": latlon
                })
        return out
    except urllib.error.HTTPError as e:
        if e.code == 504:
            print(f"[WIKIDATA] Timeout for '{label}'")
        elif e.code == 429:
            attempt += 1
            wait = 2
            if attempt == 2:
                wait = 5
            elif attempt == 3:
                wait = 10
            if attempt <= 3:
                print(f"[WIKIDATA] Too many requests for '{label}'. Retrying attempt {attempt}...")
                time.sleep(wait)
                query_candidates(sparql, label, lang, attempt)
            else:
                print(f"[WIKIDATA] Too many requests for '{label}'. Skipping...")
        else:
            print(f"[WIKIDATA] Error: {e} for '{label}'")
            return []
    except Exception as e:
        print(f"[WIKIDATA] Error: '{label}': {e}")
        return []


def do_geosparql(label, qid, entities, fts):
    geometry = retrieve_geometry(None, label, qid,
                                 entities, set(), True,
                                 True)

    if geometry:
        for g in geometry:
            if g["vkt"]:
                feature_id = f"wd:{g['qid']}"
                geometry_obj = {
                    "@id": f"{feature_id}-geom",
                    "@type": "Geometry",
                    "asWKT": f"SRID=4326;{g['vkt']}"
                }
                feature_geo = {
                    "@id": feature_id,
                    "@type": "Feature",
                    "label": g["label"],
                    "description": g["description"],
                    "qid": g["qid"],
                    "wikidata": g["wikidata_url"],
                    "osm_id": g["osm_id"],
                    "hasGeometry": geometry_obj
                }

                fts.append(feature_geo)


def download_zip(fts, geo_doc):
    filename = f"entities{uuid4().hex}.json"  # path = f"/tmp/{filename}"
    filename2 = f"geosparql{uuid4().hex}.json"
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, filename)
    path2 = os.path.join(tmpdir, filename2)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fts, f, ensure_ascii=False, indent=2)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(geo_doc, f, ensure_ascii=False, indent=2)

    zip_filename = f"export_{uuid4().hex}.zip"
    zip_path = os.path.join(tmpdir, zip_filename)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(path, arcname="entities.json")
        zipf.write(path2, arcname="geosparql.json")

    return FileResponse(zip_path, media_type="application/zip", filename="files.zip")


def download_features(fts):
    filename = f"entities{uuid4().hex}.json"  # path = f"/tmp/{filename}"
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fts, f, ensure_ascii=False, indent=2)

    return FileResponse(path, media_type="application/json", filename=filename)


def append_feature(row, label, qid):
    feature = {
        "text_label": label,
        "Wikidata_ID": qid
    }
    row["entities"].append(feature)


# ======= FastAPI endpoints =======

# redirect from root to /docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/geolinks-text", tags=["Old GeoLinks"])
def read_text(data: TextInput, download: bool = True):
    """
        Input: a JSON containing the text and the language as input, for example {"text": "your_text", "lang": "en"}.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response.
    """
    try:
        lang = data.lang.lower()
        if lang not in SUPPORTED_LANGUAGES:
            return {not_supported_message}
        results = analyze_text(data.text, lang=lang)

        features = []
        for res in results:
            if res["vkt"]:
                feature_id = f"wd:{res['qid']}"
                geometry_obj = {
                    "@id": f"{feature_id}-geom",
                    "@type": "Geometry",
                    "asWKT": f"SRID=4326;{res['vkt']}"
                }
                feature = {
                    "@id": feature_id,
                    "@type": "Feature",
                    "label": res["label"],
                    "description": res["description"],
                    "qid": res["qid"],
                    "wikidata": res["wikidata_url"],
                    "osm_id": res["osm_id"],
                    "hasGeometry": geometry_obj
                }
                features.append(feature)

        geosparql_doc = {
            **GEOSPARQL_CONTEXT,
            "@graph": features
        }

        if not download:
            return JSONResponse(content=geosparql_doc,
                                media_type="application/ld+json")

        filename = f"geosparql_{uuid4().hex}.jsonld"
        path = f"/tmp/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geosparql_doc, f, ensure_ascii=False, indent=2)

        return FileResponse(path, media_type="application/ld+json", filename=filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/geolinks-xml", tags=["Old GeoLinks"])
async def read_xml(file: UploadFile = File(...), lang: Optional[str] = "en", download: bool = True):
    """
        Input: an XML file.
        Output: a JSON+LD file containing information about the entities found.
        "Lang" is set to "en" by default.
        If "download" is True, a link is provided to download the response.
    """
    try:
        lang = lang.lower()
        if lang not in SUPPORTED_LANGUAGES:
            return JSONResponse(status_code=400, content={"error": not_supported_message})

        content = await file.read()
        tree = ET.ElementTree(ET.fromstring(content))
        root = tree.getroot()

        ns = {'ns': 'http://www.w3.org/2005/sparql-results#'}

        literals = root.findall(".//ns:binding[@name='o']/ns:literal", namespaces=ns)

        if not literals:
            return JSONResponse(status_code=400, content={"error": "No <text> nodes found in the XML file."})

        #full_text = " ".join([literal.text.strip() for literal in literals if literal.text])
        #if not full_text:
        #    return JSONResponse(status_code=400, content={"error": "Empty text in the XML file."})

        #results = analyze_text(full_text, lang=lang)
        #return {"results": results}

        features = []
        for literal in literals:
            text = literal.text.strip() if literal.text else ""
            if text:
                results = analyze_text(text, lang=lang)
                for res in results:
                    if res["vkt"]:
                        feature_id = f"wd:{res['qid']}"
                        geometry_obj = {
                            "@id": f"{feature_id}-geom",
                            "@type": "Geometry",
                            "asWKT": f"SRID=4326;{res['vkt']}"
                        }
                        feature = {
                            "@id": feature_id,
                            "@type": "Feature",
                            "label": res["label"],
                            "description": res["description"],
                            "qid": res["qid"],
                            "wikidata": res["wikidata_url"],
                            "osm_id": res["osm_id"],
                            "hasGeometry": geometry_obj,
                            "source_text": text
                        }
                        features.append(feature)
            else:
                print("Missing text for literal", literal)

        geosparql_doc = {
            **GEOSPARQL_CONTEXT,
            "@graph": features
        }

        if not download:
            return JSONResponse(content=geosparql_doc,
                                media_type="application/ld+json")

        filename = f"geosparql_{uuid4().hex}.jsonld"
        path = f"/tmp/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geosparql_doc, f, ensure_ascii=False, indent=2)

        return FileResponse(path, media_type="application/ld+json", filename=filename)

    except ET.ParseError:
        raise HTTPException(status_code=400, detail="XML file not valid.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/geolinks-iri", tags=["Old GeoLinks"])
async def read_iri_geonames(iri: str = Query(..., description="Geonames IRI (e.g. https://www.geonames.org/2618425/denmark.html)"), lang: str = Query("en", description="Text language"), download: bool = Query(False, description="If True, return a downloadable .jsonld")):
    """
        Input: a Geonames IRI.
        Output: a JSON+LD file containing information about the entities found.
        "Lang" is set to "en" by default.
        If "download" is True, a link is provided to download the response.
    """
    try:
        lang = lang.lower()
        if lang not in SUPPORTED_LANGUAGES:
            return JSONResponse(status_code=400, content={"error": not_supported_message})

        match = re.search(r'/(\d+)/', iri)
        if not match:
            return JSONResponse(status_code=400, content={"error": "Invalid GeoNames IRI format."})

        geonames_id = match.group(1)

        sparql_query = f"""
                SELECT ?item ?itemLabel WHERE {{
                  ?item wdt:P1566 "{geonames_id}".
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}" }}
                }}
                """

        results = perform_sparql_query(sparql_query)
        if not results:
            return JSONResponse(status_code=404,
                                content={"error": f"No Wikidata entity found for GeoNames ID {geonames_id}."})

        binding = results[0]
        label = binding.get("itemLabel", {}).get("value")

        if not label:
            return JSONResponse(status_code=404, content={"error": "No label found for matching Wikidata entity."})

        results = analyze_text(label, lang=lang)

        features = []
        for res in results:
            if res["vkt"]:
                feature_id = f"wd:{res['qid']}"
                geometry_obj = {
                    "@id": f"{feature_id}-geom",
                    "@type": "Geometry",
                    "asWKT": f"SRID=4326;{res['vkt']}"
                }
                feature = {
                    "@id": feature_id,
                    "@type": "Feature",
                    "label": res["label"],
                    "description": res["description"],
                    "qid": res["qid"],
                    "wikidata": res["wikidata_url"],
                    "osm_id": res["osm_id"],
                    "hasGeometry": geometry_obj
                }
                features.append(feature)
            else:
                print("Missing text for ", res)

        geosparql_doc = {
            **GEOSPARQL_CONTEXT,
            "@graph": features
        }

        if not download:
            return JSONResponse(content=geosparql_doc,
                                media_type="application/ld+json")

        filename = f"geosparql_{uuid4().hex}.jsonld"
        path = f"/tmp/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geosparql_doc, f, ensure_ascii=False, indent=2)

        return FileResponse(path, media_type="application/ld+json", filename=filename)

    except Exception as e:
        full_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\nTraceback:\n{full_trace}")

@app.post("/geolinks-csv", tags=["Old GeoLinks"])
async def read_csv_geonames(
    file: UploadFile = File(..., description="CSV file with a 'geonames' column containing GeoNames IRIs"),
    #lang: str = Query("en", description="Analysis language"),
    download: bool = Query(False, description="If True, return a downloadable .jsonld")
):
    """
        Input: a CSV file.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response.
    """
    try:
        #lang = lang.lower()
        #if lang not in SUPPORTED_LANGUAGES:
        #    return JSONResponse(status_code=400, content={"error": not_supported_message})

        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))

        if "geonames" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "Missing 'geonames' column in CSV."})

        features = []

        processed_geonames_id = set()
        processed_qids = set()

        for iri in df["geonames"].dropna().unique():
            entities = []

            match = re.search(r'/(\d+)', iri)
            if not match:
                logger.warning(f"\n⚠️ Skipping {iri}, it is not a valid GeoNames IRI format.")
                continue  # skip invalid IRI

            geonames_id = match.group(1)

            if geonames_id in processed_geonames_id:
                logger.warning(f"\n⚠️ Skipping '{iri}', already processed.")
                continue    # skip IRI already processed

            sparql_query = f"""
                        SELECT ?item ?itemLabel WHERE {{
                          ?item wdt:P1566 "{geonames_id}".
                          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
                        }}
                    """ # {lang}
            results = perform_sparql_query(sparql_query)

            if not results:

                # try retrieving the Wikipedia URL and linking it to a Wikidata entity
                try:

                    wikipedia_url = get_wikipedia_article_from_geonames(iri)

                    if wikipedia_url:
                        wikidata_entity = get_wikidata_entity_from_wikipedia_url(wikipedia_url)
                    else:
                        wikidata_entity = None

                    if wikidata_entity:

                        label = wikidata_entity["label"]
                        qid = wikidata_entity["id"]

                        if qid in processed_qids:
                            logger.warning(f"\n⚠️ Skipping '{iri}', already processed.")
                            continue

                        geometry = retrieve_geometry(None, label, qid, entities, processed_qids, True, False)

                        if geometry:
                            for g in geometry:
                                if g["vkt"]:
                                    feature_id = f"wd:{g['qid']}"
                                    geometry_obj = {
                                        "@id": f"{feature_id}-geom",
                                        "@type": "Geometry",
                                        "asWKT": f"SRID=4326;{g['vkt']}"
                                    }
                                    feature = {
                                        "@id": feature_id,
                                        "@type": "Feature",
                                        "label": g["label"],
                                        "description": g["description"],
                                        "qid": g["qid"],
                                        "wikidata": g["wikidata_url"],
                                        "osm_id": g["osm_id"],
                                        "hasGeometry": geometry_obj
                                    }

                                    features.append(feature)
                                    processed_geonames_id.add(geonames_id)

                                else:
                                    logger.warning(f"\n⚠️ Missing geometry for '{g['label']}', Wikidata URL: {g['wikidata_url']}. Skipping '{iri}'.")
                                    continue
                        else:
                            logger.warning(f"\n⚠️ Missing geometry for '{iri}'. Skipping...")
                            continue

                    else:

                        if wikipedia_url:
                            parsed_url = urlparse(wikipedia_url)
                            title = unquote(parsed_url.path.split("/wiki/")[-1])

                            if not title:
                                title = get_geonames_label(geonames_id)

                        else:
                            title = get_geonames_label(geonames_id)

                        if title:
                            if "_" in title:
                                title = title.replace("_", " ")

                            annotations = disambiguation_with_wikifier(title)
                            analyze(annotations, entities, processed_qids)

                            if not entities:

                                entity = search_wikidata_entity(title)

                                if not entity:
                                    logger.warning(f"\n⚠️ Skipping '{iri}', no results found.")
                                    continue

                                label = entity["label"]
                                qid = entity["id"]

                                if qid in processed_qids:
                                    logger.warning(f"\n⚠️ Skipping '{iri}', already processed.")
                                    continue

                                geometry = retrieve_geometry(None, label, qid, entities, processed_qids, True, False)

                                if geometry:
                                    for g in geometry:
                                        if g["vkt"]:
                                            feature_id = f"wd:{g['qid']}"
                                            geometry_obj = {
                                                "@id": f"{feature_id}-geom",
                                                "@type": "Geometry",
                                                "asWKT": f"SRID=4326;{g['vkt']}"
                                            }
                                            feature = {
                                                "@id": feature_id,
                                                "@type": "Feature",
                                                "label": g["label"],
                                                "description": g["description"],
                                                "qid": g["qid"],
                                                "wikidata": g["wikidata_url"],
                                                "osm_id": g["osm_id"],
                                                "hasGeometry": geometry_obj
                                            }

                                            features.append(feature)
                                            processed_geonames_id.add(geonames_id)

                                        else:
                                            logger.warning(
                                                f"\n⚠️ Missing geometry for '{g['label']}', Wikidata URL: {g['wikidata_url']}. Skipping '{iri}'.")
                                            continue
                                else:
                                    logger.warning(f"\n⚠️ Missing geometry for '{iri}'. Skipping...")
                                    continue

                                continue

                            for e in entities:
                                if e["vkt"]:
                                    feature_id = f"wd:{e['qid']}"
                                    geometry_obj = {
                                        "@id": f"{feature_id}-geom",
                                        "@type": "Geometry",
                                        "asWKT": f"SRID=4326;{e['vkt']}"
                                    }
                                    feature = {
                                        "@id": feature_id,
                                        "@type": "Feature",
                                        "label": e["label"],
                                        "description": e["description"],
                                        "qid": e["qid"],
                                        "wikidata": e["wikidata_url"],
                                        "osm_id": e["osm_id"],
                                        "hasGeometry": geometry_obj
                                    }
                                    features.append(feature)
                                    processed_geonames_id.add(geonames_id)

                                else:
                                    logger.warning(f"\n⚠️ Missing geometry for '{e['label']}, Wikidata URL: {e['wikidata_url']}'. Skipping '{iri}'.")
                                    continue

                            continue

                        else:
                            logger.warning(f"\n⚠️ Title is '{title}'. Skipping '{iri}'. Info: {wikipedia_url, wikidata_entity}.")
                            continue


                except WikipediaRateLimitException as e:
                    logger.warning(f"\n⚠️ Wikipedia rate limit exceeded: {e}. Skipping '{iri}'.")
                    continue

            else:

                binding = results[0]
                label = binding.get("itemLabel", {}).get("value")
                url = binding.get("item", {}).get("value")
                match_id = re.search(r"wikidata\.org/entity/(Q\d+)", url)
                qid = match_id.group(1)
                if not qid:
                    logger.warning(f"\n⚠️ Skipping '{iri}', qid not found.")
                    continue
                if not label:
                    logger.warning(f"\n⚠️ Skipping '{iri}', label not found.")
                    continue

                if qid in processed_qids:
                    logger.warning(f"\n⚠️ Skipping '{iri}', already processed.")
                    continue

                geometry = retrieve_geometry(None, label, qid, entities, processed_qids, True, False)

                if geometry:
                    for g in geometry:
                        if g["vkt"]:
                            feature_id = f"wd:{g['qid']}"
                            geometry_obj = {
                                "@id": f"{feature_id}-geom",
                                "@type": "Geometry",
                                "asWKT": f"SRID=4326;{g['vkt']}"
                            }
                            feature = {
                                "@id": feature_id,
                                "@type": "Feature",
                                "label": g["label"],
                                "description": g["description"],
                                "qid": g["qid"],
                                "wikidata": g["wikidata_url"],
                                "osm_id": g["osm_id"],
                                "hasGeometry": geometry_obj
                            }

                            features.append(feature)
                            processed_geonames_id.add(geonames_id)

                        else:
                            logger.warning(f"\n⚠️ Missing geometry for '{g['label']}', Wikidata URL: {g['wikidata_url']}. Skipping '{iri}'.")
                            continue
                else:
                    logger.warning(f"\n⚠️ Missing geometry for '{iri}'. Skipping...")
                    continue

        geosparql_doc = {
            **GEOSPARQL_CONTEXT,
            "@graph": features
        }

        if not download:
            return JSONResponse(content=geosparql_doc,
                                media_type="application/ld+json")

        filename = f"geosparql_{uuid4().hex}.jsonld"
        path = f"/tmp/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geosparql_doc, f, ensure_ascii=False, indent=2)

        return FileResponse(path, media_type="application/ld+json", filename=filename)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-xml", tags=["Test"])
async def read_xml_as_gold_standard(
    file: UploadFile = File(..., description="XML file containing events"),
    #lang: str = Query("en", description="Analysis language"),
    download: bool = Query(False, description="If True, return a downloadable .json")
):
    """
        Input: an XML file.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response.
    """
    try:

        content = await parse_excel_xml(file)

        features = []

        for i, event in enumerate(content):

            print(f"\nEvent content: {event}\n")

            lang = detect(event)
            doc, nlp = tokenize_text(event, lang=lang)
            entities_spacy = extract_geo_entity(doc)
            print(f"\nEntities found by spaCy: {', '.join(entities_spacy)}")

            row = {
                "row": i + 1,
                "entities": []
            }

            print(f"\nIndex: {i + 1}\n")

            for entity in entities_spacy:

                ent = search_wikidata_entity(entity)

                print(f"\nEnt: {ent}\n")

                if ent:
                    feature = {
                        "text_label": ent.get("label", ""),
                        "Wikidata_ID": ent.get("id", "")
                    }
                    row["entities"].append(feature)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        filename = f"entities{uuid4().hex}.json"
        path = f"/tmp/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)

        return FileResponse(path, media_type="application/json", filename=filename)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/test-flair", tags=["Test"])
async def ner_using_flair(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        NER is done by Flair, entity linking is done by custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        tagger = SequenceTagger.load("ner") # try flair/ner-english
        splitter = SegtokSentenceSplitter()

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            entities_flair = []

            sentences = splitter.split(event)

            tagger.predict(sentences)

            for sentence in sentences:
                for entity in sentence.get_spans('ner'):
                    if entity.get_label("ner").value == "LOC":
                        entities_flair.append({
                            "text": entity.text,
                            "context": sentence.to_original_text()
                        })

            all_coords = []

            print("Entities found:")
            for ent in entities_flair:
                print(ent['text'])

            for ent in entities_flair:
                cands = query_candidates(sparql, ent['text'])
                if cands:
                    all_coords += [c["coord"] for c in cands if c["coord"]]

            for ent in entities_flair:

                entities = []

                if all_coords:
                    label = ent['text']
                    context = ent['context']
                    candidates = query_candidates(sparql, label)
                    if candidates:
                        best = choose_best(model, candidates, context)  #, all_coords

                        if best:

                            append_feature(row, label, best['qid'])

                            if download_geosparql:
                                do_geosparql(label, best['qid'], entities, features_geosparql)

                    else:
                        entity = search_wikidata_entity(ent['text'])
                        if entity:

                            append_feature(row, entity['label'], entity['id'])

                            if download_geosparql:
                                do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                else:
                    entity = search_wikidata_entity(ent['text'])
                    if entity:

                        append_feature(row, entity['label'], entity['id'])

                        if download_geosparql:
                            do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-flair-custom-linker", tags=["Test"])
async def ner_using_flair_alternative(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        This is an alternative version of Flair + custom linker.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        tagger = SequenceTagger.load("ner")

        splitter = SegtokSentenceSplitter()

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            sentences = splitter.split(event)
            tagger.predict(sentences)

            entities_flair = []

            for sentence in sentences:
                for entity in sentence.get_spans('ner'):
                    if entity.get_label("ner").value == "LOC":
                        entities_flair.append(entity)

            linked = []
            for ent in entities_flair:
                candidates = query_wikidata(ent.text)
                e = {
                    "text": ent.text,
                    "candidates": candidates
                }
                linked.append(e)
                print(f"\nE: {e}")

            for entity in entities_flair:

                ent = search_wikidata_entity(entity.text)
                entities = []

                if ent:

                    append_feature(row, ent.get("label", ""), ent.get("id", ""))

                    if download_geosparql:
                        do_geosparql(ent.get("label", ""), ent.get("id", ""), entities, features_geosparql)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-wikifier", tags=["Test"])
async def wikifier(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        Both NER and entity linking are managed by Wikifier.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            result = call_wikifier(event)
            print(f"\nResult: {result}")
            for ann in result:

                while True:

                    try:

                        entities = []

                        if any("location" in t.lower() for t in ann.get("types", [])):

                            append_feature(row, ann['title'], ann.get('wikiDataId'))

                            if download_geosparql:
                                do_geosparql(ann['title'], ann.get('wikiDataId'), entities, features_geosparql)

                        break

                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            print(f"[WIKIDATA] Too many requests for entity '{ann['title']}'. Retrying...")
                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-flair-wikifier", tags=["Test"])
async def flair_wikifier(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        Flair and Wikifier are combined to increase precision: only entities that match are stored in the results.
        Input: an XML file containing events of the narrative.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        tagger = SequenceTagger.load("ner")

        splitter = SegtokSentenceSplitter()

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            sentences = splitter.split(event)
            tagger.predict(sentences)

            entities_flair = []

            for sentence in sentences:
                for entity in sentence.get_spans('ner'):
                    if entity.get_label("ner").value == "LOC":
                        entities_flair.append(entity)

            wikifier_result = call_wikifier(event)

            filtered = filter_by_mentions(wikifier_result, entities_flair)

            for entity in filtered:

                while True:

                    try:

                        entities = []

                        append_feature(row, entity['title'], entity['wikiDataId'])

                        if download_geosparql:
                            do_geosparql(entity['title'], entity['wikiDataId'], entities, features_geosparql)

                        break

                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            print(f"[WIKIDATA] Too many requests for entity '{entity['title']}'. Retrying...")
                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-rel", tags=["Test"])
async def rel(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        NER is done by REL, entity linking is done by custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        api_url = "https://rel.cs.ru.nl/api"

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            entities_rel = []

            all_coords = []

            rel_error = False

            try:
                response = requests.post(
                    api_url,
                    json={"text": event, "spans": []},
                    headers={"Connection": "close"},
                    timeout=60
                )
                response.raise_for_status()
                el_result = response.json()

            except requests.exceptions.RequestException as e:
                print(f"[Requests Error] Row {i + 1}: failed request - {e}")
                el_result = []
                rel_error = True

            except json.JSONDecodeError as e:
                print(f"[JSON Error] Row {i + 1}: the answer is not a valid JSON - {e}")
                el_result = []
                rel_error = True

            if not rel_error:

                print(f"\nRow: {i+1}, entities found by REL: {el_result}")

                for entity in el_result:

                    if entity[-1] == "LOC":
                        entities_rel.append({
                            "text": entity[3],
                            "context": event
                        })

            print("Entities found:")
            for ent in entities_rel:
                print(ent['text'])

            for ent in entities_rel:
                cands = query_candidates(sparql, ent['text'])
                if cands:
                    all_coords += [c["coord"] for c in cands if c["coord"]]

            for ent in entities_rel:

                while True:

                    try:

                        entities = []

                        if all_coords:
                            label = ent['text']
                            context = ent['context']
                            candidates = query_candidates(sparql, label)
                            if candidates:
                                best = choose_best(model, candidates, context)  #, all_coords

                                if best:

                                    append_feature(row, label, best['qid'])

                                    if download_geosparql:
                                        do_geosparql(label, best['qid'], entities, features_geosparql)

                            else:
                                entity = search_wikidata_entity(ent['text'])
                                if entity:

                                    append_feature(row, entity['label'], entity['id'])

                                    if download_geosparql:
                                        do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        else:
                            entity = search_wikidata_entity(ent['text'])
                            if entity:

                                append_feature(row, entity['label'], entity['id'])

                                if download_geosparql:
                                    do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        break

                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            print(f"[WIKIDATA] Too many requests for entity '{ent['text']}'. Retrying...")
                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-rel-flair", tags=["Test"])
async def rel_flair(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        NER is managed by REL, and Flair when the first fails, entity linking is done by custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        api_url = "https://rel.cs.ru.nl/api"

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        tagger = SequenceTagger.load("ner")
        splitter = SegtokSentenceSplitter()

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            entities_rel = []

            all_coords = []

            rel_error = False

            try:
                response = requests.post(
                    api_url,
                    json={"text": event, "spans": []},
                    headers={"Connection": "close"},
                    timeout=60
                )
                response.raise_for_status()
                el_result = response.json()

            except requests.exceptions.RequestException as e:
                print(f"[Requests Error] Row {i + 1}: failed request - {e}. Trying with Flair...")
                el_result = []
                rel_error = True

            except json.JSONDecodeError as e:
                print(f"[JSON Error] Row {i + 1}: the answer is not a valid JSON - {e}. Trying with Flair...")
                el_result = []
                rel_error = True

            if not rel_error:

                print(f"\nRow: {i+1}, entities found by REL: {el_result}")

                for entity in el_result:

                    if entity[-1] == "LOC":
                        entities_rel.append({
                            "text": entity[3],
                            "context": event
                        })

            else:

                sentences = splitter.split(event)

                tagger.predict(sentences)

                for sentence in sentences:
                    for entity in sentence.get_spans('ner'):
                        if entity.get_label("ner").value == "LOC":
                            entities_rel.append({
                                "text": entity.text,
                                "context": sentence.to_original_text()
                            })

            print("Entities found:")
            for ent in entities_rel:
                print(ent['text'])

            for ent in entities_rel:
                cands = query_candidates(sparql, ent['text'])
                if cands:
                    all_coords += [c["coord"] for c in cands if c["coord"]]

            for ent in entities_rel:

                while True:

                    try:

                        entities = []

                        if all_coords:
                            label = ent['text']
                            context = ent['context']
                            candidates = query_candidates(sparql, label)
                            if candidates:
                                best = choose_best(model, candidates, context)  #, all_coords

                                if best:

                                    append_feature(row, label, best['qid'])

                                    if download_geosparql:
                                        do_geosparql(label, best['qid'], entities, features_geosparql)

                            else:
                                entity = search_wikidata_entity(ent['text'])
                                if entity:

                                    append_feature(row, entity['label'], entity['id'])

                                    if download_geosparql:
                                        do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        else:
                            entity = search_wikidata_entity(ent['text'])
                            if entity:

                                append_feature(row, entity['label'], entity['id'])

                                if download_geosparql:
                                    do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        break

                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            print(f"[WIKIDATA] Too many requests for entity '{ent['text']}'. Retrying...")
                            time.sleep(5)

                    except requests.RequestException:
                        print(f"[WIKIDATA] Too many requests for entity '{ent['text']}'. Retrying...")
                        time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/spacy", tags=["GeoLinks"])
async def spacy_ner(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        NER is done by spaCy, entity linking is done by custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            #lang = detect(event)
            doc, nlp = tokenize_text(event, lang="en")
            entities_spacy = extract_geo_entity(doc, event)

            all_coords = []

            print("Entities found:")
            for ent in entities_spacy:
                print(ent['text'])

            for ent in entities_spacy:
                cands = query_candidates(sparql, ent['text'])
                if cands:
                    all_coords += [c["coord"] for c in cands if c["coord"]]

            for ent in entities_spacy:

                while True:

                    try:

                        entities = []

                        if all_coords:
                            label = ent['text']
                            context = ent['context']
                            candidates = query_candidates(sparql, label)
                            if candidates:
                                best = choose_best(model, candidates, context)  #, all_coords

                                if best:

                                    append_feature(row, label, best['qid'])

                                    if download_geosparql:
                                        do_geosparql(label, best['qid'], entities, features_geosparql)

                            else:
                                entity = search_wikidata_entity(ent['text'])
                                if entity:

                                    append_feature(row, entity['label'], entity['id'])

                                    if download_geosparql:
                                        do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        else:
                            entity = search_wikidata_entity(ent['text'])
                            if entity:

                                append_feature(row, entity['label'], entity['id'])

                                if download_geosparql:
                                    do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        break

                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            print(f"[WIKIDATA] Too many requests for entity '{ent['text']}'. Retrying...")
                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-spacy-el", tags=["Test"])
async def spacy_ner_el(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        NER is done by spaCy, entity linking is done by both spaCy and custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        nlp = spacy.load("en_core_web_trf")

        nlp.add_pipe("entityLinker", last=True)

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            doc = nlp(event)
            #all_linked_entities = doc._.linkedEntities
            ents = doc.ents
            entities_spacy = []

            for ent in ents:

                if ent.label_ in ["LOC", "GPE"]:

                    entity = {
                        "text": ent.text,
                        "context": event
                    }

                    entities_spacy.append(entity)

            all_coords = []

            print("Entities found:")
            for ent in entities_spacy:
                print(ent['text'])

            for ent in entities_spacy:
                cands = query_candidates(sparql, ent['text'])
                if cands:
                    all_coords += [c["coord"] for c in cands if c["coord"]]

            for ent in entities_spacy:

                while True:

                    try:

                        entities = []

                        if all_coords:
                            label = ent['text']
                            context = ent['context']
                            candidates = query_candidates(sparql, label)
                            if candidates:
                                best = choose_best(model, candidates, context)  #, all_coords

                                if best:

                                    append_feature(row, label, best['qid'])

                                    if download_geosparql:

                                        do_geosparql(label, best['qid'], entities, features_geosparql)

                            else:
                                entity = search_wikidata_entity(ent['text'])
                                if entity:

                                    append_feature(row, entity['label'], entity['id'])

                                    if download_geosparql:

                                        do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        else:
                            entity = search_wikidata_entity(ent['text'])
                            if entity:

                                append_feature(row, entity['label'], entity['id'])

                                if download_geosparql:

                                    do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        break

                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            print(f"[WIKIDATA] Too many requests for entity '{ent['text']}'. Retrying...")
                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/spacy-flair", tags=["GeoLinks"])
async def ner_spacy_flair(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        NER is done by spaCy and Flair, taking into account all the entities found, entity linking is done by custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        tagger = SequenceTagger.load("ner")
        splitter = SegtokSentenceSplitter()

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            doc, nlp = tokenize_text(event, lang="en")
            entities_spacy = extract_geo_entity(doc, event)

            sentences = splitter.split(event)

            tagger.predict(sentences)

            text_entities = [item["text"] for item in entities_spacy]

            for sentence in sentences:
                for entity in sentence.get_spans('ner'):
                    if entity.get_label("ner").value == "LOC" and entity.text not in text_entities:
                        entities_spacy.append({
                            "text": entity.text,
                            "context": sentence.to_original_text()
                        })

            all_coords = []

            print("Entities found:")
            for ent in entities_spacy:
                print(ent['text'])

            for ent in entities_spacy:
                cands = query_candidates(sparql, ent['text'])
                if cands:
                    all_coords += [c["coord"] for c in cands if c["coord"]]

            for ent in entities_spacy:

                while True:

                    try:

                        entities = []

                        if all_coords:
                            label = ent['text']
                            context = ent['context']
                            candidates = query_candidates(sparql, label)
                            if candidates:
                                best = choose_best(model, candidates, context)  #, all_coords

                                if best:

                                    append_feature(row, label, best['qid'])

                                    if download_geosparql:

                                        do_geosparql(label, best['qid'], entities, features_geosparql)

                            else:
                                entity = search_wikidata_entity(ent['text'])
                                if entity and is_geographic_entity(entity['id']):

                                    append_feature(row, entity['label'], entity['id'])

                                    if download_geosparql:

                                        do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        else:
                            entity = search_wikidata_entity(ent['text'])
                            if entity and is_geographic_entity(entity['id']):

                                append_feature(row, entity['label'], entity['id'])

                                if download_geosparql:

                                    do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                        break

                    except urllib.error.HTTPError as e:

                        if e.code == 429:
                            print(f"[WIKIDATA] Too many requests for entity '{ent['text']}'. Retrying...")

                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-geoparser-wikidata", tags=["Test"])
async def geoparser_wikidata(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        Geoparser + Wikidata, the former is used for NER, the latter for entity linking. When Wikifier fails, a custom algorithm is used.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        geoparser = Geoparser(
            spacy_model="en_core_web_trf",
            transformer_model="dguzh/geo-all-MiniLM-L6-v2", # dguzh/geo-all-distilroberta-v1
            gazetteer="geonames"
        )

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            docs = geoparser.parse([event])

            all_coords = []

            for doc in docs:
                for toponym in doc.toponyms:

                    while True:

                        try:
                            cands = query_candidates(sparql, toponym.text)
                            if cands:
                                all_coords += [c["coord"] for c in cands if c["coord"]]

                            break

                        except urllib.error.HTTPError as e:
                            if e.code == 429:
                                print(f"[WIKIDATA] Too many requests for entity '{toponym.text}'. Retrying...")
                                time.sleep(5)

            print("\nEntities found:")
            for doc in docs:
                for toponym in doc.toponyms:

                    while True:

                        try:

                            entities = []

                            print(f"- Toponym: {toponym.text}")
                            location = toponym.location

                            if location:
                                geonames_id = location['geonameid']
                                iri = 'https://www.geonames.org/' + geonames_id
                                wikidata_entity = get_wikidata_entity_from_geonames(iri)

                                if wikidata_entity:
                                    if wikidata_entity and is_geographic_entity(wikidata_entity['id']):

                                        append_feature(row, wikidata_entity['label'], wikidata_entity['id'])

                                        if download_geosparql:

                                            do_geosparql(wikidata_entity['label'], wikidata_entity['id'], entities, features_geosparql)

                                else:
                                    print(f"Entity {location['name']} could not be resolved. Trying with Wikidata method...")

                                    if all_coords:

                                        label = toponym.text
                                        context = event
                                        candidates = query_candidates(sparql, label)
                                        if candidates:
                                            best = choose_best(model, candidates, context)  #, all_coords

                                            if best:

                                                append_feature(row, label, best['qid'])

                                                if download_geosparql:

                                                    do_geosparql(label, best['qid'], entities, features_geosparql)

                                        else:
                                            entity = search_wikidata_entity(label)
                                            if entity and is_geographic_entity(entity['id']):

                                                append_feature(row, entity['label'], entity['id'])

                                                if download_geosparql:

                                                    do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                                    else:
                                        entity = search_wikidata_entity(toponym.text)
                                        if entity and is_geographic_entity(entity['id']):

                                            append_feature(row, entity['label'], entity['id'])

                                            if download_geosparql:

                                                do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                            else:
                                print(f"Location could not be resolved. Trying with Wikidata method...")

                                if all_coords:

                                    label = toponym.text
                                    context = event
                                    candidates = query_candidates(sparql, label)
                                    if candidates:
                                        best = choose_best(model, candidates, context)  #, all_coords

                                        if best:

                                            append_feature(row, label, best['qid'])

                                            if download_geosparql:

                                                do_geosparql(label, best['qid'], entities, features_geosparql)

                                    else:
                                        entity = search_wikidata_entity(label)
                                        if entity and is_geographic_entity(entity['id']):

                                            append_feature(row, entity['label'], entity['id'])

                                            if download_geosparql:

                                                do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                                else:
                                    entity = search_wikidata_entity(toponym.text)
                                    if entity and is_geographic_entity(entity['id']):

                                        append_feature(row, entity['label'], entity['id'])

                                        if download_geosparql:

                                            do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                            break

                        except urllib.error.HTTPError as e:
                            if e.code == 429:
                                print(f"[WIKIDATA] Too many requests for entity '{toponym.text}'. Retrying...")
                                time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-geoparser-spacy", tags=["Test"])
async def geoparser_spacy(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        Geoparser + spaCy, when the former fails, the latter is used. Entity linking is managed by a custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        geoparser = Geoparser(
            spacy_model="en_core_web_trf",
            transformer_model="dguzh/geo-all-MiniLM-L6-v2", # dguzh/geo-all-distilroberta-v1
            gazetteer="geonames"
        )

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            docs = geoparser.parse([event])
            doc, nlp = tokenize_text(event, lang="en")
            entities_spacy = extract_geo_entity(doc, event)

            all_coords = []

            for ent in entities_spacy:
                cands = query_candidates(sparql, ent['text'])
                if cands:
                    all_coords += [c["coord"] for c in cands if c["coord"]]

            print("\nEntities found:")
            for doc in docs:
                for toponym in doc.toponyms:

                    while True:

                        try:

                            entities = []

                            print(f"- Toponym: {toponym.text}")
                            location = toponym.location

                            entities_spacy_text = [item['text'] for item in entities_spacy if item['text']]

                            match = find_similar_string(toponym.text, entities_spacy_text, threshold=0.7)

                            if location:
                                geonames_id = location['geonameid']
                                iri = 'https://www.geonames.org/' + geonames_id
                                wikidata_entity = get_wikidata_entity_from_geonames(iri)

                                if wikidata_entity:
                                    if wikidata_entity and is_geographic_entity(wikidata_entity['id']):

                                        append_feature(row, wikidata_entity['label'], wikidata_entity['id'])

                                        if download_geosparql:

                                            do_geosparql(wikidata_entity['label'], wikidata_entity['id'], entities, features_geosparql)

                                else:
                                    print(f"Entity {location['name']} could not be resolved. Trying with spacy...")

                                    if all_coords and match:

                                        label = match
                                        context = event
                                        candidates = query_candidates(sparql, label)
                                        if candidates:
                                            best = choose_best(model, candidates, context)  #, all_coords

                                            if best:

                                                append_feature(row, label, best['qid'])

                                                if download_geosparql:

                                                    do_geosparql(label, best['qid'], entities, features_geosparql)

                                        else:
                                            entity = search_wikidata_entity(label)
                                            if entity and is_geographic_entity(entity['id']):

                                                append_feature(row, entity['label'], entity['id'])

                                                if download_geosparql:

                                                    do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                                    else:
                                        entity = search_wikidata_entity(match)
                                        if entity and is_geographic_entity(entity['id']):

                                            append_feature(row, entity['label'], entity['id'])

                                            if download_geosparql:

                                                do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                            else:
                                print(f"Location could not be resolved. Trying with spacy...")

                                if all_coords and match:

                                    label = match
                                    context = event
                                    candidates = query_candidates(sparql, label)
                                    if candidates:
                                        best = choose_best(model, candidates, context)  #, all_coords

                                        if best:

                                            append_feature(row, label, best['qid'])

                                            if download_geosparql:

                                                do_geosparql(label, best['qid'], entities, features_geosparql)

                                    else:
                                        entity = search_wikidata_entity(label)
                                        if entity and is_geographic_entity(entity['id']):

                                            append_feature(row, entity['label'], entity['id'])

                                            if download_geosparql:

                                                do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                                else:
                                    entity = search_wikidata_entity(match)
                                    if entity and is_geographic_entity(entity['id']):

                                        append_feature(row, entity['label'], entity['id'])

                                        if download_geosparql:

                                            do_geosparql(entity['label'], entity['id'], entities, features_geosparql)

                            break

                        except urllib.error.HTTPError as e:
                            if e.code == 429:
                                print(f"[WIKIDATA] Too many requests for entity '{toponym.text}'. Retrying...")
                                time.sleep(5)

                        except requests.RequestException:
                            print(f"[WIKIDATA] Too many requests for entity '{toponym.text}'. Retrying...")
                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        filename, lineno, func, text = tb[-1]  # last call in stack
        error_message = f"{str(e)} (File \"{filename}\", line {lineno}, in {func}: {text})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/test-geoparser", tags=["Test"])
async def geoparser(
    file: UploadFile = File(..., description="XML file containing events"),
    download: bool = Query(False, description="If True, return a downloadable .json"),
    download_geosparql: bool = Query(False, description="If True, return a downloadable .zip that contains a json file for the evaluation and a jsonld file in geosparql format.")
):
    """
        NER is managed by Geoparser, entity linking is managed by a custom algorithm.
        Input: an XML file containing events.
        Output: a JSON+LD file containing information about the entities found.
        If "download" is True, a link is provided to download the response in JSON format.
        If "download_geosparql" is True, a link is provided to download a .zip containing a JSON file and a JSON+LD file.
    """
    try:

        content = await parse_excel_xml(file)

        features = []
        features_geosparql = []

        geoparser = Geoparser(
            spacy_model="en_core_web_trf",
            transformer_model="dguzh/geo-all-MiniLM-L6-v2", # dguzh/geo-all-distilroberta-v1
            gazetteer="geonames"
        )

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        for i, event in enumerate(content):

            row = {
                "row": i + 1,
                "entities": []
            }

            docs = geoparser.parse([event])

            print("\nEntities found:")
            for doc in docs:
                for toponym in doc.toponyms:

                    while True:

                        try:

                            entities = []

                            print(f"- Toponym: {toponym.text}")
                            location = toponym.location

                            if location:
                                geonames_id = location['geonameid']
                                iri = 'https://www.geonames.org/' + geonames_id
                                wikidata_entity = get_wikidata_entity_from_geonames(iri)

                                if wikidata_entity:
                                    if wikidata_entity and is_geographic_entity(wikidata_entity['id']):

                                        append_feature(row, wikidata_entity['label'], wikidata_entity['id'])

                                        if download_geosparql:

                                            do_geosparql(wikidata_entity['label'], wikidata_entity['id'], entities, features_geosparql)

                                else:
                                    print(f"Entity {location['name']} could not be resolved. Skipping...")

                            else:
                                print(f"Location could not be resolved. Skipping {toponym.text}...")

                            break

                        except urllib.error.HTTPError as e:
                            if e.code == 429:
                                print(f"[WIKIDATA] Too many requests for entity '{toponym.text}'. Retrying...")
                                time.sleep(5)

                        except requests.RequestException:
                            print(f"[WIKIDATA] Too many requests for entity '{toponym.text}'. Retrying...")
                            time.sleep(5)

            print(f"\nRow: {row}\n")
            features.append(row)

        if not download:
            return JSONResponse(content=features,
                                media_type="application/json")

        if download_geosparql:

            geosparql_doc = {
                **GEOSPARQL_CONTEXT,
                "@graph": features_geosparql
            }
            return download_zip(features, geosparql_doc)

        else:

            return download_features(features)

    except Exception as e:
        full_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\nTraceback:\n{full_trace}")