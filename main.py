# ======= Import libraries =======

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi import UploadFile, File
import xml.etree.ElementTree as ET
from pydantic import BaseModel
from typing import Optional
from langdetect import detect
from uuid import uuid4
import spacy
import requests
import time
import pandas as pd
import json
import os
import re

# ======= Init =======

GEOSPARQL_CONTEXT = {
    "@context": {
        "geo":        "http://www.opengis.net/ont/geosparql#",
        "schema":     "http://schema.org/",
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

not_supported_message = "Language not supported. Please insert one value among \'en\' (English), \'it\' (Italian), \'fr\' (French), \'de\' (Deutsch), \'ru\' (Russian), \'pt\' (Portuguese), \'es\' (Spanish), \'nl\' (Dutch) , \'pl\' (Polish) or \'xx\' (for multi language texts)."

app = FastAPI()

SPACY_MODELS = {
    "en": "en_core_web_sm",
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

WIKIFIER_API_KEY = os.getenv("WIKIFIER_API_KEY")
if not WIKIFIER_API_KEY:
    raise EnvironmentError("WIKIFIER_API_KEY not defined in environment.")


# ======= Pydantic model =======
class TextInput(BaseModel):
    text: str
    lang: Optional[str] = "en"


# ======= Utility functions =======

def get_spacy_model(lang="en"):
    model_name = SPACY_MODELS.get(lang, "en_core_web_sm")
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = spacy.load(model_name)
        except OSError:
            print(f"⚠️ spaCy model '{model_name}' not found. Use fallback 'en_core_web_sm'.")
            model_name = "en_core_web_sm"
            loaded_models[model_name] = spacy.load(model_name)
    return loaded_models[model_name]

def tokenize_text(text, lang="en"):
    nlp = get_spacy_model(lang)
    doc = nlp(text)
    return doc, nlp

def extract_geo_entity(doc):
    return [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE", "NOUN", "PROPN"]]

def disambiguation_with_wikifier(text, lang="en"):
    url = "http://www.wikifier.org/annotate-article"
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

def is_geographic_entity(qid):
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

import requests

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

def retrieve_geometry(annotation, label, qid, entities, processed_qids, only_geometry):
    try:
        if not only_geometry:
            if annotation.get("cosine", 1.0) < 0.5 or not is_geographic_entity(qid):
                return
        else:
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
        processed_qids.add(qid)
        time.sleep(3)  # Avoid rate limit

        if only_geometry:
            return entities
    except Exception as e:
        print(f"❌ Error with {label}: {e}")

def process_annotation(annotation, processed_qids, entities):
    try:
        qid = annotation["wikiDataItemId"]
        label = annotation["title"]
    except KeyError as e:
        print(f"\n⚠️ Warning. The key {e} is missing from the annotation '{annotation['title']}'.")
        return

    if qid in processed_qids:
        return

    retrieve_geometry(annotation, label, qid, entities, processed_qids, False)

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


# ======= FastAPI endpoints =======

@app.post("/geosparql")
def analyze_from_input(data: TextInput, download: bool = True):
    """
        Return JSON‑LD compliant with GeoSPARQL.
        ?download=false --> JSON inline
        else downloadable .jsonld file
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


@app.post("/analyze-from-xml")
async def analyze_from_xml(file: UploadFile = File(...), lang: Optional[str] = "en", download: bool = True):
    """
    Parse an uploaded XML file,
    extract text from a specific node,
    and start to analyze.
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

@app.post("/analyze-from-iri")
async def analyze_geonames_iri(iri: str = Query(..., description="IRI from Geonames (e.g. https://www.geonames.org/2618425/denmark.html)"), lang: str = Query("en", description="Analysis language"), download: bool = Query(False, description="If True, return a downloadable .jsonld")):
    """
    Analyze a GeoNames data page using IRI.
    Extract the main content and apply the geographic disambiguation process.
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-from-csv")
async def analyze_geonames_csv(
    file: UploadFile = File(..., description="CSV file with a 'geonames' column containing GeoNames IRIs"),
    #lang: str = Query("en", description="Analysis language"),
    download: bool = Query(False, description="If True, return a downloadable .jsonld")
):
    """
    Analyze a CSV file containing GeoNames IRIs in the 'geonames' column.
    Extract the main content and apply the disambiguation process.
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

        for iri in df["geonames"].dropna().unique():
            match = re.search(r'/(\d+)/', iri)
            if not match:
                continue  # skip invalid IRI

            geonames_id = match.group(1)

            if geonames_id in processed_geonames_id:
                print(f"\n⚠️ Skipping '{iri}', already processed.")
                continue    # skip IRI already processed

            sparql_query = f"""
                        SELECT ?item ?itemLabel WHERE {{
                          ?item wdt:P1566 "{geonames_id}".
                          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
                        }}
                    """ # {lang}
            results = perform_sparql_query(sparql_query)
            if not results:
                print(f"\n⚠️ Skipping '{iri}', query returned no results.")
                continue

            binding = results[0]
            label = binding.get("itemLabel", {}).get("value")
            url = binding.get("item", {}).get("value")
            match_id = re.search(r"wikidata\.org/entity/(Q\d+)", url)
            qid = match_id.group(1)
            if not qid:
                raise HTTPException(status_code=500, detail="Wikidata ID not found.")
            if not label:
                print(f"\n⚠️ Skipping '{iri}', label not found.")
                continue

            entities = []
            processed_qids = set()

            geometry = retrieve_geometry(None, label, qid, entities, processed_qids, True)

            #results = analyze_text(label, lang=lang)

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
                else:
                    print("Missing text for ", g)

            processed_geonames_id.add(iri)

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