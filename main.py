# ======= Import libraries =======

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import spacy
import requests
import time
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import geojson
import os


# ======= Init =======

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

WIKIFIER_API_KEY = os.getenv("WIKIFIER_API_KEY")
if not WIKIFIER_API_KEY:
    raise EnvironmentError("WIKIFIER_API_KEY not defined in environment.")


# ======= Pydantic model =======
class TextInput(BaseModel):
    text: str


# ======= Utility functions =======

def extract_geo_entity(text):
    doc = nlp(text)
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
        "threshold": "0.8"
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

def save_geojson(file, filename="output.geojson"):
    features = []

    for res in file:
        vkt_value = res.get("vkt")
        if not vkt_value:
            continue
        try:
            shape = wkt.loads(vkt_value)
            geojson_geom = geojson.Feature(
                geometry=geojson.loads(geojson.dumps(shape.__geo_interface__)),
                properties={
                    "label": res["label"],
                    "qid": res["qid"],
                    "description": res.get("description"),
                    "wikidata_url": res["wikidata_url"],
                    "osm_id": res.get("osm_id")
                }
            )
            features.append(geojson_geom)
        except Exception as e:
            print(f"❌ Error converting GeoJSON to {res['label']}: {e}")

    feature_collection = geojson.FeatureCollection(features)
    with open(filename, "w", encoding="utf-8") as f:
        geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)
    print(f"\n✅ GeoJSON saved in: {filename}")

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


def analyze_text(text):
    entities_spacy = extract_geo_entity(text)
    print(f"\nEntities found by spaCy: {entities_spacy}")

    annotations = disambiguation_with_wikifier(text)
    entities = []
    processed_qids = set()

    def process_annotation(annotation):
        try:
            qid = annotation["wikiDataItemId"]
            label = annotation["title"]
        except KeyError:
            return

        if qid in processed_qids:
            return

        try:
            if annotation.get("cosine", 1.0) < 0.5 or not is_geographic_entity(qid):
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
            entities.append({
                "label": label,
                "qid": qid,
                "description": annotation.get("description"),
                "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
                "osm_id": osm_id,
                "vkt": vkt
            })
            processed_qids.add(qid)
            time.sleep(1)  # Avoid rate limit
        except Exception as e:
            print(f"❌ Error with {label}: {e}")

    for ent_text in entities_spacy:
        ent_annotations = disambiguation_with_wikifier(ent_text)
        for ann in ent_annotations:
            process_annotation(ann)

    for ann in annotations:
        process_annotation(ann)

    return entities


# ======= FastAPI endpoints =======

@app.post("/analyze")
def analyze_endpoint(payload: TextInput):
    try:
        results = analyze_text(payload.text)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/geojson")
def get_geojson(data: TextInput):
    try:
        results = analyze_text(data.text)

        features = []
        for res in results:
            if res["vkt"]:
                geometry = wkt.loads(res["vkt"])
                gj_geometry = geojson.Feature(
                    geometry=geojson.loads(geojson.dumps(geometry.__geo_interface__)),
                    properties={
                        "label": res["label"],
                        "qid": res["qid"],
                        "wikidata_url": res["wikidata_url"],
                        "osm_id": res["osm_id"],
                        "description": res["description"]
                    }
                )
                features.append(gj_geometry)

        feature_collection = geojson.FeatureCollection(features)

        filename = f"output_{uuid4().hex}.geojson"
        path = f"/tmp/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)

        return FileResponse(path, media_type="application/geo+json", filename=filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))