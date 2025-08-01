import json
import requests
import time
from pathlib import Path

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

def fetch_geographic_entities(qids, retries=3, delay=5):
    if not qids:
        return set()

    query = f"""
    SELECT DISTINCT ?qid WHERE {{
      VALUES ?qid {{ {' '.join(f'wd:{q}' for q in qids)} }}
      ?qid wdt:P31/wdt:P279* wd:Q618123 .
    }}
    """

    headers = {"Accept": "application/sparql-results+json"}
    for attempt in range(retries):
        try:
            response = requests.get(WIKIDATA_SPARQL_URL, params={"query": query}, headers=headers, timeout=60)
            response.raise_for_status()
            results = response.json()["results"]["bindings"]
            return {r["qid"]["value"].split("/")[-1] for r in results}
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Errore di connessione: {e}, retry tra {delay}s...")
            time.sleep(delay)
            delay *= 2  # backoff esponenziale
    print("[ERRORE] Query fallita dopo i retry.")
    return set()


def filter_places(input_path, output_path, batch_size=20):
    # Carica dati di input
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_qids = {e["Wikidata_ID"] for row in data for e in row["entities"] if e.get("Wikidata_ID")}
    print(f"Trovati {len(all_qids)} QID unici, verifica in batch da {batch_size}...")

    geo_qids = set()
    qid_list = list(all_qids)

    for i in range(0, len(qid_list), batch_size):
        batch = qid_list[i:i+batch_size]
        print(f"[INFO] Elaboro batch {i//batch_size + 1}/{-(-len(qid_list)//batch_size)}: {batch}")
        geo_qids |= fetch_geographic_entities(batch)

        # Aggiorna file JSON parziale dopo ogni batch
        filtered_data = []
        for row in data:
            filtered_entities = [e for e in row["entities"] if e.get("Wikidata_ID") in geo_qids]
            filtered_data.append({
                "row": row["row"],
                "sentence": row["sentence"],
                "entities": filtered_entities
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        print(f"[INFO] File aggiornato: {output_path} (QID geografici trovati finora: {len(geo_qids)})")

        # Pausa tra batch per non sovraccaricare il server
        time.sleep(15)

    print(f"[FINE] Identificati {len(geo_qids)} QID geografici. File finale salvato in: {output_path}")


input_file = "relik_entities.json"        # <-- file di input con entità trovate da Relik
output_file = "filtered_places.json"
filter_places(input_file, output_file)