import json
import csv
import re

if __name__ == '__main__':

    def jsonld_entity_counter(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = data.get('@graph', [])

        seen_qids = set()
        unique_entities = []

        for e in graph:
            if e.get('@type') == 'Feature':
                qid = e.get('qid')
                print(qid)
                if qid == 'Q12181500':
                    print(e)
                if qid and qid not in seen_qids:
                    seen_qids.add(qid)
                    unique_entities.append(e)

        return len(unique_entities)


    def geonames_value_counter(filepath, no_duplicates=False):
        if no_duplicates:
            count = set()
        else:
            count = 0
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("geonames", "").strip()
                if value:
                    if no_duplicates:
                        match = re.search(r'(\d+)', value)
                        if match:
                            geonames_id = match.group(1)
                            count.add(geonames_id)
                    else:
                        count += 1

        if no_duplicates:
            return len(count)
        else:
            return count


    entity_counter = jsonld_entity_counter('geosparql.jsonld')
    geonames_counter = geonames_value_counter('results_filtered.csv')

    if geonames_counter == 0:
        print("Unable to calculate the percentage: no 'geonames' values found.")
    else:
        percentage = (entity_counter / geonames_counter) * 100
        print(f"\n{entity_counter}/{geonames_counter} --> {percentage:.2f}%")