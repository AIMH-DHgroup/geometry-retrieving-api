import json
import csv

if __name__ == '__main__':

    def jsonld_entity_counter(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = data.get('@graph', [])

        entities = [e for e in graph if e.get('@type') == 'Feature']

        return len(entities)


    def geonames_value_counter(filepath):
        count = 0
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("geonames", "").strip()
                if value:
                    count += 1

        return count


    entity_counter = jsonld_entity_counter('geosparql.jsonld')
    geonames_counter = geonames_value_counter('results_filtered.csv')

    if geonames_counter == 0:
        print("Unable to calculate the percentage: no 'geonames' values found.")
    else:
        percentage = (entity_counter / geonames_counter) * 100
        print(f"\n{entity_counter}/{geonames_counter} --> {percentage:.2f}%")