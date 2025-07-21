import os
import json
import re

def normalize(text):
    """Tokenize and clean the text for the comparison"""
    tokens = re.findall(r'\w+', text.lower())
    return set(tokens)

def jaccard_similarity(set1, set2):
    """Calculate the Jaccard similarity between two strings"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def labels_match(label1, label2):
    """Check if two labels corresponds to each other according to the rules"""
    if not label1 or not label2:
        return False

    label1 = label1.strip().lower()
    label2 = label2.strip().lower()

    if label1 == label2:
        return True
    if label1 in label2 or label2 in label1:
        return True
    if jaccard_similarity(normalize(label1), normalize(label2)) > 0.5:
        return True

    return False

def load_json_files(folder_path, filename):
    """
    Load a JSON file.
    """
    data = {}
    if not os.path.isdir(folder_path):
        print(f"\nFolder does not exist: {folder_path}, {os.path.isdir(folder_path)}")
        return data
    if filename.endswith(".json"):
        try:
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                data[filename] = json.load(file)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error loading file '{filename}': {e}")
    return data

def count_entities(data):
    """Count the number of entities in the dataset"""
    count = 0
    count_duplicates = 0
    for i in range(len(data)):
        processed_ids = set()
        for j in range(len(data[i]['entities'])):
            count += 1
            if data[i]['entities'][j]['Wikidata_ID'] in processed_ids:
                count_duplicates += 1
            processed_ids.add(data[i]['entities'][j]['Wikidata_ID'])
    return count, count_duplicates

def calculate_metrics_only_ner(gold_data, predicted_data):
    """
    Calculate precision, recall e F1 score for ID retrieving.
    """
    # Metrics for keyword extraction
    entity_true_positive = 0
    entity_false_positive = 0
    entity_false_negative = 0

    # Calculate total entities
    (gold_count, gold_duplicates_count) = count_entities(gold_data)
    (predicted_count, predicted_duplicates_count) = count_entities(predicted_data)
    tot = gold_count + predicted_count
    print(f"\nNumber of total entities: {gold_count} (gold standard), {predicted_count} (predicted), {tot} (overall)")
    print(f"Number of total duplicated entities (of the same event): {gold_duplicates_count} (gold standard), {predicted_duplicates_count} (predicted)")

    total_processed_labels = 0

    if len(gold_data) != len(predicted_data):
        raise ValueError(f"The files does not have the same number of rows ({len(gold_data)}, {len(predicted_data)}).")

    for i in range(len(gold_data)):

        gold_entities = gold_data[i].get('entities', [])
        predicted_entities = predicted_data[i].get('entities', [])

        processed_labels = set()

        if not gold_entities:
            if not predicted_entities:
                continue
            else:

                wikidata_labels = [ent["text_label"] for ent in predicted_entities]

                for predicted in predicted_entities:
                    predicted_label = predicted.get('text_label')

                    if predicted_label not in processed_labels:
                        count_predicted_label = wikidata_labels.count(predicted_label)

                        entity_false_positive += count_predicted_label
                        processed_labels.add(predicted_label)

                continue

        if not predicted_entities:

            wikidata_labels = [ent["text_label"] for ent in gold_entities]

            for gold in gold_entities:
                gold_label = gold.get('text_label')

                if gold_label not in processed_labels:
                    count_gold_label = wikidata_labels.count(gold_label)

                    entity_false_negative += count_gold_label
                    processed_labels.add(gold_label)

            total_processed_labels += len(processed_labels)
            continue

        gold_values = [e['text_label'] for e in gold_entities if 'text_label' in e]

        for gold in gold_entities:

            gold_label = gold.get('text_label')

            labels_gold = [ent["text_label"] for ent in gold_entities]
            labels_predicted = [ent["text_label"] for ent in predicted_entities]

            count_gold_label = labels_gold.count(gold_label)
            count_predicted_label = labels_predicted.count(gold_label)

            diff_labels = count_gold_label - count_predicted_label

            if gold_label not in processed_labels:

                if gold_label not in labels_predicted:

                    entity_false_negative += diff_labels

                else:

                    if diff_labels == 0:

                        entity_true_positive += count_predicted_label

                    elif diff_labels > 0:

                        entity_true_positive += count_predicted_label
                        count_gold_label -= count_predicted_label
                        count_predicted_label -= count_predicted_label
                        diff_labels = count_gold_label - count_predicted_label
                        entity_false_negative += diff_labels

                    elif diff_labels < 0:

                        entity_true_positive += count_gold_label
                        count_predicted_label -= count_gold_label
                        count_gold_label -= count_gold_label
                        diff_labels =  count_predicted_label - count_gold_label
                        entity_false_positive += abs(diff_labels)

                processed_labels.add(gold_label)

        for predicted in predicted_entities:
            predicted_label = predicted.get('text_label')

            if predicted_label not in processed_labels:

                wikidata_labels = [ent["text_label"] for ent in predicted_entities]

                count_predicted_label = wikidata_labels.count(predicted_label)

                if predicted_label not in gold_values:

                    entity_false_positive += count_predicted_label

                processed_labels.add(predicted_label)

        total_processed_labels += len(processed_labels)

    # Metrics for keyword extraction
    entity_precision = entity_true_positive / (entity_true_positive + entity_false_positive) if (entity_true_positive + entity_false_positive) > 0 else 0
    entity_recall = entity_true_positive / (entity_true_positive + entity_false_negative) if (entity_true_positive + entity_false_negative) > 0 else 0
    entity_f1_score = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0

    print(f"Total unique processed IDs of the gold standard: {total_processed_labels}")

    return entity_precision, entity_recall, entity_f1_score, entity_true_positive, entity_false_positive, entity_false_negative

def calculate_metrics_only_id(gold_data, predicted_data):
    """
    Calculate precision, recall e F1 score for ID retrieving.
    """
    # Metrics for keyword extraction
    entity_true_positive = 0
    entity_false_positive = 0
    entity_false_negative = 0

    # Calculate total entities
    (gold_count, gold_duplicates_count) = count_entities(gold_data)
    (predicted_count, predicted_duplicates_count) = count_entities(predicted_data)
    tot = gold_count + predicted_count
    print(f"\nNumber of total entities: {gold_count} (gold standard), {predicted_count} (predicted), {tot} (overall)")
    print(f"Number of total duplicated entities (of the same event): {gold_duplicates_count} (gold standard), {predicted_duplicates_count} (predicted)")

    total_processed_ids = 0

    if len(gold_data) != len(predicted_data):
        raise ValueError(f"The files does not have the same number of rows ({len(gold_data)}, {len(predicted_data)}).")

    for i in range(len(gold_data)):

        gold_entities = gold_data[i].get('entities', [])
        predicted_entities = predicted_data[i].get('entities', [])

        processed_ids = set()

        if not gold_entities:
            if not predicted_entities:
                continue
            else:

                wikidata_ids = [ent["Wikidata_ID"] for ent in predicted_entities]

                for predicted in predicted_entities:
                    predicted_id = predicted.get('Wikidata_ID')

                    if predicted_id not in processed_ids:
                        count_predicted_id = wikidata_ids.count(predicted_id)

                        entity_false_positive += count_predicted_id
                        processed_ids.add(predicted_id)

                continue

        if not predicted_entities:

            wikidata_ids = [ent["Wikidata_ID"] for ent in gold_entities]

            for gold in gold_entities:
                gold_id = gold.get('Wikidata_ID')

                if gold_id not in processed_ids:
                    count_gold_id = wikidata_ids.count(gold_id)

                    entity_false_negative += count_gold_id
                    processed_ids.add(gold_id)

            total_processed_ids += len(processed_ids)
            continue

        gold_values = [e['Wikidata_ID'] for e in gold_entities if 'Wikidata_ID' in e]

        for gold in gold_entities:

            gold_id = gold.get('Wikidata_ID')

            id_gold = [ent["Wikidata_ID"] for ent in gold_entities]
            id_predicted = [ent["Wikidata_ID"] for ent in predicted_entities]

            count_gold_id = id_gold.count(gold_id)
            count_predicted_id = id_predicted.count(gold_id)

            diff_labels = count_gold_id - count_predicted_id

            if gold_id not in processed_ids:

                if gold_id not in id_predicted:

                    entity_false_negative += diff_labels

                else:

                    if diff_labels == 0:

                        entity_true_positive += count_predicted_id

                    elif diff_labels > 0:

                        entity_true_positive += count_predicted_id
                        count_gold_id -= count_predicted_id
                        count_predicted_id -= count_predicted_id
                        diff_labels = count_gold_id - count_predicted_id
                        entity_false_negative += diff_labels

                    elif diff_labels < 0:

                        entity_true_positive += count_gold_id
                        count_predicted_id -= count_gold_id
                        count_gold_id -= count_gold_id
                        diff_labels =  count_predicted_id - count_gold_id
                        entity_false_positive += abs(diff_labels)

                processed_ids.add(gold_id)

        for predicted in predicted_entities:
            predicted_id = predicted.get('Wikidata_ID')

            if predicted_id not in processed_ids:

                wikidata_ids = [ent["Wikidata_ID"] for ent in predicted_entities]

                count_predicted_id = wikidata_ids.count(predicted_id)

                if predicted_id not in gold_values:

                    entity_false_positive += count_predicted_id

                processed_ids.add(predicted_id)

        total_processed_ids += len(processed_ids)

    # Metrics for keyword extraction
    entity_precision = entity_true_positive / (entity_true_positive + entity_false_positive) if (entity_true_positive + entity_false_positive) > 0 else 0
    entity_recall = entity_true_positive / (entity_true_positive + entity_false_negative) if (entity_true_positive + entity_false_negative) > 0 else 0
    entity_f1_score = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0

    print(f"Total unique processed IDs of the gold standard: {total_processed_ids}")

    return entity_precision, entity_recall, entity_f1_score, entity_true_positive, entity_false_positive, entity_false_negative

def calculate_metrics(gold_data, predicted_data):
    """
    Calculate precision, recall e F1 score for ID retrieving.
    """
    # Metrics for keyword extraction
    entity_true_positive = 0
    entity_false_positive = 0
    entity_false_negative = 0

    # Calculate total entities
    (gold_count, gold_duplicates_count) = count_entities(gold_data)
    (predicted_count, predicted_duplicates_count) = count_entities(predicted_data)
    tot = gold_count + predicted_count
    print(f"\nNumber of total entities: {gold_count} (gold standard), {predicted_count} (predicted), {tot} (overall)")
    print(f"Number of total duplicated entities (of the same event): {gold_duplicates_count} (gold standard), {predicted_duplicates_count} (predicted)")

    total_processed_ids = 0

    if len(gold_data) != len(predicted_data):
        raise ValueError(f"The files does not have the same number of rows ({len(gold_data)}, {len(predicted_data)}).")

    for i in range(len(gold_data)):

        gold_entities = gold_data[i].get('entities', [])
        predicted_entities = predicted_data[i].get('entities', [])

        processed_ids = set()

        if not gold_entities:
            if not predicted_entities:
                continue
            else:

                wikidata_ids = [ent["Wikidata_ID"] for ent in predicted_entities]

                for predicted in predicted_entities:
                    predicted_id = predicted.get('Wikidata_ID')

                    if predicted_id not in processed_ids:
                        count_predicted_id = wikidata_ids.count(predicted_id)

                        entity_false_positive += count_predicted_id
                        processed_ids.add(predicted_id)

                continue

        if not predicted_entities:

            wikidata_ids = [ent["Wikidata_ID"] for ent in gold_entities]

            for gold in gold_entities:
                gold_id = gold.get('Wikidata_ID')

                if gold_id not in processed_ids:
                    count_gold_id = wikidata_ids.count(gold_id)

                    entity_false_negative += count_gold_id
                    processed_ids.add(gold_id)

            total_processed_ids += len(processed_ids)
            continue

        gold_values = [e['Wikidata_ID'] for e in gold_entities if 'Wikidata_ID' in e]
        predicted_values = [e['Wikidata_ID'] for e in predicted_entities if 'Wikidata_ID' in e]

        for gold in gold_entities:

            gold_id = gold.get('Wikidata_ID')
            gold_label = gold.get('text_label')

            if gold_id not in processed_ids:

                wikidata_gold_ids = [ent["Wikidata_ID"] for ent in gold_entities]
                wikidata_predicted_ids = [ent["Wikidata_ID"] for ent in predicted_entities]

                count_gold_id = wikidata_gold_ids.count(gold_id)

                labels_gold = [ent["text_label"] for ent in gold_entities]
                labels_predicted = [ent["text_label"] for ent in predicted_entities]

                count_gold_label = labels_gold.count(gold_label)
                count_predicted_label = labels_predicted.count(gold_label)

                diff_labels = count_gold_label - count_predicted_label

                if gold_id in predicted_values:

                    count_predicted_id = wikidata_predicted_ids.count(gold_id)

                    if count_gold_id == count_predicted_id:
                        entity_true_positive += count_predicted_id
                    else:

                        diff = count_gold_id - count_predicted_id

                        if diff > 0:

                            if diff_labels == 0:

                                entity_true_positive += count_predicted_id

                            elif diff_labels > 0:

                                entity_true_positive += count_predicted_id
                                count_gold_id -= count_predicted_id
                                count_predicted_id -= count_predicted_id
                                diff = count_gold_id - count_predicted_id

                                entity_false_negative += diff

                            elif diff_labels < 0:

                                entity_true_positive += count_predicted_id
                                count_gold_id -= count_predicted_id
                                count_predicted_id -= count_predicted_id
                                diff = count_gold_id - count_predicted_id

                                entity_false_negative += diff

                                count_gold_label -= count_predicted_id
                                count_predicted_label -= count_predicted_id

                                count_gold_label -= count_gold_id
                                count_predicted_label -= count_gold_id

                                diff_labels = count_gold_label - count_predicted_label

                                entity_false_positive += abs(diff_labels)

                        elif diff < 0:

                            entity_false_positive += abs(diff)

                        elif diff == 0:

                            entity_true_positive += count_predicted_id

                else:

                    while count_gold_label > 0 or count_predicted_label > 0:

                        if gold_label in labels_predicted and diff_labels > 0:
                            entity_false_positive += count_gold_label
                        elif gold_label not in labels_predicted and diff_labels > 0:
                            entity_false_negative += count_gold_label

                        if count_gold_label > 0:
                            count_gold_label -= 1
                        if count_predicted_label > 0:
                            count_predicted_label -= 1

                        diff_labels = count_gold_label - count_predicted_label

                processed_ids.add(gold_id)

        for predicted in predicted_entities:
            predicted_id = predicted.get('Wikidata_ID')

            if predicted_id not in processed_ids:

                wikidata_ids = [ent["Wikidata_ID"] for ent in predicted_entities]

                count_predicted_id = wikidata_ids.count(predicted_id)

                if predicted_id not in gold_values:

                    entity_false_positive += count_predicted_id

                processed_ids.add(predicted_id)

        total_processed_ids += len(processed_ids)

    # Metrics for keyword extraction
    entity_precision = entity_true_positive / (entity_true_positive + entity_false_positive) if (entity_true_positive + entity_false_positive) > 0 else 0
    entity_recall = entity_true_positive / (entity_true_positive + entity_false_negative) if (entity_true_positive + entity_false_negative) > 0 else 0
    entity_f1_score = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0

    print(f"Total unique processed IDs of the gold standard: {total_processed_ids}")

    return entity_precision, entity_recall, entity_f1_score, entity_true_positive, entity_false_positive, entity_false_negative


if __name__ == "__main__":

    gold_file = load_json_files("./data", "entities_gold.json")
    predicted_file = load_json_files("./data", "entities_pred.json")

    print("\nNER")
    metrics = calculate_metrics_only_ner(gold_file['entities_gold.json'], predicted_file['entities_pred.json'])
    (precision, recall, f1_score, true_positive, false_positive, false_negative) = metrics

    print(f"\nPrecision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")
    print(f"True positive: {true_positive}")
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")

    print("\nID")
    metrics = calculate_metrics_only_id(gold_file['entities_gold.json'], predicted_file['entities_pred.json'])
    (precision, recall, f1_score, true_positive, false_positive, false_negative) = metrics

    print(f"\nPrecision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")
    print(f"True positive: {true_positive}")
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")

    print("\nALL")
    metrics = calculate_metrics(gold_file['entities_gold.json'], predicted_file['entities_pred.json'])
    (precision, recall, f1_score, true_positive, false_positive, false_negative) = metrics

    print(f"\nPrecision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")
    print(f"True positive: {true_positive}")
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")