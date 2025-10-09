# GeoLinks API
<!--This API allows to retrieve information about entities in an input text. After the Named-Entity Recognition phase, it uses ``Wikifier`` to disambiguate them, then a SPARQL query is performed to obtain Wikidata and OpenStreetMap IDs. Finally, the geometries are stored in a GeoJSON file.
Tested with Python 3.9.
GeoLinks is a multilingual API that processes either an input text or a GeoNames IRI to identify geographical entities and enrich them with geospatial information. Indeed, for entities detected from text, the API returns their corresponding coordinates (latitude and longitude) and polygon geometry, as well as the corresponding Wikidata IRI. For entities provided as GeoNames IRIs, coordinates are not retrieved since this information is already available in Geonames. The geographic data are automatically retrieved from Wikidata and OpenStreetMap, and the results are provided in JSON-LD format.-->

GeoLinks is a multilingual API that processes either an input text or a GeoNames IRI to enrich geographical entities with geospatial information, including an associated Wikidata IRI for each entity. For input text, the API first extracts the geographical entities and then returns their corresponding coordinates (latitude and longitude), polygon geometry, and the associated Wikidata IRI. For entities provided as GeoNames IRIs, coordinates are not retrieved since this information is already available in GeoNames; however, the API retrieves the corresponding polygon geometry and the related Wikidata IRIs. The geographic data are automatically obtained from Wikidata and OpenStreetMap, and the results are provided in JSON-LD format.

In conclusion, the API produces a graph of interconnected geographical entities enriched with spatial information.

## Installation
Create a Python environment and install the requirements.txt using the command:

```shell
pip install -r requirements.txt
```

After that, run the API using the command:

```shell
uvicorn main:app
```

To access the API web interface, open this link [https://127.0.0.1:8000/docs](https://127.0.0.1:8000/docs) in your browser.

<!--After that, go to the [Wikifier website](https://wikifier.org/register.html) and create a user. Then, copy the key and paste it into the following command:

```shell
export WIKIFIER_API_KEY="your_api_key"
```

Lastly, run the API with:

```shell
uvicorn main:app
```

If you want to use the web interface open this [tab](http://127.0.0.1:8000/docs) in your browser, otherwise you can use this command:

```shell
curl -X POST "http://127.0.0.1:8000/geosparql" \
     -H "Content-Type: application/json" \
     -d '{"text":"your_text"}'
```

The endpoints are ``/analyze`` and ``/geosparql`` and the latter has the ``download`` option set to ``true`` by default but you can pass ``false`` with: ``http://127.0.0.1:8000/geosparql?download=false``.
-->

## Usage Instructions
### Input:Text 
When provided with a text input, GeoLink offers two endpoints for Named Entity Recognition (NER):
- SpaCy 3.8-based endpoint: https://127.0.0.1:8000/spacy — uses SpaCy for NER.
- SpaCy 3.8+Flair endpoint: https://127.0.0.1:8000/spacy-flair — uses a hybrid SpaCy + Flair model for enhanced entity recognition.

To use the API, send a POST request to one of the URLs above, including the text you want to analyze in the request body.
For example, to analyze the text Paris is the capital city of France, the body of the POST request should contain the following JSON object:  
{ "text": "Paris is the capital city of France." }

### Input: GeoNames IRI
For GeoNames IRIs, GeoLink provides the following endpoint:

- https://127.0.0.1:8000/iri

To use the API, send a GET request by appending the GeoNames IRI of the entity you want to query to the endpoint URL. 
For example, to retrieve information for Paris (GeoNames ID 2988507), whose GeoNames page is https://www.geonames.org/2988507/, the URL to load is:
https://127.0.0.1:8000/iri?iri=https://www.geonames.org/2988507/

## Evaluation
### Input:Text 
The evaluation of the API is based on a corpus consisting of # narratives, comprising a total of # events, created within the Horizon Europe Craeft project (https://www.craeft.eu/). The narratives and events were retrieved via the SPARQL endpoint of the project platform and are available in the file Craeft_corpus.txt.

For the evaluation, we created a gold standard composed of # narratives and # events. These numbers were calculated using the sample size determination formula with finite population correction. The gold standard corpus is available in Json format in the file gold_standard.json.

The gold standard was manually annotated, identifying the geographical entities and their corresponding Wikidata IDs. To retrieve the entities and their IDs, we tested several Named Entity Recognition (NER) systems, also in combination with entity linking software. 

The evaluation metrics (Precision, Recall, and F1) are reported in the following article: XXXX. 

For each retrieved entity associated with a Wikidata ID, we were able to obtain both the geographical coordinates and the polygon geometry. 

The best results were obtained using SpaCy and the combination SpaCy+Flair, which achieved very similar performance for the Named Entity Recognition task, while the retrieval of Wikidata IDs was performed via SPARQL queries to the Wikidata endpoint, leveraging the semantic similarity between the contexts in which the NER-extracted entities appear and their descriptions as reported on Wikidata.

### Input:Geonames IRI 
In the Craeft corpus, the narratives were associated with metadata, including some GeoNames IRIs for certain geographical entities. These IRIs were extracted and used as input for the corresponding API endpoint. We evaluated the percentage of entities for which the API was able to correctly associate a polygon. 

The results are reported in the following article: XXXX.

## Supported Languages
The list was taken by Spacy documentation: "en" (English - UK), "it" (Italian), "de" (German), "fr" (French - France), "es" (Spanish - Spain), "ru" (Russian), "pl" (Polish), "pt" (Portuguese - Portugal) and "gr", etc.


## Help/Feedback
If you need help or want to leave feedback, check out the discussions [here](https://github.com/AIMH-DHgroup/geometry-retrieving-api/discussions) or start a new one.
