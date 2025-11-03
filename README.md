# GeoLinks API
<!--This API allows to retrieve information about entities in an input text. After the Named-Entity Recognition phase, it uses ``Wikifier`` to disambiguate them, then a SPARQL query is performed to obtain Wikidata and OpenStreetMap IDs. Finally, the geometries are stored in a GeoJSON file.
Tested with Python 3.9. -->
GeoLinks is a multilingual API that processes either an input text or a GeoNames IRI to identify geographical entities. For entities detected from text, the API returns their corresponding coordinates (latitude and longitude) and polygon geometry. For entities provided as GeoNames IRIs, coordinates are not retrieved since this information is already available in Geonames. The geographic data are automatically retrieved from Wikidata and OpenStreetMap, and the results are provided as a GeoSPARQL knowledge graph in JSON-LD format.

## Installation
Create a Python environment and install the requirements.txt using the command:

```shell
pip install -r requirements.txt
```

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
- SpaCy-based endpoint: https://gel.isti.cnr.it/spacy — uses SpaCy for NER.
- SpaCy+Flair endpoint: https://gel.isti.cnr.it/spacy-flair — uses a hybrid SpaCy + Flair model for enhanced entity recognition.

To use the API, send a POST request to one of the URLs above, including the text you want to analyze in the request body.
For example, to analyze the text Paris is the capital city of France, the body of the POST request should contain the following JSON object:  
{ "text": "Paris is the capital city of France." }

### Input: GeoNames IRI
For GeoNames IRIs, GeoLink provides the following endpoint:

- https://gel.isti.cnr.it/iri

To use the API, send a GET request by appending the GeoNames IRI of the entity you want to query to the endpoint URL. 
For example, to retrieve information for Paris (GeoNames ID 2988507), whose GeoNames page is https://www.geonames.org/2988507/, the URL to load is:
https://gel.isti.cnr.it/iri?iri=https://www.geonames.org/2988507/

## Evaluation
### Input:Text 
The evaluation of the API is based on a corpus consisting of 678 events, created within the Horizon Europe Craeft project (https://www.craeft.eu/). The events were retrieved via the SPARQL endpoint of the project platform and are available in the file [Creaeft_event_corpus.csv](data/Creaeft_event_corpus.csv).

For the evaluation, we created a gold standard composed of 237 events. These numbers were calculated using the sample size determination formula with finite population correction. The gold standard corpus is available in Json format in the file [gold-standard.json](data
/gold_standard_geographic_entities/gold-standard.json). 

The gold standard was manually annotated, identifying the geographical entities and their corresponding Wikidata IDs. To retrieve the entities and their IDs, we tested several Named Entity Recognition (NER) systems, also in combination with entity linking software. The evaluation metrics (Precision, Recall, and F1) are reported in the following article: [https://zenodo.org/records/17422250](https://zenodo.org/records/17422250). 
For each retrieved entity associated with a Wikidata ID, we were able to obtain both the geographical coordinates and the polygon geometry. 

The best results were obtained using SpaCy and the combination SpaCy+Flair, which achieved very similar performance for the Named Entity Recognition task, while the retrieval of Wikidata IDs was performed via SPARQL queries to the Wikidata endpoint, leveraging the semantic similarity between the contexts in which the NER-extracted entities appear and their descriptions as reported on Wikidata.

### Input:Geonames IRI 
In the Craeft corpus, the narratives were associated with metadata, including some GeoNames IRIs for certain geographical entities. These IRIs were extracted and used as input for the corresponding API endpoint. We evaluated the percentage of entities for which the API was able to correctly associate a polygon. The API was tested on 459 IRIs of places extracted from the Craeft knowledge base. The list of IRIs is available in the file [Geonames_IRI_corpus.csv](data/Geonames_IRI_corpus.csv).

The results are reported in the following article: [https://zenodo.org/records/17422250](https://zenodo.org/records/17422250).

## Supported Languages
The list was taken by Spacy documentation: "en" (English - UK), "it" (Italian), "de" (German), "fr" (French - France), "es" (Spanish - Spain), "ru" (Russian), "pl" (Polish), "pt" (Portuguese - Portugal) and "gr", etc.


## Help/Feedback
If you need help or want to leave feedback, check out the discussions [here](https://github.com/AIMH-DHgroup/geometry-retrieving-api/discussions) or start a new one.
