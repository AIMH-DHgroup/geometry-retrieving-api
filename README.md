# geometry-retrieving-api
This API allows to retrieve information about entities in an input text. After the Named-Entity Recognition phase, it uses ``Wikifier`` to disambiguate them, then a SPARQL query is performed to obtain Wikidata and OpenStreetMap IDs. Finally, the geometries are stored in a GeoJSON file.

## Installation
Create a Python environment and install the requirements.txt using the command:

```shell
pip install -r requirements.txt
```

After that, go to the [Wikifier website](https://wikifier.org/register.html) and create a user. Then, copy the key and paste it into the following command:

```shell
export WIKIFIER_API_KEY="your_api_key"
```

Lastly, run the API with:

```shell
uvicorn main:app
```

If you want to use the web interface open this [tab](http://127.0.0.1:8000/docs) in your browser, otherwise you can use this command from terminal:

```shell
curl -X POST "http://127.0.0.1:8000/geosparql" \
     -H "Content-Type: application/json" \
     -d '{"text":"your_text"}'
```

The endpoints are ``/analyze`` and ``/geosparql`` and the latter has the ``download`` option set to ``true`` by default but you can pass ``false`` with: ``http://127.0.0.1:8000/geosparql?download=false``.

## Help/Feedback
If you need help or want to leave feedback, check out the discussions [here](https://github.com/AIMH-DHgroup/geometry-retrieving-api/discussions) or start a new one.
