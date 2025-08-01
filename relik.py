import json
import torch

from relik import Relik
from relik.inference.data.objects import RelikOutput

from kilt.knowledge_source import KnowledgeSource

import ezodf

# Apri il file .ods
doc = ezodf.opendoc("ANN_245_training_resultsEvents_EN_noduplicate_noProcess.ods")

# Prendi il primo foglio
sheet = doc.sheets[0]

# Crea una lista per salvare i nuovi dati
new_data = []

# Inizializza il modello relik
relik = Relik.from_pretrained(
    "sapienzanlp/relik-entity-linking-base",
    device="cuda" if torch.cuda.is_available() else "cpu",
    precision="fp16" if torch.cuda.is_available() else "fp32",
    skip_metadata=True  # don't load index metadata to keep low memory requirements
)

# Itera sulle righe
for i, row in enumerate(sheet.rows(), start=1):
    # Estrai il testo delle celle (rimuovendo celle vuote)
    values = [cell.value for cell in row if cell.value is not None]
    if values: 
        
        sen = str(values[0])  # Supponiamo che la prima cella contenga il testo
        #print(f"Testo: {sen}")
        # Qui puoi fare altre operazioni con "testo"
        
        #apply relik on sentence
        relik_out: RelikOutput = relik(sen)
        print(relik_out.spans) 
        

        new_item = {
            "row": i,
            "sentence": sen,
            "entities": []
        } 
        
        # Aggiungi l'elemento alla lista
        new_data.append(new_item)   
        
        #find kilt wikidata id for each entity recognized by relik
        for entityRelik in relik_out.spans:
            #print(entityRelik)
            titleEntity = entityRelik.label
            
            # Utilizza KILT Knowledge Source per ottenere informazioni aggiuntive
            ks = KnowledgeSource()
            ks.get_num_pages()
            page = ks.get_page_by_title(titleEntity)
                            
            wikipedia_title = page['wikipedia_title']
            wikidata_info = page['wikidata_info']['wikidata_id']
        
            # Crea un nuovo elemento con lo stesso contenuto
            new_item["entities"].append(
                {
                    "text_label": wikipedia_title,
                    "type": "",
                    "Wikidata_ID": "" + wikidata_info + ""
                }
            )
            

            # Salva il nuovo JSON in un file
            with open('relik_entities.json', 'w') as f:
                json.dump(new_data, f, indent=4)