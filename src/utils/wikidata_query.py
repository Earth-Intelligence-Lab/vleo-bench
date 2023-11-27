import json
import sys

import geopandas
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def main():
    endpoint_url = "https://query.wikidata.org/sparql"
    shp_file_path = "./data/Landmarks/all_unions_polygons_convexhulls_metadata_wikidata_us.gpkg"

    shp = geopandas.read_file(shp_file_path)
    shp["wikidata"] = [json.loads(x) for x in shp["wikidata"]]
    shp["name"] = [list(x["entities"].values())[0]["labels"]["en"]["value"] for x in shp["wikidata"]]
    print(shp)

    instance_of_labels = []
    instance_of_ids = []
    distractor_ids = []
    distractor_labels = []
    for entity_id in tqdm(shp["wikidata_entity_id"]):
        query = f"""SELECT ?instanceOf ?instanceOfLabel WHERE {{
            wd:{entity_id} wdt:P31 ?instanceOf .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,fr,zh,it,es". }}
        }}"""

        results = pd.json_normalize(get_results(endpoint_url, query)['results']['bindings'])

        instance_of_labels.append(json.dumps(results["instanceOfLabel.value"].tolist()))
        instance_of_ids.append(json.dumps(results["instanceOf.value"].tolist()))

        entity_distractor_ids = []
        entity_distractor_labels = []
        for instance_of_id in results["instanceOf.value"]:
            query_distractors = f"""SELECT DISTINCT ?item ?itemLabel WHERE {{
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,fr,zh,it,es". }}
                {{
                    SELECT DISTINCT ?item WHERE {{
                        ?item p:P31 ?statement0.
                        ?statement0 (ps:P31/(wdt:P279*)) wd:{instance_of_id.split('/')[-1]}.
                        {{
                            ?item wdt:P17 wd:Q30.
                        }} UNION {{
                            ?item wdt:P30 wd:Q49.
                        }}
                    }}
                    LIMIT 10
                }}
            }}"""
            results = get_results(endpoint_url, query_distractors)['results']['bindings']
            if len(results) <= 1:
                query_distractors = f"""SELECT DISTINCT ?item ?itemLabel WHERE {{
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                    {{
                        SELECT DISTINCT ?item WHERE {{
                            ?item p:P31 ?statement0.
                            ?statement0 (ps:P31/(wdt:P279*)) wd:{instance_of_id.split('/')[-1]}.
                        }}
                        LIMIT 10
                    }}
                }}"""
                results = get_results(endpoint_url, query_distractors)['results']['bindings']

            results = pd.json_normalize(results)
            entity_distractor_ids.extend(results["item.value"].tolist())
            entity_distractor_labels.extend(results["itemLabel.value"].tolist())

        distractor_ids.append(json.dumps(entity_distractor_ids))
        distractor_labels.append(json.dumps(entity_distractor_labels))

    shp["instanceOfIDs"] = instance_of_ids
    shp["instanceOfLabels"] = instance_of_labels
    shp["distractorIDs"] = distractor_ids
    shp["distractorLabels"] = distractor_labels
    shp.to_file(shp_file_path, index=False, driver="GPKG")


if __name__ == "__main__":
    main()
