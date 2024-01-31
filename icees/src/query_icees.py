from openai import OpenAI
import argparse
import os
import pandas as pd
from dotenv import load_dotenv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_question', type=str,
                        default='What genes or proteins interact with gene KRAS?',
                        help='input question to llm model')

    input_query_template = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "ids": [],
                        "categories": ["biolink:category"]
                    },
                    "n1": {
                        "categories": ["biolink:category"]
                    }
                },
                "edges": {
                    "e1": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:predicates"]
                    }
                }
            }
        }
    }

    args = parser.parse_args()
    input_question = args.input_question

    meta_kg = pd.read_csv('../metadata/metaKG.csv')
    predicate_list = list(set(meta_kg['KG_category']))
    kg_category = list(set(list(meta_kg['Subject'].unique()) + list(meta_kg['Object'].unique())))
    all_predicates = ','.join(predicate_list)
    print(f'all_predicates: {all_predicates}')
    all_categories = ','.join(kg_category)
    print(f'all_categories: {all_categories}')
    query_json = str(input_query_template)
    input_text = f'We know the available predicates in the KG are: "{all_predicates}". We know the ' \
                 f'available categories in the KGs are "{all_categories}". We know a TRAPI message template ' \
                 f'is "{query_json}". With the question of "{input_question}" What is the json format of message ' \
                 f'to represent this question? Please follow rules below for the output: ' \
                 f'1) The result must be in a json format with the same structure as the template; 2) categories ' \
                 f'should be replaced from the categories in the KG; 3) predicates should be replaced from the ' \
                 f'predicates in the KG; 4) the name can be used to fill the ids; ' \
                 f'At least one ids should be given and no annotations are needed!"'

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        n=1,
        response_format={"type": "json_object"},
        temperature=1.2,
        messages=[
            {"role": "user",
             "content": input_text
             }
            ]
        )
    print(response.choices[0].message.content)
