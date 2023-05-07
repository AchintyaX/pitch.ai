import numpy as np
from pymilvus import connections
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from pymilvus import utility
from time import time
from loguru import logger 
import openai 
from openai.embeddings_utils import  get_embedding
import os 
from typing import List
from ast import literal_eval
import copy 

class PitchCollection: 
    def __init__(self) -> None :
        self.connections = connections.connect("default", host="localhost", port="19530")
        self.current_collections = {
            'bbc': Collection('bbc_news')
        }
        os.environ["OPENAI_API_KEY"] = ""
        openai.api_key = os.getenv("OPENAI_API_KEY")

        for key in self.current_collections.keys():
            self.current_collections[key].load()
            logger.info(f"Loaded {key} collection for search")

        if str(utility.list_collections() ) != '[]':
            logger.info("Successfully Connected to Milvus DB")

        self.search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }

    def openai_embedder(self, text: str): 
        return np.array(get_embedding(text, engine="text-embedding-ada-002"))
    
    def search_index(self, query_list: List[str], collection_name: str = 'bbc') :
        '''
        slow code - for loading collection in memory when search is called
        ---------
        topic_collection = Collection(collection_name)
        topic_collection.load()
        '''
        vectors_to_search = [self.openai_embedder(i) for i in query_list]
        results = self.current_collections[collection_name].search(vectors_to_search, 'summary_embedding', self.search_params, limit=10, output_fields=['title', 'content', 'summary', 'keywords'])
        #topic_collection.release()
        """
        The results is a 2D list with number of queries as len and 
        contains 10 matches per query.
        The matches are sorted in descencding order
        """
        result_list = []
        for out in results: 
            hit_list = []
            for hit in out:
                hit_dict = {}
                hit_dict['filename'] = hit.id 
                hit_dict['title'] = hit.entity.get('title')
                hit_dict['content'] = hit.entity.get('content')
                hit_dict['summary'] = hit.entity.get('summary')
                hit_dict['keywords'] = literal_eval(hit.entity.get('keywords'))
                hit_dict['score'] = hit.distance

                hit_list.append(hit_dict)
            result_list.append(hit_list)

        return result_list[0]
    

    def keyword_relevance(self, user_keywords, search_results): 
        """
        Args: 
            user_keywords: string of keywords seperated by commas 
            search_results: list of results from vector search 
        """
        search_output = copy.deepcopy(search_results)
        keywords = user_keywords.split(',')
        keywords = [i.lstrip() for i in keywords]
        keywords = [i.rstrip() for i in keywords]
        keywords = set(keywords)

        for result in search_output:
            result_keywords = set(result['keywords'])
            intersection_score = len(keywords.intersection(result_keywords))/len(keywords)
            result['intersection_score'] = intersection_score
        
        return search_output
    

    def combine_score(self, search_results):

        search_output = copy.deepcopy(search_results)
        keyword_weights = 0.4
        search_weights = 0.6 

        for result in search_output:
            final_score = (keyword_weights * result['intersection_score']) +( search_weights * result['score'])
            result['final_score'] = final_score
        
        return search_output 
    
    def __call__(self, query_list: List[str], user_keywords: str ,collection_name: str = 'bbc'):

        search_results = self.search_index(query_list, collection_name)
        search_results = self.keyword_relevance(user_keywords, search_results)
        search_results = self.combine_score(search_results)
        search_results = sorted(search_results, key=lambda d: d['final_score'], reverse=True) 


        return search_results
        



