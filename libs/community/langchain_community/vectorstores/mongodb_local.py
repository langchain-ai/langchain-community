#--------VERSION 1--------
#This version will only have the 'flat' and 'hnsw' index_types with all 3 metrics (euclidean,dot,cosine)
#This is just like what Atlas gives


#Docs for me to remember cause i forgot what ive done:
"""
FAISS Index types:
    Flat
    HNSW
by now ive made a class that holds this info:
    collection: Collection[MongoDBDocumentType],
    embedder_model: Embeddings,
    *,
    index_name:str='default',
    text_key:str='text',
    embedding_key:str='embedding',
    faiss_config: Optional[Dict[str, Any]] = None,
    faiss_index_path: Optional[str] = None,
    faiss_id_map_path: Optional[str] = None,
    self._faiss_index = None #faiss index
    self._faiss_to_mongo_id = {} #id mapping dict
    self._deleted_faiss_ids=set() #all deleted ids dict
    self._next_faiss_id:int=1 #helper variable for data insertion

so i have these functions:
    _initialize_faiss_index():
            Gets the dimensions from a sample embedding
            Then get the metric,index type and check if metric exists as a possibility in the faiss_creator and same for index type 
            Then i get the index params and for each param i check if its valid, if not it raises ValueError
            Once all params validated i assign the classes faiss index (_faiss_index) to the faiss creator with metric index type and param

    load_faiss(index_path,id_map_path):
            tries to read the faiss index from disk using faiss.read_index
            Opens the path to id mapping using pickle and assigns it to (_faiss_to_mongo_id) and logs it as successfully done
            Otherwise there is an exception

    save_faiss(index_path,id_map_path):
            tried to write the faiss index to index path using faiss.write_index
            Then opens the filepath with pickle and dumps the mongo to faiss ids and logs success
            otherwise an exception which gets logged

    from_connection_string: 
    add_texts: that batches text,metadata and uses helper func insert_texts to ins data once text is encoded into the embedding
    insert_texts: that inserts the text,metadata and embedding into the collection and faiss index
    _get_all_embeddings: that gets all the embeddings from the collection for training
    queryfilter: that gets the faiss ids and mongo ids for a given query filter
    delete: 
            All this essentially does is takes a query
            gets the mongo and faiss ids related to the query, converts the faiss ids into an np array
            Then if we can completely remove the id with 'remove_ids' we do that, else just add the id into the deleted faiss ids set
            Then for all faiss ids to delete, we jus del from the mapping and completely delete all data related to that query

            so if we dont have the remove with ids, then we have data in the deleted faiss ids set
            So that when im adding data to the db, it will take the deleted faiss id set, make new faiss ids and then for every data item to be inserted,
            it will insert the data along with 'faiss_id': faiss_ids[i]
            So that the faiss ids will be re usable.

            IN DELETE FUNCTION, THERE WILL BE NO MONGO TO ID MAPPING FOR THE DELETED FAISS IDS 
            UNTIL WE RE ADD IT WITH A NEW ID

    update: that updates the documents with the query filter with new text and/or metadata
    similarity_search_with_score: that performs similarity search with score for top k* something results since im not retraining the index for every chnge
    ,

"""
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import faiss
import os
import pickle
from pymongo import UpdateOne

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from pymongo.collection import Collection

MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 100

#FAISS index creation helper functions
def euclid_flat_faiss(dimensions:int):
    """Returns a FAISS Flat Euclidean index (exact search)."""
    return faiss.IndexFlatL2(dimensions)

def euclid_hnsw_faiss(
        dimensions:int,
        neighbours:int=32, 
        efSearch:int=64, 
        efConstruction:int=200
    ):
    """Returns a FAISS HNSW index with Euclidean distance."""
    index = faiss.IndexHNSWFlat(dimensions, neighbours, faiss.METRIC_L2)
    index.hnsw.efSearch = efSearch
    index.hnsw.efConstruction = efConstruction
    return index

# def euclid_ivf(dimensions, clusters=100):
#     """Returns an IVF index with Euclidean distance (untrained)."""
#     quantizer = faiss.IndexFlatL2(dimensions)
#     return faiss.IndexIVFFlat(quantizer, dimensions, clusters, faiss.METRIC_L2)

# def euclid_ivf_pq(dimensions, clusters=100, m=8, nbits=8):
#     """Returns an IVFPQ index with Euclidean distance (untrained)."""
#     quantizer = faiss.IndexFlatL2(dimensions)
#     return faiss.IndexIVFPQ(quantizer, dimensions, clusters, m, nbits, faiss.METRIC_L2)

# def euclid_ivf_pq_hnsw(dimensions, clusters=100, m=8, nbits=8, neighbours=32):
#     """Returns an IVFPQ index with HNSW quantizer (Euclidean distance)."""
#     quantizer = faiss.IndexHNSWFlat(dimensions, neighbours, faiss.METRIC_L2)
#     return faiss.IndexIVFPQ(quantizer, dimensions, clusters, m, nbits, faiss.METRIC_L2)

# def euclid_ann_faiss(dimensions, clusters=100, neighbours=32, m=8, nbits=8):
#     """
#     Returns a 'best-effort' ANN index using IVFPQ + HNSW (Euclidean).
#     This can be your default ANN choice.
#     """
#     quantizer = faiss.IndexHNSWFlat(dimensions, neighbours, faiss.METRIC_L2)
#     return faiss.IndexIVFPQ(quantizer, dimensions, clusters, m, nbits, faiss.METRIC_L2)

def dot_flat_faiss(dimensions):
    """Returns a FAISS Flat index with Dot Product (exact search)."""
    return faiss.IndexFlat(dimensions, faiss.METRIC_INNER_PRODUCT)

def dot_hnsw_faiss(
        dimensions:int, 
        neighbours:int=32, 
        efSearch:int=64, 
        efConstruction:int=200
    ):
    """Returns a FAISS HNSW index with Dot Product."""
    index = faiss.IndexHNSWFlat(dimensions, neighbours, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = efSearch
    index.hnsw.efConstruction = efConstruction
    return index

# def dot_ivf(dimensions, clusters=100):
#     """Returns an IVF index with Dot Product (untrained)."""
#     quantizer = faiss.IndexFlat(dimensions, faiss.METRIC_INNER_PRODUCT)
#     return faiss.IndexIVFFlat(quantizer, dimensions, clusters, faiss.METRIC_INNER_PRODUCT)

# def dot_ivf_pq(dimensions, clusters=100, m=8, nbits=8):
#     """Returns an IVFPQ index with Dot Product (untrained)."""
#     quantizer = faiss.IndexFlat(dimensions, faiss.METRIC_INNER_PRODUCT)
#     return faiss.IndexIVFPQ(quantizer, dimensions, clusters, m, nbits, faiss.METRIC_INNER_PRODUCT)

# def dot_ann_faiss(dimensions, clusters=100, neighbours=32, m=8, nbits=8):
#     """Returns an ANN index using IVFPQ + HNSW with Dot Product."""
#     quantizer = faiss.IndexHNSWFlat(dimensions, neighbours, faiss.METRIC_INNER_PRODUCT)
#     return faiss.IndexIVFPQ(quantizer, dimensions, clusters, m, nbits, faiss.METRIC_INNER_PRODUCT)

def cosine_flat_faiss(dimensions):
    """Returns a FAISS Flat index with Cosine similarity (via inner product)."""
    return faiss.IndexFlat(dimensions, faiss.METRIC_INNER_PRODUCT)

def cosine_hnsw_faiss(
        dimensions:int, 
        neighbours:int=32, 
        efSearch:int=64, 
        efConstruction:int=200
    ):
    """Returns a FAISS HNSW index with Cosine similarity."""
    index = faiss.IndexHNSWFlat(dimensions, neighbours, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = efSearch
    index.hnsw.efConstruction = efConstruction
    return index

# def cosine_ivf(dimensions, clusters=100):
#     """Returns an IVF index with Cosine similarity (untrained)."""
#     quantizer = faiss.IndexFlat(dimensions, faiss.METRIC_INNER_PRODUCT)
#     return faiss.IndexIVFFlat(quantizer, dimensions, clusters, faiss.METRIC_INNER_PRODUCT)

# def cosine_ivf_pq(dimensions, clusters=100, m=8, nbits=8):
#     """Returns an IVFPQ index with Cosine similarity (untrained)."""
#     quantizer = faiss.IndexFlat(dimensions, faiss.METRIC_INNER_PRODUCT)
#     return faiss.IndexIVFPQ(quantizer, dimensions, clusters, m, nbits, faiss.METRIC_INNER_PRODUCT)

# def cosine_ann_faiss(dimensions, clusters=100, neighbours=32, m=8, nbits=8):
#     """Returns an ANN index using IVFPQ + HNSW with Cosine similarity."""
#     quantizer = faiss.IndexHNSWFlat(dimensions, neighbours, faiss.METRIC_INNER_PRODUCT)
#     return faiss.IndexIVFPQ(quantizer, dimensions, clusters, m, nbits, faiss.METRIC_INNER_PRODUCT)

# FAISS index map
faiss_creator = {
    'euclidean': {
        'flat': euclid_flat_faiss,
        'hnsw': euclid_hnsw_faiss,
        # 'ivf': euclid_ivf,
        # 'ivf_pq': euclid_ivf_pq,
        # 'ivf_pq_hnsw': euclid_ivf_pq_hnsw,
        # 'ann': euclid_ann_faiss
    },
    'dot': {
        'flat': dot_flat_faiss,
        'hnsw': dot_hnsw_faiss,
        # 'ivf': dot_ivf,
        # 'ivf_pq': dot_ivf_pq,
        # 'ann': dot_ann_faiss
    },
    'cosine': {
        'flat': cosine_flat_faiss,
        'hnsw': cosine_hnsw_faiss,
        # 'ivf': cosine_ivf,
        # 'ivf_pq': cosine_ivf_pq,
        # 'ann': cosine_ann_faiss
    }
}

valid_params=['metric','index_type','index_params','efSearch','efConstruction','neighbours','dimensions']

def is_param_valid(param:str)->bool:
    if param in valid_params:
        return True
    return False

class MongoDBLocalVectorSearch(VectorStore):
    """`MongoDB Local Vector Search` vector store.

    To use, install:
    - the ``pymongo`` python package
    - the ``pickle`` python package
    - the ``os`` python package
    - the ``numpy`` python package
    - the ``faiss-cpu`` or ``faiss-gpu`` python package
    - A MongoDB connection

    Example:
        .. code-block:: python
            from langchain_community.vectorstores import MongoDBLocalVectorSearch
            from langhchain_community.embeddings.openai import OpenAIEmbeddings
            from pymongo import MongoClient

            mongo_client=MongoClient('<Mongo_URI_Connection_String>')
            collection=mongo_client['<database_name>']['<collection_name>']
            embedder=OpenAIEmbeddings()
            vectorstore=MongoDBLocalVectorStore(collection,embedder)
        ..

    This vector store will enable a developer to simulate Atlas vector store in their local testing server. """
    def __init__(
            self,
            collection: Collection[MongoDBDocumentType],
            embedder_model: Embeddings,
            *,
            index_name:str='default',
            text_key:str='text',
            embedding_key:str='embedding',
            faiss_config: Optional[Dict[str, Any]] = None,
            faiss_index_path: Optional[str] = None,
            faiss_id_map_path: Optional[str] = None,
            # ivf_train_delay:int=DEFAULT_INSERT_BATCH_SIZE,
    ):
        """Args:
            collection: The MongoDB collection that we add data to.
            embedder_model: The embedding model used.
            index_name: Name of the vector search index.
            text_key: MongoDB field that contains data for each document.
            embedding_key: MongoDB field that contains the embedded data for the documents.
            faiss_config: FAISS configuration('metric': The similarity score to use, 'index_type': The type of FAISS index to use)
            next_faiss_id
        """
        self._collection=collection
        self._embedder_model=embedder_model
        self._index_name=index_name
        self._text_key=text_key
        self._embedding_key=embedding_key
        self._faiss_config = faiss_config or {
            'metric': 'euclidean',
            'index_type': 'hnsw'
        }
        # self._ivf_train_delay=ivf_train_delay
        
        #Initializes the index to none and the faiss_to_mongo id mapping to an empty dict
        self._faiss_index = None #faiss index
        self._faiss_to_mongo_id = {} #id mapping dict
        self._deleted_faiss_ids=set() #all deleted ids dict
        self._next_faiss_id:int=1 #helper variable for data insertion
        # self._total_embeddings:int=0 #helper for ivf training

        #loads or inits a faiss index
        if faiss_index_path and faiss_id_map_path:
            self.load_faiss(faiss_index_path, faiss_id_map_path)
        else:
            self._initialize_faiss_index()

    
    def _initialize_faiss_index(self):
        """
        Initializes the FAISS index based on the current configuration(faiss_config)
        """
        sample_embedding=self._embedder_model.embed_query('sample_query')
        dimensions=len(sample_embedding)
        #Gets the metric,index_type from the config
        metric=self._faiss_config.get('metric','cosine')
        index_type=self._faiss_config.get('index_type','flat')

        if metric not in faiss_creator or index_type not in faiss_creator[metric]:
            raise ValueError(f"Unsupported Faiss metric:{metric} or index_type: {index_type}")
        
        index_params=self._faiss_config.get('index_params',{})
        for param in index_params:
            if not is_param_valid(param):
                raise ValueError(f"Invalid FAISS parameter: {param}. Valid parameters are: {valid_params}")
        self._faiss_index=faiss_creator[metric][index_type](dimensions,**index_params)
        # self._requires_training = 'ivf' in index_type

    def load_faiss(
            self,
            index_path:str,
            id_map_path:str
    ):
        """
        Loads the faiss index to the class variable and the id mapping from disk.
        """
        try:
            # self._faiss_index=faiss.read(index_path)
            self._faiss_index=faiss.read_index(index_path)
            with open(id_map_path,'rb') as f:
                self._faiss_to_mongo_id=pickle.load(f)
            logger.info(f"Loaded FAISS index from {index_path} and id_mapping from {id_map_path}")
        except Exception as e:
            logger.exception(f"Failed to load FAISS index error:{e}")
            raise
    
    def save_faiss(
            self,
            index_path:str,
            id_map_path:str
    ):
        """Saves FAISS index and id mapping to disk.
        """
        try: 
            # faiss.write(self._faiss_index,index_path)
            faiss.write_index(self._faiss_index,index_path)
            with open(id_map_path,'wb') as f:
                pickle.dump(self._faiss_to_mongo_id,f)
            logger.info(f"Saved FAISS index to {index_path} and id mapping to {id_map_path}")
        except Exception as e:
            logger.exception(f"Unable to save FAISS index at {index_path} and id mapping at {id_map_path} with error: {e}")
            raise
     
    @property
    def embedder(self)->Embeddings:
        return self._embedder_model
    
    @classmethod
    def from_connection_string(
        cls,
        connection_string:str,
        namespace:str,
        embedder:Embeddings,
        **kwargs: Any
    ):
        """
        Constructs a `MongoDBLocalVectorSearch` vector store using a Mongo URI, `db.collection` namespace and an embedder.
        
        Args:
            connection_string: A valid MongoDB connection URI.
            namespace: A valid MongoDB namespace (database and collection).
            embedding: The text embedding model to use for the vector store.

        Returns:
            A new MongoDVLocalVectorSearch instance.


        """
        try:
            from importlib.metadata import version
            from pymongo import MongoClient
            from pymongo.driver_info import DriverInfo
        except ImportError:
            raise ImportError(
                "Could not import pymongo, install it with:" 
                "`pip install pymongo`."
            )
        client:MongoClient=MongoClient(
            connection_string,
            driver=DriverInfo(name="langchain",version=version('langchain')),
        )
        db_name,collection_name=namespace.split('.')
        collection=client[db_name][collection_name]
        return cls(collection,embedder,**kwargs)


    #im thinking that instead of updating on every batch of data inserted,
    #ill just update(train) the faiss ivf index at the end of all inserts
    def add_texts(
            self,
            texts:Iterable[str],
            metadatas:Optional[List[Dict[str,Any]]]=None,
            **kwargs:Any
    )->List:
        """ This helps batch insertion of text
            Batching Text for a specified or default size to insert into the collection.
            Batched list of text is embedded and inserted into the MongoDB collection

            Args:
                texts: Iterable of strings to add to the vectorstore.
                metadatas: Optional list of metadatas associated with the texts.
                batch_size: Optional integer for size of data to be inserted at a time (Default=100).

            Returns:
                List of ids from adding the texts into the vectorstore.
        """        
        batch_size=kwargs.get('batch_size',DEFAULT_INSERT_BATCH_SIZE)
        _metadatas:Union[List|Generator]=metadatas or [{} for _ in texts]
        text_batch=[]
        metadata_batch=[]
        result_ids=[]
        for i,(text,metadata) in enumerate(zip(texts,_metadatas)):
            # self._total_embeddings+=1
            text_batch.append(text)
            metadata_batch.append(metadata)
            if (i+1) % batch_size==0:
                # self._requires_training= 'ivf' in self._faiss_config.get('index_type','flat')
                result_ids.extend(self.insert_texts(text_batch,metadata_batch))
                text_batch=[]
                metadata_batch=[]
        #for any remaining text in text_batch
        if text_batch:
            result_ids.extend(self.insert_texts(text_batch,metadata_batch))
        #function that trains the faiss index if 'ivf' in the index_type 
        #MUST FIX: if possible train the ivf in batches for memory efficiency, also try to train it in the background, 
        # since this holds up the processes
        # self._requires_training= 'ivf' in self._faiss_config.get('index_type','flat')
        # if self._requires_training:
        #     self.retrain_index()
        return result_ids


    def insert_texts(
            self,
            texts:List[str],
            metadatas:List[Dict[str,Any]]
    )->List:
        if not texts:
            return []
        embeddings=self._embedder_model.embed_documents(texts)
        embeddings_np=np.array(embeddings).astype('float32')

        if self._faiss_config.get('metric')=='cosine':
            faiss.normalize_L2(embeddings_np)
        #------------------------------------------------------------BUILDING FAISS IDS WITH RE USING DELETED IDS----------------------------------------------
        faiss_ids=[]
        if hasattr(self._faiss_index,'add_with_ids'):
            while self._deleted_faiss_ids and len(faiss_ids)<len(texts):
                reused_id=self._deleted_faiss_ids.pop()
                faiss_ids.append(reused_id)

        remaining_length=len(texts)-len(faiss_ids)
        
        if remaining_length>0:
            if self._faiss_to_mongo_id:
                start_index=max(self._next_faiss_id,max(self._faiss_to_mongo_id.keys())+1)
            else:
                start_index=self._next_faiss_id
            new_ids=np.arange(start_index,start_index+remaining_length).astype('int64')
            faiss_ids.extend(new_ids)
            self._next_faiss_id=start_index+remaining_length
        faiss_ids=np.array(faiss_ids).astype('int64')

        #---------------------------------------------------------------INSERTING DATA--------------------------------------------------------------------------
        
        #inserting all old and new ids int the db with the new text,metadata etc.
        #the old ids are now re used here, but
        inserting=[]
        for i,(text,metadata,embedding) in enumerate(zip(texts,metadatas,embeddings)):
            doc = {
                self._text_key: text,
                self._embedding_key: embedding,
                **metadata,
                'faiss_id': int(faiss_ids[i])
            }
            inserting.append(doc)
        #inserting the data
        insertion_result=self._collection.insert_many(inserting)

        if embeddings_np.shape[0] != len(faiss_ids):
            raise ValueError("Embedding count and FAISS ID count mismatch.")
        
        if hasattr(self._faiss_index,'add_with_ids'):
            self._faiss_index.add_with_ids(embeddings_np,faiss_ids)
        else:
            self._faiss_index.add(embeddings_np)
        #we always save the ids to the mapping dict after insertion
        result_ids=[]
        for i,inserted_id in enumerate(insertion_result.inserted_ids):
            faiss_id=int(faiss_ids[i])
            self._faiss_to_mongo_id[faiss_id]=str(inserted_id)
            result_ids.append(inserted_id)
        
        return result_ids
    
    def _get_all_embeddings(self)->np.ndarray:
        """
        Gets all the embeddings from MongoDB for training.
        """
        docs= self._collection.find({},{self._embedding_key:1})
        embeddings=np.array( [doc[self._embedding_key] for doc in docs] ).astype('float32')
        return embeddings if embeddings else np.empty( (0,0) )
    
    def queryfilter(
            self,
            query:Dict[str,Any]
    )->Tuple[List[int],List[str]]:
        """
        Helper function to get the faiss ids and document ids for any query on the db.
        """
        #returns the faiss ids and mongodb _ids for the query
        docs=list(self._collection.find(query,{'_id':1,'faiss_id':1}))
        
        faiss_ids= [int(doc['faiss_id']) for doc in docs]
        mongo_ids=[str(doc['_id']) for doc in docs]
        if len(faiss_ids)!=len(mongo_ids):
            raise ValueError("Mismatch between length of faiss ids and mongo ids found by queryfilter().")
        return faiss_ids,mongo_ids

    def delete(
            self,
            query:Dict[str,Any],
            # ids:Optional[List[str]] = None,
            **kwargs:Any 
    )->List:
        """Deletes the documents with the query filter from the vectorstore and faiss index.

            Args:
                query: A MongoDB query filter to identify documents to delete.

            Returns:
                List of ids that were deleted from the vectorstore.
        """
        faiss_ids_to_delete,mongo_ids=self.queryfilter(query)
        if not faiss_ids_to_delete:
            return []
       
        faiss_ids_np=np.array(faiss_ids_to_delete).astype('int64')
        if hasattr(self._faiss_index,'remove_ids'):
            #removes it directly from the faiss index
            self._faiss_index.remove_ids(faiss_ids_np)
        else:
            #adds the ids to the deleted ids set for re use during insertion 
            self._deleted_faiss_ids.update(faiss_ids_to_delete)
            logger.warning("FAISS index does not support removing ids. Deleted ids will be ignored or replaced during search/inserts.It is recommended to re-train the index to maintain a good search experience or add more vectors to replace the faiss inde")
        #removes the ids from the mapping dict but its ok 
        #since i will be replacing the id with new data whenever new data is added
        for fid in faiss_ids_to_delete:
            del self._faiss_to_mongo_id[fid]
        deletion_result=self._collection.delete_many({'faiss_id':{'$in':faiss_ids_to_delete}})
        # return [str(id) for id in ids]
        return mongo_ids
    
    def update(
            self,
            query:Dict[str,Any],
            texts:Optional[List[str]],
            metadatas:Optional[List[Dict[str,Any]]]=None,
            **kwargs:Any
    )->List:
        """Updates documents that match the query filter with new text and/or metadata.

            Args:
                query: A MongoDB query filter to identify documents to update.
                texts: Optional list of new texts to update the documents.
                metadatas: Optional list of new metadatas to update the documents.
            
            Returns:
                List of ids that were updated in the vectorstore.
        """

        #2 ways to update the faiss index and database
        """if faiss has the attribute to add with ids, then i can remove all ids that are to be removed, then add them back with the new embeddings
           so basically calling delete(query) then add_text(texts,metadatas)
           
           Otherwise i have to perform a delete for those ids, which means adding them into the deleted set and then remaking/adding data with new ids into the db 
           basically,

           update data(query):
            since no attr
            i have to delete the ids related to the query, can only add ids to delted set and skip over them if found in sim search
            then have to re add the data into the faiss index
        """
        _metadatas:Union[List|Generator]=metadatas or [{} for _ in texts]
        batch_size=kwargs.get('batch_size',DEFAULT_INSERT_BATCH_SIZE)

        faiss_ids_to_update,mongo_ids=self.queryfilter(query)
        if not faiss_ids_to_update:
            return []
        
        faiss_ids_to_update_np=np.array(faiss_ids_to_update).astype('int64')
        
        # updated_ids=[] #to return
        text_embeddings=self._embedder_model.embed_documents(texts) # all text embeddings
        if self._faiss_config.get('metric') == 'cosine':
            faiss.normalize_L2(text_embeddings)

        text_embeddings=np.array(text_embeddings).astype('float32')
        faiss_ids=[]
        if hasattr(self._faiss_index,'add_with_ids'):
            self._faiss_index.remove_ids(faiss_ids_to_update_np)
            self._faiss_index.add_with_ids(text_embeddings,faiss_ids_to_update_np)
            #dont have to change faiss_to_mongoid mapping since none of the faiss ids or mongoids are changing, only the data
            #and the vector in faiss is the only thing that changes essentially, which doesnt need remap
            #so i make no appends to faiss_ids arr
            faiss_ids=faiss_ids_to_update
        else:
            #Basically just delete and add again
            #add all faiss ids to be deleted into the deleted set
            self._deleted_faiss_ids.update(faiss_ids_to_update) #essentially marked as removed, so in sim search we skip these
            #must delete mapping right now
            #if i delete the mapping, i cannot re use it ofc so i need to make new mappings with new faiss ids but same mongo ids
            for fid in faiss_ids_to_update:
                del self._faiss_to_mongo_id[fid]
            
            remaining_length=len(faiss_ids_to_update)
            if self._faiss_to_mongo_id:
                start_index=max(self._next_faiss_id,max(self._faiss_to_mongo_id.keys())+1)
            else:
                start_index=self._next_faiss_id
            new_ids=np.arange(start_index,start_index+remaining_length).astype('int64')
            faiss_ids.extend(new_ids)
            self._next_faiss_id=start_index+remaining_length

            #add the embeddings into faiss
            self._faiss_index.add(text_embeddings)

            logger.warning("FAISS index does not support removing ids. Deleted ids will be ignored or replaced during search/inserts." \
            "It is recommended to re-train the index to maintain a good search experience.")
    
        for fid,mid in zip(faiss_ids,mongo_ids):
            self._faiss_to_mongo_id[fid]=mid
        
        updates=[]
        for mid, fid, text, embedding, metadata in zip(mongo_ids,faiss_ids,texts,text_embeddings,_metadatas):
            updates.append(
                UpdateOne(
                    {"_id": mid},
                    {"$set": {
                        self._text_key: text,
                        self._embedding_key: embedding,
                        **metadata,
                        "faiss_id": int(fid)
                    }}
                )
            )
        if updates:
            self._collection.bulk_write(updates, ordered=False)
        return mongo_ids
    
    def _similarity_search_with_score(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        k: int = 4,
        post_filter_query: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform FAISS similarity search and return top-k (Document, score) tuples."""
        #ensure no error
        if self._faiss_index is None or not self._faiss_to_mongo_id:
            logger.warning("No FAISS index or mappings found; cannot perform search.")
            return []

        #query embedding
        query_embedding = np.array(
            query_embedding or self._embedder_model.embed_query(query),
            dtype='float32'
        ).reshape(1, -1)

        if self._faiss_config.get('metric') == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        overfetch_factor = self._faiss_config.get('overfetch_factor', 5)
        distances, indices = self._faiss_index.search(query_embedding, k * overfetch_factor)

        allowed_mongo_ids = None
        if post_filter_query:
            _, mids = self.queryfilter(post_filter_query)
            allowed_mongo_ids = set(mids)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx not in self._faiss_to_mongo_id or idx in self._deleted_faiss_ids:
                continue
            mongo_id = self._faiss_to_mongo_id[idx]
            if allowed_mongo_ids and mongo_id not in allowed_mongo_ids:
                continue
            candidates.append((idx, mongo_id, dist))
            # if len(candidates) >= k * 2:  # safety cap
            #     break
        if not candidates:
            return []

        #batch fetching
        mongo_ids = [mid for _, mid, _ in candidates]
        docs = list(self._collection.find(
            {"_id": {"$in": mongo_ids}},
            {self._embedding_key: 0}
        ))
        doc_map = {doc["_id"]: doc for doc in docs}

        results = []
        for fid, mid, dist in candidates:
            doc = doc_map.get(mid)
            if not doc:
                continue
            text = doc.get(self._text_key, "")
            metadata = {
                k: v for k, v in doc.items()
                if k not in [self._text_key, self._embedding_key, "_id", "faiss_id"]
            }
            results.append((Document(page_content=text, metadata=metadata), float(dist)))
            if len(results) >= k:
                break
        return results

    
    def similarity_search_with_score(
            self,
            query:str,
            query_embedding:List['float']=None,
            k:int=5,
            post_filter_query: Optional[List[Dict[str,Any]]]=None,
        )->List[Tuple[Document,float]]:

            result=self._similarity_search_with_score(query,query_embedding,k,post_filter_query)
            return result            
            

    # def retrain_index(self):
    #     """
    #     Trains the faiss index on all the data stored in MongoDB.
    #     """
    #     if 'ivf' not in self._faiss_config.get('index_type'):
    #         logger.warning(f"FAISS index not a trainable type")
    #         raise
    #     embeddings=self._get_all_embeddings()
    #     self._faiss_index.train(embeddings)

    #### SIMILARITY SEARCH WITH SCORE FOR TOP K: RETURNS SCORE AND DATA
    #### SIMILARITY SEARCH FOR TOP K: RETURNS DATA