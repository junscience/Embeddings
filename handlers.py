from fastapi import APIRouter
from sentence_transformers.util import semantic_search
from fastapi.responses import JSONResponse
import semantic_search_model
from forms import RequestParams, ResponsePredict
from loguru import logger
import pickle
import torch
from DB_connect import dataset_embeddings, DB
from model_params import query
import pandas as pd

logger.add('debug.log', format="{time} {level} {message}", level="DEBUG", rotation="1 weeks", compression='zip')

router = APIRouter()


with open('model.pkl', 'wb') as dump_out:
    pickle.dump(query, dump_out)
router.model_embeddings = pickle.load(open("model.pkl", "rb"))
router.model_sentiment = semantic_search_model.model_hits
@router.get('/status')
def index():
    return {'status': 'OK'}


@router.post('/predict/Embeddings', response_model=ResponsePredict, name='prediction')
def predict_model(descriptions: RequestParams):
    logger.info('Run /predict/Embeddings')
    try:
        test_data = descriptions
        modeling = router.model_embeddings(test_data)
        query_embeddings = torch.FloatTensor(modeling)
        hits = semantic_search(query_embeddings, dataset_embeddings, top_k=1)
        predict = DB.iloc[router.model_sentiment(hits).predict]
        predict_quality = router.model_sentiment(hits).predict_quality
        return ResponsePredict(descriptions = descriptions, predict = predict, predict_quality = predict_quality)
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(status_code=500, content={'message': str(e)})