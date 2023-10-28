def model_hits(hits):
    predict = []
    predict_quality = []
    for element in hits:
        for i in element:
            predict = predict.append(i['corpus_id'])
            predict_quality = predict_quality.append(i['score'])
    dictionary = {'predict': predict, 'predict_quality': predict_quality}
    return dictionary


