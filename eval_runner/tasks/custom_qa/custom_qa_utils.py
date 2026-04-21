def process_results(doc, results):
    prediction = results[0].strip().lower()
    answer = doc["answer"].strip().lower()
    match = int(answer in prediction or prediction in answer)
    return {"contains_match": match}
