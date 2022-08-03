import math

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


class SentimentClassifier():
    sentiment_analysis = pipeline("sentiment-analysis")
    #review_stars_predictor = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    def __init__(self):
        pass

    def predict(self, X):
        pred_1 = self.sentiment_analysis(X)
        #pred_2 = self.review_stars_predictor(X)

        def decode_1(text: str) -> int:
            if text == "NEGATIVE":
                return 0
            elif text == "POSITIVE":
                return 1
            else:
                print(f"Wrong label: {text}")
                raise Exception("Wrong label")

        def decode_2(text: str) -> int:
            """

            :param text: expected '1 stars', '2 stars', ... '5 stars'
            :return:
            """
            return math.log(int(text[0]))
        # Using both ? TODO
        #data = [[decode_1(pred_1[i]['label']), decode_2(pred_2[i]['label'])] for i, pred in enumerate(pred_1)]

        data = [[decode_1(pred_1[i]['label']) for i, pred in enumerate(pred_1)]]

        return data
