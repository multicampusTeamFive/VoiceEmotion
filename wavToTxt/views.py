from django.shortcuts import render
from datasets import load_dataset

import torch
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
# Create your views here.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델과 토크나이저 불러오기
model = BertForSequenceClassification.from_pretrained("여기에 모델이 저장된 '폴더' 경로를 입력")
#모델 : 분할된 문장에 벡터값을 부여하고 벡터값에 학습된 감정 점수 부여
tokenizer = BertTokenizer.from_pretrained("여기에 모델이 저장된 '폴더' 경로를 입력")
#토크나이저 : 문장을 음소 단위로 분할
model.to(device)

# 음성인식 모델 import
from voiceToWav.migrations import voiceToWavFun
def emotion_model(request): #input data : 음성인식 모델에서 들어온 한글 문장(string)
    txt_data=[]
    input_string = voiceToWavFun()
    txt_data.append(input_string)

    def analyze_predictions(predictions):
        emotion_labels = ['pleasant', 'actived']
        scores = predictions.tolist()
        results = []

        for score in scores:
            result = {}
            for i, emotion_label in enumerate(emotion_labels):
                result[emotion_label] = score[i]
            results.append(result)
        return results

    input_ids = []
    attention_masks = []
    for sentence in txt_data:
        #문장 토큰화
        encoded = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

        # 토큰 ID와 어텐션 마스크를 텐서로 변환
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        predictions = outputs.logits

    scores = analyze_predictions(predictions)

    result_scores = []
    for sentence, score in zip(txt_data, scores):
        result_scores.append([score['pleasant'], score['actived']])
    rounded_result = [[round(num) % 10 for num in sublist] for sublist in result_scores] #결과값 반올림
    value = rounded_result[0][0] + 1
    mapped_value = 1 if value <= 2 else 2 if value <= 4 else 3 if value <= 6 else 4 if value <= 8 else 5 #간단한 점수로 치환

    result_final = [] #정수 2개가 들어간 리스트로 반환
    result_final.append(mapped_value)
    result_final.append(rounded_result[0][1])

    return result_final[0] #여건상 pleasant 점수만 사용, 1~5

return render(request, '여기에 html 입력')