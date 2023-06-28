# 음성 파일 로드
import torchaudio
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import librosa
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from . import JamoFusion

repo_name = "daeinbangeu/wav2vec2-large-xls-r-300m-korean-g-TW3"
processor = Wav2Vec2Processor.from_pretrained(repo_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(repo_name)

def transcribe_audio_file(file_path):
    # 음성 파일 로드
    data = wavfile.read(file_path)
    framerate = data[0]
    sounddata = data[1]

    # 음성 파일 전처리
    time = np.arange(0,len(sounddata))/framerate

    # 모델 불러오기
    model = Wav2Vec2ForCTC.from_pretrained(repo_name)

    # 데이터 시계열 처리
    input_audio, _ = librosa.load(file_path, sr=16000)
    # 음성을 텍스트로 변환
    input_values = torch.tensor(input_audio).unsqueeze(0)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    # 자음 모음 합치기
    resultsFinal = JamoFusion.join_jamos(transcription)

    return resultsFinal