import Wav2Vec2Korean

file_path = "C:/Users/Mephisto851/Downloads/mulitcampus/deeplearning/pythonProject/RecordAudio.wav"
transcription = Wav2Vec2Korean.transcribe_audio_file(file_path)

print(transcription)

output_file_path = "C:/Users/Mephisto851/Downloads/mulitcampus/deeplearning/pythonProject/outputTxt.txt"
with open(output_file_path, 'w') as f:
    f.write(transcription)