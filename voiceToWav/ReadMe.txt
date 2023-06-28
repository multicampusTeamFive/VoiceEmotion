Python 3.9 가상환경으로 작업했습니다

아래 pip 인스톨 필요합니다.

***********************************************************************************************************
pip install datasets==1.18.3
pip install torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers scipy ftfy accelerate
pip install librosa
pip install pyaudio
pip install wave
***********************************************************************************************************

파일 설명

1. JamoFusion.py
Wav2Vec2Korean.py 에서 쓰이는 함수입니다. 너무길어서 따로 만들었습니다.

2. main.py
이건 별거 아닙니다. 가상환경 처음 생성될떄 만들어지는 테스트 코드입니다.

3. MainProcess.py
함수들로 메인 과정이 발생하는 코드입니다. 두번째 인공지능 모델로 넘어가는 outputTxt.txt를 생성합니다

4. MicAudioRecord.py
연결된 마이크로 사용자 음성을 녹음하는 부분입니다. 이 프로그램이 버튼을 통해 사용자 음성을 입력받는 부분이라고 보시면 됩니다.
이 코드로 RecordAudio.wav 오디오 파일이 생성됩니다.

5. Wav2Vec2Korean.py
transcribe_audio_file() 함수가 있는 코드입니다. 음성을 텍스트로 변환해줍니다.
해당 코드로 인해서 인공지능 모델을 불러와 설치합니다. 최초 1회만 작동합니다.

6. RecordAudio.wav
최초 입력인 사용자 음성 파일입니다. MicAudioRecord.py에서 생성됩니다.

7. outputTxt.txt
사용자 음성 오디오를 텍스트로 변환한 결과물입니다.

************************************************************************************************************

실행시 안될 수도 있습니다. 문의 주시기 바랍니다.