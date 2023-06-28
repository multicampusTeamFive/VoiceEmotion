import pyaudio
import wave

def MicRecordWav():
    # 녹음 설정
    CHUNK = 1024  # 버퍼 크기
    FORMAT = pyaudio.paInt16  # 샘플 포맷
    CHANNELS = 1  # 채널 개수 (단일 모노)
    RATE = 16000  # 샘플 레이트 (Hz)
    RECORD_SECONDS = 5  # 녹음 시간 (초)
    OUTPUT_FILENAME = 'voice/RecordAudio.wav'  # 출력 파일 이름

    # PyAudio 객체 생성
    audio = pyaudio.PyAudio()

    # 입력 스트림 열기
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("녹음 시작...")

    # 버퍼링할 데이터 저장할 리스트
    frames = []

    # 녹음 데이터 수집
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("녹음 종료.")

    # 입력 스트림 닫기
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # WAV 파일로 저장
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"{OUTPUT_FILENAME} 파일로 저장되었습니다.")

