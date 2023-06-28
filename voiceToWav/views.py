from django.shortcuts import render
from pathlib import Path
from django.http import HttpResponse

# from django.http import HttpResponse
# from voiceToWav.models import MusicUrl
# Create your views here.

transcription = ''
def index(request):
    # musicurl = MusicUrl.objects.all()
    # transcription = voiceToWavFun(request).mainProcess()
    print("=============================================")
    # print(musicurl)
    print("=============================================")
    context = {
        'test': "test",
        'trascription' : transcription,
    }

    return render(request, 'index.html', context)

def voiceToWavFun(request):
    from . import Wav2Vec2Korean
    def mainProcess():
        global transcription
        BASE_DIR = Path(__file__).resolve().parent.parent   # 경로를 실행하는 파일 위치 기준으로 확인 필요
        file_path = BASE_DIR/"Voice/RecordAudio.wav"        # 여기서 수정하면 될듯

        transcription = Wav2Vec2Korean.transcribe_audio_file(file_path)

        print(transcription)

        return transcription

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

    import pyaudio
    import wave

    def MicRecordWav():
        # 녹음 설정
        CHUNK = 1024  # 버퍼 크기
        FORMAT = pyaudio.paInt16  # 샘플 포맷
        CHANNELS = 1  # 채널 개수 (단일 모노)
        RATE = 16000  # 샘플 레이트 (Hz)
        RECORD_SECONDS = 5  # 녹음 시간 (초)
        OUTPUT_FILENAME = 'Voice/RecordAudio.wav'  # 출력 파일 이름

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
        
        return HttpResponse(f"{OUTPUT_FILENAME} 파일로 저장되었습니다.")



    ### 한글 자모음 분리 ###
    __all__ = ["split_syllable_char", "split_syllables",
               "join_jamos", "join_jamos_char",
               "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"]

    import itertools

    # 한글 ASCII 코드
    INITIAL = 0x001
    MEDIAL = 0x010
    FINAL = 0x100
    CHAR_LISTS = {
        INITIAL: list(map(chr, [
            0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
            0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
            0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
            0x314e
        ])),
        MEDIAL: list(map(chr, [
            0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
            0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
            0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
            0x3161, 0x3162, 0x3163
        ])),
        FINAL: list(map(chr, [
            0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
            0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
            0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
            0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
            0x314c, 0x314d, 0x314e
        ]))
    }
    CHAR_INITIALS = CHAR_LISTS[INITIAL]
    CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
    CHAR_FINALS = CHAR_LISTS[FINAL]
    CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
    CHARSET = set(itertools.chain(*CHAR_SETS.values()))
    CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                    for k, v in CHAR_LISTS.items()}


    def is_hangul_syllable(c):
        return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


    def is_hangul_jamo(c):
        return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


    def is_hangul_compat_jamo(c):
        return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


    def is_hangul_jamo_exta(c):
        return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


    def is_hangul_jamo_extb(c):
        return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


    def is_hangul(c):
        return (is_hangul_syllable(c) or
                is_hangul_jamo(c) or
                is_hangul_compat_jamo(c) or
                is_hangul_jamo_exta(c) or
                is_hangul_jamo_extb(c))


    def is_supported_hangul(c):
        return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


    def check_hangul(c, jamo_only=False):
        if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
            raise ValueError(f"'{c}' is not a supported hangul character. "
                             f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                             f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                             f"supported at the moment.")


    def get_jamo_type(c):
        check_hangul(c)
        assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
        return sum(t for t, s in CHAR_SETS.items() if c in s)


    def split_syllable_char(c):

        check_hangul(c)
        if len(c) != 1:
            raise ValueError("Input string must have exactly one character.")

        init, med, final = None, None, None
        if is_hangul_syllable(c):
            offset = ord(c) - 0xac00
            x = (offset - offset % 28) // 28
            init, med, final = x // 21, x % 21, offset % 28
            if not final:
                final = None
            else:
                final -= 1
        else:
            pos = get_jamo_type(c)
            if pos & INITIAL == INITIAL:
                pos = INITIAL
            elif pos & MEDIAL == MEDIAL:
                pos = MEDIAL
            elif pos & FINAL == FINAL:
                pos = FINAL
            idx = CHAR_INDICES[pos][c]
            if pos == INITIAL:
                init = idx
            elif pos == MEDIAL:
                med = idx
            else:
                final = idx
        return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
                     for pos, idx in
                     zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


    def split_syllables(s, ignore_err=True, pad=None):

        def try_split(c):
            try:
                return split_syllable_char(c)
            except ValueError:
                if ignore_err:
                    return (c,)
                raise ValueError(f"encountered an unsupported character: "
                                 f"{c} (0x{ord(c):x})")

        s = map(try_split, s)
        if pad is not None:
            tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
        else:
            tuples = map(lambda x: filter(None, x), s)
        return "".join(itertools.chain(*tuples))


    def join_jamos_char(init, med, final=None):
        """
        Combines jamos into a single syllable.

        Arguments:
            init (str): Initial jao.
            med (str): Medial jamo.
            final (str): Final jamo. If not supplied, the final syllable is made
                without the final. (default: None)

        Returns:
            A Korean syllable.
        """
        chars = (init, med, final)
        for c in filter(None, chars):
            check_hangul(c, jamo_only=True)

        idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                    for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
        init_idx, med_idx, final_idx = idx
        # final index must be shifted once as
        # final index with 0 points to syllables without final
        final_idx = 0 if final_idx is None else final_idx + 1
        return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)


    def join_jamos(s, ignore_err=True):

        last_t = 0
        queue = []
        new_string = ""

        def flush(n=0):
            new_queue = []
            while len(queue) > n:
                new_queue.append(queue.pop())
            if len(new_queue) == 1:
                if not ignore_err:
                    raise ValueError(f"invalid jamo character: {new_queue[0]}")
                result = new_queue[0]
            elif len(new_queue) >= 2:
                try:
                    result = join_jamos_char(*new_queue)
                except (ValueError, KeyError):
                    # Invalid jamo combination
                    if not ignore_err:
                        raise ValueError(f"invalid jamo characters: {new_queue}")
                    result = "".join(new_queue)
            else:
                result = None
            return result

        for c in s:
            if c not in CHARSET:
                if queue:
                    new_c = flush() + c
                else:
                    new_c = c
                last_t = 0
            else:
                t = get_jamo_type(c)
                new_c = None
                if t & FINAL == FINAL:
                    if not (last_t == MEDIAL):
                        new_c = flush()
                elif t == INITIAL:
                    new_c = flush()
                elif t == MEDIAL:
                    if last_t & INITIAL == INITIAL:
                        new_c = flush(1)
                    else:
                        new_c = flush()
                last_t = t
                queue.insert(0, c)
            if new_c:
                new_string += new_c
        if queue:
            new_string += flush()
        return new_string
    ### ###

    return transcription
