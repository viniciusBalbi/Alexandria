import ffmpeg
import numpy as np


def carregar_audio(file, sr):
    try:
         # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
         # Isso inicia um subprocesso para decodificar o áudio durante a mixagem e reamostragem conforme necessário.
         # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.

        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        ) # Para evitar que iniciantes copiem caminhos com espaços à esquerda ou à direita, aspas e quebras de linha.
        out,  _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )

    except Exception as e:
        raise RuntimeError(f"Falha ao carregar áudio: {e}")
    
    return np.frombuffer(out, np.float32).flatten()