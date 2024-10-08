from utils import remover_arquivos
import os, shutil, subprocess, time, shlex, sys # noqa
from loggingsetup import logger
import json

ERROR_INCORRECT_CODEC_PARAMETERS = [
    "prores",  # mov
    "ffv1",  # mkv
    "msmpeg4v3",  # avi
    "wmv2",  # wmv
    "theora",  # ogv
]  # fix final merge

TESTED_CODECS = [
    "h264",  # mp4
    "h265",  # mp4
    "hevc",
    "vp9",  # webm
    "mpeg4",  # mp4
    "mpeg2video",  # mpg
    "mjpeg",  # avi
]


class OperationFailedError(Exception):
    def __init__(self, message="A operação não foi concluída com êxito."):
        self.message = message
        super().__init__(self.message)


def get_video_codec(video_file):
    command_base = rf'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of json "{video_file}"'
    command = shlex.split(command_base)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        output, _ = process.communicate()
        codec_info = json.loads(output.decode('utf-8'))
        codec_name = codec_info['streams'][0]['codec_name']
        return codec_name
    except Exception as error:
        logger.debug(str(error))
        return None


def audio_preprocessor(preview, base_audio, audio_wav, use_cuda=False):
    base_audio = base_audio.strip()
    previous_files_to_remove = [audio_wav]
    remover_arquivos(previous_files_to_remove)

    if preview:
        logger.warning(
            "Criando um vídeo de visualização de 10 segundos, para desativar "
            "esta opção, vá para configurações avançadas e desative a visualização."
        )
        wav_ = f'ffmpeg -y -i "{base_audio}" -ss 00:00:20 -t 00:00:10 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav'
    else:
        wav_ = f'ffmpeg -y -i "{base_audio}" -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav'

    # Execute o processo cmd
    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    }
    wav_ = shlex.split(wav_)
    result_convert_audio = subprocess.Popen(wav_, **sub_params)
    output, errors = result_convert_audio.communicate()
    time.sleep(1)
    if result_convert_audio.returncode in [1, 2] or not os.path.exists(
        audio_wav
    ):
        raise OperationFailedError(f"Erro não é possível criar o arquivo de áudio:\n{errors.decode('utf-8')}")


def audio_video_preprocessor(
    preview, video, OutputFile, audio_wav, use_cuda=False
):
    video = video.strip()
    previous_files_to_remove = [OutputFile, "audio.webm", audio_wav]
    remover_arquivos(previous_files_to_remove)

    if os.path.exists(video):
        if preview:
            logger.warning(
                "Criando um vídeo de visualização de 10 segundos, "
                "para desativar esta opção, vá para avançado "
                "configurações e desative a visualização."
            )
            mp4_ = f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental Video.mp4'
        else:
            video_codec = get_video_codec(video)
            if not video_codec:
                logger.debug("Nenhum codec de vídeo encontrado no vídeo")
            else:
                logger.info(f"Video codec: {video_codec}")

            # Verifique se o arquivo termina com extensão ".mp4" ou é um codec válido
            if video.endswith(".mp4") or video_codec in TESTED_CODECS:
                destination_path = os.path.join(os.getcwd(), "Video.mp4")
                shutil.copy(video, destination_path)
                time.sleep(0.5)
                if os.path.exists(OutputFile):
                    mp4_ = "ffmpeg -h"
                else:
                    mp4_ = f'ffmpeg -y -i "{video}" -c copy Video.mp4'
            else:
                logger.warning(
                    "O arquivo não possui a extensão '.mp4' ou um "
                    "codec suportado. Convertendo vídeo para mp4 (codec: h264)."
                )
                mp4_ = f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental Video.mp4'
    else:
        if preview:
            logger.warning(
                "Criando uma visualização a partir do link, 10 segundos "
                "para desativar esta opção, vá para avançado "
                "configurações e desative a visualização."
            )
            # https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = f"python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {video}"

    # Run cmd process
    mp4_ = shlex.split(mp4_)
    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    }

    if os.path.exists(video):
        logger.info("Vídeo do processo...")
        result_convert_video = subprocess.Popen(mp4_, **sub_params)
        # result_convert_video.wait()
        output, errors = result_convert_video.communicate()
        time.sleep(1)
        if result_convert_video.returncode in [1, 2] or not os.path.exists(
            OutputFile
        ):
            raise OperationFailedError(f"Erro ao processar o vídeo:\n{errors.decode('utf-8')}")
        logger.info("Processar áudio...")
        wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
        wav_ = shlex.split(wav_)
        result_convert_audio = subprocess.Popen(wav_, **sub_params)
        output, errors = result_convert_audio.communicate()
        time.sleep(1)
        if result_convert_audio.returncode in [1, 2] or not os.path.exists(
            audio_wav
        ):
            raise OperationFailedError(f"Erro não é possível criar o arquivo de áudio:\n{errors.decode('utf-8')}")

    else:
        wav_ = shlex.split(wav_)
        if preview:
            result_convert_video = subprocess.Popen(mp4_, **sub_params)
            output, errors = result_convert_video.communicate()
            time.sleep(0.5)
            result_convert_audio = subprocess.Popen(wav_, **sub_params)
            output, errors = result_convert_audio.communicate()
            time.sleep(0.5)
            if result_convert_audio.returncode in [1, 2] or not os.path.exists(
                audio_wav
            ):
                raise OperationFailedError(
                    f"Erro, não é possível criar o arquivo de visualização:\n{errors.decode('utf-8')}"
                )
        else:
            logger.info("Processar áudio...")
            result_convert_audio = subprocess.Popen(wav_, **sub_params)
            output, errors = result_convert_audio.communicate()
            time.sleep(1)
            if result_convert_audio.returncode in [1, 2] or not os.path.exists(
                audio_wav
            ):
                raise OperationFailedError(f"Erro não consigo baixar o áudio:\n{errors.decode('utf-8')}")
            logger.info("Processar vídeo...")
            result_convert_video = subprocess.Popen(mp4_, **sub_params)
            output, errors = result_convert_video.communicate()
            time.sleep(1)
            if result_convert_video.returncode in [1, 2] or not os.path.exists(
                OutputFile
            ):
                raise OperationFailedError(f"Erro não consigo baixar o vídeo:\n{errors.decode('utf-8')}")


def old_audio_video_preprocessor(preview, video, OutputFile, audio_wav):
    previous_files_to_remove = [OutputFile, "audio.webm", audio_wav]
    remover_arquivos(previous_files_to_remove)

    if os.path.exists(video):
        if preview:
            logger.warning(
                "Criando um vídeo de visualização de 10 segundos, "
                "para desativar esta opção, vá para avançado "
                "configurações e desative a visualização."
            )
            command = f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental Video.mp4'
            result_convert_video = subprocess.run(
                command, capture_output=True, text=True, shell=True
            )
        else:
            # Check if the file ends with ".mp4" extension
            if video.endswith(".mp4"):
                destination_path = os.path.join(os.getcwd(), "Video.mp4")
                shutil.copy(video, destination_path)
                result_convert_video = {}
                result_convert_video = subprocess.run(
                    "echo Video copied",
                    capture_output=True,
                    text=True,
                    shell=True,
                )
            else:
                logger.warning(
                    "O arquivo não possui a extensão '.mp4'. Convertendo vídeo."
                )
                command = f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental Video.mp4'
                result_convert_video = subprocess.run(
                    command, capture_output=True, text=True, shell=True
                )

        if result_convert_video.returncode in [1, 2]:
            raise OperationFailedError("Erro não é possível converter o vídeo")

        for i in range(120):
            time.sleep(1)
            logger.info("Processar vídeo...")
            if os.path.exists(OutputFile):
                time.sleep(1)
                command = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
                result_convert_audio = subprocess.run(
                    command, capture_output=True, text=True, shell=True
                )
                time.sleep(1)
                break
            if i == 119:
                # if not os.path.exists(OutputFile):
                raise OperationFailedError("Erro ao processar o vídeo")

        if result_convert_audio.returncode in [1, 2]:
            raise OperationFailedError(
                f"Erro não é possível criar o arquivo de áudio: {result_convert_audio.stderr}"
            )

        for i in range(120):
            time.sleep(1)
            logger.info("Processar áudio...")
            if os.path.exists(audio_wav):
                break
            if i == 119:
                raise OperationFailedError("Erro não é possível criar o arquivo de áudio")

    else:
        video = video.strip()
        if preview:
            logger.warning(
                "Criando uma visualização a partir do link, 10 "
                "segundos para desativar esta opção, vá para "
                "configurações avançadas e desative a visualização."
            )
            # https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
            result_convert_video = subprocess.run(
                mp4_, capture_output=True, text=True, shell=True
            )
            result_convert_audio = subprocess.run(
                wav_, capture_output=True, text=True, shell=True
            )
            if result_convert_audio.returncode in [1, 2]:
                raise OperationFailedError("Error can't download a preview")
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = f"python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {video}"

            result_convert_audio = subprocess.run(
                wav_, capture_output=True, text=True, shell=True
            )

            if result_convert_audio.returncode in [1, 2]:
                raise OperationFailedError("Erro não consigo baixar o áudio")

            for i in range(120):
                time.sleep(1)
                logger.info("Processar áudio...")
                if os.path.exists(audio_wav) and not os.path.exists(
                    "audio.webm"
                ):
                    time.sleep(1)
                    result_convert_video = subprocess.run(
                        mp4_, capture_output=True, text=True, shell=True
                    )
                    break
                if i == 119:
                    raise OperationFailedError("Erro ao baixar o áudio")

            if result_convert_video.returncode in [1, 2]:
                raise OperationFailedError("Erro não consigo baixar o vídeo")