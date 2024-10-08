from pydub import AudioSegment
from tqdm import tqdm
from utils import run_command
from loggingsetup import logger
import numpy as np


class Mixer:
    def __init__(self):
        self.parts = []

    def __len__(self):
        parts = self._sync()
        seg = parts[0][1]
        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        return int(1000.0 * frame_count / seg.frame_rate)

    def overlay(self, sound, position=0):
        self.parts.append((position, sound))
        return self

    def _sync(self):
        positions, segs = zip(*self.parts)

        frame_rate = segs[0].frame_rate
        array_type = segs[0].array_type # noqa

        offsets = [int(frame_rate * pos / 1000.0) for pos in positions]
        segs = AudioSegment.empty()._sync(*segs)
        return list(zip(offsets, segs))

    def acrescentar(self, sound):
        self.overlay(sound, position=len(self))

    def para_segmento_audio(self):
        parts = self._sync()
        seg = parts[0][1]
        channels = seg.channels

        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        sample_count = int(frame_count * seg.channels)

        output = np.zeros(sample_count, dtype="int32")
        for offset, seg in parts:
            sample_offset = offset * channels
            samples = np.frombuffer(seg.get_array_of_samples(), dtype="int32")
            samples = np.int16(samples/np.max(np.abs(samples)) * 32767)
            start = sample_offset
            end = start + len(samples)
            output[start:end] += samples

        return seg._spawn(
            output, overrides={"sample_width": 4}).normalize(headroom=0.0)


def criar_traducao_de_adio(
    result_diarize, audio_files, final_file, concat=False, avoid_overlap=False,
):
    total_duration = result_diarize["segmentos"][-1]["fim"]  # em segundos

    if concat:
        """
        file .\audio\1.ogg
        file .\audio\2.ogg
        file .\audio\3.ogg
        file .\audio\4.ogg
        ...
        """

        # Escreva os caminhos dos arquivos em list.txt
        with open("list.txt", "w") as file:
            for i, audio_file in enumerate(audio_files):
                if i == len(audio_files) - 1:  # Verifique se é o último item
                    file.write(f"file {audio_file}")
                else:
                    file.write(f"file {audio_file}\n")

        # command = f"ffmpeg -f concat -safe 0 -i list.txt {final_file}"
        command = (
            f"ffmpeg -f concat -safe 0 -i list.txt -c:a pcm_s16le {final_file}"
        )
        run_command(command)

    else:
        # áudio silencioso com total_duration
        base_audio = AudioSegment.silent(
            duration=int(total_duration * 1000), frame_rate=41000
        )
        combined_audio = Mixer()
        combined_audio.overlay(base_audio)

        logger.debug(
            f"Duração de audio: {total_duration // 60} "
            f"minutos e{int(total_duration % 60)} segundos"
        )

        last_end_time = 0
        previous_speaker = ""
        for line, audio_file in tqdm(
            zip(result_diarize["segmentos"], audio_files)
        ):
            start = float(line["começo"])

            # Sobrepor cada áudio no horário correspondente
            try:
                audio = AudioSegment.from_file(audio_file)
                # audio_a = audio.speedup(playback_speed=1.5)

                if avoid_overlap:
                    speaker = line["speaker"]
                    if (last_end_time - 0.500) > start:
                        overlap_time = last_end_time - start
                        if previous_speaker and previous_speaker != speaker:
                            start = (last_end_time - 0.500)
                        else:
                            start = (last_end_time - 0.200)
                        if overlap_time > 2.5:
                            start = start - 0.3
                        logger.info(
                              f"Evite sobreposição para {str(audio_file)} "
                              f"Com {str(start)}"
                        )

                    previous_speaker = speaker

                    duration_tts_seconds = len(audio) / 1000.0  # no seg
                    last_end_time = (start + duration_tts_seconds)

                start_time = start * 1000  # para
                combined_audio = combined_audio.overlay(
                    audio, position=start_time
                )
            except Exception as error:
                logger.debug(str(error))
                logger.error(f"Erro no arquivo de áudio {audio_file}")

        # áudio combinado como um arquivo
        combined_audio_data = combined_audio.to_audio_segment()
        combined_audio_data.export(
            final_file, format="wav"
        )  # melhor que ogg, mude se o áudio estiver anômalo