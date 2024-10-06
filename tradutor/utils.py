import os, zipfile, rarfile, shutil, subprocess, shlex, sys # noqa
from loggingsetup import logger
from urllib.parse import urlparse
from IPython.utils import capture
import re
import soundfile as sf
import numpy as np


Extensãode_video  = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp"
]


Extensãode_audio = [
    ".mp3",
    ".wav",
    ".aiff",
    ".aif",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".m4a",
    ".alac",
    ".pcm",
    ".opus",
    ".ape",
    ".amr",
    ".ac3",
    ".vox",
    ".caf"
]

Extensãode_legenda = [
    ".srt",
    ".vtt",
    ".ass"
]



def run_command(command):
    logger.debug(command)
    if isinstance(command, str):
        command = shlex.split(command)

    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    }
    process_command = subprocess.Popen(command, **sub_params)
    output, errors = process_command.communicate()
    if (
        process_command.returncode != 0
    ):  # or not os.path.exists(mono_path) or os.path.getsize(mono_path) == 0:
        logger.error("Error comnand")
        raise Exception(errors.decode())



def escrita_em_pedaços(
    file,
    data,
    samplerate,
    subtype=None,
    endian=None,
    format=None,
    closefd=True,
    chunk_size=0x1000
):

    data = np.asarray(data)
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]
    with sf.SoundFile(
        file, 'w', samplerate, channels,
        subtype, endian, format, closefd
    ) as f:
        num_chunks = (len(data) + chunk_size - 1) // chunk_size
        for chunk in np.array_split(data, num_chunks, axis=0):
            f.write(chunk)



def print_tree_directory(root_dir, indent=""):
    if not os.path.exists(root_dir):
        logger.error(f"{indent} Invalid directory or file: {root_dir}")
        return

    items = os.listdir(root_dir)

    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        is_last_item = index == len(items) - 1

        if os.path.isfile(item_path) and item_path.endswith(".zip"):
            with zipfile.ZipFile(item_path, "r") as zip_file:
                print(
                    f"{indent}{'└──' if is_last_item else '├──'} {item} (zip file)"
                )
                zip_contents = zip_file.namelist()
                for zip_item in sorted(zip_contents):
                    print(
                        f"{indent}{'    ' if is_last_item else '│   '}{zip_item}"
                    )
        else:
            print(f"{indent}{'└──' if is_last_item else '├──'} {item}")

            if os.path.isdir(item_path):
                new_indent = indent + ("    " if is_last_item else "│   ")
                print_tree_directory(item_path, new_indent)


def carregar_lista_de_modelos():
    weight_root = "weights"
    models = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            models.append("weights/" + name)
    if models:
        logger.debug(models)

    index_root = "logs"
    index_paths = [None]
    for name in os.listdir(index_root):
        if name.endswith(".index"):
            index_paths.append("logs/" + name)
    if index_paths:
        logger.debug(index_paths)

    return models, index_paths


def manual_download(url, dst):
    if "drive.google" in url:
        logger.info("Drive url")
        if "folders" in url:
            logger.info("folder")
            os.system(f'gdown --folder "{url}" -O {dst} --fuzzy -c')
        else:
            logger.info("single")
            os.system(f'gdown "{url}" -O {dst} --fuzzy -c')
    elif "huggingface" in url:
        logger.info("HuggingFace url")
        if "/blob/" in url or "/resolve/" in url:
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/")
            gerenciador_de_dowload(url=url, path=dst, overwrite=True, progress=True)
        else:
            os.system(f"git clone {url} {dst+'repo/'}")
    elif "http" in url:
        logger.info("URL")
        gerenciador_de_dowload(url=url, path=dst, overwrite=True, progress=True)
    elif os.path.exists(url):
        logger.info("Path")
        copiar_arquivos(url, dst)
    else:
        logger.error(f"No valid URL: {url}")


def download_list(text_downloads):
    try:
        urls = [elem.strip() for elem in text_downloads.split(",")]
    except Exception as error:
        raise ValueError(f"No valid URL. {str(error)}")

    criar_diretorio(["downloads", "logs", "weights"])

    path_download = "downloads/"
    for url in urls:
        manual_download(url, path_download)

    # Tree
    print("####################################")
    print_tree_directory("downloads", indent="")
    print("####################################")

    # Place files
    selecione_arquivos_zip_e_rar("downloads/")

    models, _ = carregar_lista_de_modelos()

    # hf space models files delete
    remover_conteudo_de_diretorio("downloads/repo")

    return f"Downloaded = {models}"


def selecione_arquivos_zip_e_rar(directory_path="downloads/"):
    # filter
    zip_files = []
    rar_files = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".zip"):
            zip_files.append(file_name)
        elif file_name.endswith(".rar"):
            rar_files.append(file_name)

    # extract
    for file_name in zip_files:
        file_path = os.path.join(directory_path, file_name)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(directory_path)

    for file_name in rar_files:
        file_path = os.path.join(directory_path, file_name)
        with rarfile.RarFile(file_path, "r") as rar_ref:
            rar_ref.extractall(directory_path)


    # Definindo caminho
    def move_files_with_extension(src_dir, extension, destination_dir):
        for root, _, files in os.walk(src_dir):
            for file_name in files:
                if file_name.endswith(extension):
                    source_file = os.path.join(root, file_name)
                    destination = os.path.join(destination_dir, file_name)
                    shutil.move(source_file, destination)

    move_files_with_extension(directory_path, ".index", "logs/")
    move_files_with_extension(directory_path, ".pth", "weights/")

    return "Download complete"


def arquivo_com_extenssoes(string_path, extensions):
    return any(string_path.lower().endswith(ext) for ext in extensions)

def arquivo_de_video(string_path):
    return arquivo_com_extenssoes(string_path, Extensãode_video)

def arquivo_de_audio(string_path):
    return arquivo_com_extenssoes(string_path, Extensãode_audio)

def arquivo_de_legenda(string_path):
    return arquivo_com_extenssoes(string_path, Extensãode_legenda)

def arquivos_de_diretorio(directory):
    audio_files = []
    video_files = []
    sub_files = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):

            if arquivo_de_audio(item_path):
                audio_files.append(item_path)
            
            elif arquivo_de_video(item_path):
                video_files.append(item_path)

            elif arquivo_de_legenda(item_path):
                sub_files.append(item_path)

    logger.info(
        f"Files in path ({directory}): "
        f"{str(audio_files + video_files + sub_files)}"
    )

    return audio_files, video_files, sub_files


def obter_arquivos_validos(paths):
    valid_paths = []
    for path in paths:
        if os.path.isdir(path):
            audio_files, video_files, sub_files = arquivos_de_diretorio(path)
            valid_paths.extend(audio_files)
            valid_paths.extend(video_files)
            valid_paths.extend(sub_files)
        else:
            valid_paths.append(path)

    return valid_paths


def extrair_links_de_video(link):

    params_dlp = {"quiet": False, "no_warnings": True, "noplaylist": False}

    try:
        from yt_dlp import YoutubeDL
        with capture.capture_output() as cap:
            with YoutubeDL(params_dlp) as ydl:
                info_dict = ydl.extract_info( # noqa
                    link, download=False, process=True
                )

        urls = re.findall(r'\[youtube\] Extracting URL: (.*?)\n', cap.stdout)
        logger.info(f"List of videos in ({link}): {str(urls)}")
        del cap
    except Exception as error:
        logger.error(f"{link} >> {str(error)}")
        urls = [link]

    return urls


def obter_lista_de_Links(urls):
    valid_links = []
    for url_video in urls:
        if "youtube.com" in url_video and "/watch?v=" not in url_video:
            url_links = extrair_links_de_video(url_video)
            valid_links.extend(url_links)
        else:
            valid_links.append(url_video)
    return valid_links

# =====================================
# Download Manager
# =====================================



def carregar_arquivo_da_url(
    url: str,
    model_dir: str,
    file_name: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """Baixe um arquivo de `url` em `model_dir`,
    usando o arquivo presente, se possível.

    Retorna o caminho para o arquivo baixado.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    # Substituir
    if os.path.exists(cached_file):
        if overwrite or os.path.getsize(cached_file) == 0:
            remover_arquivos(cached_file)

    # Download
    if not os.path.exists(cached_file):
        logger.info(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    else:
        logger.debug(cached_file)

    return cached_file


def nome_amigavel(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name, extension


def gerenciador_de_dowload(
    url: str,
    path: str,
    extension: str = "",
    overwrite: bool = False,
    progress: bool = True,
):
    url = url.strip()

    name, ext = nome_amigavel(url)
    name += ext if not extension else f".{extension}"

    if url.startswith("http"):
        filename = carregar_arquivo_da_url(
            url=url,
            model_dir=path,
            file_name=name,
            overwrite=overwrite,
            progress=progress,
        )
    else:
        filename = path

    return filename



# =====================================
# Gerenciamento de arquivos
# =====================================


# remover apenas arquivos
def remover_arquivos(file_list):
    if isinstance(file_list, str):
        file_list = [file_list]

    for file in file_list:
        if os.path.exists(file):
            os.remove(file)



def remover_conteudo_de_diretorio(directory_path):
    """
    Remove todos os arquivos e subdiretórios de um diretório.

    Parâmetros:
    directory_path (str): Caminho para o diretório cujo
    o conteúdo precisa ser removido.
    """
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Falha ao excluir {file_path}. Razão: {e}")
        logger.info(f"Conteúdo em '{directory_path}' removido.")
    else:
        logger.error(f"Diretório '{directory_path}' não existe.")



# Cria diretório se não existir
def criar_diretorio(directory_path):
    if isinstance(directory_path, str):
        directory_path = [directory_path]
    for one_dir_path in directory_path:
        if not os.path.exists(one_dir_path):
            os.makedirs(one_dir_path)
            logger.debug(f"Diretorio '{one_dir_path}' criado.")



def mover_arquivo(source_dir, destination_dir, extension=""):
    """
    Move arquivos do caminho de origem para o caminho de destino.

    Parâmetros:
    source_dir (str): Caminho para o diretório de origem.
    destination_dir (str): Caminho para o diretório de destino.
    extensão (str): mova apenas arquivos com esta extensão.
    """
    criar_diretorio(destination_dir)

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        if extension and not filename.endswith(extension):
            continue
        os.replace(source_path, destination_path)



def copiar_arquivos(source_path, destination_path):
    """
    Copia um arquivo ou vários arquivos de um caminho de origem para um caminho de destino.

    Parâmetros:
    source_path (str ou list): Caminho ou lista de caminhos para a origem
    arquivo(s) ou diretório.
    destination_path (str): Caminho para o diretório de destino.
    """
    criar_diretorio(destination_path)

    if isinstance(source_path, str):
        source_path = [source_path]

    if os.path.isdir(source_path[0]):
        # Copie todos os arquivos do diretório de origem para o diretório de destino
        base_path = source_path[0]
        source_path = os.listdir(source_path[0])
        source_path = [
            os.path.join(base_path, file_name) for file_name in source_path
        ]

    for one_source_path in source_path:
        if os.path.exists(one_source_path):
            shutil.copy2(one_source_path, destination_path)
            logger.debug(
                f"arquivo '{one_source_path}' copiado '{destination_path}'."
            )
        else:
            logger.error(f"arquivo '{one_source_path}' não existe.")

def renomar_arquivo(current_name, new_name):
    file_directory = os.path.dirname(current_name)

    if os.path.exists(current_name):
        dir_new_name_file = os.path.join(file_directory, new_name)
        os.rename(current_name, dir_new_name_file)
        logger.debug(f"File '{current_name}' renamed to '{new_name}'.")
        return dir_new_name_file
    else:
        logger.error(f"File '{current_name}' does not exist.")
        return None