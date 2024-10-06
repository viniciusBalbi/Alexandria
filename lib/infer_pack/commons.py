import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def pesos_iniciais(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def obter_preenchimento(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def converçor_formato_de_bloco(formato_de_bloco):
    l = formato_de_bloco[::-1]
    formato_de_bloco = [item for sublist in l for item in sublist]
    return formato_de_bloco

def kl_divergente(m_p, logs_p, m_q, logs_q):
    """KL(P|Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    ) 
    return kl


def rand_gumbel(shape):
    """Amostra da distribuição Gumbel, protegida contra transbordamentos."""
    amostras_uniformes = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(amostras_uniformes))

def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g

def fatia_de_segmentos(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def fatia_de_segmentos2(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret

def fatia_de_segmentos_rand(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = fatia_de_segmentos(x, ids_str, segment_size)
    return ret, ids_str

def sinal_de_temporização_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    posição = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales =  min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    tempo_escalonado = posição.unsqueeze(0) * inv_timescales.unsqueeze(1)
    sinal = torch.cat([torch.sin(tempo_escalonado), torch.cos(tempo_escalonado)], 0)
    sinal = F.pad(sinal, [0, 0, 0,  channels % 2])
    sinal = sinal.view(1,  channels, length)
    return sinal

def adicione_sinal_de_temporização_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = sinal_de_temporização_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype,  device=x.device)

def gato_desinal_de_temporização_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = adicione_sinal_de_temporização_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def mascara_subsequente(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask

@torch.jit.script
def adicionar_tanh_sigmoide_multiplicar(input_a, input_b, n_canais):
    n_canais_int = n_canais[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_canais_int, :])
    s_act = torch.sigmoid(in_act[:, n_canais_int:, :])
    acts = t_act * s_act
    return acts

def converçor_de_formato_de_bloco(formato_bloco):
    l = formato_bloco[::-1]
    formato_bloco = [item for sublist in l for item in sublist]
    return formato_bloco

def turno_1d(x):
    x = F.pad(x, converçor_de_formato_de_bloco([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x

def sequencia_de_mascaras(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def gerar_caminho(duração, mascara):
    """
    duração: [b, 1, t_x]
    mascara: [b, 1, t_y, t_x]
    """

    dispositivo = duração.dispositivo

    b, _, t_y, t_x = mascara.formato

    duração2 = torch.cumsum(duração, -1)

    duração3 = duração2.view(b * t_x)
    path = sequencia_de_mascaras(duração3, t_y).to(mascara.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, converçor_de_formato_de_bloco([[0, 0], [1, 0], [0,  0]]))[:,  :-1]
    return path

def valor_de_duração_de_video(parametros, clip_value, norm_type=2):
    if isinstance(parametros, torch.Tensor):
        parametros = [parametros]
    parametros = list(filter(lambda p: p.grad is not  None, parametros))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in  parametros:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm =  total_norm ** (1.0 / norm_type)
    return total_norm










