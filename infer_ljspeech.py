"""StyleTTS2 LJSpeech CLI inference (based on Demo/Inference_LJSpeech.ipynb)."""
from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from pathlib import Path

import nltk
import numpy as np
import phonemizer
import soundfile as sf
import torch
import torchaudio
import yaml
from munch import Munch
from nltk.tokenize import word_tokenize

# PyTorch 2.6+ defaults weights_only=True; StyleTTS2 checkpoints need full unpickle.
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from models import build_model, load_ASR_models, load_F0_models
from text_utils import TextCleaner
from utils import recursive_munch


def tokenize_words(text: str) -> list[str]:
    """Prefer NLTK punkt; fall back to whitespace split if data missing/offline."""
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def load_checkpoint(model, params: dict) -> None:
    for key in model:
        if key not in params:
            continue
        print(f"{key} loaded")
        try:
            model[key].load_state_dict(params[key])
        except Exception:
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model[key].load_state_dict(new_state_dict, strict=False)


def synthesize(
    model,
    sampler,
    text_cleaner: TextCleaner,
    phonemizer_backend,
    device: str,
    text: str,
    noise: torch.Tensor,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
) -> np.ndarray:
    text = text.strip().replace('"', "")
    ps = phonemizer_backend.phonemize([text])
    ps = " ".join(tokenize_words(ps[0]))
    tokens = text_cleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise,
            embedding=bert_dur[0].unsqueeze(0),
            num_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        ).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder(
            (t_en @ pred_aln_trg.unsqueeze(0).to(device)),
            F0_pred,
            N_pred,
            ref.squeeze().unsqueeze(0),
        )
    return out.squeeze().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="StyleTTS2 LJSpeech inference")
    parser.add_argument(
        "--text",
        default=(
            "StyleTTS 2 is a text-to-speech model that leverages style diffusion "
            "and adversarial training with large speech language models to achieve "
            "human-level text-to-speech synthesis."
        ),
    )
    parser.add_argument("--out", default="outputs/ljspeech_demo.wav")
    parser.add_argument("--config", default="Models/LJSpeech/config.yml")
    parser.add_argument("--checkpoint", default="Models/LJSpeech/epoch_2nd_00100.pth")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--embedding-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    os_chdir = root
    import os

    os.chdir(os_chdir)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("nltk punkt missing; using whitespace tokenize fallback", flush=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    print("loading config/aux models...", flush=True)
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    text_aligner = load_ASR_models(config.get("ASR_path", False), config.get("ASR_config", False))
    pitch_extractor = load_F0_models(config.get("F0_path", False))
    plbert = load_plbert(config.get("PLBERT_dir", False))

    print("building model...", flush=True)
    model = build_model(recursive_munch(config["model_params"]), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    print(f"loading checkpoint {args.checkpoint}...", flush=True)
    params_whole = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_checkpoint(model, params_whole["net"])
    _ = [model[key].eval() for key in model]

    print("init sampler/phonemizer...", flush=True)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    phonemizer_backend = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    text_cleaner = TextCleaner()

    noise = torch.randn(1, 1, 256).to(device)
    t0 = time.time()
    wav = synthesize(
        model,
        sampler,
        text_cleaner,
        phonemizer_backend,
        device,
        args.text,
        noise,
        diffusion_steps=args.steps,
        embedding_scale=args.embedding_scale,
    )
    elapsed = time.time() - t0
    rtf = elapsed / (len(wav) / 24000)
    print(f"RTF={rtf:.5f} duration={len(wav)/24000:.2f}s elapsed={elapsed:.2f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, 24000)
    print(f"wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
