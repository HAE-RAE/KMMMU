from PIL import Image
from io import BytesIO
import base64

from vllm.multimodal.utils import fetch_image


SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You MUST output the final answer in boxed format as LaTeX with dollar signs, exactly like:\n"
    "$\\boxed{ANSWER}$\n"
    "Do NOT add any extra text outside the box."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator.\n"
    "Decide if the model's answer matches the gold answer.\n"
    "Use the parsed model answer (from Math-Verify) if helpful.\n"
    "Return ONLY: true or false."
)


def pil_to_base64(img: Image.Image, *, img_format="PNG", add_data_uri=False, quality=95):
    if img_format.upper() == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    buf = BytesIO()
    save_kwargs = {}
    if img_format.upper() == "JPEG":
        save_kwargs.update(dict(quality=quality, optimize=True))
    img.save(buf, format=img_format.upper(), **save_kwargs)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    if add_data_uri:
        return f"data:image/{img_format.lower()};base64,{b64}"
    return b64
