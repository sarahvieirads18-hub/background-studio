from __future__ import annotations

import io
import os
import threading
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from rembg import new_session, remove

BASE = Path(__file__).resolve().parent
MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 32_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2
app = FastAPI(title="Background Studio")
_lock = threading.Lock()
_session = None


def get_session():
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                _session = new_session("u2net")
    return _session


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/api/remove")
def remove_background(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Imagem acima do limite de 25 MB")
    if not data:
        raise HTTPException(400, "Arquivo vazio")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(415, "Formato não aceito. Use PNG, JPG ou WEBP")
            if im.width * im.height > MAX_PIXELS:
                raise HTTPException(413, "Imagem acima do limite de 32 megapixels")
            original = ImageOps.exif_transpose(im).convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Não foi possível abrir a imagem")
    # Original is never resampled or resized: segmentation works at the original dimensions.
    try:
        session = get_session()
        with _lock:  # inference thread safety and predictable CPU utilization
            result = remove(original, session=session).convert("RGBA")
    except Exception as exc:
        raise HTTPException(500, f"Falha no motor de IA: {type(exc).__name__}") from exc
    out = io.BytesIO()
    result.save(out, format="PNG", compress_level=6)
    return Response(content=out.getvalue(), media_type="image/png", headers={
        "X-Image-Width": str(original.width), "X-Image-Height": str(original.height)
    })


@app.post("/api/zip")
def make_zip(files: list[UploadFile] = File(...)):
    if not files or len(files) > 30:
        raise HTTPException(400, "Envie entre 1 e 30 resultados")
    out = io.BytesIO()
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for i, upload in enumerate(files, 1):
            raw = upload.file.read(MAX_BYTES + 1)
            total += len(raw)
            if len(raw) > MAX_BYTES or total > 200 * 1024 * 1024:
                raise HTTPException(413, "Resultados muito grandes para um único ZIP")
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    if im.format != "PNG":
                        raise ValueError("not png")
                    im.verify()
            except (UnidentifiedImageError, ValueError, OSError):
                raise HTTPException(400, f"Arquivo {i} não é PNG válido")
            z.writestr(f"{i:02d}_{Path(upload.filename or 'imagem').stem[:70]}.png", raw)
    out.seek(0)
    return StreamingResponse(out, media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="background_studio_resultados.zip"'
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
