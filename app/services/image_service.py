import io
import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile
from PIL import Image


async def save_upload_file(upload_file: UploadFile, upload_dir: str = "tmp/uploads") -> str:
    """
    将 UploadFile 保存到临时目录，返回本地文件路径。
    使用 UUID 避免并发冲突，并归一化为 JPEG 格式。
    """
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    content = await upload_file.read()

    try:
        img = Image.open(io.BytesIO(content))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        content = buf.getvalue()
        suffix = ".jpg"
    except Exception:
        suffix = Path(upload_file.filename or "image.jpg").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".jpg"

    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = upload_path / temp_name

    async with aiofiles.open(temp_path, "wb") as f:
        await f.write(content)

    return str(temp_path)


async def cleanup_file(file_path: str):
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass
