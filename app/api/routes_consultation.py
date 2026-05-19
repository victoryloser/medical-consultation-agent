from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from app.schemas.consultation import ConsultationResponse
from app.services.consultation_service import ConsultationService
from app.services.image_service import cleanup_file, save_upload_file

router = APIRouter(prefix="/api", tags=["consultation"])
_service = ConsultationService()


@router.post("/consultation", response_model=ConsultationResponse)
async def consultation(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="症状描述"),
    model_provider: str = Form("auto", description="auto | ollama | api"),
    image: UploadFile | None = File(None, description="可选图片"),
):
    image_path = None
    if image and image.filename:
        from app.config import get_config
        cfg = get_config()
        image_path = await save_upload_file(image, cfg.app.upload_dir)
        background_tasks.add_task(cleanup_file, image_path)

    return await _service.run(text=text, image_path=image_path, model_provider=model_provider)
