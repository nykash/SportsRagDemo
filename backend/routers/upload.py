from fastapi import APIRouter, UploadFile, File, HTTPException
from file_system import write_file  # make sure file_system.py is in your project

router = APIRouter(
    prefix="/upload",
    tags=["Videos"]
)

@router.post("/video")
async def upload_video(file: UploadFile = File(...)):
    # Validate file type (optional)
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    # Read content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
    finally:
        await file.close()

    # Save using your file_system abstraction
    try:
        saved_path = write_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    return {
        "message": "Video uploaded successfully",
        "file_path": saved_path
    }
