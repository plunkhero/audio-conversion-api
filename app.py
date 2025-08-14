from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import librosa
import soundfile as sf
import numpy as np
import tempfile
import os
from typing import Optional, List
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil

app = FastAPI(
    title="Audio Conversion API for AI Training",
    description="API สำหรับแปลงนามสกุลไฟล์เสียงและปรับแต่งพารามิเตอร์เพื่อการเทรน AI โมเดล",
    version="1.0.0"
)

class ConversionParams(BaseModel):
    target_format: str = "wav"
    sample_rate: Optional[int] = 22050  # ค่ามาตรฐานสำหรับ AI
    channels: Optional[int] = 1  # mono สำหรับ AI
    bit_depth: Optional[int] = 16  # 16-bit PCM
    normalize: Optional[bool] = True  # normalize amplitude
    trim_silence: Optional[bool] = True  # ตัดเสียงเงียบ
    max_duration: Optional[float] = None  # จำกัดความยาว (วินาที)
    min_duration: Optional[float] = None  # ความยาวขั้นต่ำ

class AudioInfo(BaseModel):
    filename: str
    original_format: str
    duration: float
    sample_rate: int
    channels: int
    file_size: int

# รูปแบบไฟล์ที่รองรับ
SUPPORTED_INPUT_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
SUPPORTED_OUTPUT_FORMATS = {'.wav', '.flac'}

def get_audio_info(file_path: str) -> dict:
    """ดึงข้อมูลไฟล์เสียง"""
    try:
        # โหลดไฟล์เสียงแบบไม่บังคับ mono
        y, sr = librosa.load(file_path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)
        file_size = os.path.getsize(file_path)
        
        # ตรวจสอบจำนวน channel
        if y.ndim == 1:
            channels = 1
        else:
            channels = y.shape[0]

        return {
            "duration": duration,
            "sample_rate": sr,
            "channels": channels,
            "file_size": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading audio file: {str(e)}")

def process_audio(input_path: str, params: ConversionParams) -> tuple:
    """ประมวลผลไฟล์เสียงตามพารามิเตอร์ที่กำหนด"""
    try:
        # โหลดไฟล์เสียง
        y, sr = librosa.load(input_path, sr=params.sample_rate, mono=(params.channels == 1))
        
        # ตัดเสียงเงียบ
        if params.trim_silence:
            y, _ = librosa.effects.trim(y, top_db=20)
        
        # จำกัดความยาว
        if params.max_duration:
            max_samples = int(params.max_duration * sr)
            if len(y) > max_samples:
                y = y[:max_samples]
        
        # ตรวจสอบความยาวขั้นต่ำ
        if params.min_duration:
            min_samples = int(params.min_duration * sr)
            if len(y) < min_samples:
                # Pad with zeros if too short
                y = np.pad(y, (0, min_samples - len(y)), mode='constant')
        
        # Normalize
        if params.normalize:
            y = librosa.util.normalize(y)
        
        # แปลงเป็น multi-channel หากต้องการ
        if params.channels > 1 and len(y.shape) == 1:
            y = np.tile(y, (params.channels, 1))
        
        return y, sr
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

@app.post("/convert", response_class=FileResponse)
async def convert_audio(
    file: UploadFile = File(...),
    target_format: str = Form("wav"),
    background_tasks: BackgroundTasks = None
):
    """แปลงไฟล์เสียง รับค่าจากผู้ใช้แค่ file และ target_format"""

    # ตั้งค่าพารามิเตอร์ default
    params = ConversionParams(
        target_format=target_format,
        sample_rate=22050,
        channels=1,
        bit_depth=16,
        normalize=True,
        trim_silence=True,
        max_duration=None,
        min_duration=None
    )

    # ตรวจสอบนามสกุลไฟล์
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_INPUT_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported input format. Supported: {SUPPORTED_INPUT_FORMATS}"
        )

    if f".{target_format}" not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format. Supported: {list(SUPPORTED_OUTPUT_FORMATS)}"
        )

    # สร้างไฟล์ชั่วคราวสำหรับ input
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    try:
        content = await file.read()
        temp_input.write(content)
        temp_input.flush()
        temp_input.close()  # ปิดก่อนใช้งาน

        # ประมวลผลเสียง
        y, sr = process_audio(temp_input.name, params)

        # สร้างไฟล์ output
        output_filename = f"{Path(file.filename).stem}.{target_format}"
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=f".{target_format}")
        temp_output.close()

        # บันทึกไฟล์
        sf.write(
            temp_output.name,
            y.T if len(y.shape) > 1 else y,
            sr,
            subtype='PCM_16' if params.bit_depth == 16 else 'PCM_24'
        )

        # ลบไฟล์ temp หลังส่ง response เสร็จ
        if background_tasks:
            background_tasks.add_task(lambda: [os.unlink(temp_input.name), os.unlink(temp_output.name)])

        return FileResponse(
            path=temp_output.name,
            filename=output_filename,
            media_type=f"audio/{target_format}"
        )

    except Exception as e:
        if os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/convert_batch")
# async def convert_batch(
#     files: List[UploadFile] = File(...),
#     target_format: str = Form("wav"),
#     sample_rate: int = Form(22050),
#     channels: int = Form(1),
#     normalize: bool = Form(True),
#     trim_silence: bool = Form(True)
# ):
#     """แปลงไฟล์เสียงหลายไฟล์พร้อมกัน"""
    
#     results = []
#     temp_dir = tempfile.mkdtemp()
    
#     try:
#         for file in files:
#             file_ext = Path(file.filename).suffix.lower()
#             if file_ext not in SUPPORTED_INPUT_FORMATS:
#                 results.append({
#                     "filename": file.filename,
#                     "status": "error",
#                     "message": f"Unsupported format: {file_ext}"
#                 })
#                 continue
            
#             try:
#                 # บันทึกไฟล์ input
#                 input_path = os.path.join(temp_dir, file.filename)
#                 with open(input_path, "wb") as f:
#                     content = await file.read()
#                     f.write(content)
                
#                 # ประมวลผล
#                 params = ConversionParams(
#                     target_format=target_format,
#                     sample_rate=sample_rate,
#                     channels=channels,
#                     normalize=normalize,
#                     trim_silence=trim_silence
#                 )
                
#                 y, sr = process_audio(input_path, params)
                
#                 # บันทึก output
#                 output_filename = f"{Path(file.filename).stem}.{target_format}"
#                 output_path = os.path.join(temp_dir, output_filename)
#                 sf.write(output_path, y.T if len(y.shape) > 1 else y, sr)
                
#                 results.append({
#                     "filename": file.filename,
#                     "output_filename": output_filename,
#                     "status": "success",
#                     "file_size": os.path.getsize(output_path)
#                 })
                
#             except Exception as e:
#                 results.append({
#                     "filename": file.filename,
#                     "status": "error",
#                     "message": str(e)
#                 })
    
#     finally:
#         # ทำความสะอาดไฟล์ชั่วคราว
#         shutil.rmtree(temp_dir, ignore_errors=True)
    
#     return {"results": results}

@app.post("/audio_info", response_model=AudioInfo)
async def get_file_info(file: UploadFile = File(...)):
    """ดึงข้อมูลไฟล์เสียง"""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()
        temp_path = temp_file.name
        
    try:
        info = get_audio_info(temp_path)
        return AudioInfo(
            filename=file.filename,
            original_format=suffix.lstrip('.'),
            duration=info["duration"],
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            file_size=info["file_size"]
        )
    finally:
        os.unlink(temp_path)

@app.get("/supported_formats")
async def get_supported_formats():
    """รายการรูปแบบไฟล์ที่รองรับ"""
    return {
        "input_formats": list(SUPPORTED_INPUT_FORMATS),
        "output_formats": list(SUPPORTED_OUTPUT_FORMATS),
        "recommended_ai_settings": {
            "sample_rate": 22050,
            "channels": 1,
            "format": "wav, flac",
            "bit_depth": 16,
            "normalize": True,
            "trim_silence": True
        }
    }

@app.get("/")
async def root():
    """หน้าแรกของ API"""
    return {
        "message": "Audio Conversion API for AI Training",
        "version": "1.0.0",
        "endpoints": [
            "/convert - แปลงไฟล์เสียงเดี่ยว",
            # "/convert_batch - แปลงไฟล์เสียงหลายไฟล์",
            "/audio_info - ดึงข้อมูลไฟล์เสียง",
            "/supported_formats - รูปแบบไฟล์ที่รองรับ",
            "/docs - API documentation"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)