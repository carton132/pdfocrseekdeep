"""
DeepSeek PDF OCR - FastAPI Server
"""

import os
import io
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdf2image

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/deepseek-ocr")
PDF_DPI = int(os.environ.get("PDF_DPI", "300"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("deepseek-ocr")

class OCRModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load(self):
        if self.loaded:
            return
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        start = time.time()
        
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager"
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)
        self.loaded = True
        
        logger.info(f"Model loaded in {time.time() - start:.2f}s")
    
    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.loaded = False

ocr_model = OCRModel()

PROMPTS = {
    "document": "<image>\n<|grounding|>Convert the document to markdown.",
    "free": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
    "describe": "<image>\nDescribe this image in detail.",
}

RESOLUTIONS = {
    "tiny": {"base_size": 512, "image_size": 512},
    "small": {"base_size": 640, "image_size": 640},
    "base": {"base_size": 1024, "image_size": 640},
    "large": {"base_size": 1280, "image_size": 640},
}

class OCRResponse(BaseModel):
    success: bool
    content: str
    processing_time_ms: int
    page_number: Optional[int] = None
    error: Optional[str] = None

class PDFResponse(BaseModel):
    success: bool
    total_pages: int
    processed_pages: int
    content: str
    pages: list[OCRResponse]
    processing_time_ms: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None

def process_image(image, mode="document", resolution="base", page_number=None):
    start = time.time()
    
    try:
        prompt = PROMPTS.get(mode, PROMPTS["document"])
        res = RESOLUTIONS.get(resolution, RESOLUTIONS["base"])
        
        temp_path = f"/tmp/ocr_{time.time_ns()}.png"
        image.save(temp_path, "PNG")
        
        try:
            result = ocr_model.model.infer(
                ocr_model.tokenizer,
                prompt=prompt,
                image_file=temp_path,
                base_size=res["base_size"],
                image_size=res["image_size"],
                crop_mode=True,
                save_results=False,
                test_compress=False
            )
            
            content = result.get("text", str(result)) if isinstance(result, dict) else str(result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return OCRResponse(
            success=True,
            content=content,
            processing_time_ms=int((time.time() - start) * 1000),
            page_number=page_number
        )
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return OCRResponse(
            success=False,
            content="",
            processing_time_ms=int((time.time() - start) * 1000),
            page_number=page_number,
            error=str(e)
        )

@asynccontextmanager
async def lifespan(app):
    ocr_model.load()
    yield
    ocr_model.unload()

app = FastAPI(title="DeepSeek PDF OCR API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return HealthResponse(
        status="healthy" if ocr_model.loaded else "loading",
        model_loaded=ocr_model.loaded,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name
    )

@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    mode: str = Form(default="document"),
    resolution: str = Form(default="base")
):
    if not ocr_model.loaded:
        raise HTTPException(503, "Model not loaded")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return process_image(image, mode, resolution)

@app.post("/ocr/pdf", response_model=PDFResponse)
async def ocr_pdf(
    file: UploadFile = File(...),
    mode: str = Form(default="document"),
    resolution: str = Form(default="base"),
    start_page: int = Form(default=1),
    end_page: Optional[int] = Form(default=None)
):
    if not ocr_model.loaded:
        raise HTTPException(503, "Model not loaded")
    
    start = time.time()
    pdf_bytes = await file.read()
    
    images = pdf2image.convert_from_bytes(pdf_bytes, dpi=PDF_DPI)
    total_pages = len(images)
    
    start_idx = start_page - 1
    end_idx = end_page if end_page else total_pages
    images_to_process = images[start_idx:end_idx]
    
    page_results = []
    combined = []
    
    for i, img in enumerate(images_to_process):
        page_num = start_idx + i + 1
        logger.info(f"Processing page {page_num}/{total_pages}")
        
        result = process_image(img, mode, resolution, page_num)
        page_results.append(result)
        
        if result.success:
            combined.append(f"\n<!-- Page {page_num} -->\n")
            combined.append(result.content)
    
    return PDFResponse(
        success=any(r.success for r in page_results),
        total_pages=total_pages,
        processed_pages=sum(1 for r in page_results if r.success),
        content="\n\n".join(combined),
        pages=page_results,
        processing_time_ms=int((time.time() - start) * 1000)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
