"""
DeepSeek PDF OCR - FastAPI Server
=================================
Provides REST API for PDF/image OCR using DeepSeek-OCR model.

Endpoints:
    POST /ocr/image     - OCR a single image
    POST /ocr/pdf       - OCR an entire PDF
    GET  /health        - Health check
    GET  /info          - Model and server info
"""

import os
import io
import base64
import time
import logging
from pathlib import Path
from typing import Optional, Literal
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import pdf2image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Model
    model_path: str = Field(default="/workspace/models/deepseek-ocr")
    model_dtype: str = Field(default="bfloat16")
    max_tokens: int = Field(default=8192)
    default_resolution: str = Field(default="base")
    default_crop_mode: bool = Field(default=True)
    
    # PDF Processing
    pdf_dpi: int = Field(default=300)
    max_pages_per_request: int = Field(default=100)
    page_timeout: int = Field(default=120)
    
    # Output
    default_output_format: str = Field(default="markdown")
    include_confidence: bool = Field(default=True)
    include_page_numbers: bool = Field(default=True)
    
    # Server
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepseek-ocr")

# -----------------------------------------------------------------------------
# Global Model Instance
# -----------------------------------------------------------------------------

class OCRModel:
    """Singleton wrapper for DeepSeek-OCR model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.loaded = False
    
    def load(self):
        """Load model into GPU memory."""
        if self.loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading model from {settings.model_path}...")
        start_time = time.time()
        
        from transformers import AutoModel, AutoTokenizer
        
        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(settings.model_dtype, torch.bfloat16)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModel.from_pretrained(
            settings.model_path,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation='flash_attention_2'
        )
        
        self.model = self.model.eval().cuda().to(dtype)
        self.device = next(self.model.parameters()).device
        self.loaded = True
        
        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s on {self.device}")
    
    def unload(self):
        """Unload model from GPU memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.loaded = False
            logger.info("Model unloaded")


# Global model instance
ocr_model = OCRModel()

# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------

PROMPTS = {
    "document": "<image>\n<|grounding|>Convert the document to markdown.",
    "free": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
    "table": "<image>\n<|grounding|>Convert the document to markdown.",  # Tables use document mode
    "describe": "<image>\nDescribe this image in detail.",
}

# Resolution settings (image_size parameter)
RESOLUTION_MAP = {
    "tiny": {"base_size": 512, "image_size": 512},
    "small": {"base_size": 640, "image_size": 640},
    "base": {"base_size": 1024, "image_size": 640},
    "large": {"base_size": 1280, "image_size": 640},
    "gundam": {"base_size": 1024, "image_size": 640},  # Dynamic resolution
}

# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------

class OCRRequest(BaseModel):
    """Request model for OCR endpoint (when using JSON body)."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    mode: Literal["document", "free", "figure", "table", "describe"] = Field(
        default="document",
        description="OCR mode"
    )
    resolution: Literal["tiny", "small", "base", "large", "gundam"] = Field(
        default="base",
        description="Resolution setting"
    )
    crop_mode: bool = Field(default=True, description="Enable crop mode for better accuracy")


class OCRResponse(BaseModel):
    """Response model for OCR endpoint."""
    success: bool
    content: str = Field(description="Extracted text in markdown format")
    mode: str
    resolution: str
    processing_time_ms: int
    page_number: Optional[int] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class PDFResponse(BaseModel):
    """Response model for PDF OCR endpoint."""
    success: bool
    total_pages: int
    processed_pages: int
    content: str = Field(description="Combined markdown for all pages")
    pages: list[OCRResponse] = Field(description="Per-page results")
    processing_time_ms: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_free_gb: Optional[float] = None


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Starting up...")
    ocr_model.load()
    yield
    logger.info("Shutting down...")
    ocr_model.unload()


app = FastAPI(
    title="DeepSeek PDF OCR API",
    description="Convert PDFs and images to accessible markdown using DeepSeek-OCR",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def process_image(
    image: Image.Image,
    mode: str = "document",
    resolution: str = "base",
    crop_mode: bool = True,
    page_number: Optional[int] = None
) -> OCRResponse:
    """Process a single image through the OCR model."""
    
    start_time = time.time()
    
    try:
        # Get prompt and resolution settings
        prompt = PROMPTS.get(mode, PROMPTS["document"])
        res_settings = RESOLUTION_MAP.get(resolution, RESOLUTION_MAP["base"])
        
        # Save image to temp file (model expects file path)
        temp_path = f"/tmp/ocr_temp_{time.time_ns()}.png"
        image.save(temp_path, "PNG")
        
        try:
            # Run inference
            result = ocr_model.model.infer(
                ocr_model.tokenizer,
                prompt=prompt,
                image_file=temp_path,
                base_size=res_settings["base_size"],
                image_size=res_settings["image_size"],
                crop_mode=crop_mode,
                save_results=False,
                test_compress=False
            )
            
            # Extract text from result
            if isinstance(result, dict):
                content = result.get("text", result.get("content", str(result)))
            else:
                content = str(result)
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return OCRResponse(
            success=True,
            content=content,
            mode=mode,
            resolution=resolution,
            processing_time_ms=elapsed_ms,
            page_number=page_number
        )
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}", exc_info=True)
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return OCRResponse(
            success=False,
            content="",
            mode=mode,
            resolution=resolution,
            processing_time_ms=elapsed_ms,
            page_number=page_number,
            error=str(e)
        )


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    """Convert PDF to list of PIL Images."""
    return pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server and model health."""
    gpu_name = None
    gpu_memory_free = None
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_free = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9 
            - torch.cuda.memory_allocated(0) / 1e9, 
            2
        )
    
    return HealthResponse(
        status="healthy" if ocr_model.loaded else "model_not_loaded",
        model_loaded=ocr_model.loaded,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_free_gb=gpu_memory_free
    )


@app.get("/info")
async def server_info():
    """Get server and model information."""
    return {
        "model_path": settings.model_path,
        "default_resolution": settings.default_resolution,
        "max_tokens": settings.max_tokens,
        "pdf_dpi": settings.pdf_dpi,
        "max_pages_per_request": settings.max_pages_per_request,
        "available_modes": list(PROMPTS.keys()),
        "available_resolutions": list(RESOLUTION_MAP.keys()),
    }


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    mode: str = Form(default="document"),
    resolution: str = Form(default="base"),
    crop_mode: bool = Form(default=True)
):
    """
    OCR a single image file.
    
    - **file**: Image file (PNG, JPG, etc.)
    - **mode**: OCR mode (document, free, figure, table, describe)
    - **resolution**: Resolution setting (tiny, small, base, large, gundam)
    - **crop_mode**: Enable crop mode for better accuracy
    """
    if not ocr_model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate inputs
    if mode not in PROMPTS:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    if resolution not in RESOLUTION_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid resolution: {resolution}")
    
    # Read and process image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    return process_image(image, mode, resolution, crop_mode)


@app.post("/ocr/image/base64", response_model=OCRResponse)
async def ocr_image_base64(request: OCRRequest):
    """
    OCR an image from base64 encoded string.
    
    Useful for programmatic API access without multipart form data.
    """
    if not ocr_model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    
    return process_image(image, request.mode, request.resolution, request.crop_mode)


@app.post("/ocr/pdf", response_model=PDFResponse)
async def ocr_pdf(
    file: UploadFile = File(...),
    mode: str = Form(default="document"),
    resolution: str = Form(default="base"),
    crop_mode: bool = Form(default=True),
    start_page: int = Form(default=1, ge=1),
    end_page: Optional[int] = Form(default=None)
):
    """
    OCR an entire PDF document.
    
    - **file**: PDF file
    - **mode**: OCR mode for all pages
    - **resolution**: Resolution setting
    - **crop_mode**: Enable crop mode
    - **start_page**: First page to process (1-indexed)
    - **end_page**: Last page to process (None = all pages)
    
    Returns combined markdown and per-page results.
    """
    if not ocr_model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Read PDF
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")
    
    # Convert to images
    try:
        logger.info(f"Converting PDF to images at {settings.pdf_dpi} DPI...")
        images = pdf_to_images(pdf_bytes, dpi=settings.pdf_dpi)
        total_pages = len(images)
        logger.info(f"PDF has {total_pages} pages")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert PDF: {e}")
    
    # Apply page range
    start_idx = start_page - 1
    end_idx = end_page if end_page else total_pages
    
    if start_idx >= total_pages:
        raise HTTPException(status_code=400, detail=f"start_page {start_page} exceeds PDF pages ({total_pages})")
    
    # Limit pages
    if settings.max_pages_per_request > 0:
        end_idx = min(end_idx, start_idx + settings.max_pages_per_request)
    
    images_to_process = images[start_idx:end_idx]
    
    # Process each page
    page_results = []
    combined_content = []
    
    for i, image in enumerate(images_to_process):
        page_num = start_idx + i + 1
        logger.info(f"Processing page {page_num}/{total_pages}...")
        
        result = process_image(
            image, 
            mode=mode, 
            resolution=resolution, 
            crop_mode=crop_mode,
            page_number=page_num
        )
        page_results.append(result)
        
        if result.success and result.content:
            # Add page marker and content
            if settings.include_page_numbers:
                combined_content.append(f"\n<!-- Page {page_num} -->\n")
            combined_content.append(result.content)
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    processed_count = sum(1 for r in page_results if r.success)
    
    return PDFResponse(
        success=processed_count > 0,
        total_pages=total_pages,
        processed_pages=processed_count,
        content="\n\n".join(combined_content),
        pages=page_results,
        processing_time_ms=elapsed_ms
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False
    )
