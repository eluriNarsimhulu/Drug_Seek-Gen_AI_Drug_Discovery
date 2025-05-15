from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import os
import uuid
import shutil
import cv2
import numpy as np
import nibabel as nib
from pathlib import Path
import tensorflow as tf
from patchify import patchify
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = BASE_DIR / "temp_nii_uploads"
SLICES_FOLDER = BASE_DIR / "temp_nii_slices"
OUTPUT_MASKS_FOLDER = BASE_DIR / "temp_nii_masks"
TEMP_FILES_FOLDER = BASE_DIR / "temp_files"  # New temporary folder for viewer files

# Create directories
for folder in [STATIC_DIR, UPLOAD_FOLDER, SLICES_FOLDER, OUTPUT_MASKS_FOLDER, TEMP_FILES_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/viewer-static", StaticFiles(directory="static"), name="viewer-static")
app.mount("/temp", StaticFiles(directory=str(TEMP_FILES_FOLDER)), name="temp")  # Mount temp directory

# Setup templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Model configuration
MODEL_PATHS = {
    "lung": str(BASE_DIR / "model (4).keras")
}

# Configuration dictionary
cf = {
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "flattened_patch_dim": 16 * 16 * 3,
}

# Dictionary to track temporary files with their creation timestamps
temp_files = {}

# Configure file expiration time (in seconds)
FILE_EXPIRATION_TIME = 3600  # 1 hour

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def load_model(model_type: str):
    model_path = MODEL_PATHS.get(model_type)
    if not model_path or not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"

    try:
        model = tf.keras.models.load_model(model_path, 
                                         custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

def cleanup_old_files():
    """Remove temporary files that have exceeded their expiration time"""
    current_time = time.time()
    expired_files = []
    
    for file_path, timestamp in list(temp_files.items()):
        if current_time - timestamp > FILE_EXPIRATION_TIME:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                expired_files.append(file_path)
                logger.info(f"Removed expired file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")
    
    # Remove expired files from the tracking dictionary
    for file_path in expired_files:
        temp_files.pop(file_path, None)

def cleanup_temp_directories():
    """Clean temporary directories"""
    for folder in [SLICES_FOLDER, OUTPUT_MASKS_FOLDER]:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(exist_ok=True)

def delayed_file_cleanup(file_paths):
    """Remove specified files after they've been served"""
    time.sleep(FILE_EXPIRATION_TIME)
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Cleaned up temporary file: {path}")
                temp_files.pop(path, None)
        except Exception as e:
            logger.error(f"Error removing file {path}: {e}")

@app.get("/viewer")
async def get_viewer(request: Request, original: str, predicted: str):
    # Clean up old files whenever the viewer is accessed
    cleanup_old_files()
    
    return templates.TemplateResponse("index1.html", {
        "request": request,
        "file1_path": original,
        "file2_path": predicted
    })

@app.post("/api/upload_nii")
async def upload_nii_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.lower().endswith('.nii.gz'):
        raise HTTPException(status_code=400, detail="File must be a .nii.gz file")

    model_type = "lung"
    model, error = load_model(model_type)
    if error:
        raise HTTPException(status_code=500, detail=error)

    # Clean temporary directories at the start
    cleanup_temp_directories()
    
    # Clean up old files
    cleanup_old_files()

    try:
        # Generate unique filenames for temporary files
        input_filename = f"uploaded_{uuid.uuid4().hex}_{file.filename}"
        input_nii_path = UPLOAD_FOLDER / input_filename
        
        # Create temporary files for viewing
        temp_original_filename = f"temp_original_{uuid.uuid4().hex}_{file.filename}"
        temp_original_path = TEMP_FILES_FOLDER / temp_original_filename
        
        temp_predicted_filename = f"temp_predicted_{uuid.uuid4().hex}_{file.filename}"
        temp_predicted_path = TEMP_FILES_FOLDER / temp_predicted_filename

        # Save uploaded file
        with open(input_nii_path, "wb") as f:
            while contents := await file.read(1024 * 1024):
                f.write(contents)

        # Process NIfTI file
        nii_img = nib.load(input_nii_path)
        nii_data = nii_img.get_fdata()
        depth = nii_data.shape[0]

        # Create slices
        for i in range(depth):
            slice_img = nii_data[i, :, :]
            slice_img = np.clip(slice_img, 0, 255).astype(np.uint8)
            slice_resized = cv2.resize(slice_img, (512, 512), interpolation=cv2.INTER_AREA)
            slice_rgb = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2BGR)
            slice_path = SLICES_FOLDER / f"slice_{i:03d}.png"
            cv2.imwrite(str(slice_path), slice_rgb)

        # Predict masks
        image_files = sorted(os.listdir(SLICES_FOLDER))
        for image_name in image_files:
            input_image_path = SLICES_FOLDER / image_name
            image = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            resized = cv2.resize(image, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_LANCZOS4)
            norm = resized / 255.0
            patches = patchify(norm, (cf["patch_size"], cf["patch_size"], cf["num_channels"]), cf["patch_size"])
            patches = patches.reshape(-1, cf["flattened_patch_dim"])
            patches = np.expand_dims(patches, axis=0)
            pred = model.predict(patches, verbose=0)[0]
            pred = (pred * 255).astype(np.uint8)
            if len(pred.shape) == 3 and pred.shape[-1] > 1:
                pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            mask_resized = cv2.resize(pred, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_NEAREST)
            _, mask_thresh = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            output_mask_path = OUTPUT_MASKS_FOLDER / f"mask_{image_name}"
            cv2.imwrite(str(output_mask_path), mask_thresh)

        # Reconstruct NIfTI from masks
        mask_files = sorted(os.listdir(OUTPUT_MASKS_FOLDER), key=lambda x: int("".join(filter(str.isdigit, x))))
        mask_slices = []
        for fname in mask_files:
            path = OUTPUT_MASKS_FOLDER / fname
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_slices.append(resized)

        if not mask_slices:
            raise HTTPException(status_code=500, detail="No masks were generated")

        predicted_volume = np.stack(mask_slices, axis=0)
        nii_pred = nib.Nifti1Image(predicted_volume, affine=np.eye(4))
        
        # Save temporary files for viewer
        shutil.copy(str(input_nii_path), str(temp_original_path))
        nib.save(nii_pred, temp_predicted_path)
        
        # Track temporary files with creation timestamp
        current_time = time.time()
        temp_files[str(temp_original_path)] = current_time
        temp_files[str(temp_predicted_path)] = current_time
        
        # Delete the uploaded file since we've copied it
        os.remove(str(input_nii_path))
        
        # Schedule cleanup task for temporary files
        if background_tasks:
            background_tasks.add_task(
                delayed_file_cleanup, 
                [str(temp_original_path), str(temp_predicted_path)]
            )
        
        # Return the paths to the frontend
        return {
            "success": True,
            "original_path": f"/temp/{temp_original_filename}",
            "predicted_path": f"/temp/{temp_predicted_filename}"
        }

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Always clean up temporary processing directories
        cleanup_temp_directories()

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...), model_type: str = "lung"):
    model, error = load_model(model_type)
    if error:
        raise HTTPException(status_code=500, detail=error)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail=f"Unable to read image: {file.filename}")

        image = cv2.resize(image, (256, 256)) / 255.0
        patches = patchify(image, (16, 16, 3), 16).reshape(1, -1, 16 * 16 * 3)

        pred = model.predict(patches, verbose=0)[0]
        pred_mask = np.where(pred > 0.5, 255, 0).astype(np.uint8)
        pred_mask_resized = pred_mask.reshape((256, 256))

        # Save the mask to temp folder
        mask_filename = f"temp_mask_{uuid.uuid4().hex}_{file.filename}"
        mask_output_path = TEMP_FILES_FOLDER / mask_filename
        
        if cv2.imwrite(str(mask_output_path), pred_mask_resized):
            # Track temporary file
            temp_files[str(mask_output_path)] = time.time()
            
            return {
                "success": True,
                "mask_path": f"/temp/{mask_filename}"
            }

        raise HTTPException(status_code=500, detail="Failed to save prediction mask")

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.on_event("startup")
async def startup_event():
    # Clean all temporary directories on startup
    for folder in [UPLOAD_FOLDER, SLICES_FOLDER, OUTPUT_MASKS_FOLDER, TEMP_FILES_FOLDER]:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    # Clean all temporary directories on shutdown
    for folder in [UPLOAD_FOLDER, SLICES_FOLDER, OUTPUT_MASKS_FOLDER, TEMP_FILES_FOLDER]:
        shutil.rmtree(folder, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)