from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlmodel import Session
import aiofiles
import nibabel as nib
import shutil
import os
import gc
from src.database.db import get_session, init_db
from src.database import crud, models
from src.core.predict import load_brain, predict_volume

app = FastAPI(title="Liver Segmentation API")

print("Initializing AI Model")
AI_MODEL = load_brain()
print("Server Ready.")


def get_recommended_procedure(pct: float) -> str:
    if pct is None:
        return "Unknown"
    if pct < 1.0:
        return "Observation / Routine Checkup"
    if pct < 30.0:
        return "Surgical Resection"
    if pct < 70.0:
        return "Chemotherapy / TACE"
    return "Transplant Assessment / Palliative"


@app.post("/scans/", response_model=models.Scan)
def create_new_scan(scan: models.Scan, session: Session = Depends(get_session)):
    return crud.create_scan(session=session, scan=scan)


@app.get("/scans/", response_model=list[models.Scan])
def read_all_scans(
    skip: int = 0, limit: int = 10, session: Session = Depends(get_session)
):
    return crud.get_scans(session=session, skip=skip, limit=limit)


@app.get("/scans/{scan_id}", response_model=models.Scan)
def read_single_scan(scan_id: int, session: Session = Depends(get_session)):
    db_scan = crud.get_scan_by_id(session=session, scan_id=scan_id)
    if not db_scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return db_scan


@app.post("/upload/")
async def upload_scan(
    my_upload: UploadFile = File(...), session: Session = Depends(get_session)
):
    init_db()

    upload_dir = "data/scans"
    os.makedirs(upload_dir, exist_ok=True)

    safe_filename = my_upload.filename or "unknown.nii"
    original_file_location = f"{upload_dir}/{safe_filename}"

    async with aiofiles.open(original_file_location, "wb") as out_file:
        while content := await my_upload.read(1024 * 1024):
            await out_file.write(content)

    fixed_filename = f"fixed_{safe_filename}"
    fixed_file_location = f"{upload_dir}/{fixed_filename}"

    try:
        nifti_img = nib.load(original_file_location)

        canonical_img = nib.as_closest_canonical(nifti_img)

        nib.save(canonical_img, fixed_file_location)
        print(f"✅ Sanitized to: {fixed_filename}")

        del nifti_img
        del canonical_img
        gc.collect()

    except Exception as e:
        print(f"⚠️ Orientation fix failed: {e}")
        fixed_file_location = original_file_location


    mask_location = None
    if AI_MODEL:
        try:
            print(f"Processing {fixed_filename}...")
            results = predict_volume(AI_MODEL, fixed_file_location)

            liver_vol = results["liver_volume_cm3"]
            tumor_vol = results["tumor_volume_cm3"]
            tumor_pct = results["tumor_percentage"]
            mask_location = results.get("mask_path")

            status = "Processed"
            procedure = get_recommended_procedure(tumor_pct)

        except Exception as e:
            print(f"AI Prediction Failed: {e}")
            liver_vol, tumor_vol, tumor_pct = 0.0, 0.0, 0.0
            status = "Error"
            procedure = "AI Failed"
    else:
        status = "No Model"
        procedure = "System Offline"
        liver_vol, tumor_vol, tumor_pct = 0.0, 0.0, 0.0

    new_scan = models.Scan(
        patient_id=safe_filename.split(".")[0],
        filename=safe_filename,
        status=status,
        liver_volume_cm3=liver_vol,
        tumor_volume_cm3=tumor_vol,
        tumor_percentage=tumor_pct,
        procedure=procedure,
    )

    saved_scan = crud.create_scan(session=session, scan=new_scan)

    response_data = saved_scan.dict()
    response_data["original_file_path"] = fixed_file_location
    response_data["mask_file_path"] = mask_location if mask_location else ""

    return response_data
