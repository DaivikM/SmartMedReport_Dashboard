from fastapi import FastAPI, File, UploadFile, HTTPException
from app.medical_analyzer import MedicalReportAnalyzer
from app.config import GROQ_API_KEY
import uvicorn
import os
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Medical Report API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://127.0.0.1:5500"],  # Update with your frontend's URL
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Initialize analyzer
analyzer = MedicalReportAnalyzer(GROQ_API_KEY)

# Define file storage paths
UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_files(medical_file: Optional[UploadFile], policy_file: Optional[UploadFile]) -> tuple[Optional[str], Optional[str]]:
    """Save the uploaded files and return their paths."""
    try:
        # Remove existing files in the upload folder
        for file_name in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        medical_path = None
        policy_path = None

        # Save medical report if provided
        if medical_file:
            medical_path = os.path.join(UPLOAD_FOLDER, "latest_medical.pdf")
            with open(medical_path, "wb") as f:
                f.write(medical_file.file.read())

        # Save policy document if provided
        if policy_file:
            policy_path = os.path.join(UPLOAD_FOLDER, "latest_policy.pdf")
            with open(policy_path, "wb") as f:
                f.write(policy_file.file.read())

        return medical_path, policy_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save files: {str(e)}")


from typing import Optional

def get_stored_files() -> tuple[Optional[bytes], Optional[bytes]]:
    """Retrieve the stored files."""
    try:
        medical_path = os.path.join(UPLOAD_FOLDER, "latest_medical.pdf")
        policy_path = os.path.join(UPLOAD_FOLDER, "latest_policy.pdf")

        medical_content = None
        policy_content = None

        # Check and read the medical file if it exists
        if os.path.exists(medical_path):
            with open(medical_path, "rb") as f:
                medical_content = f.read()

        # Check and read the policy file if it exists
        if os.path.exists(policy_path):
            with open(policy_path, "rb") as f:
                policy_content = f.read()

        # Raise an error if both files are missing
        if not medical_content and not policy_content:
            raise HTTPException(
                status_code=404, 
                detail="No stored files found. Please upload files first using the /upload endpoint."
            )

        return medical_content, policy_content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read stored files: {str(e)}")


@app.post("/upload")
async def upload_documents(
    medical_file: UploadFile = File(None),
    policy_file: UploadFile = File(None)
):
    """Upload and store medical report and policy documents."""
    try:
        # Ensure at least one file is uploaded
        if not medical_file and not policy_file:
            raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

        # Save the files (pass None for missing files)
        medical_path, policy_path = save_files(medical_file, policy_file)

        # Construct the response dynamically
        response = {"message": "Files uploaded successfully"}
        if medical_path:
            response["stored_medical_path"] = medical_path
        if policy_path:
            response["stored_policy_path"] = policy_path

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-stored")
async def analyze_stored_report():
    """Analyze the stored medical report."""
    try:
        # Get stored files
        medical_content, _ = get_stored_files()

        # Analyze the report
        relevance_data, cost_data, fraud_analysis = analyzer.analyze_report(medical_content)

        return {
            "relevance_data": relevance_data,
            "cost_data": cost_data,
            "fraud_analysis": fraud_analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-stored")
async def verify_stored_documents():
    """Verify the stored medical report against the stored policy document."""
    try:
        # Get stored files
        medical_content, policy_content = get_stored_files()

        # Perform verification
        verification_results = analyzer.verify_policy_documents(
            medical_content,
            policy_content
        )

        return verification_results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/predict-days")
async def predict_daywise_tests():
    """Predict day-by-day test requirements and conditions from stored medical report."""
    try:
        # Retrieve stored medical file
        medical_path = os.path.join(UPLOAD_FOLDER, "latest_medical.pdf")
        if not os.path.exists(medical_path):
            raise HTTPException(status_code=404, detail="No stored medical report found. Please upload a report first.")

        # Extract content and analyze
        with open(medical_path, "rb") as f:
            medical_content = f.read()

        # Extract text from PDF
        text_content = analyzer.extract_text_from_pdf(medical_content)

        # Perform day-wise predictions
        daywise_predictions = analyzer.parse_and_predict_daywise(text_content)

        return {
            "message": "Day-wise predictions completed successfully",
            "daywise_predictions": daywise_predictions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
