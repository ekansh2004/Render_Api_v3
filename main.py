import os
import io
import re
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Model Loading ---
app = FastAPI(title="Concrete Defect Analysis API")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Load the TFLite Model using TensorFlow's Interpreter ---
try:
    interpreter = tf.lite.Interpreter(model_path="spalling_model.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model 'model.tflite' loaded successfully using TensorFlow.")
except Exception as e:
    raise RuntimeError(f"Error loading TFLite model: {e}")

# Configure Gemini
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_vision_model = genai.GenerativeModel('gemini-2.5-flash')

# --- Stage 1: Local TFLite Model for Binary Classification ---
def run_local_binary_classifier(image_bytes: bytes) -> str:
    """
    Runs the local TFLite model to classify an image.
    """
    try:
        # Preprocess the image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array /= 255.0
        
        # Add a batch dimension to match model's input shape
        input_data = np.expand_dims(img_array, axis=0)
        
        # Set the value of the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run the inference
        interpreter.invoke()
        
        # Get the prediction score from the output tensor
        prediction_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        if prediction_score > 0.5:
            print(f"Local model: Defect Detected (Score: {prediction_score:.2f})")
            return "Defect Detected"
        else:
            print(f"Local model: No Defect (Score: {prediction_score:.2f})")
            return "No Defect"
            
    except Exception as e:
        print(f"Error during TFLite model prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing image with local model.")

# --- Helper Function to Parse Gemini's Response ---
def parse_gemini_response(text: str) -> dict:
    """
    Parses the structured text from Gemini into a clean dictionary.
    Example input: 'Defect Type : Spalling\nSeverity: High'
    """
    try:
        defect_type = re.search(r"Defect Type\s*:\s*(.*)", text, re.IGNORECASE).group(1).strip()
        severity = re.search(r"Severity\s*:\s*(.*)", text, re.IGNORECASE).group(1).strip()
        return {"Defect_Type": defect_type, "Severity": severity}
    except (AttributeError, IndexError):
        # Fallback if the parsing fails for any reason
        print(f"Warning: Could not parse Gemini response: '{text}'")
        return {"Defect_Type": "Unspecified Defect", "Severity": "Unknown"}

# --- Stage 2: Gemini Vision for Detailed Analysis ---
def get_gemini_vision_analysis(image_bytes: bytes) -> dict:
    """
    Sends the image to Gemini and returns a parsed dictionary.
    """
    try:
        image_for_gemini = Image.open(io.BytesIO(image_bytes))
        
        # This prompt asks for the specific key-value format
        prompt = """
        Analyze the provided image of a concrete surface.
        A preliminary check has indicated a defect is present.
        
        Please provide the following in this format only:
        1.  **Defect Type:** Identify the specific type of defect out of Spalling, Honeycomb, Voids, Cracks
        2.  **Severity:** Assess the severity on a scale of Low, Medium, or High.
        
        Give one word answers like
        Defect Type : Spalling
        Severity: High 
        This how you response should be
        """
        
        response = gemini_vision_model.generate_content([prompt, image_for_gemini])
        
        # Parse the raw text response into a clean dictionary
        return parse_gemini_response(response.text)

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analysis from Gemini Vision API.")

# --- API Endpoint ---
@app.post("/analyze-concrete-image/")
async def analyze_concrete_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image and returns the standardized defect analysis JSON.
    """
    image_bytes = await file.read()
    
    local_result = run_local_binary_classifier(image_bytes)
    
    if local_result == "Defect Detected":
        print("Passing to Gemini for detailed analysis...")
        final_analysis = get_gemini_vision_analysis(image_bytes)
        return final_analysis # Returns the parsed {'Defect_Type': '...', 'Severity': '...'}
    else:
        # Return the desired standardized format for "No Defect"
        return {'Defect_Type': 'No Defect', 'Severity': 'NA'}