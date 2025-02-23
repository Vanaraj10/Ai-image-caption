from flask import Flask, request, render_template, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import os

app = Flask(__name__)

# Load AI Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Upload folder setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Process image
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Return caption & image path
    return jsonify({
        "caption": caption,
        "image_url": image_path
    })

if __name__ == "__main__":
    app.run(debug=True)
