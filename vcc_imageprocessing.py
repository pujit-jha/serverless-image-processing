from flask import Flask, request, jsonify, send_file
import os
import uuid
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from rembg import remove
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

# Install required dependencies
# !pip install rembg
# !pip install rembg onnxruntime
# !pip install facenet-pytorch
# !pip install torch torchvision matplotlib

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MTCNN detector for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained DeepLabV3 model for segmentation
segmentation_model = deeplabv3_resnet101(pretrained=True).eval()

def generate_blip_caption(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def high_contrast_lattice_sketch_denoised(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    inverted = 255 - sharpened
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)
    dodge = cv2.divide(sharpened, 255 - blur, scale=256)
    sketch = cv2.equalizeHist(dodge)
    sketch = cv2.fastNlMeansDenoising(sketch, h=75, templateWindowSize=7, searchWindowSize=21)
    pil_sketch = Image.fromarray(sketch)
    return pil_sketch

def is_mostly_face(sketch_path, face_area_threshold=0.25, visualize=False):
    img = Image.open(sketch_path).convert("RGB")
    width, height = img.size
    image_area = width * height
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return False
    face_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    max_face_area = max(face_areas)
    face_ratio = max_face_area / image_area
    if visualize:
        plt.imshow(img)
        ax = plt.gca()
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
        plt.axis('off')
        plt.show()
    return face_ratio >= face_area_threshold

def segment_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.Resize(520), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    def decode_segmap(mask):
        colors = np.array([
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
            (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
            (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
            (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
        ], dtype=np.uint8)
        r = np.zeros_like(mask, dtype=np.uint8)
        g = np.zeros_like(mask, dtype=np.uint8)
        b = np.zeros_like(mask, dtype=np.uint8)
        for l in range(len(colors)):
            idx = mask == l
            r[idx] = colors[l][0]
            g[idx] = colors[l][1]
            b[idx] = colors[l][2]
        return np.stack([r, g, b], axis=2)
    
    segmentation_colored = decode_segmap(output_predictions)
    segmentation_image = Image.fromarray(segmentation_colored).resize(img.size)
    return Image.blend(img, segmentation_image, alpha=0.6)

def remove_background(image):
    return remove(image)

@app.route('/')
def index():
    return '''
    <h1>Image Processing API</h1>
    <p>POST to /process with image and task type (caption, sketch, background, segmentation, face)</p>
    '''

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files or 'task' not in request.form:
        return jsonify({'error': 'Missing image or task'}), 400

    image_file = request.files['image']
    task = request.form['task']

    img_id = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.png")
    image_file.save(img_path)

    result = None
    if task == 'caption':
        caption = generate_blip_caption(img_path)
        return jsonify({'caption': caption})
    elif task == 'sketch':
        output = high_contrast_lattice_sketch_denoised(img_path)
    elif task == 'background':
        image = Image.open(img_path)
        output = remove_background(image)
    elif task == 'segmentation':
        output = segment_image(img_path)
    elif task == 'face':
        image = Image.open(img_path)
        if is_mostly_face(img_path):
            output = image
        else:
            return jsonify({'error': 'No face detected or face not prominent enough'}), 400
    else:
        return jsonify({'error': 'Invalid task'}), 400

    # Save output image and return it
    output_buffer = io.BytesIO()
    output.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    return send_file(output_buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
