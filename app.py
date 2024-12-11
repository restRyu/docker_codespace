import os
from flask import Flask, request, render_template, send_from_directory
from model.sam_model import SAMModel
import matplotlib.pyplot as plt

app = Flask(__name__)

# Inisialisasi model SAM
checkpoint_path = "models/sam_vit_b_01ec64.pth"
sam_model = SAMModel(checkpoint_path)

# Halaman utama dan upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menerima gambar dan memprosesnya
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    if file and (file.filename.endswith('.tif') or file.filename.endswith('.tiff')):
        filename = os.path.join('static/images', file.filename)
        file.save(filename)
    
        # Segmentasi gambar menggunakan model SAM
        masks = sam_model.segment_image(filename)
        
        # Mendapatkan latitude dan longitude dari titik tengah gambar
        image = sam_model.read_multiband_tiff(filename)
        row, col = image.shape[0] // 2, image.shape[1] // 2
        lat, lon = sam_model.get_lat_lon(filename, row, col)

        # Menyimpan hasil segmentasi
        result_image_path = os.path.join('static/results', f"segmented_{file.filename}")
        plt.imshow(masks[0], alpha=0.5, cmap='viridis')
        plt.savefig(result_image_path)
        
        # Mengirimkan latitude, longitude, dan hasil segmentasi ke template
        return render_template('index.html', original_image=file.filename, result_image=f"segmented_{file.filename}", latitude=lat, longitude=lon)
    
    return "Invalid file format", 400

# Menyajikan gambar statis
@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)
