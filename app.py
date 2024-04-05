from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

app = Flask(__name__)

def dark_channel_prior_dehaze(image, index=2.0):
    normalized_image = image / 255.0
    index_corrected = np.power(normalized_image, index) * 255.0
    threshold = 230
    index_corrected[index_corrected > threshold] = threshold
    return index_corrected.astype(np.uint8)

def cnn_dehaze(image, index=2.2):
    index_corrected = np.power(image / 255.0, index) * 255.0
    return index_corrected.astype(np.uint8)

def calculate_metrics(original_image, dehazed_image):
    psnr = peak_signal_noise_ratio(original_image, dehazed_image)
    mse = mean_squared_error(original_image, dehazed_image)
    mae = np.mean(np.abs(original_image - dehazed_image))
    rmse = np.sqrt(mse)
    return psnr, mae, rmse

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file from the request
        file = request.files['file']
        if file:
            # Read the image file and convert it to a numpy array
            image = Image.open(file).convert('RGB')
            image_array = np.array(image)

            # Dehaze the image using DCP and CNN algorithms
            dehazed_image_dcp = dark_channel_prior_dehaze(image_array)
            dehazed_image_cnn = cnn_dehaze(image_array)

            # Calculate metrics
            psnr_dcp, mae_dcp, rmse_dcp = calculate_metrics(image_array, dehazed_image_dcp)
            psnr_cnn, mae_cnn, rmse_cnn = calculate_metrics(image_array, dehazed_image_cnn)

            # Convert the dehazed images to base64 for display
            buffered_dcp = BytesIO()
            Image.fromarray(dehazed_image_dcp).save(buffered_dcp, format="JPEG")
            image_stream_dcp = base64.b64encode(buffered_dcp.getvalue()).decode('utf-8')

            buffered_cnn = BytesIO()
            Image.fromarray(dehazed_image_cnn).save(buffered_cnn, format="JPEG")
            image_stream_cnn = base64.b64encode(buffered_cnn.getvalue()).decode('utf-8')

            return render_template('index.html', image_stream_dcp=image_stream_dcp, image_stream_cnn=image_stream_cnn,
                                   psnr_dcp=psnr_dcp, mae_dcp=mae_dcp, rmse_dcp=rmse_dcp,
                                   psnr_cnn=psnr_cnn, mae_cnn=mae_cnn, rmse_cnn=rmse_cnn)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
