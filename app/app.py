import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import cv2
import lime
from lime import lime_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load DenseNet model
try:
    logger.info("Loading DenseNet model...")
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'tb_detector_model_densenet.h5'))
    logger.info("DenseNet model loaded successfully")
except Exception as e:
    logger.error(f"Error loading DenseNet model: {str(e)}")
    raise

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_gradcam(model, image_array, interpolant=0.5):
    original_img = image_array[0]
    
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model.")

    gradient_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv2d_out, prediction = gradient_model(image_array)
        loss = prediction[:, 0]

    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    activation_map = cv2.resize(activation_map.numpy(), (original_img.shape[1], original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = np.uint8(255 * activation_map)

    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant))

def generate_lime_explanation(model, image_array):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_array[0].astype('double'),
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=True
    )
    
    return temp, mask

def save_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def image_to_base64(img_array):
    # Convert numpy array to PIL Image
    img = Image.fromarray(img_array)
    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        image_bytes = file.read()
        
        try:
            image_array = preprocess_image(image_bytes)
            prediction = model.predict(image_array)[0]
            
            if len(prediction) == 1:
                prob = float(prediction[0])
                result = {
                    'prediction': prob,
                    'class': 'Tuberculosis' if prob > 0.5 else 'Normal'
                }
            else:
                prob = float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
                result = {
                    'prediction': prob,
                    'class': 'Tuberculosis' if prob > 0.5 else 'Normal'
                }
            
            try:
                # Generate and encode GradCAM
                gradcam = generate_gradcam(model, image_array)
                result['gradcam'] = image_to_base64(gradcam)
            except Exception as e:
                logger.error(f"Error generating GradCAM: {str(e)}")
                result['gradcam'] = None
            
            try:
                # Generate and encode LIME
                lime_temp, lime_mask = generate_lime_explanation(model, image_array)
                # Create a figure with the LIME visualization
                plt.figure(figsize=(10, 10))
                plt.imshow(lime_temp)
                plt.imshow(lime_mask, alpha=0.5)
                plt.axis('off')
                result['lime'] = save_plot_to_base64(plt.gcf())
            except Exception as e:
                logger.error(f"Error generating LIME: {str(e)}")
                result['lime'] = None
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=debug) 
