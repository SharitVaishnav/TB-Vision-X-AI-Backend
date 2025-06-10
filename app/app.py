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
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure TensorFlow for better performance
tf.config.optimizer.set_jit(True)  # Enable XLA optimization
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": True,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": True
})

# Load DenseNet model with memory optimization
try:
    logger.info("Loading DenseNet model from local directory...")
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    gc.collect()

    # Load model from local directory
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'tb_detector_model_densenet.h5')
    model = tf.keras.models.load_model(model_path, compile=False)

    # Define inference function
    @tf.function
    def predict_function(x):
        return model(x, training=False)

    logger.info("DenseNet model loaded successfully")

except Exception as e:
    logger.error(f"Error loading DenseNet model: {str(e)}")
    raise

def preprocess_image(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Resize to match model's expected input shape
        img = cv2.resize(img, (256, 256))  # Changed back to 256x256
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def generate_gradcam(model, image_array, interpolant=0.5):
    try:
        original_img = image_array[0]
        
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model.")

        # Create a model that outputs the last conv layer and the prediction
        gradient_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv2d_out, prediction = gradient_model(image_array)
            loss = prediction[:, 0]

        # Calculate gradients
        gradients = tape.gradient(loss, conv2d_out)
        output = conv2d_out[0]
        weights = tf.reduce_mean(gradients[0], axis=(0, 1))

        # Generate heatmap
        activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
        for idx, weight in enumerate(weights):
            activation_map += weight * output[:, :, idx]

        # Process heatmap
        activation_map = cv2.resize(activation_map.numpy(), (original_img.shape[1], original_img.shape[0]))
        activation_map = np.maximum(activation_map, 0)
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        activation_map = np.uint8(255 * activation_map)

        # Create visualization
        heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
        original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
        cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant))
    except Exception as e:
        logger.error(f"Error in GradCAM generation: {str(e)}")
        raise

def generate_lime_explanation(model, image_array):
    try:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_array[0].astype('double'),
            predict_function,  # Use the optimized prediction function
            top_labels=1,
            hide_color=0,
            num_samples=50  # Reduced from 100 to make it faster
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=3,  # Reduced from 5 to make it faster
            hide_rest=True
        )
        
        return temp, mask
    except Exception as e:
        logger.error(f"Error in LIME generation: {str(e)}")
        raise

def save_plot_to_base64(fig):
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=72)  # Reduced DPI for faster processing
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    except Exception as e:
        logger.error(f"Error in saving plot: {str(e)}")
        raise

def image_to_base64(img_array):
    try:
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)  # Added optimize=True for faster processing
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    except Exception as e:
        logger.error(f"Error in image conversion: {str(e)}")
        raise

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        logger.info("Received prediction request")
        
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
            
        logger.info(f"Processing image: {file.filename}")
        image_bytes = file.read()
        
        try:
            logger.info("Preprocessing image")
            image_array = preprocess_image(image_bytes)
            
            logger.info("Making prediction")
            prediction = predict_function(image_array)[0]  # Use the optimized prediction function
            
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
            
            logger.info(f"Prediction result: {result}")
            
            # Clear memory after prediction
            tf.keras.backend.clear_session()
            gc.collect()
            
            try:
                logger.info("Generating GradCAM")
                gradcam = generate_gradcam(model, image_array)
                result['gradcam'] = image_to_base64(gradcam)
                logger.info("GradCAM generated successfully")
            except Exception as e:
                logger.error(f"Error generating GradCAM: {str(e)}")
                result['gradcam'] = None
            
            try:
                logger.info("Generating LIME explanation")
                lime_temp, lime_mask = generate_lime_explanation(model, image_array)
                plt.figure(figsize=(8, 8))  # Reduced figure size
                plt.imshow(lime_temp)
                plt.imshow(lime_mask, alpha=0.5)
                plt.axis('off')
                result['lime'] = save_plot_to_base64(plt.gcf())
                logger.info("LIME explanation generated successfully")
            except Exception as e:
                logger.error(f"Error generating LIME: {str(e)}")
                result['lime'] = None
            
            # Final memory cleanup
            tf.keras.backend.clear_session()
            gc.collect()
            
            logger.info("Successfully completed prediction request")
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
