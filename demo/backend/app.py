from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from api.retrieve import handle_retrieve
import os

app = Flask(__name__)

# Enable CORS for all routes with explicit configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE", "HEAD"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False
    }
})

# Add after_request hook to ensure CORS headers are always set
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin:
        response.headers['Access-Control-Allow-Origin'] = origin
    else:
        response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE, HEAD'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Credentials'] = 'false'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

# Create temp_videos directory if it doesn't exist
TEMP_VIDEO_DIR = 'temp_videos'
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# Serve static files from temp_videos directory
@app.route('/temp_videos/<path:filename>')
def serve_video(filename):
    """Serve video files from temp_videos directory"""
    return send_from_directory(TEMP_VIDEO_DIR, filename)

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    """API root endpoint"""
    return jsonify({
        "message": "Volleyball Highlights API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "retrieve": "/retrieve (POST)",
            "test": "/test"
        }
    }), 200

# Add health check endpoint
@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

# Add a simple test endpoint for CORS
@app.route('/test', methods=['GET', 'POST', 'OPTIONS'])
def test_cors():
    return jsonify({"message": "CORS is working", "origin": request.headers.get('Origin', 'No origin')}), 200

# Add retrieve endpoint
@app.route('/retrieve', methods=['POST', 'OPTIONS'])
def retrieve():
    return handle_retrieve()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

