from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, 
    static_folder='../static',
    template_folder='../templates'
)

# Configure CORS and SocketIO
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/api/start_journey', methods=['POST'])
def start_journey():
    try:
        data = request.get_json()
        destination = data.get('destination')
        if not destination:
            return jsonify({'error': 'Destination is required'}), 400
        
        return jsonify({'status': 'success', 'message': 'Journey started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start server
if __name__ == '__main__':
    print("Starting Rover Control Server...")
    print("Access the interface at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
import cv2
import threading
import time
import json
from typing import Optional
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

class RoverController:
    def __init__(self):
        self.current_destination = None
        self.journey_status = "idle"
        self.progress = 0
        self.camera = None
        self.is_streaming = False
        self._stream_thread = None

    def start_journey(self, destination: str) -> bool:
        if self.journey_status != "idle":
            return False
        
        self.current_destination = destination
        self.journey_status = "in_progress"
        self.progress = 0
        self.start_camera()
        
        # Start progress simulation in a separate thread
        threading.Thread(target=self._simulate_journey).start()
        return True

    def _simulate_journey(self):
        """Simulate journey progress - Replace with actual rover control logic"""
        while self.progress < 100 and self.journey_status == "in_progress":
            time.sleep(1)  # Update every second
            self.progress += 2  # Simulate 2% progress per second
            socketio.emit('journey_progress', {
                'progress': self.progress,
                'status': self.journey_status,
                'destination': self.current_destination
            })
        
        if self.progress >= 100:
            self.journey_status = "completed"
            socketio.emit('journey_completed', {
                'destination': self.current_destination,
                'time_taken': 50  # Replace with actual time calculation
            })
            self.stop_camera()

    def start_camera(self):
        """Initialize and start camera feed"""
        if not self.camera:
            self.camera = cv2.VideoCapture(0)  # Use 0 for default camera, adjust for rover's camera
        self.is_streaming = True
        if not self._stream_thread:
            self._stream_thread = threading.Thread(target=self._stream_camera)
            self._stream_thread.start()

    def stop_camera(self):
        """Stop camera streaming"""
        self.is_streaming = False
        if self.camera:
            self.camera.release()
            self.camera = None

    def _stream_camera(self):
        """Stream camera feed through websocket"""
        while self.is_streaming and self.camera:
            success, frame = self.camera.read()
            if success:
                # Convert frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    # Send frame through websocket
                    socketio.emit('camera_frame', {
                        'frame': jpeg.tobytes().decode('latin1')
                    })
            time.sleep(0.1)  # Limit frame rate

rover = RoverController()

@app.route('/api/start_journey', methods=['POST'])
def start_journey():
    data = request.json
    destination = data.get('destination')
    if not destination:
        return jsonify({'error': 'Destination is required'}), 400
    
    success = rover.start_journey(destination)
    if success:
        return jsonify({'status': 'Journey started', 'destination': destination})
    return jsonify({'error': 'Journey already in progress'}), 400

@socketio.on('connect')
def handle_connect():
    if rover.journey_status == "in_progress":
        emit('journey_progress', {
            'progress': rover.progress,
            'status': rover.journey_status,
            'destination': rover.current_destination
        })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
