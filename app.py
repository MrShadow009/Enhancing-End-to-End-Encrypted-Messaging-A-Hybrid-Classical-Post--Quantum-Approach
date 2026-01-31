from flask import Flask, render_template, request, jsonify
from passlib.hash import pbkdf2_sha256
import uuid
import json
import time

# Import quantum modules (with error handling for Vercel)
try:
    from quantum_state_manager import QuantumState, QuantumChannel
    from bb84_protocol import BB84Protocol
    from quantum_teleportation import QuantumTeleportation, QuantumMessagePacket
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Quantum modules not available: {e}")
    QUANTUM_AVAILABLE = False
    # Define dummy classes to prevent crashes
    class QuantumState:
        pass
    class QuantumChannel:
        pass
    class BB84Protocol:
        pass
    class QuantumTeleportation:
        pass
    class QuantumMessagePacket:
        pass

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-a-secure-secret'

# In-memory room store (prototype). Replace with DB for production.
# Structure: rooms[room_id] = {name, algorithm, pass_hash, participants: {username: True}, messages: []}
rooms = {}

@app.route('/')
def index():
    return render_template('index_socketio.html')

# Create a room (HTTP POST)
@app.route('/api/create-room', methods=['POST'])
def create_room():
    data = request.json or {}
    room_name = data.get('room_name')
    room_description = data.get('room_description', '')
    algorithm = data.get('algorithm', 'RSA')
    passphrase = data.get('passphrase')
    creator = data.get('creator', 'unknown')

    if not room_name or not passphrase:
        return jsonify({'error': 'room_name and passphrase required'}), 400

    room_id = str(uuid.uuid4())
    pass_hash = pbkdf2_sha256.hash(passphrase)
    rooms[room_id] = {
        'name': room_name,
        'description': room_description,
        'algorithm': algorithm,
        'pass_hash': pass_hash,
        'creator': creator,
        'participants': {},  # username -> True
        'messages': [],  # list of {'id':, 'username':, 'message':}
        'next_message_id': 0
    }
    return jsonify({'room_id': room_id, 'message': 'room created', 'algorithm': algorithm})

# Join-check (HTTP POST) - verifies passphrase, returns success
@app.route('/api/join-room', methods=['POST'])
def join_room_http():
    data = request.json or {}
    room_id = data.get('room_id')
    passphrase = data.get('passphrase')
    username = data.get('username') or 'guest'

    print(f"Join room request: room_id={room_id}, username={username}")

    # Validate input
    if not room_id or not passphrase:
        return jsonify({'error': 'room_id and passphrase required'}), 400

    if not username or len(username.strip()) == 0:
        username = f'user_{int(time.time())}'

    room = rooms.get(room_id)
    if not room:
        print(f"Room not found: {room_id}. Available rooms: {list(rooms.keys())}")
        return jsonify({'error': 'room not found'}), 404

    # Check participant limit (35 max)
    current_participants = len(room['participants'])
    if current_participants >= 35:
        return jsonify({'error': 'room is full (max 35 participants)'}), 403

    # Check if username already exists
    if username in room['participants']:
        return jsonify({'error': 'username already taken in this room'}), 409

    if not pbkdf2_sha256.verify(passphrase, room['pass_hash']):
        print(f"Invalid passphrase for room: {room_id}")
        return jsonify({'error': 'invalid passphrase'}), 403

    # Add participant
    room['participants'][username] = True
    print(f"User {username} joined room {room_id}. Participants: {len(room['participants'])}/{35}")

    return jsonify({
        'room_id': room_id,
        'message': 'joined successfully',
        'algorithm': room['algorithm'],
        'name': room['name'],
        'participant_count': len(room['participants']),
        'max_participants': 35
    })

# Polling endpoints for real-time simulation

@app.route('/api/send-message', methods=['POST'])
def send_message():
    data = request.json or {}
    room_id = data.get('room_id')
    username = data.get('username')
    message = data.get('message')

    print(f"Send message request: room_id={room_id}, username={username}, message_length={len(message) if message else 0}")

    # Validate input
    if not room_id or not username or not message:
        print(f"Missing required fields: room_id={room_id}, username={username}, message={message}")
        return jsonify({'error': 'room_id, username, and message required'}), 400

    room = rooms.get(room_id)
    if not room:
        print(f"Room not found: {room_id}")
        return jsonify({'error': 'room not found'}), 404

    # Check if user is a participant
    if username not in room['participants']:
        print(f"User {username} not in room {room_id} participants: {list(room['participants'].keys())}")
        return jsonify({'error': 'user not in room'}), 403

    # Add message
    msg_id = room['next_message_id']
    room['next_message_id'] += 1
    room['messages'].append({'id': msg_id, 'username': username, 'message': message})

    print(f"Message sent successfully: {username} in room {room_id}, total messages: {len(room['messages'])}")
    return jsonify({'success': True})

@app.route('/api/get-messages', methods=['GET'])
def get_messages():
    room_id = request.args.get('room_id')
    room = rooms.get(room_id)
    if not room:
        return jsonify({'error': 'room not found'}), 404
    return jsonify({'messages': room['messages'], 'next_id': room['next_message_id']})

@app.route('/api/get-participants', methods=['GET'])
def get_participants():
    room_id = request.args.get('room_id')
    room = rooms.get(room_id)
    if not room:
        return jsonify({'error': 'room not found'}), 404
    return jsonify({'participants': list(room['participants'])})

@app.route('/api/leave-room', methods=['POST'])
def leave_room():
    data = request.json or {}
    room_id = data.get('room_id')
    username = data.get('username')
    room = rooms.get(room_id)
    if room and username in room['participants']:
        del room['participants'][username]
    return jsonify({'success': True})

@app.route('/api/list-rooms', methods=['GET'])
def list_rooms():
    """List rooms created by the current user only"""
    creator = request.args.get('creator', 'web')  # Default to 'web' for web clients
    room_list = []
    for room_id, room_data in rooms.items():
        if room_data['creator'] == creator:
            room_list.append({
                'id': room_id,
                'name': room_data['name'],
                'algorithm': room_data['algorithm'],
                'creator': room_data['creator'],
                'participant_count': len(room_data['participants']),
                'message_count': len(room_data['messages'])
            })
    return jsonify({'rooms': room_list})

# Export Flask app for Vercel
application = app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
