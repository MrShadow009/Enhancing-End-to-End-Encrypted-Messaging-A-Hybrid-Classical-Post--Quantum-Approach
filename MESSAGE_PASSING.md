# Message Passing Documentation - SecureChat

## Overview
SecureChat implements a multi-layered message passing architecture supporting both traditional socket-based TCP communication and web-based HTTP polling. Messages are encrypted using various cryptographic algorithms (classical and post-quantum) and passed through a central server to multiple clients.

---

## 1. Architecture Overview

### Components
- **Server** (`server.py`): Central message hub that routes encrypted messages
- **Client** (`client.py`): Desktop client with TKinter GUI for sending/receiving messages
- **Web App** (`app.py`): Flask-based web interface with HTTP polling and room-based messaging
- **Network Protocol**: TCP sockets (port 1234) for desktop clients, HTTP/REST for web clients

### Message Flow Diagram
```
CLIENT 1                    SERVER                    CLIENT 2
   |                          |                          |
   |----Encrypted Msg-------->|                          |
   |                          |----Broadcast Msg-------->|
   |                          |                          |
   |<------Encrypted Msg------|<-----Encrypted Msg-------|
```

---

## 2. Desktop Client-Server Communication (TCP Sockets)

### 2.1 Connection Establishment

**Step 1: Client connects to server**
```python
# client.py - connect() function
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))  # Connect to 127.0.0.1:1234
```

**Step 2: Send username to server**
```python
username = username_textbox.get()
client.sendall(username.encode())  # Send as UTF-8 string
print("SEND : ", username.encode())
```

**Step 3: Server receives username and registers client**
```python
# server.py - client_handler() function
username = client.recv(2048).decode('utf-8')
active_clients.append((username, client, key))  # Register in active clients list
```

### 2.2 Message Format Structure

Messages use a **delimiter-based format** with `~` as separator:

```
[username]~[message_content]~[session_key]~[algorithm_flag]~[public_key_1]~[public_key_2]
```

**Example:**
```
john~Hello World~a1b2c3d4~3~public_key_value~rsa_params
```

**Components:**
- **username**: Sender identifier
- **message_content**: Encrypted message body
- **session_key**: Server-generated session key (8 hex characters)
- **algorithm_flag**: Encryption method identifier (1-18)
- **public_key_1**: ElGamal/alternative public key
- **public_key_2**: RSA/alternative key parameters

### 2.3 Encryption Process (Client Side)

**Step 1: User types message**
```python
# client.py - send_message()
message = message_textbox.get()
```

**Step 2: Select encryption based on algorithm flag**
```python
if flagMethod == 1:  # DES
    message = DES_Encrypt.startDesEncryption(message, key)
    
elif flagMethod == 2:  # ElGamal
    message = el_gamal.incrypt_gamal(
        int(elgamalkey[0]),  # p
        int(elgamalkey[1]),  # a
        int(elgamalkey[2]),  # YA (public key)
        message
    )
    
elif flagMethod == 3:  # RSA
    pla, mes = RSA.preprocess_message(message, int(rsa_string[0]))
    message = RSA.to_cipher(int(rsa_string[1]), int(rsa_string[0]), pla)
    message = ",".join([str(x) for x in message])  # Convert to CSV format
```

**Step 3: Send encrypted message to server**
```python
client.sendall(message.encode("utf-8"))
print("SEND :", message.encode())
```

### 2.4 Server Broadcasting

**Step 1: Server receives encrypted message**
```python
# server.py - listen_for_messages()
message = client.recv(2048).decode('utf-8')
```

**Step 2: Construct full message packet with metadata**
```python
final_msg = username + '~' + message + '~' + key + "~" + flagmethod + "~" + elgamapublickey + "~" + rsa_string
send_messages_to_all(final_msg)
```

**Step 3: Broadcast to all connected clients**
```python
# server.py - send_messages_to_all()
def send_messages_to_all(message):
    for user in active_clients:
        send_message_to_client(user[1], message)  # user[1] is socket
        
def send_message_to_client(client, message):
    client.sendall(message.encode('utf-8'))
```

### 2.5 Decryption Process (Receiving Client)

**Step 1: Receive and parse message**
```python
# client.py - listen_for_messages_from_server()
message = client.recv(2048).decode('utf-8')
message = message.split("~")  # Parse delimiter-separated format
```

**Step 2: Extract message components**
```python
username = message[0]
content = message[1]           # Encrypted content
key = message[2]               # Session key
flagMethod = int(message[3])   # Algorithm type
elgamalkey = message[4].split(",")  # Parse public keys
rsa_string = message[5].split(",")  # Parse RSA params
```

**Step 3: Decrypt based on algorithm**
```python
if flagMethod == 1:  # DES
    content = DES_Decrypt.startDesDecryption(content, key)
    content = bytes.fromhex(content).decode('utf-8')
    
elif flagMethod == 2:  # ElGamal
    content = el_gamal.decrept_gamal(content, int(elgamalkey[3]))  # XA (private key)
    
elif flagMethod == 3:  # RSA
    content = content.split(",")
    content = [int(x) for x in content]
    content = RSA.to_plain(int(rsa_string[2]), int(rsa_string[0]), content, mes)
```

**Step 4: Display message**
```python
add_message(f"[{username}] {content}")
```

---

## 3. Web-Based Message Passing (HTTP Polling)

### 3.1 Room Creation Flow

**HTTP Endpoint:** `POST /api/create-room`

**Request:**
```json
{
  "room_name": "SecureRoom1",
  "algorithm": "RSA",
  "passphrase": "secret123",
  "creator": "web_user"
}
```

**Server Processing:**
```python
# app.py - create_room()
room_id = str(uuid.uuid4())
pass_hash = pbkdf2_sha256.hash(passphrase)
rooms[room_id] = {
    'name': room_name,
    'algorithm': algorithm,
    'pass_hash': pass_hash,
    'creator': creator,
    'participants': {},
    'messages': [],
    'next_message_id': 0
}
```

**Response:**
```json
{
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "room created",
  "algorithm": "RSA"
}
```

### 3.2 Join Room Flow

**HTTP Endpoint:** `POST /api/join-room`

**Request:**
```json
{
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "passphrase": "secret123",
  "username": "alice"
}
```

**Server Processing:**
```python
# app.py - join_room_http()
room = rooms.get(room_id)

# Verify passphrase
if not pbkdf2_sha256.verify(passphrase, room['pass_hash']):
    return jsonify({'error': 'invalid passphrase'}), 403

# Add participant
room['participants'][username] = True
return jsonify({'room_id': room_id, 'algorithm': room['algorithm']})
```

### 3.3 Message Sending (HTTP Polling)

**HTTP Endpoint:** `POST /api/send-message`

**Request:**
```json
{
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "alice",
  "message": "{encrypted_content}"
}
```

**Server Processing:**
```python
# app.py - send_message()
room = rooms.get(room_id)
if not room or username not in room['participants']:
    return jsonify({'error': 'unauthorized'}), 403

msg_id = room['next_message_id']
room['next_message_id'] += 1
room['messages'].append({
    'id': msg_id,
    'username': username,
    'message': message
})
return jsonify({'success': True})
```

### 3.4 Message Polling (HTTP GET)

**HTTP Endpoint:** `GET /api/get-messages?room_id={room_id}`

**Server Response:**
```json
{
  "messages": [
    {"id": 0, "username": "alice", "message": "{encrypted}"},
    {"id": 1, "username": "bob", "message": "{encrypted}"}
  ],
  "next_id": 2
}
```

**Client-Side Polling Logic:**
```javascript
// socket_main.js - startPolling()
messageInterval = setInterval(async () => {
    const res = await fetch(`/api/get-messages?room_id=${currentRoomId}`);
    const data = await res.json();
    
    // Process only new messages
    data.messages.forEach(msg => {
        if (msg.id > lastMessageId) {
            appendMessage(`[${msg.username}] ${msg.message}`);
            lastMessageId = msg.id;
        }
    });
}, 1000);  // Poll every second
```

### 3.5 Participant Tracking

**HTTP Endpoint:** `GET /api/get-participants?room_id={room_id}`

**Response:**
```json
{
  "participants": ["alice", "bob", "charlie"]
}
```

---

## 4. Encryption Algorithms Supported

### Classical Algorithms
| Flag | Algorithm | Key Exchange | Encryption | Signature |
|------|-----------|--------------|-----------|-----------|
| 1 | DES | Symmetric | Yes | N/A |
| 2 | ElGamal | Asymmetric | Yes | Yes |
| 3 | RSA | Asymmetric | Yes | Yes |

### Post-Quantum Algorithms
| Flag | Algorithm | Category | Type |
|------|-----------|----------|------|
| 4 | CRYSTALS-Kyber | Lattice | KEM |
| 5 | CRYSTALS-Dilithium | Lattice | Signature |
| 6 | Falcon | Lattice | Signature |
| 7 | SABER | Lattice | KEM |
| 8 | NewHope | Lattice | KEM |
| 9 | FrodoKEM | Lattice | KEM |
| 10 | NTRUEncrypt | Lattice | Encryption |
| 11 | NTRUPrime | Lattice | KEM |
| 12 | Classic McEliece | Code-based | KEM |
| 13 | BIKE | Code-based | KEM |
| 14 | HQC | Code-based | KEM |
| 15 | Rainbow | Multivariate | Signature |
| 16 | SPHINCS+ | Hash-based | Signature |
| 17 | CSIDH | Isogeny | Key Exchange |
| 18 | Picnic | Symmetric | Signature |

### Key Generation per Algorithm
```python
# server.py - client_handler()
if flagmethod == "4":  # Kyber
    public_key, secret_key = kyber.generate_keys()
    rsa_string = public_key.hex()
    elgamalpublickey = secret_key.hex()
# ... similar for other algorithms
```

---

## 5. Session Management

### 5.1 Session Key Generation

**When client connects:**
```python
# server.py - client_handler()
key = secrets.token_hex(8).upper()  # Generate 16-char hex string
print(f"Session key for {username}: {key}")
```

**Example session keys:**
- `A1B2C3D4E5F6G7H8`
- `X9Y8Z7W6V5U4T3S2`

### 5.2 Key Distribution

Keys are sent with every message packet:
```
[username]~[message]~[key]~[algorithm]~[pubkey1]~[pubkey2]
```

### 5.3 Room-Level Session State

**Web Rooms:**
```python
rooms = {
    'room_uuid': {
        'name': str,
        'algorithm': str,
        'pass_hash': str,  # PBKDF2 hashed passphrase
        'participants': {username: True},
        'messages': [{id, username, message}],
        'next_message_id': int
    }
}
```

**Desktop Clients:**
```python
active_clients = [
    (username, socket, key)  # Tuple per connected client
]
```

---

## 6. Error Handling

### Socket Communication Errors

```python
# client.py - connect()
try:
    client.connect((HOST, PORT))
except:
    messagebox.showerror(
        "Unable to connect to server",
        f"Unable to connect to server {HOST} {PORT}"
    )

# client.py - send_message()
try:
    client.sendall(message.encode("utf-8"))
except Exception as e:
    print(f"Send error: {e}")
```

### HTTP API Errors

```javascript
// socket_main.js - error handling
try {
    const res = await fetch('/api/join-room', {...});
    const data = await res.json();
    if (data.error) {
        alert('Join failed: ' + data.error);
    }
} catch (error) {
    console.error('Network error:', error);
    alert('Network error during join');
}
```

### Server-Side Validation

```python
# app.py - join_room_http()
if not room:
    return jsonify({'error': 'room not found'}), 404

if not pbkdf2_sha256.verify(passphrase, room['pass_hash']):
    return jsonify({'error': 'invalid passphrase'}), 403

if username not in room['participants']:
    return jsonify({'error': 'unauthorized'}), 403
```

---

## 7. Message Queue & Broadcasting

### Desktop Clients Queue

```python
# server.py - send_messages_to_all()
def send_messages_to_all(message):
    for user in active_clients:  # Iterate all connected clients
        send_message_to_client(user[1], message)

def send_message_to_client(client, message):
    client.sendall(message.encode('utf-8'))  # Send immediately (not queued)
```

**Characteristics:**
- **Synchronous**: Blocking send to each client
- **No queue**: Messages sent in real-time
- **Broadcast**: All clients receive same packet

### Web Rooms Message Queue

```python
# app.py - message storage
rooms[room_id]['messages'].append({
    'id': room['next_message_id'],
    'username': username,
    'message': message
})
room['next_message_id'] += 1
```

**Characteristics:**
- **Stored in memory**: Messages persisted in room object
- **Polling-based**: Clients retrieve new messages via HTTP GET
- **ID-tracked**: Each message has unique ID for polling

---

## 8. Concurrency & Threading

### Desktop Server (Thread Per Client)

```python
# server.py - main()
while True:
    client, address = server.accept()
    threading.Thread(target=client_handler, args=(client, key)).start()
```

**One thread per connected client:**
- `client_handler()`: Waits for username
- `listen_for_messages()`: Listens for incoming messages

### Desktop Client (Listener Thread)

```python
# client.py - connect()
threading.Thread(target=listen_for_messages_from_server, args=(client,)).start()
```

**Separate thread for receiving messages:**
- Main thread: User input & GUI
- Listener thread: Blocks on `client.recv()` for incoming messages

### Web App (Flask Single Process)

```python
# app.py - REST endpoints
@app.route('/api/send-message', methods=['POST'])
def send_message():
    # Handles HTTP request
    
@app.route('/api/get-messages', methods=['GET'])
def get_messages():
    # Returns room messages
```

**Characteristics:**
- Stateless HTTP requests
- No persistent connections
- Polling-based message retrieval

---

## 9. Security Considerations

### Message Encryption

**In Transit:**
- DES: Symmetric block cipher
- RSA/ElGamal: Asymmetric encryption
- PQC Algorithms: Quantum-resistant alternatives

### Authentication

**Web Rooms:**
```python
# Passphrase verification with PBKDF2
pass_hash = pbkdf2_sha256.hash(passphrase)
if not pbkdf2_sha256.verify(passphrase, pass_hash):
    return error
```

**Desktop Clients:**
- No explicit authentication
- Relies on session key + socket connection

### Potential Vulnerabilities

⚠️ **Issues to Address:**

1. **Session Keys in Messages**: Session key sent in plaintext within every message packet
   ```
   [msg]~[plaintext_key]~[algo]  ❌ Insecure
   ```
   - **Fix**: Establish session before messaging, use TLS/SSL

2. **No Message Integrity**: No HMAC or digital signatures
   - **Fix**: Add HMAC-SHA256 for authenticity

3. **Replay Attacks**: No timestamp or nonce
   - **Fix**: Include timestamp and increment counter

4. **Memory Storage**: Messages stored unencrypted in Python dicts
   - **Fix**: Encrypt at-rest, use database

---

## 10. Configuration & Startup

### Server Startup

```python
# server.py - main()
HOST = '127.0.0.1'
PORT = 1234
LISTENER_LIMIT = 5

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(LISTENER_LIMIT)
```

**User selects encryption algorithm:**
```
---------Welcome to our secure chat
1- DES (Data encryption standard)
2- ElGamal encryption system
3- RSA (Rivest–Shamir–Adleman)
...
Choose the encryption system: 3
RSA mode has been started
```

### Client Startup

```python
# client.py - main()
HOST = '127.0.0.1'
PORT = 1234
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
root = tk.Tk()
root.geometry("600x600")
root.mainloop()
```

### Web App Startup

```python
# app.py
flask run --host=127.0.0.1 --port=5000
```

---

## 11. Data Flow Examples

### Example 1: DES Message Flow

```
CLIENT                          SERVER                      CLIENT 2
   |                              |                             |
   | "Hello" (plaintext)          |                             |
   | ↓                            |                             |
   | DES Encrypt: "8F42B1C9"      |                             |
   |                              |                             |
   | Send: "alice~8F42B1C9~...~1" |                             |
   |------------------------------>|                             |
   |                              | Broadcast full packet       |
   |                              |----------------------------->|
   |                              |                             | Parse packet
   |                              |                             | ↓
   |                              |                             | DES Decrypt: "Hello"
   |                              |                             | Display: "[alice] Hello"
```

### Example 2: RSA Message Flow

```
CLIENT 1                    SERVER                      CLIENT 2
   |                          |                             |
   | Generate keys            |                             |
   | ↓                        |                             |
   | E = 65537, N = 1021      |                             |
   |                          |                             |
   | Message: "Hi" → [11, 8]  |                             |
   | ↓                        |                             |
   | RSA Encrypt: [123, 456]  |                             |
   |                          |                             |
   | Send: "alice~123,456~..~3" |                          |
   |-------------------------->|                             |
   |                          | Broadcast                    |
   |                          |------------------------------>|
   |                          |                             | Receive keys
   |                          |                             | ↓
   |                          |                             | Extract D (private)
   |                          |                             | ↓
   |                          |                             | RSA Decrypt: "Hi"
```

### Example 3: Web Room HTTP Flow

```
CLIENT (Web)                    FLASK SERVER               DATABASE
   |                              |                           |
   | POST /api/create-room        |                           |
   | {room_name, algorithm, pass} |                           |
   |----------------------------->|                           |
   |                              | Generate room_id          |
   |                              | Hash passphrase           |
   |                              | Create room object        |
   |                              |---->Store in rooms dict   |
   |                              |                           |
   | ✓ room_id: "xyz-123"         |                           |
   |<------------------------------|                           |
   |                              |                           |
   | POST /api/join-room          |                           |
   | {room_id, pass, username}    |                           |
   |----------------------------->|                           |
   |                              | Verify passphrase         |
   |                              | Add participant           |
   |                              |                           |
   | ✓ Joined                     |                           |
   |<------------------------------|                           |
   |                              |                           |
   | POST /api/send-message       |                           |
   | {room_id, username, msg}     |                           |
   |----------------------------->|                           |
   |                              | Append to messages        |
   |                              | Increment next_id         |
   |                              |                           |
   | ✓ Message sent               |                           |
   |<------------------------------|                           |
   |                              |                           |
   | GET /api/get-messages        |                           |
   | (polling every 1 sec)        |                           |
   |----------------------------->|                           |
   |                              | Return new messages       |
   | [msg1, msg2, ...]            |                           |
   |<------------------------------|                           |
```

---

## 12. Performance Characteristics

### Desktop TCP Sockets
- **Latency**: ~1-5ms (local network)
- **Throughput**: Depends on encryption algorithm
- **Connections**: Limited by thread pool (LISTENER_LIMIT = 5)
- **Message Format**: Delimiter-separated strings

### Web HTTP Polling
- **Latency**: ~50-200ms (HTTP overhead)
- **Poll Interval**: 1 second (configurable)
- **Scalability**: Better for many clients (stateless)
- **Storage**: In-memory (room messages)

### Encryption Overhead
- **DES**: ~1-2ms per message
- **RSA**: ~10-50ms per message (key size dependent)
- **PQC Algorithms**: ~50-500ms per message

---

## 13. Best Practices & Recommendations

### Security
- ✅ Use TLS/SSL for all connections
- ✅ Hash passphrases (already done with PBKDF2)
- ✅ Add message timestamps to prevent replay attacks
- ✅ Implement HMAC for message integrity
- ✅ Use random IVs for symmetric encryption

### Scalability
- ✅ Web app better than desktop sockets for many users
- ✅ Consider message queuing system (Redis, RabbitMQ)
- ✅ Move to database instead of in-memory storage
- ✅ Implement connection pooling for web app

### Reliability
- ✅ Add reconnection logic with exponential backoff
- ✅ Implement message acknowledgments
- ✅ Add heartbeat/keep-alive signals
- ✅ Handle socket disconnections gracefully

### Monitoring
- ✅ Log all message transfers
- ✅ Track active connections
- ✅ Monitor encryption/decryption times
- ✅ Alert on failed authentications

---

## Summary

SecureChat implements a **hybrid message passing system**:

1. **Desktop clients** use **TCP sockets** for low-latency, persistent connections
2. **Web clients** use **HTTP polling** for stateless, scalable access
3. **Messages are encrypted** with 18 different algorithms (classical + post-quantum)
4. **Server broadcasts** messages to all participants
5. **Session keys** are generated per client and sent with each message

The architecture prioritizes **security through encryption** while supporting multiple protocols and encryption algorithms.
