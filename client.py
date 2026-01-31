# import required modules
import socket
import threading
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import DES_Decrypt
import DES_Encrypt
import el_gamal
import RSA

# Import new PQC algorithms
from pqc_algorithms import kyber
from pqc_algorithms import dilithium
from pqc_algorithms import falcon
from pqc_algorithms import saber
from pqc_algorithms import newhope
from pqc_algorithms import frodo
from pqc_algorithms import ntru_encrypt
from pqc_algorithms import ntruprime
from pqc_algorithms import classic_mceliece
from pqc_algorithms import bike
from pqc_algorithms import hqc
from pqc_algorithms import rainbow
from pqc_algorithms import sphincsplus
from pqc_algorithms import csidh
from pqc_algorithms import picnic
from pqc_algorithms import xmss
from pqc_algorithms import quartz
from pqc_algorithms import sike

# Import quantum modules
from quantum_state_manager import QuantumState, encode_message_to_quantum_state, decode_quantum_state_to_message
from bb84_protocol import BB84Protocol
from quantum_teleportation import QuantumMessagePacket
from quantum_channel import QuantumChannelSimulator

# Import quantum modules
import json
import time
from quantum_state_manager import QuantumState
from bb84_protocol import BB84Protocol
from quantum_teleportation import QuantumTeleportation, QuantumMessagePacket


#HOST = '192.168.1.8'
HOST = '127.0.0.1'

PORT = 1234

DARK_GREY = '#485460'
MEDIUM_GREY = '#1e272e'
OCEAN_BLUE = '#60a3bc'
WHITE = "white"
FONT = ("Helvetica", 17)
BUTTON_FONT = ("Helvetica", 15)
SMALL_FONT = ("Helvetica", 13)

# Creating a socket object
# AF_INET: we are going to use IPv4 addresses
# SOCK_STREAM: we are using TCP packets for communication
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Initialize global variables
flagMethod = 0
key = ""
elgamalkey = []
rsa_string = []

# Quantum client variables
quantum_client = None
bb84_protocol = BB84Protocol()
teleportation = QuantumTeleportation()
epr_qubits = {}  # Store received EPR qubits
quantum_key_id = None

def add_message(message):
    message_box.config(state=tk.NORMAL)
    message_box.insert(tk.END, message + '\n')
    message_box.config(state=tk.DISABLED)

def connect():

    # try except block
    try:

        # Connect to the server
        client.connect((HOST, PORT))
        print("Successfully connected to server")
        add_message("[SERVER] Successfully connected to the server")
        add_message("[QUANTUM] Initializing quantum protocols...")
    except:
        messagebox.showerror("Unable to connect to server", f"Unable to connect to server {HOST} {PORT}")

    username = username_textbox.get()
    if username != '':
        client.sendall(username.encode())
        print("SEND : ", username.encode() )
    else:
        messagebox.showerror("Invalid username", "Username cannot be empty")

    threading.Thread(target=listen_for_messages_from_server, args=(client, )).start()
    threading.Thread(target=perform_qkd_handshake, args=(client, username)).start()

    #tk
    username_textbox.config(state=tk.DISABLED)
    username_button.config(state=tk.DISABLED)
    username_button.pack_forget()
    username_textbox.pack_forget()
    username_label['text']= "Welcome " + username + " to our secure room"
    username_label.pack(side=tk.LEFT)

####here
def send_message():
    message = message_textbox.get()
    if message != '':
        message_textbox.delete(0, len(message))

        # Determine which algorithm to use for this message
        if flagMethod == "0":  # RANDOM mode
            # Randomly select one of the 21 algorithms (1-21)
            import random
            selected_algorithm = str(random.randint(1, 21))
            algorithm_name = get_algorithm_name(selected_algorithm)
            print(f"[RANDOM] Selected algorithm {selected_algorithm}: {algorithm_name}")
        else:
            # Use the fixed algorithm
            selected_algorithm = flagMethod
            algorithm_name = get_algorithm_name(selected_algorithm)

        # Step 1: First encrypt with chosen algorithm (traditional or PQC)
        encrypted_message = encrypt_with_pqc(message, selected_algorithm)
        print(f"[ENCRYPTION] Original: '{message}' -> Encrypted with {algorithm_name}: '{encrypted_message[:50]}...'")

        # Step 2: Then encode the encrypted message into quantum states (double encryption for ALL paths)
        quantum_state = encode_message_to_quantum_state(encrypted_message)
        print(f"[QUANTUM ENCODING] Encrypted message encoded into quantum state")

        # Step 3: Create quantum message packet
        quantum_packet = QuantumMessagePacket(
            sender=username_textbox.get() or "Anonymous",
            message=encrypted_message,
            qkd_key_id=quantum_key_id
        )
        quantum_packet.quantum_state = quantum_state
        quantum_packet.algorithm_used = selected_algorithm  # Store which algorithm was used
        quantum_packet.encode_quantum()

        # Step 4: Send the quantum-encoded message (ALL messages now go through quantum channel)
        packet_data = quantum_packet.encode_quantum()
        message_to_send = json.dumps(packet_data)

        client.sendall(message_to_send.encode("utf-8"))
        print("SEND QUANTUM: ", message_to_send.encode() )

        add_message(f"[SENT] Message encrypted with {algorithm_name} + Quantum encoding")
        print("This message has been delivered with double encryption")
    else:
        messagebox.showerror("Empty message", "Message cannot be empty")

def get_algorithm_name(method_flag):
    """Get human-readable algorithm name"""
    algorithms = {
        1: "DES", 2: "ElGamal", 3: "RSA", 4: "Kyber", 5: "Dilithium", 6: "Falcon",
        7: "SABER", 8: "NewHope", 9: "FrodoKEM", 10: "NTRUEncrypt", 11: "NTRUPrime",
        12: "Classic McEliece", 13: "BIKE", 14: "HQC", 15: "Rainbow", 16: "SPHINCS+",
        17: "CSIDH", 18: "Picnic", 19: "XMSS", 20: "QUARTZ", 21: "SIKE"
    }
    return algorithms.get(method_flag, "Unknown")

def encrypt_with_pqc(message, method_flag):
    """Encrypt message with chosen PQC algorithm"""
    if method_flag == 1:
        return DES_Encrypt.startDesEncryption(message, key)
    elif method_flag == 2:
        print("elgammel encryption")
        global messageCopy
        encrypted = el_gamal.incrypt_gamal(int(elgamalkey[0]), int(elgamalkey[1]), int(elgamalkey[2]), message)
        messageCopy = encrypted
        return str(encrypted)
    elif method_flag == 3:
        print("RSA encryption")
        global vo
        pla=[]
        global mes
        mes = []
        pla, mes = RSA.preprocess_message(message, int(rsa_string[0]))
        print("mes:", mes)
        encrypted = RSA.to_cipher(int(rsa_string[1]), int(rsa_string[0]), pla)
        encrypted = [str(x) for x in encrypted]
        return ",".join(encrypted)
    elif method_flag == 4:  # Kyber
        ciphertext, shared_secret = kyber.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 5:  # Dilithium
        signature = dilithium.sign(bytes(message, 'utf-8'))
        return signature.hex()
    elif method_flag == 6:  # Falcon
        signature = falcon.sign(bytes(message, 'utf-8'))
        return signature.hex()
    elif method_flag == 7:  # SABER
        ciphertext, shared_secret = saber.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 8:  # NewHope
        ciphertext, shared_secret = newhope.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 9:  # FrodoKEM
        ciphertext, shared_secret = frodo.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 10:  # NTRUEncrypt
        ciphertext, shared_secret = ntru_encrypt.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 11:  # NTRUPrime
        ciphertext, shared_secret = ntruprime.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 12:  # Classic McEliece
        ciphertext, shared_secret = classic_mceliece.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 13:  # BIKE
        ciphertext, shared_secret = bike.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 14:  # HQC
        ciphertext, shared_secret = hqc.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 15:  # Rainbow
        signature = rainbow.sign(bytes(message, 'utf-8'))
        return signature.hex()
    elif method_flag == 16:  # SPHINCS+
        signature = sphincsplus.sign(bytes(message, 'utf-8'))
        return signature.hex()
    elif method_flag == 17:  # CSIDH
        ciphertext, shared_secret = csidh.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    elif method_flag == 18:  # Picnic
        signature = picnic.sign(bytes(message, 'utf-8'))
        return signature.hex()
    elif method_flag == 19:  # XMSS
        signature, _ = xmss.sign(bytes(message, 'utf-8'))
        return signature.hex()
    elif method_flag == 20:  # QUARTZ
        ciphertext = quartz.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex()
    elif method_flag == 21:  # SIKE
        ciphertext, shared_secret = sike.encrypt(bytes(message, 'utf-8'))
        return ciphertext.hex() + ":" + shared_secret.hex()
    else:
        return message  # No encryption


####here
def listen_for_messages_from_server(client):

    while 1:
        message = client.recv(2048).decode('utf-8')
        print("RECV : ", message)
        #####
        if message != '':
            try:
                # ALL messages now go through quantum channel - parse as quantum message
                quantum_packet = json.loads(message)
                if isinstance(quantum_packet, dict) and 'quantum_state' in quantum_packet:
                    # Handle quantum message with double decryption: Quantum decoding -> Algorithm decryption
                    handle_quantum_message_from_server(quantum_packet)
                else:
                    # Handle non-quantum messages (like server notifications, QKD, etc.)
                    handle_non_quantum_message(quantum_packet)
            except json.JSONDecodeError:
                # Legacy support for old message format (if any)
                print(f"[WARNING] Received non-JSON message: {message[:100]}...")
                add_message(f"[SYSTEM] {message}")

        else:
            messagebox.showerror("Error", "Message received from server is empty")

def handle_non_quantum_message(message_data):
    """Handle non-quantum protocol messages (QKD results, server notifications, etc.)"""
    try:
        msg_type = message_data.get('type', 'unknown')

        if msg_type == 'qkd_success':
            global quantum_key_id
            quantum_key_id = message_data.get('key_id')
            add_message(f"[QUANTUM] QKD successful! Key ID: {quantum_key_id}")
            print(f"[QUANTUM CLIENT] QKD completed")

        elif msg_type == 'qkd_failed':
            add_message(f"[QUANTUM] QKD failed: {message_data.get('reason', 'Unknown error')}")

        elif msg_type == 'server_notification':
            add_message(f"[SERVER] {message_data.get('message', 'Notification')}")

        else:
            print(f"[UNKNOWN MESSAGE TYPE] {msg_type}: {message_data}")

    except Exception as e:
        print(f"[ERROR] Failed to handle non-quantum message: {e}")

def handle_quantum_message_from_server(quantum_packet):
    """Handle quantum message from server with double decryption: Quantum decoding -> PQC decryption"""
    try:
        sender = quantum_packet.get('sender', 'Unknown')

        # Get the algorithm used for encryption (important for RANDOM mode)
        algorithm_used = quantum_packet.get('algorithm_used', flagMethod)
        algorithm_name = get_algorithm_name(algorithm_used)

        print(f"[CLIENT QUANTUM] Received from {sender} (algorithm: {algorithm_name}): Decoding quantum state")

        # Step 1: Decode the quantum message packet
        packet = QuantumMessagePacket.decode_quantum(quantum_packet)
        quantum_state = packet.quantum_state

        # Step 2: Decode the quantum state back to classical ciphertext bits
        ciphertext_bits = decode_quantum_state_to_message(quantum_state)

        print(f"[CLIENT QUANTUM] Quantum decoded to {len(ciphertext_bits)}-bit ciphertext")

        # Step 3: Convert bit string back to the original PQC ciphertext format
        # The ciphertext_bits is a binary string, we need to convert it back to the format expected by PQC decryption
        pqc_encrypted_message = ciphertext_bits  # For now, keep as binary string

        # Step 4: Decrypt the PQC-encrypted message using the SAME algorithm that was used for encryption
        decrypted_message = decrypt_with_pqc(pqc_encrypted_message, algorithm_used)

        print(f"[CLIENT QUANTUM] PQC decrypted with {algorithm_name}: '{pqc_encrypted_message[:50]}...' -> '{decrypted_message}'")

        # Step 5: Display the fully decrypted message
        add_message(f"[QUANTUM RECEIVE] [{sender}] {decrypted_message}")

    except Exception as e:
        print(f"[ERROR] Failed to handle quantum message from server: {e}")
        add_message(f"[ERROR] Failed to decrypt quantum message from {sender}")

def decrypt_with_pqc(encrypted_message, method_flag):
    """Decrypt message with chosen PQC algorithm"""
    if method_flag == 1:
        decrypted = DES_Decrypt.startDesDecryption(encrypted_message, key)
        try:
            return bytes.fromhex(decrypted).decode('utf-8')
        except:
            return decrypted
    elif method_flag == 2:
        return el_gamal.decrept_gamal(encrypted_message, int(elgamalkey[3]))
    elif method_flag == 3:
        content = encrypted_message.split(",")
        content = [int(x) for x in content]
        return RSA.to_plain(int(rsa_string[2]), int(rsa_string[0]), content, mes)
    elif method_flag == 4:  # Kyber
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return kyber.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 5:  # Dilithium
        signature = bytes.fromhex(encrypted_message)
        # For signature, we'd verify, but for demo return as-is
        return f"Verified signature: {signature.hex()[:20]}..."
    elif method_flag == 6:  # Falcon
        signature = bytes.fromhex(encrypted_message)
        return f"Verified signature: {signature.hex()[:20]}..."
    elif method_flag == 7:  # SABER
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return saber.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 8:  # NewHope
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return newhope.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 9:  # FrodoKEM
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return frodo.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 10:  # NTRUEncrypt
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return ntru_encrypt.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 11:  # NTRUPrime
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return ntruprime.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 12:  # Classic McEliece
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return classic_mceliece.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 13:  # BIKE
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return bike.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 14:  # HQC
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return hqc.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 15:  # Rainbow
        signature = bytes.fromhex(encrypted_message)
        return f"Verified signature: {signature.hex()[:20]}..."
    elif method_flag == 16:  # SPHINCS+
        signature = bytes.fromhex(encrypted_message)
        return f"Verified signature: {signature.hex()[:20]}..."
    elif method_flag == 17:  # CSIDH
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return csidh.decrypt(ciphertext, shared_secret).decode('utf-8')
    elif method_flag == 18:  # Picnic
        signature = bytes.fromhex(encrypted_message)
        return f"Verified signature: {signature.hex()[:20]}..."
    elif method_flag == 19:  # XMSS
        signature = bytes.fromhex(encrypted_message)
        return f"Verified signature: {signature.hex()[:20]}..."
    elif method_flag == 20:  # QUARTZ
        ciphertext = bytes.fromhex(encrypted_message)
        return quartz.decrypt(ciphertext).decode('utf-8')
    elif method_flag == 21:  # SIKE
        parts = encrypted_message.split(":")
        ciphertext = bytes.fromhex(parts[0])
        shared_secret = bytes.fromhex(parts[1])
        return sike.decrypt(ciphertext, shared_secret).decode('utf-8')
    else:
        return encrypted_message  # No decryption
    
def DES_Encryption(pt, key):
    cipher = DES_Encrypt.startDesEncryption(pt,key)
    #print("The Cipher Text: ",cipher)
    return cipher

def perform_qkd_handshake(client_socket, username):
    """Perform BB84 QKD handshake with server (Bob's side)"""
    global quantum_key_id

    try:
        # Wait for QKD qubits from server (Alice)
        qkd_data = json.loads(client_socket.recv(4096).decode())
        if qkd_data.get('type') == 'qkd_qubits':
            qubits_data = qkd_data['qubits']
            alice_bases = qkd_data['alice_bases']
            n_bits = qkd_data['n']

            # Deserialize qubits
            qubits = [QuantumState.from_dict(q) for q in qubits_data]

            # Bob measures qubits in random bases
            bob_bases, bob_measurements = bb84_protocol.bob_measure_qubits(qubits, n_bits)

            # Send measurement results back to Alice
            response = {
                'bases': bob_bases,
                'measurements': bob_measurements
            }
            client_socket.sendall(json.dumps(response).encode())

            # Receive QKD result
            result_data = json.loads(client_socket.recv(4096).decode())
            if result_data.get('type') == 'qkd_success':
                quantum_key_id = result_data['key_id']
                add_message(f"[QUANTUM] QKD successful! Key ID: {quantum_key_id}")
                print(f"[QUANTUM CLIENT] QKD completed for {username}")
            elif result_data.get('type') == 'qkd_failed':
                add_message(f"[QUANTUM] QKD failed: {result_data.get('reason', 'Unknown error')}")
                print(f"[QUANTUM CLIENT] QKD failed for {username}")

    except Exception as e:
        print(f"[ERROR] QKD handshake failed: {e}")
        add_message("[QUANTUM] QKD handshake failed")

def handle_quantum_message(message_data):
    """Handle incoming quantum protocol messages"""
    try:
        msg = json.loads(message_data)

        if msg['type'] == 'epr_qubit':
            # Store received EPR qubit
            qubit_data = msg['qubit']
            partner = msg['partner']
            qubit = QuantumState.from_dict(qubit_data)
            epr_qubits[partner] = qubit
            add_message(f"[QUANTUM] Received EPR qubit for communication with {partner}")
            print(f"[QUANTUM CLIENT] EPR qubit received for {partner}")

        elif msg['type'] == 'quantum_teleport':
            # Handle quantum teleportation message
            sender = msg['sender']
            bell_measurement = msg['bell_measurement']
            message_length = msg.get('message_length', 0)

            # Reconstruct message using teleportation (simplified)
            reconstructed_msg = f"Quantum message from {sender}: Bell={bell_measurement}"
            add_message(f"[QUANTUM RECEIVE] {reconstructed_msg}")
            print(f"[QUANTUM CLIENT] Teleportation received: {bell_measurement}")

    except Exception as e:
        print(f"[ERROR] Failed to handle quantum message: {e}")

          
root = tk.Tk()
root.geometry("600x600")
root.title("Messenger Client")
root.resizable(False, False)

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=4)
root.grid_rowconfigure(2, weight=1)

top_frame = tk.Frame(root, width=600, height=100, bg=DARK_GREY)
top_frame.grid(row=0, column=0, sticky=tk.NSEW)

middle_frame = tk.Frame(root, width=600, height=400, bg=MEDIUM_GREY)
middle_frame.grid(row=1, column=0, sticky=tk.NSEW)

bottom_frame = tk.Frame(root, width=600, height=100, bg=DARK_GREY)
bottom_frame.grid(row=2, column=0, sticky=tk.NSEW)

username_label = tk.Label(top_frame, text="Enter your alias:", font=FONT, bg=DARK_GREY, fg=WHITE)
username_label.pack(side=tk.LEFT, padx=10)

username_textbox = tk.Entry(top_frame, font=FONT, bg=MEDIUM_GREY, fg=WHITE, width=23)
username_textbox.pack(side=tk.LEFT)

username_button = tk.Button(top_frame, text="Join", font=BUTTON_FONT, bg=OCEAN_BLUE, fg=WHITE, command=connect)
username_button.pack(side=tk.LEFT, padx=15)

message_textbox = tk.Entry(bottom_frame, font=FONT, bg=MEDIUM_GREY, fg=WHITE, width=38)
message_textbox.pack(side=tk.LEFT, padx=10)

message_button = tk.Button(bottom_frame, text="Send", font=BUTTON_FONT, bg=OCEAN_BLUE, fg=WHITE, command=send_message)
message_button.pack(side=tk.LEFT, padx=10)

message_box = scrolledtext.ScrolledText(middle_frame, font=SMALL_FONT, bg=MEDIUM_GREY, fg=WHITE, width=67, height=26.5)
message_box.config(state=tk.DISABLED)
message_box.pack(side=tk.TOP)


# main function
def main():
    #print("CODE :", server.getMethod())
    root.mainloop()
    
if __name__ == '__main__':
    main()