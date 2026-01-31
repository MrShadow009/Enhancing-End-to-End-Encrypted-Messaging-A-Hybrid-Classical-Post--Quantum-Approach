# Import required modules
import socket
import threading
import secrets
from tkinter import E
import el_gamal
import RSA
import json
import time

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
from quantum_state_manager import QuantumState, QuantumChannel, create_bell_state
from bb84_protocol import BB84Protocol
from quantum_teleportation import QuantumTeleportation, QuantumMessagePacket, simulate_quantum_network
from quantum_channel import QuantumChannelSimulator

#HOST = '192.168.1.8'
HOST = '127.0.0.1'
PORT = 1234 # to 65535
LISTENER_LIMIT = 5
active_clients = [] # List of all currently connected users

    
#Function to choose which security method to use
def chooseMethod():
    lst = [
        "Random",
        "DES",
        "ELGAMAL",
        "RSA",
        "CRYSTALS-Kyber",
        "CRYSTALS-Dilithium",
        "Falcon",
        "SABER",
        "NewHope",
        "FrodoKEM",
        "NTRUEncrypt",
        "NTRUPrime",
        "Classic McEliece",
        "BIKE",
        "HQC",
        "Rainbow",
        "SPHINCS+",
        "CSIDH",
        "Picnic",
        "XMSS",
        "QUARTZ",
        "SIKE"
    ]
    print("---------Welcome to our secure chat")
    print("0- RANDOM (Unpredictable algorithm selection per message)")
    print("1- DES (Data encryption standard)")
    print("2- ElGamal encryption system")
    print("3- RSA (Rivest–Shamir–Adleman)")
    print("4- CRYSTALS-Kyber (Post-Quantum KEM)")
    print("5- CRYSTALS-Dilithium (Post-Quantum Signature)")
    print("6- Falcon (Post-Quantum Signature)")
    print("7- SABER (Post-Quantum KEM)")
    print("8- NewHope (Post-Quantum KEM)")
    print("9- FrodoKEM (Post-Quantum KEM)")
    print("10- NTRUEncrypt (Post-Quantum Encryption)")
    print("11- NTRUPrime (Post-Quantum KEM)")
    print("12- Classic McEliece (Post-Quantum KEM)")
    print("13- BIKE (Post-Quantum KEM)")
    print("14- HQC (Post-Quantum KEM)")
    print("15- Rainbow (Post-Quantum Signature)")
    print("16- SPHINCS+ (Post-Quantum Signature)")
    print("17- CSIDH (Post-Quantum Key Exchange)")
    print("18- Picnic (Post-Quantum Signature)")
    print("19- XMSS (Post-Quantum Hash-Based Signature)")
    print("20- QUARTZ (Post-Quantum Multivariate Cryptography)")
    print("21- SIKE (Post-Quantum Isogeny-Based KEM)")

    while True:
        try:
            # Default to RANDOM mode (press Enter or input 0)
            num = input("Choose the encryption system (0=RANDOM [default], 1-21=specific) [0]: ").strip()
            if num == "":
                num = "0"  # Default to random

            choice = int(num)
            if choice == 0:
                print("RANDOM mode selected - Algorithm will change unpredictably per message")
                print("Each of the 21 algorithms has equal probability (~4.76% chance each)")
                return "0"  # Random mode
            elif 1 <= choice <= 21:
                print(lst[choice-1] + " mode has been started")
                return str(choice)
            else:
                print("Invalid choice. Please enter 0 for RANDOM or 1-21 for specific algorithm.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def getMethod():
    return flagmethod
   
# Function to listen for upcoming messages from a client
def listen_for_messages(client, username,key,elgamapublickey,rsa_string):

    while 1:

        message = client.recv(2048).decode('utf-8')
        print("RECV : ",message)
        if message != '':
            try:
                # Try to parse as quantum message
                quantum_packet = json.loads(message)
                if isinstance(quantum_packet, dict) and 'quantum_state' in quantum_packet:
                    # Handle quantum message
                    handle_quantum_message_from_client(quantum_packet, username, client)
                    continue
            except json.JSONDecodeError:
                # Not a quantum message, process traditionally
                pass

            ####### send traditional message
            final_msg = username + '~' + message + '~' + key + "~" +flagmethod+"~"+elgamapublickey+"~"+rsa_string
            send_messages_to_all(final_msg)
            print("rsaaaaaaa:   ",final_msg)

        else:
            print(f"The message send from client {username} is empty")

def handle_quantum_message_from_client(quantum_packet, sender_username, sender_client):
    """Handle quantum message from client with double encryption/decryption"""
    try:
        # Step 1: Decode the quantum message packet
        packet = QuantumMessagePacket.decode_quantum(quantum_packet)
        pqc_encrypted_message = packet.original_message

        print(f"[SERVER QUANTUM] Received from {sender_username}: Quantum decoded to PQC encrypted")

        # Step 2: Extract algorithm information from the packet (for RANDOM mode)
        algorithm_used = quantum_packet.get('algorithm_used', flagmethod)

        # Step 3: The PQC encrypted message is already in the packet, broadcast it to all clients
        # Each client will handle their own quantum decoding and PQC decryption
        quantum_broadcast_data = quantum_packet.copy()
        quantum_broadcast_data['sender'] = sender_username
        quantum_broadcast_data['algorithm_used'] = algorithm_used  # Include algorithm info

        # Broadcast to all other clients
        for user_tuple in active_clients:
            user_username, user_client, user_key = user_tuple
            if user_username != sender_username:
                try:
                    user_client.sendall(json.dumps(quantum_broadcast_data).encode('utf-8'))
                    print(f"[QUANTUM BROADCAST] Sent to {user_username} (algorithm: {algorithm_used})")
                except Exception as e:
                    print(f"[ERROR] Failed to send quantum message to {user_username}: {e}")

        print(f"[SERVER QUANTUM] Quantum message from {sender_username} broadcasted to all clients (algorithm: {algorithm_used})")

    except Exception as e:
        print(f"[ERROR] Failed to handle quantum message from {sender_username}: {e}")


# Function to send message to a single client
def send_message_to_client(client, message):

    client.sendall(message.encode())
    print("SEND : ", message.encode() )

# Function to send any new message to all the clients that
# are currently connected to this server
    #####here
def send_messages_to_all(message):
    
    for user in active_clients:
        
        # Start the security phase using message then pass the message to client
        send_message_to_client(user[1], message)

# Function to handle client
def client_handler(client,key):
    
    # Server will listen for client message that will
    # Contain the username
    while 1:

        username = client.recv(2048).decode('utf-8')
        print("RECV : ",username)
        if username != '':
            active_clients.append((username, client,key))
            # generate session key
            key = secrets.token_hex(8).upper()

            rsa_string = ""
            elgamalpublickey = ""

            # Handle key generation based on chosen method
            if flagmethod == "1":  # DES
                pass  # DES handled elsewhere

            elif flagmethod == "2":  # ElGamal
                string_ints = [str(x) for x in ElgamalKey]
                elgamalpublickey = ",".join(string_ints)
                print("elgamal public key", elgamalpublickey)

            elif flagmethod == "3":  # RSA
                n, E, D = RSA.calc()
                print("public and private key parameters: ")
                print("n: ", n)
                print("E: ", E)
                print("D: ", D)
                rsa_string += str(n) + "," + str(E) + "," + str(D) + ","

            elif flagmethod == "4":  # CRYSTALS-Kyber
                public_key, secret_key = kyber.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "5":  # CRYSTALS-Dilithium
                public_key, secret_key = dilithium.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "6":  # Falcon
                public_key, secret_key = falcon.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "7":  # SABER
                public_key, secret_key = saber.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "8":  # NewHope
                public_key, secret_key = newhope.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "9":  # FrodoKEM
                public_key, secret_key = frodo.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "10":  # NTRUEncrypt
                public_key, secret_key = ntru_encrypt.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "11":  # NTRUPrime
                public_key, secret_key = ntruprime.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "12":  # Classic McEliece
                public_key, secret_key = classic_mceliece.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "13":  # BIKE
                public_key, secret_key = bike.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "14":  # HQC
                public_key, secret_key = hqc.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "15":  # Rainbow
                public_key, secret_key = rainbow.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "16":  # SPHINCS+
                public_key, secret_key = sphincsplus.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "17":  # CSIDH
                public_key, secret_key = csidh.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "18":  # Picnic
                public_key, secret_key = picnic.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "19":  # XMSS
                public_key, secret_key = xmss.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "20":  # QUARTZ
                public_key, secret_key = quartz.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            elif flagmethod == "21":  # SIKE
                public_key, secret_key = sike.generate_keys()
                rsa_string = public_key.hex()
                elgamalpublickey = secret_key.hex()

            else:
                print("Unknown encryption method selected")





            #########send
            prompt_message = "SERVER~" + f"{username} added to the chat~" + key + "~" +flagmethod +"~" + elgamalpublickey +"~"+rsa_string 
            send_messages_to_all(prompt_message)
            
            print("Sessison key successfully generated for " + f"{username } ==>",key)

            break
        else:
            print("Client username is empty")

    threading.Thread(target=listen_for_messages, args=(client, username, key,elgamalpublickey,rsa_string, )).start()


class QuantumServer:
    def __init__(self, host='127.0.0.1', port=1234):
        self.host = host
        self.port = port
        self.active_clients = []
        self.entanglement_registry = {}
        self.bb84_protocol = BB84Protocol()
        self.teleportation = QuantumTeleportation()
        self.quantum_channel = QuantumChannelSimulator(noise_level=0.01)
        self.quantum_keys = {}  # Store QKD keys for each client pair

    def establish_quantum_connection(self, client_socket, username):
        """Perform QKD handshake with client"""
        print(f"[QUANTUM] Establishing QKD with client {username}...")

        # BB84 Protocol
        n_bits = 256
        alice_bits, alice_bases, qubits = self.bb84_protocol.alice_prepare_qubits(n_bits)

        # Serialize and send qubits
        qubit_data = self.serialize_qubits(qubits)
        qkd_packet = {
            'type': 'qkd_qubits',
            'qubits': qubit_data,
            'n': n_bits,
            'alice_bases': alice_bases
        }
        client_socket.sendall(json.dumps(qkd_packet).encode())

        # Receive Bob's measurement bases and results
        try:
            bob_response = json.loads(client_socket.recv(4096).decode())
            bob_bases = bob_response['bases']
            bob_measurements = bob_response['measurements']
        except:
            print(f"[QUANTUM] Failed to receive QKD response from {username}")
            return None

        # Sift key
        shared_key = self.bb84_protocol.sift_key(alice_bits, alice_bases, bob_bases, bob_measurements)

        # Eavesdropping detection
        qber, is_secure = self.bb84_protocol.detect_eavesdropping(alice_bits, shared_key)

        if not is_secure:
            print(f"[SECURITY ALERT] QBER = {qber:.2%} - Eavesdropping detected for {username}!")
            client_socket.sendall(json.dumps({'type': 'qkd_failed', 'reason': 'eavesdropping'}).encode())
            return None

        # Store the quantum key
        key_id = f"qkd_{username}_{int(time.time())}"
        self.quantum_keys[key_id] = ''.join(map(str, shared_key[:128]))  # Use first 128 bits

        print(f"[QUANTUM] QKD successful for {username}. Key length: {len(shared_key)} bits, QBER: {qber:.2%}")

        # Confirm successful QKD
        client_socket.sendall(json.dumps({
            'type': 'qkd_success',
            'key_id': key_id,
            'key_length': len(shared_key)
        }).encode())

        return key_id

    def serialize_qubits(self, qubits):
        """Serialize quantum states for transmission"""
        return [qubit.to_dict() for qubit in qubits]

    def distribute_entanglement(self):
        """Pre-distribute EPR pairs to all client pairs"""
        client_names = [client['username'] for client in self.active_clients]

        for i, client_a in enumerate(self.active_clients):
            for client_b in self.active_clients[i+1:]:
                # Create EPR pair
                qubit_a, qubit_b = self.teleportation.create_epr_pair()

                # Store in registry
                pair_id = f"{client_a['username']}-{client_b['username']}"
                self.entanglement_registry[pair_id] = {
                    'qubit_a': qubit_a,
                    'qubit_b': qubit_b,
                    'created_at': time.time(),
                    'client_a': client_a['username'],
                    'client_b': client_b['username']
                }

                # Send qubits to clients
                self.send_epr_qubit(client_a['socket'], qubit_a, f"EPR for {client_b['username']}")
                self.send_epr_qubit(client_b['socket'], qubit_b, f"EPR for {client_a['username']}")

                print(f"[ENTANGLEMENT] Distributed EPR pair between {client_a['username']} and {client_b['username']}")

    def send_epr_qubit(self, client_socket, qubit, partner):
        """Send EPR qubit to client"""
        try:
            epr_packet = {
                'type': 'epr_qubit',
                'qubit': qubit.to_dict(),
                'partner': partner,
                'timestamp': time.time()
            }
            client_socket.sendall(json.dumps(epr_packet).encode())
        except Exception as e:
            print(f"[ERROR] Failed to send EPR qubit: {e}")

    def quantum_broadcast(self, sender_client, message):
        """Broadcast message using quantum teleportation"""
        # First encrypt with PQC, then encode quantum
        encrypted_message = self.encrypt_with_pqc(message, sender_client)

        for client in self.active_clients:
            if client['username'] != sender_client['username']:
                try:
                    # Get shared EPR pair
                    pair_id = f"{sender_client['username']}-{client['username']}"
                    if pair_id not in self.entanglement_registry:
                        pair_id = f"{client['username']}-{sender_client['username']}"

                    if pair_id in self.entanglement_registry:
                        epr_pair = self.entanglement_registry[pair_id]

                        # Perform quantum teleportation
                        decoded_msg, bell_bits = self.teleportation.teleport_message(
                            encrypted_message,
                            epr_pair['qubit_a'],
                            epr_pair['qubit_b']
                        )

                        # Send classical bits to receiver
                        teleport_packet = {
                            'type': 'quantum_teleport',
                            'sender': sender_client['username'],
                            'receiver': client['username'],
                            'bell_measurement': bell_bits,
                            'message_length': len(encrypted_message),
                            'timestamp': time.time()
                        }
                        client['socket'].sendall(json.dumps(teleport_packet).encode())

                        print(f"[QUANTUM TELEPORT] {sender_client['username']} → {client['username']}: Bell={bell_bits}")
                    else:
                        print(f"[WARNING] No entanglement found between {sender_client['username']} and {client['username']}")

                except Exception as e:
                    print(f"[ERROR] Quantum broadcast failed for {client['username']}: {e}")

    def encrypt_with_pqc(self, message, client):
        """Encrypt message using the client's PQC method"""
        # This is a placeholder - in practice, you'd use the client's chosen PQC algorithm
        # For now, return the message as-is (quantum layer will handle encoding)
        return message

    def handle_quantum_message(self, client_socket, data):
        """Handle incoming quantum protocol messages"""
        try:
            msg = json.loads(data.decode())

            if msg['type'] == 'quantum_message':
                # Handle direct quantum message
                sender = msg.get('sender')
                receiver = msg.get('receiver')
                bell_measurement = msg.get('bell_measurement')

                # Find sender client
                sender_client = None
                for client in self.active_clients:
                    if client['username'] == sender:
                        sender_client = client
                        break

                if sender_client:
                    # Reconstruct message using teleportation
                    # This is simplified - in practice, you'd use the stored EPR pair
                    reconstructed_msg = f"Quantum message from {sender}: {bell_measurement}"
                    self.quantum_broadcast(sender_client, reconstructed_msg)

        except Exception as e:
            print(f"[ERROR] Failed to handle quantum message: {e}")

    def add_client(self, username, client_socket, traditional_key):
        """Add a client to the quantum server"""
        client_info = {
            'username': username,
            'socket': client_socket,
            'traditional_key': traditional_key,
            'quantum_key_id': None,
            'epr_qubits': {}
        }
        self.active_clients.append(client_info)
        return client_info

    def remove_client(self, username):
        """Remove a client and clean up entanglement"""
        self.active_clients = [c for c in self.active_clients if c['username'] != username]

        # Remove related entanglement pairs
        pairs_to_remove = []
        for pair_id, pair_data in self.entanglement_registry.items():
            if pair_data['client_a'] == username or pair_data['client_b'] == username:
                pairs_to_remove.append(pair_id)

        for pair_id in pairs_to_remove:
            del self.entanglement_registry[pair_id]

# Global quantum server instance
quantum_server = QuantumServer()


# Main function
def main():
    global ElgamalKey
    ElgamalKey = el_gamal.generate_public_key()

    # Choose encryption method
    global flagmethod
    flagmethod = chooseMethod()

    # Initialize quantum server
    print("[QUANTUM] Initializing Quantum Secure Chat Server...")
    print("[QUANTUM] Quantum channel noise level: 1%")
    print("[QUANTUM] BB84 QKD protocol ready")
    print("[QUANTUM] Quantum teleportation protocol ready")

    # Creating the socket class object
    # AF_INET: we are going to use IPv4 addresses
    # SOCK_STREAM: we are using TCP packets for communication
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Creating a try catch block
    try:
        server.bind((HOST, PORT))
        print(f"[QUANTUM] Running Quantum Secure Chat Server on {HOST} {PORT}")
        print("[QUANTUM] Server supports hybrid encryption: PQC + Quantum protocols")
    except:
        print(f"Unable to bind to host {HOST} and port {PORT}")
        return

    # Set server limit
    server.listen(LISTENER_LIMIT)

    # This while loop will keep listening to client connections
    while True:
        client, address = server.accept()
        print(f"[QUANTUM] New connection from {address[0]} {address[1]}")
        key = ""
        threading.Thread(target=quantum_client_handler, args=(client, key)).start()


if __name__ == '__main__':
    main()