import numpy as np
from typing import List, Tuple, Dict, Any
from quantum_state_manager import QuantumState, EntangledState, create_bell_state, encode_message_to_quantum_state, decode_quantum_state_to_message
import time

class QuantumTeleportation:
    def __init__(self):
        self.bell_states = {
            'phi_plus': np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),   # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            'phi_minus': np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]),  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            'psi_plus': np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]),    # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            'psi_minus': np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])   # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        }
        self.teleportation_log = []

    def create_epr_pair(self, state_type: str = 'phi_plus') -> Tuple[QuantumState, QuantumState]:
        """Create an entangled EPR pair for teleportation"""
        entangled_state = create_bell_state(state_type)

        # Split into two qubits (Alice's and Bob's)
        qubit_a = QuantumState(
            amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
            basis='bell',
            num_qubits=1
        )
        qubit_b = QuantumState(
            amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
            basis='bell',
            num_qubits=1
        )

        # Mark as entangled
        qubit_a.entangled_with = qubit_b
        qubit_b.entangled_with = qubit_a

        return qubit_a, qubit_b

    def encode_message_to_qubit(self, message: str) -> QuantumState:
        """Encode a classical message as a quantum state for teleportation"""
        return encode_message_to_quantum_state(message)

    def perform_bell_measurement(self, message_qubit: QuantumState,
                               alice_epr_qubit: QuantumState) -> Tuple[int, int]:
        """Perform Bell basis measurement on Alice's side"""
        # Combine message qubit with Alice's EPR qubit
        # This creates a 4-qubit state, but we'll simulate the measurement

        # Simplified: randomly select one of the 4 Bell states based on probabilities
        # In a real quantum computer, this would be a true Bell measurement

        # Calculate joint probabilities (simplified model)
        bell_probabilities = np.array([0.25, 0.25, 0.25, 0.25])  # Equal probability for simulation

        # Sample which Bell state
        bell_measurement = np.random.choice(4, p=bell_probabilities)

        # Bell measurement gives 2 classical bits
        # 00, 01, 10, 11 corresponding to Φ⁺, Φ⁻, Ψ⁺, Ψ⁻
        bit1 = bell_measurement // 2  # First bit
        bit2 = bell_measurement % 2   # Second bit

        return bit1, bit2

    def apply_pauli_correction(self, bob_epr_qubit: QuantumState,
                             bell_measurement_bits: Tuple[int, int]) -> QuantumState:
        """Bob applies Pauli operators based on Bell measurement results"""
        bit1, bit2 = bell_measurement_bits

        corrected_qubit = QuantumState(
            np.copy(bob_epr_qubit.amplitudes),
            bob_epr_qubit.basis,
            bob_epr_qubit.num_qubits
        )

        # Apply corrections based on Bell measurement
        # 00: I ⊗ I (no correction)
        # 01: I ⊗ X
        # 10: Z ⊗ I
        # 11: Z ⊗ X

        if bit2 == 1:  # X correction
            corrected_qubit.apply_pauli_x()

        if bit1 == 1:  # Z correction
            corrected_qubit.apply_pauli_z()

        return corrected_qubit

    def teleport_quantum_state(self, message_qubit: QuantumState,
                              alice_epr_qubit: QuantumState,
                              bob_epr_qubit: QuantumState) -> Tuple[QuantumState, Tuple[int, int]]:
        """Complete quantum teleportation protocol"""
        # Step 1: Bell measurement by Alice
        bell_bits = self.perform_bell_measurement(message_qubit, alice_epr_qubit)

        # Step 2: Send classical bits to Bob (over classical channel)
        # This is done by the calling function

        # Step 3: Bob applies corrections
        teleported_state = self.apply_pauli_correction(bob_epr_qubit, bell_bits)

        # Log the teleportation
        self.teleportation_log.append({
            'timestamp': time.time(),
            'original_state': message_qubit.to_dict(),
            'bell_measurement': bell_bits,
            'teleported_state': teleported_state.to_dict(),
            'fidelity': self.calculate_teleportation_fidelity(message_qubit, teleported_state)
        })

        return teleported_state, bell_bits

    def teleport_message(self, message: str, alice_epr_qubit: QuantumState,
                        bob_epr_qubit: QuantumState) -> Tuple[str, Tuple[int, int]]:
        """Teleport a classical message using quantum teleportation"""
        # Encode message as quantum state
        message_state = self.encode_message_to_qubit(message)

        # Perform teleportation
        teleported_state, bell_bits = self.teleport_quantum_state(
            message_state, alice_epr_qubit, bob_epr_qubit
        )

        # Decode back to message
        decoded_message = decode_quantum_state_to_message(teleported_state)

        return decoded_message, bell_bits

    def calculate_teleportation_fidelity(self, original_state: QuantumState,
                                       teleported_state: QuantumState) -> float:
        """Calculate how well the teleportation preserved the quantum state"""
        # Simplified fidelity calculation
        overlap = np.abs(np.dot(np.conj(original_state.amplitudes), teleported_state.amplitudes)) ** 2
        return float(overlap)

    def get_teleportation_stats(self) -> Dict[str, Any]:
        """Get statistics about teleportation performance"""
        if not self.teleportation_log:
            return {'avg_fidelity': 1.0, 'success_rate': 1.0, 'total_teleportations': 0}

        fidelities = [entry['fidelity'] for entry in self.teleportation_log]

        return {
            'avg_fidelity': np.mean(fidelities),
            'min_fidelity': np.min(fidelities),
            'max_fidelity': np.max(fidelities),
            'success_rate': sum(1 for f in fidelities if f > 0.9) / len(fidelities),  # >90% fidelity
            'total_teleportations': len(self.teleportation_log)
        }

class QuantumMessagePacket:
    """Represents a quantum-encoded message packet"""
    def __init__(self, sender: str, message: str, qkd_key_id: str = None,
                 entanglement_id: str = None):
        self.sender = sender
        self.original_message = message
        self.qkd_key_id = qkd_key_id or f"qkd_{int(time.time())}"
        self.entanglement_id = entanglement_id or f"epr_{int(time.time())}"
        self.timestamp = time.time()
        self.quantum_state = None
        self.bell_measurement_results = None
        self.integrity_check = None

    def encode_quantum(self) -> Dict[str, Any]:
        """Encode the message as a quantum packet"""
        # Create quantum state from message
        self.quantum_state = encode_message_to_quantum_state(self.original_message)

        # Generate integrity check (simplified hash)
        self.integrity_check = hash(self.original_message + str(self.timestamp))

        return {
            'sender': self.sender,
            'quantum_state': self.quantum_state.to_dict(),
            'entanglement_id': self.entanglement_id,
            'qkd_key_id': self.qkd_key_id,
            'algorithm_used': self.algorithm_used,  # Include algorithm information
            'timestamp': self.timestamp,
            'integrity_check': self.integrity_check,
            'bell_measurement_results': self.bell_measurement_results
        }

    @classmethod
    def decode_quantum(cls, packet_data: Dict[str, Any]) -> 'QuantumMessagePacket':
        """Decode a quantum packet back to message"""
        packet = cls(
            sender=packet_data['sender'],
            message="",  # Will be decoded from quantum state
            qkd_key_id=packet_data.get('qkd_key_id'),
            entanglement_id=packet_data.get('entanglement_id')
        )

        packet.timestamp = packet_data['timestamp']
        packet.integrity_check = packet_data['integrity_check']
        packet.bell_measurement_results = packet_data.get('bell_measurement_results')
        packet.algorithm_used = packet_data.get('algorithm_used')  # Extract algorithm information

        # Decode quantum state
        quantum_state_data = packet_data['quantum_state']
        packet.quantum_state = QuantumState.from_dict(quantum_state_data)
        packet.original_message = decode_quantum_state_to_message(packet.quantum_state)

        return packet

    def verify_integrity(self) -> bool:
        """Verify message integrity"""
        expected_check = hash(self.original_message + str(self.timestamp))
        return expected_check == self.integrity_check

def simulate_quantum_network(nodes: List[str]) -> Dict[str, Dict[str, EntangledState]]:
    """Simulate a quantum network with entanglement between all node pairs"""
    entanglement_registry = {}

    for i, node_a in enumerate(nodes):
        for node_b in nodes[i+1:]:
            # Create EPR pair between nodes
            epr_pair = create_bell_state('phi_plus')
            pair_id = f"{node_a}-{node_b}"

            entanglement_registry[pair_id] = epr_pair
            # Also store reverse lookup
            entanglement_registry[f"{node_b}-{node_a}"] = epr_pair

    return entanglement_registry

def route_quantum_message(sender: str, receiver: str,
                         entanglement_registry: Dict[str, Dict[str, EntangledState]],
                         message: str) -> Tuple[str, Tuple[int, int]]:
    """Route a message using quantum teleportation through the network"""
    teleportation = QuantumTeleportation()

    # Get shared EPR pair
    pair_id = f"{sender}-{receiver}"
    if pair_id not in entanglement_registry:
        pair_id = f"{receiver}-{sender}"

    if pair_id not in entanglement_registry:
        raise ValueError(f"No entanglement found between {sender} and {receiver}")

    epr_pair = entanglement_registry[pair_id]

    # For simplicity, use the first qubit as Alice's, second as Bob's
    alice_qubit = epr_pair.qubits[0]
    bob_qubit = epr_pair.qubits[1]

    # Perform teleportation
    decoded_message, bell_bits = teleportation.teleport_message(
        message, alice_qubit, bob_qubit
    )

    return decoded_message, bell_bits
