import random
import numpy as np
from typing import List, Tuple, Dict, Any
from quantum_state_manager import QuantumState

class BB84Protocol:
    def __init__(self):
        self.bases = ['rectilinear', 'diagonal']  # + and × bases

    def alice_prepare_qubits(self, n: int) -> Tuple[List[int], List[str], List[QuantumState]]:
        """Alice prepares random qubits in random bases"""
        bits = [random.randint(0, 1) for _ in range(n)]
        bases = [random.choice(self.bases) for _ in range(n)]
        qubits = []

        for bit, basis in zip(bits, bases):
            qubit = QuantumState(amplitudes=np.zeros(2), basis='computational', num_qubits=1)

            if bit == 0:
                qubit.amplitudes[0] = 1.0  # |0⟩
            else:
                qubit.amplitudes[1] = 1.0  # |1⟩

            if basis == 'diagonal':
                qubit.apply_hadamard()  # Create |+⟩ or |−⟩

            qubits.append(qubit)

        return bits, bases, qubits

    def bob_measure_qubits(self, qubits: List[QuantumState], n: int) -> Tuple[List[str], List[int]]:
        """Bob measures qubits in random bases"""
        bob_bases = [random.choice(self.bases) for _ in range(n)]
        measurements = []

        for qubit, basis in zip(qubits, bob_bases):
            if basis == 'diagonal':
                qubit.apply_hadamard()  # Change to diagonal basis

            outcome, _ = qubit.measure('computational')
            measurements.append(outcome)

        return bob_bases, measurements

    def sift_key(self, alice_bits, alice_bases, bob_bases, bob_measurements):
        """Extract shared key from matching bases"""
        shared_key = []
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                shared_key.append(bob_measurements[i])
        return shared_key

    def detect_eavesdropping(self, alice_bits, shared_key, sample_size: int = 50):
        """Check for eavesdropping via error rate"""
        if len(shared_key) < sample_size:
            sample_size = len(shared_key)

        if sample_size == 0:
            return 0.0, True

        sample_indices = random.sample(range(len(shared_key)), sample_size)
        errors = sum(1 for i in sample_indices if alice_bits[i] != shared_key[i])
        qber = errors / sample_size  # Quantum Bit Error Rate

        return qber, qber < 0.11  # Threshold for security
