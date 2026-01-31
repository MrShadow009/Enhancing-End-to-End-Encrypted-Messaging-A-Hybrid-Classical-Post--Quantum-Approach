import numpy as np
import random
import time
from typing import List, Dict, Any
from quantum_state_manager import QuantumState

class QuantumChannelSimulator:
    def __init__(self, noise_level: float = 0.01, eavesdropping_probability: float = 0.0):
        self.noise_level = noise_level
        self.eavesdropping_probability = eavesdropping_probability
        self.set_eavesdropping(eavesdropping_probability)
        self.transmission_log = []

    def set_eavesdropping(self, probability: float):
        """Enable/disable eavesdropping with given probability"""
        self.eavesdropping_enabled = probability > 0
        self.eavesdropping_probability = probability

    def transmit_quantum_state(self, quantum_state: QuantumState, sender: str = "Alice", receiver: str = "Bob") -> QuantumState:
        """Simulate transmission of quantum state through noisy channel"""
        transmitted_state = QuantumState(
            np.copy(quantum_state.amplitudes),
            quantum_state.basis,
            quantum_state.num_qubits
        )

        # Add noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, len(transmitted_state.amplitudes))
            noisy_amplitudes = transmitted_state.amplitudes + noise
            # Renormalize
            norm = np.linalg.norm(noisy_amplitudes)
            if norm > 0:
                transmitted_state.amplitudes = noisy_amplitudes / norm

        # Simulate eavesdropping
        if self.eavesdropping_enabled and random.random() < self.eavesdropping_probability:
            self.simulate_eavesdropping(transmitted_state, sender, receiver)

        # Log transmission
        self.transmission_log.append({
            'timestamp': time.time(),
            'sender': sender,
            'receiver': receiver,
            'original_state': quantum_state.to_dict(),
            'transmitted_state': transmitted_state.to_dict(),
            'noise_level': self.noise_level,
            'eavesdropped': self.eavesdropping_enabled and random.random() < self.eavesdropping_probability
        })

        return transmitted_state

    def simulate_eavesdropping(self, quantum_state: QuantumState, sender: str, receiver: str):
        """Simulate an eavesdropper intercepting and measuring the quantum state"""
        # Eavesdropper chooses random basis
        eavesdropper_basis = random.choice(['computational', 'hadamard'])

        # Measure in chosen basis (this collapses the state)
        outcome, collapsed_state = quantum_state.measure(eavesdropper_basis)

        # Log eavesdropping attempt
        print(f"[EAVESDROPPING ALERT] Eavesdropper intercepted message from {sender} to {receiver}")
        print(f"[EAVESDROPPING ALERT] Measured in {eavesdropper_basis} basis, got outcome: {outcome}")

        # Resend the collapsed state (in a real scenario, this would be detected)
        quantum_state.amplitudes = collapsed_state.amplitudes
        quantum_state.basis = collapsed_state.basis

    def get_channel_stats(self) -> Dict[str, Any]:
        """Get statistics about channel performance"""
        total_transmissions = len(self.transmission_log)
        eavesdropped_count = sum(1 for log in self.transmission_log if log['eavesdropped'])

        return {
            'total_transmissions': total_transmissions,
            'eavesdropped_transmissions': eavesdropped_count,
            'eavesdropping_rate': eavesdropped_count / total_transmissions if total_transmissions > 0 else 0,
            'noise_level': self.noise_level,
            'eavesdropping_probability': self.eavesdropping_probability
        }

    def reset_logs(self):
        """Clear transmission logs"""
        self.transmission_log = []

    def enable_error_correction(self, enable: bool = True):
        """Enable or disable active error correction in the channel"""
        self.error_correction_enabled = enable

    def enable_active_mitigation(self, enable: bool = True):
        """Enable or disable active error mitigation"""
        self.active_error_mitigation = enable

    def transmit_with_error_correction(self, quantum_state: QuantumState, sender: str = "Alice", receiver: str = "Bob") -> QuantumState:
        """Transmit quantum state with active error correction"""
        # First, encode the state with error correction if enabled
        if self.error_correction_enabled and quantum_state.error_correction:
            encoded_state = self._encode_with_error_correction(quantum_state)
        else:
            encoded_state = quantum_state

        # Transmit through noisy channel
        transmitted_state = self.transmit_quantum_state(encoded_state, sender, receiver)

        # Apply active error mitigation if enabled
        if self.active_error_mitigation and quantum_state.error_correction:
            corrected_state = self._apply_active_error_mitigation(transmitted_state, quantum_state)
            return corrected_state
        else:
            return transmitted_state

    def _encode_with_error_correction(self, quantum_state: QuantumState) -> QuantumState:
        """Encode quantum state with error correction codes"""
        # This is a simplified encoding - real implementation would be much more complex
        if quantum_state.error_correction == quantum_state.error_correction.REPETITION:
            # Simple repetition encoding
            encoded_amplitudes = np.tile(quantum_state.amplitudes, quantum_state.stabilizer_code.code_distance)
            return QuantumState(
                encoded_amplitudes,
                quantum_state.basis,
                quantum_state.num_qubits * quantum_state.stabilizer_code.code_distance,
                error_correction=quantum_state.error_correction,
                stabilizer_code=quantum_state.stabilizer_code
            )
        else:
            # For other codes, return original state (simplified)
            return quantum_state

    def _apply_active_error_mitigation(self, transmitted_state: QuantumState, original_state: QuantumState) -> QuantumState:
        """Apply active error mitigation techniques"""
        # Measure syndrome
        syndrome = transmitted_state.measure_syndrome()

        # Apply error correction
        transmitted_state.apply_error_correction(syndrome)

        # Verify correction with additional syndrome measurement
        verification_syndrome = transmitted_state.measure_syndrome()

        # If syndromes don't match, attempt additional correction
        if not np.array_equal(syndrome.syndrome_bits, verification_syndrome.syndrome_bits):
            # Additional correction attempt (simplified)
            transmitted_state.apply_error_correction(verification_syndrome)

        return transmitted_state

    def perform_channel_calibration(self, num_samples: int = 100) -> Dict[str, Any]:
        """Calibrate channel parameters for optimal error correction"""
        calibration_results = {
            'optimal_noise_threshold': 0.0,
            'recommended_error_correction': None,
            'channel_fidelity': 0.0,
            'calibration_samples': num_samples
        }

        # Test different noise levels and error correction schemes
        test_states = []
        for _ in range(num_samples):
            # Create random test state
            amplitudes = np.random.rand(4) + 1j * np.random.rand(4)
            amplitudes /= np.linalg.norm(amplitudes)
            test_state = QuantumState(amplitudes, 'computational', 2)
            test_states.append(test_state)

        # Test without error correction
        fidelities_no_ec = []
        for state in test_states:
            transmitted = self.transmit_quantum_state(state)
            fidelity = self._calculate_fidelity(state, transmitted)
            fidelities_no_ec.append(fidelity)

        avg_fidelity_no_ec = np.mean(fidelities_no_ec)

        # Test with repetition code
        fidelities_with_ec = []
        for state in test_states:
            state.enable_error_correction(state.error_correction.REPETITION, 3)
            transmitted = self.transmit_with_error_correction(state)
            fidelity = self._calculate_fidelity(state, transmitted)
            fidelities_with_ec.append(fidelity)

        avg_fidelity_with_ec = np.mean(fidelities_with_ec)

        # Determine optimal settings
        if avg_fidelity_with_ec > avg_fidelity_no_ec:
            calibration_results['recommended_error_correction'] = 'repetition'
            calibration_results['channel_fidelity'] = avg_fidelity_with_ec
        else:
            calibration_results['recommended_error_correction'] = None
            calibration_results['channel_fidelity'] = avg_fidelity_no_ec

        calibration_results['optimal_noise_threshold'] = self.noise_level * 0.8  # Conservative threshold

        return calibration_results

    def _calculate_fidelity(self, state1: QuantumState, state2: QuantumState) -> float:
        """Calculate quantum state fidelity"""
        overlap = np.abs(np.dot(np.conj(state1.amplitudes), state2.amplitudes)) ** 2
        return float(overlap)

    def get_error_mitigation_stats(self) -> Dict[str, Any]:
        """Get statistics about error mitigation performance"""
        total_transmissions = len(self.transmission_log)
        error_corrected_transmissions = sum(1 for log in self.transmission_log
                                          if log.get('error_correction_applied', False))

        return {
            'total_transmissions': total_transmissions,
            'error_corrected_transmissions': error_corrected_transmissions,
            'error_correction_rate': error_corrected_transmissions / total_transmissions if total_transmissions > 0 else 0,
            'error_correction_enabled': self.error_correction_enabled,
            'active_mitigation_enabled': self.active_error_mitigation
        }
