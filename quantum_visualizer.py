import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from quantum_state_manager import QuantumState
import time

class QuantumVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.current_states = []

    def plot_bloch_sphere(self, quantum_state: QuantumState, title="Quantum State on Bloch Sphere"):
        """Display qubit state on Bloch sphere"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.clear()

        # Create Bloch sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot sphere
        self.ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

        # Calculate Bloch vector coordinates
        if quantum_state.num_qubits == 1:
            amplitudes = quantum_state.amplitudes
            if quantum_state.basis == 'computational':
                # |ψ⟩ = α|0⟩ + β|1⟩
                alpha, beta = amplitudes[0], amplitudes[1]
            elif quantum_state.basis == 'hadamard':
                # Convert from Hadamard basis
                H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                computational = H @ amplitudes
                alpha, beta = computational[0], computational[1]
            else:
                alpha, beta = amplitudes[0], amplitudes[1]

            # Bloch coordinates
            theta = 2 * np.arccos(abs(alpha))
            phi = np.angle(beta) - np.angle(alpha)

            x_bloch = np.sin(theta) * np.cos(phi)
            y_bloch = np.sin(theta) * np.sin(phi)
            z_bloch = np.cos(theta)

            # Plot state vector
            self.ax.quiver(0, 0, 0, x_bloch, y_bloch, z_bloch,
                          color='red', linewidth=3, arrow_length_ratio=0.1)

            # Plot |0⟩ and |1⟩ states
            self.ax.quiver(0, 0, 0, 0, 0, 1, color='black', alpha=0.5, linewidth=2)
            self.ax.quiver(0, 0, 0, 0, 0, -1, color='black', alpha=0.5, linewidth=2)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([-1.5, 1.5])

        plt.draw()
        plt.pause(0.1)

    def plot_probability_distribution(self, quantum_state: QuantumState,
                                    title="Measurement Probabilities"):
        """Bar chart of measurement probabilities"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))

        self.ax.clear()

        probabilities = np.abs(quantum_state.amplitudes) ** 2
        states = [f"|{i}⟩" for i in range(len(probabilities))]

        bars = self.ax.bar(states, probabilities, color='skyblue', alpha=0.7)

        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '.3f', ha='center', va='bottom')

        self.ax.set_xlabel('Quantum States')
        self.ax.set_ylabel('Probability')
        self.ax.set_title(title)
        self.ax.set_ylim(0, 1.1)
        self.ax.grid(True, alpha=0.3)

        plt.draw()
        plt.pause(0.1)

    def animate_quantum_channel(self, states_over_time, title="Quantum Channel Evolution"):
        """Animate state evolution through channel"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            self.ax.clear()
            current_state = states_over_time[frame]

            probabilities = np.abs(current_state.amplitudes) ** 2
            states = [f"|{i}⟩" for i in range(len(probabilities))]

            bars = self.ax.bar(states, probabilities, color='skyblue', alpha=0.7)

            self.ax.set_xlabel('Quantum States')
            self.ax.set_ylabel('Probability')
            self.ax.set_title(f"{title} - Time Step {frame}")
            self.ax.set_ylim(0, 1.1)
            self.ax.grid(True, alpha=0.3)

            return bars

        anim = FuncAnimation(self.fig, update, frames=len(states_over_time),
                           interval=500, repeat=True)
        plt.show()

    def plot_qkd_statistics(self, alice_bits, bob_bits, matching_bases,
                           title="BB84 QKD Statistics"):
        """Plot QKD protocol statistics"""
        if self.fig is None:
            self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Bit comparison
        ax1.plot(alice_bits[:100], 'b-', label='Alice', alpha=0.7)
        ax1.plot(bob_bits[:100], 'r-', label='Bob', alpha=0.7)
        ax1.set_title('Alice vs Bob Bits (First 100)')
        ax1.set_xlabel('Bit Position')
        ax1.set_ylabel('Bit Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Matching bases
        ax2.plot(matching_bases[:100], 'g-', linewidth=2)
        ax2.set_title('Matching Bases')
        ax2.set_xlabel('Bit Position')
        ax2.set_ylabel('Match (1) / No Match (0)')
        ax2.grid(True, alpha=0.3)

        # Error rate over time
        window_size = 10
        errors = [1 if a != b else 0 for a, b in zip(alice_bits, bob_bits)]
        error_rates = []

        for i in range(window_size, len(errors)):
            window_errors = sum(errors[i-window_size:i])
            error_rates.append(window_errors / window_size)

        ax3.plot(error_rates, 'r-', linewidth=2)
        ax3.axhline(y=0.11, color='black', linestyle='--', label='Security Threshold (11%)')
        ax3.set_title('Quantum Bit Error Rate (QBER)')
        ax3.set_xlabel('Window Position')
        ax3.set_ylabel('Error Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Key distribution
        ax4.hist([alice_bits, bob_bits], bins=2, alpha=0.7,
                label=['Alice', 'Bob'], color=['blue', 'red'])
        ax4.set_title('Bit Distribution')
        ax4.set_xlabel('Bit Value')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def plot_teleportation_fidelity(self, fidelities_over_time,
                                   title="Quantum Teleportation Fidelity"):
        """Plot teleportation success over time"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))

        self.ax.clear()

        self.ax.plot(fidelities_over_time, 'b-', linewidth=2, marker='o', markersize=4)
        self.ax.axhline(y=0.9, color='green', linestyle='--', label='Good Fidelity (90%)')
        self.ax.axhline(y=0.5, color='red', linestyle='--', label='Poor Fidelity (50%)')

        self.ax.set_xlabel('Teleportation Event')
        self.ax.set_ylabel('Fidelity')
        self.ax.set_title(title)
        self.ax.set_ylim(0, 1.1)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # Add statistics
        avg_fidelity = np.mean(fidelities_over_time)
        success_rate = sum(1 for f in fidelities_over_time if f > 0.9) / len(fidelities_over_time)

        self.ax.text(0.02, 0.98, '.3f',
                    transform=self.ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.draw()
        plt.pause(0.1)

    def create_dashboard(self, quantum_data):
        """Create a comprehensive quantum dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Channel fidelity over time
        if 'channel_fidelity_history' in quantum_data:
            ax1.plot(quantum_data['channel_fidelity_history'], 'b-', linewidth=2)
            ax1.set_title('Quantum Channel Fidelity')
            ax1.set_xlabel('Transmission')
            ax1.set_ylabel('Fidelity')
            ax1.grid(True, alpha=0.3)

        # QKD key generation rate
        if 'qkd_keys_generated' in quantum_data:
            ax2.plot(quantum_data['qkd_keys_generated'], 'g-', linewidth=2)
            ax2.set_title('QKD Keys Generated')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Keys')
            ax2.grid(True, alpha=0.3)

        # Teleportation success rate
        if 'teleportation_fidelities' in quantum_data:
            ax3.plot(quantum_data['teleportation_fidelities'], 'r-', linewidth=2)
            ax3.axhline(y=0.9, color='black', linestyle='--', alpha=0.7)
            ax3.set_title('Teleportation Success Rate')
            ax3.set_xlabel('Teleportation Event')
            ax3.set_ylabel('Fidelity')
            ax3.grid(True, alpha=0.3)

        # Active quantum connections
        if 'active_connections' in quantum_data:
            ax4.plot(quantum_data['active_connections'], 'purple', linewidth=2)
            ax4.set_title('Active Quantum Connections')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Connections')
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Quantum Secure Chat Dashboard', fontsize=16)
        plt.tight_layout()
        plt.show()

def visualize_quantum_demo():
    """Demonstrate quantum visualization capabilities"""
    visualizer = QuantumVisualizer()

    # Create a superposition state |+⟩ = (|0⟩ + |1⟩)/√2
    plus_state = QuantumState(amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                              basis='hadamard', num_qubits=1)

    print("Visualizing |+⟩ state on Bloch sphere...")
    visualizer.plot_bloch_sphere(plus_state, "Superposition State |+⟩")

    time.sleep(2)

    # Create a mixed state
    mixed_state = QuantumState(amplitudes=np.array([0.8, 0.6]),
                               basis='computational', num_qubits=1)

    print("Visualizing mixed state...")
    visualizer.plot_bloch_sphere(mixed_state, "Mixed Quantum State")

    time.sleep(2)

    # Show probability distribution
    print("Showing probability distribution...")
    visualizer.plot_probability_distribution(plus_state, "Probability Distribution for |+⟩")

    time.sleep(2)

    # Simulate channel evolution
    print("Simulating quantum channel with noise...")
    states_over_time = [plus_state]
    for i in range(10):
        # Simulate noisy channel
        noisy_amplitudes = plus_state.amplitudes + np.random.normal(0, 0.1, 2)
        noisy_amplitudes /= np.linalg.norm(noisy_amplitudes)
        noisy_state = QuantumState(amplitudes=noisy_amplitudes,
                                  basis='hadamard', num_qubits=1)
        states_over_time.append(noisy_state)

    visualizer.animate_quantum_channel(states_over_time, "Noisy Quantum Channel")

if __name__ == "__main__":
    visualize_quantum_demo()
