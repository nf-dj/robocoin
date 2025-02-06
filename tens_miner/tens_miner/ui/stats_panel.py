"""Statistics panel for mining progress."""

from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel
from PyQt6.QtCore import Qt
from ..mining import OPS_PER_HASH
from .time_format import format_time_estimate

class StatsPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()
        self.setLayout(layout)

        # Create labels for stats
        self.elapsed_label = QLabel("Time: 0.0s")
        self.hashrate_label = QLabel("Hash Rate: 0 h/s")
        self.tops_label = QLabel("TOPS: 0.000000")
        self.total_label = QLabel("Total Hashes: 0")
        self.coins_label = QLabel("TensCoins Mined: 0")
        self.best_label = QLabel("Best: 0 bits")
        self.target_label = QLabel("Target: ? zeros")
        self.estimate_label = QLabel("Est. Time: N/A")

        # Add stats labels to grid
        layout.addWidget(QLabel("Stats:"), 0, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.elapsed_label, 1, 0)
        layout.addWidget(self.hashrate_label, 1, 1)
        layout.addWidget(self.tops_label, 2, 0)
        layout.addWidget(self.total_label, 2, 1)
        layout.addWidget(self.target_label, 3, 0)
        layout.addWidget(self.best_label, 3, 1)
        layout.addWidget(self.estimate_label, 4, 0)
        layout.addWidget(self.coins_label, 4, 1)

    def update_stats(self, data):
        """Update stats from progress data."""
        elapsed = data['elapsed_time']
        attempts = data['attempts']
        best_bits = data['best_zero_bits']
        
        # Calculate hash rate and TOPS
        hash_rate = attempts / elapsed if elapsed > 0 else 0
        tops = (hash_rate * OPS_PER_HASH) / 1e12
        
        # Update stat labels
        self.elapsed_label.setText(f"Time: {elapsed:.1f}s")
        self.hashrate_label.setText(f"Hash Rate: {hash_rate:.1f} h/s")
        self.tops_label.setText(f"TOPS: {tops:.6f}")
        self.total_label.setText(f"Total Hashes: {attempts:,}")
        self.best_label.setText(f"Best: {best_bits} bits")
        
        # Update time estimate
        target_text = self.target_label.text()
        try:
            target_bits = int(target_text.split()[1])
            if hash_rate > 0:
                avg_hashes_needed = 2 ** target_bits
                seconds = avg_hashes_needed / hash_rate
                self.estimate_label.setText(f"Est. Time: {format_time_estimate(seconds)}")
        except:
            pass

    def set_target(self, target_bits):
        """Set target difficulty."""
        self.target_label.setText(f"Target: {target_bits} zeros")

    def increment_coins(self):
        """Increment mined coins counter."""
        current = int(self.coins_label.text().split(":")[1])
        self.coins_label.setText(f"TensCoins Mined: {current + 1}")

    def reset(self):
        """Reset all stats to initial values."""
        self.elapsed_label.setText("Time: 0.0s")
        self.hashrate_label.setText("Hash Rate: 0 h/s")
        self.tops_label.setText("TOPS: 0.000000")
        self.total_label.setText("Total Hashes: 0")
        self.best_label.setText("Best: 0 bits")
        self.estimate_label.setText("Est. Time: N/A")