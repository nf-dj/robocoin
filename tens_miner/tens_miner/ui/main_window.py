"""Main window implementation for the TensCoin miner GUI."""

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QMessageBox
from .mining_controls import MiningControls
from .stats_panel import StatsPanel
from .test_mining import TestMiningThread, TEST_SEED, TEST_DIFFICULTY, EXPECTED_NONCE, EXPECTED_HASH
from ..worker import MiningWorker
from ..wallet import generate_keypair, save_keys, load_wallet
from ..network import RPCClient
from ..utils import print_hex_msb

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensCoin Miner")
        self.worker = None
        self.test_worker = None
        self.rpc = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Mining controls
        self.controls = MiningControls()
        self.controls.start_button.clicked.connect(self.start_mining)
        self.controls.stop_button.clicked.connect(self.stop_mining)
        self.controls.test_button.clicked.connect(self.start_test)
        self.controls.generate_address_button.clicked.connect(self.generate_new_address)
        layout.addWidget(self.controls)

        # Load any existing address
        wallet = load_wallet()
        if wallet:
            self.controls.address_input.setText(wallet[-1]['address'])

        # Stats panel
        self.stats = StatsPanel()
        layout.addWidget(self.stats)

        # Status display
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)

        # Add status message for loaded address
        if wallet and len(wallet) > 0:
            self.status_text.append(f"Loaded saved address: {wallet[-1]['address']}")

        # Window setup
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

    def start_test(self):
        """Start test mining."""
        batch_size = self.controls.get_batch_size()
        if not batch_size:
            self.status_text.append("Error: Invalid batch size")
            return

        device = self.controls.device_combo.currentText()

        # Clear status and reset stats
        self.status_text.clear()
        self.status_text.append(f"Starting test mining with seed: {TEST_SEED}")
        self.status_text.append(f"Target difficulty: {TEST_DIFFICULTY} leading zeros")
        self.status_text.append(f"Expected nonce: {EXPECTED_NONCE}")
        self.status_text.append(f"Expected hash: {EXPECTED_HASH}")

        # Reset stats
        self.stats.reset()
        self.stats.set_target(TEST_DIFFICULTY)

        # Disable controls
        self.controls.set_mining_state(True)

        # Start test mining thread
        self.test_worker = TestMiningThread(device, batch_size)
        self.test_worker.progress.connect(self.update_progress)
        self.test_worker.solution.connect(self.solution_found)
        self.test_worker.status.connect(self.status_update)
        self.test_worker.start()

    def generate_new_address(self):
        """Generate a new TensCoin address locally."""
        try:
            # Generate new keypair
            private_key, address = generate_keypair()
            
            # Save to wallet file
            wallet_path = save_keys(private_key, address)
            
            # Update UI
            self.controls.address_input.setText(address)
            self.status_text.append(f"Generated new address: {address}")
            self.status_text.append(f"Private key saved to: {wallet_path}")
            
            # Show warning about backing up private key
            QMessageBox.warning(
                self,
                "Backup Warning",
                f"Private key saved to {wallet_path}\n\n" \
                "Please back up this file! If you lose it, you will lose access to any mined coins."
            )
            
        except Exception as e:
            self.status_text.append(f"Failed to generate address: {e}")

    def show_test_result(self, nonce_hex, hash_hex):
        """Show test result in a popup."""
        if nonce_hex == EXPECTED_NONCE:
            QMessageBox.information(
                self,
                "Test Result",
                "✓ TEST PASSED!\n\n" \
                f"Found correct nonce:\n" \
                f"Nonce: {nonce_hex}\n" \
                f"Hash:  {hash_hex}"
            )
        else:
            QMessageBox.warning(
                self,
                "Test Result",
                "✗ TEST FAILED!\n\n" \
                f"Found nonce: {nonce_hex}\n" \
                f"Expected:    {EXPECTED_NONCE}\n\n" \
                f"Found hash:  {hash_hex}\n" \
                f"Expected:    {EXPECTED_HASH}"
            )

    def start_mining(self):
        """Start mining to the network."""
        # Get node address
        host, port = self.controls.get_node_address()
        if not host or not port:
            self.status_text.append("Error: Invalid node address (use format host:port)")
            return

        # Validate mining address
        address = self.controls.address_input.text().strip()
        if not address:
            self.status_text.append("Error: Mining address required")
            return

        # Get batch size
        batch_size = self.controls.get_batch_size()
        if not batch_size:
            self.status_text.append("Error: Invalid batch size")
            return

        device = self.controls.device_combo.currentText()

        # Clear status and reset stats
        self.status_text.clear()
        self.status_text.append(f"Connecting to node {host}:{port}...")
        self.stats.reset()

        # Initialize RPC client
        self.rpc = RPCClient(host, port)

        try:
            # Get initial mining info via RPC
            mining_info = self.rpc.get_mining_info()
            target = mining_info.get('target', 'unknown')
            target_bits = mining_info.get('target_bits', 0)
            self.stats.set_target(target_bits)
            self.status_text.append(f"Connected successfully. Need {target_bits} leading zeros")
            self.status_text.append(f"Target: {target}")
            self.status_text.append(f"Mining to address: {address}")
        except Exception as e:
            self.status_text.append(f"Failed to connect to node: {e}")
            return

        # Disable controls
        self.controls.set_mining_state(True)

        # Start mining thread
        self.worker = MiningWorker(self.rpc, batch_size, device, address)
        self.worker.progress.connect(self.update_progress)
        self.worker.solution.connect(self.solution_found)
        self.worker.status.connect(self.status_update)
        self.worker.start()

    def stop_mining(self):
        """Stop mining."""
        # Stop either mining or test worker
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        if self.test_worker:
            self.test_worker.stop()
            self.test_worker.wait()
            self.test_worker = None

        # Re-enable controls
        self.controls.set_mining_state(False)
        self.status_text.append("Mining stopped.")

    def update_progress(self, data):
        """Handle progress updates from worker."""
        self.stats.update_stats(data)

    def status_update(self, message):
        """Handle status updates from worker."""
        self.status_text.append(message)
        # Scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def solution_found(self, data):
        """Handle found solutions."""
        self.stats.increment_coins()

        nonce = print_hex_msb(data['nonce'])
        hash_str = print_hex_msb(data['hash'])
        attempts = data['attempts']
        total_time = data['time']

        self.status_text.append("\nSolution found!")
        self.status_text.append(f"Nonce: {nonce}")
        self.status_text.append(f"Hash:  {hash_str}")

        # If this was a test run, show result popup
        if self.test_worker:
            self.show_test_result(nonce, hash_str)

        # Handle RPC update if not in test mode
        if not self.test_worker:
            try:
                # Get updated mining info after found block
                mining_info = self.rpc.get_mining_info()
                target = mining_info.get('target', 'unknown')
                target_bits = mining_info.get('target_bits', 0)
                self.stats.set_target(target_bits)
                self.status_text.append(f"Updated target: need {target_bits} leading zeros")
                self.status_text.append(f"Target: {target}")
            except Exception as e:
                self.status_text.append(f"Failed to update target: {e}")
