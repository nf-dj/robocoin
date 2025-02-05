"""API endpoints for TensCoin miner."""

from flask import Flask, request, jsonify
from tens_miner import TensHashMiner

app = Flask(__name__)
miner = TensHashMiner()

@app.route('/api/generate_address', methods=['POST'])
def generate_address():
    """Generate new TensCoin address."""
    try:
        address = miner.generate_new_address()
        return jsonify({'address': address})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_mining', methods=['POST'])
def start_mining():
    """Start mining to specified address."""
    try:
        data = request.get_json()
        address = data.get('address')
        if not address:
            return jsonify({'error': 'Address required'}), 400
            
        miner.start(address=address)
        return jsonify({'status': 'Mining started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_mining', methods=['POST'])
def stop_mining():
    """Stop mining."""
    try:
        miner.stop()
        return jsonify({'status': 'Mining stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)