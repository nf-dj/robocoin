"""Local wallet management for TensCoin."""

import os
import json
import time
import hashlib
import ecdsa
from typing import Tuple

def bech32_polymod(values):
    """Internal function that computes the Bech32 checksum."""
    generator = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk

def bech32_hrp_expand(hrp):
    """Expand the HRP into values for checksum computation."""
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

def bech32_verify_checksum(hrp, data):
    """Verify a checksum given HRP and converted data characters."""
    return bech32_polymod(bech32_hrp_expand(hrp) + data) == 1

def bech32_create_checksum(hrp, data):
    """Compute the checksum values given HRP and data."""
    values = bech32_hrp_expand(hrp) + data
    polymod = bech32_polymod(values + [0,0,0,0,0,0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

def convertbits(data, frombits, tobits, pad=True):
    """General power-of-2 base conversion."""
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    max_acc = (1 << (frombits + tobits - 1)) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = ((acc << frombits) | value) & max_acc
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret

def encode_bech32(hrp: str, witver: int, witprog: bytes) -> str:
    """Encode a segwit address."""
    CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    
    data = [witver] + convertbits(witprog, 8, 5)
    checksum = bech32_create_checksum(hrp, data)
    return hrp + '1' + ''.join([CHARSET[d] for d in data + checksum])

def generate_keypair() -> Tuple[str, str]:
    """Generate a new private/public key pair for TensCoin.
    
    Returns:
        Tuple of (private_key_wif, public_address)
    """
    # Generate private key
    private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
    private_bytes = private_key.to_string()
    
    # Generate public key
    public_key = private_key.get_verifying_key().to_string()
    
    # Hash public key
    sha256_hash = hashlib.sha256(public_key).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    
    # Create bech32 address with "tens" prefix
    address = encode_bech32("tens", 0, ripemd160_hash)
    
    # Create WIF private key
    version = b'\x80'  # mainnet private key
    extended_key = version + private_bytes
    checksum = hashlib.sha256(hashlib.sha256(extended_key).digest()).digest()[:4]
    wif = ''.join(['{:02x}'.format(b) for b in extended_key + checksum])
    
    return wif, address

def save_keys(private_key: str, address: str) -> str:
    """Save key pair to file.
    
    Args:
        private_key: Private key in WIF format
        address: Bech32 address starting with 'tens1'
        
    Returns:
        Path to wallet file
    """
    # Create ~/.tenscoin directory if it doesn't exist
    home = os.path.expanduser("~")
    wallet_dir = os.path.join(home, ".tenscoin")
    os.makedirs(wallet_dir, exist_ok=True)

    wallet_file = os.path.join(wallet_dir, "wallet.json")
    
    # Read existing wallet if it exists
    wallet = []
    if os.path.exists(wallet_file):
        try:
            with open(wallet_file, 'r') as f:
                content = f.read().strip()
                if content:  # Only try to load if file is not empty
                    existing_data = json.loads(content)
                    # Convert to list if it's a single entry
                    if isinstance(existing_data, dict):
                        wallet = [existing_data]
                    elif isinstance(existing_data, list):
                        wallet = existing_data
                    else:
                        # Create backup if data is invalid
                        backup_path = wallet_file + '.bak'
                        import shutil
                        shutil.copy2(wallet_file, backup_path)
                        print(f"Created backup of wallet at {backup_path}")
        except (json.JSONDecodeError, Exception) as e:
            # If file is corrupted, create a backup before starting fresh
            if os.path.getsize(wallet_file) > 0:
                backup_path = wallet_file + f'.bak.{int(time.time())}'
                import shutil
                shutil.copy2(wallet_file, backup_path)
                print(f"Created backup of corrupted wallet at {backup_path}")
    
    # Check if address already exists
    for entry in wallet:
        if entry.get('address') == address:
            print(f"Address {address} already exists in wallet")
            return wallet_file

    # Add new key pair
    new_entry = {
        'private_key': private_key,
        'address': address,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    wallet.append(new_entry)

    # Save with atomic write to prevent corruption
    temp_file = wallet_file + '.tmp'
    try:
        with open(temp_file, 'w') as f:
            json.dump(wallet, f, indent=2)
            f.write('\n')  # Add newline at end of file
        os.replace(temp_file, wallet_file)  # Atomic operation
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)  # Clean up temp file if write failed
        raise e
    
    return wallet_file

def load_wallet() -> list:
    """Load saved wallet file.
    
    Returns:
        List of wallet entries, each containing 'private_key' and 'address'
    """
    wallet_file = os.path.join(os.path.expanduser("~"), ".tenscoin", "wallet.json")
    
    if not os.path.exists(wallet_file):
        return []
        
    try:
        with open(wallet_file, 'r') as f:
            data = json.load(f)
            # Always return as list
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
            return []
    except json.JSONDecodeError:
        print(f"Warning: Wallet file {wallet_file} is corrupted")
        return []
    except Exception as e:
        print(f"Error loading wallet: {e}")
        return []