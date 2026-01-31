"""
XMSS (eXtended Merkle Signature Scheme) - Post-Quantum Hash-Based Signature
Based on NIST SP 800-208 and RFC 8391
"""

import hashlib
import secrets
import math
from typing import Tuple, List, Optional

class XMSS:
    def __init__(self, n: int = 32, h: int = 10, w: int = 16):
        """
        Initialize XMSS parameters
        n: security parameter (output length of hash function)
        h: height of the hyper tree
        w: Winternitz parameter
        """
        self.n = n
        self.h = h
        self.w = w
        self.total_signatures = 2 ** h

        # Calculate Winternitz parameters
        self.len1 = math.ceil((8 * n + math.log2(w - 1)) / math.log2(w)) + 1
        self.len2 = math.floor(math.log2(self.len1 * (w - 1)) / math.log2(w)) + 1
        self.len = self.len1 + self.len2

    def generate_keys(self) -> Tuple[bytes, bytes]:
        """
        Generate XMSS key pair
        Returns: (public_key, private_key)
        """
        # Generate seed for private key
        sk_seed = secrets.token_bytes(self.n)
        sk_prf = secrets.token_bytes(self.n)
        pub_seed = secrets.token_bytes(self.n)

        # Generate WOTS+ key pairs for each leaf
        wots_keys = []
        for i in range(self.total_signatures):
            wots_pk, wots_sk = self._generate_wots_keypair(sk_seed, sk_prf, pub_seed, i)
            wots_keys.append(wots_pk)

        # Build Merkle tree
        root = self._build_merkle_tree(wots_keys, pub_seed)

        # Construct public key: (OID, root, pub_seed)
        public_key = pub_seed + root

        # Construct private key: (index, sk_seed, sk_prf, pub_seed, root)
        private_key = (0).to_bytes(4, 'big') + sk_seed + sk_prf + pub_seed + root

        return public_key, private_key

    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """
        Sign a message using XMSS private key
        """
        if len(private_key) < 4 + 4 * self.n:
            raise ValueError("Invalid private key")

        # Parse private key
        index = int.from_bytes(private_key[:4], 'big')
        sk_seed = private_key[4:4+self.n]
        sk_prf = private_key[4+self.n:4+2*self.n]
        pub_seed = private_key[4+2*self.n:4+3*self.n]
        root = private_key[4+3*self.n:]

        if index >= self.total_signatures:
            raise ValueError("All signatures used")

        # Generate WOTS+ signature
        wots_sig = self._wots_sign(sk_seed, sk_prf, pub_seed, index, message)

        # Generate authentication path
        auth_path = self._generate_auth_path(sk_seed, sk_prf, pub_seed, index)

        # Construct XMSS signature: (index, R, sig_wots, auth_path)
        signature = index.to_bytes(4, 'big') + wots_sig + auth_path

        # Update private key index (in a real implementation, this would be stored securely)
        new_index = index + 1
        updated_private_key = new_index.to_bytes(4, 'big') + sk_seed + sk_prf + pub_seed + root

        return signature, updated_private_key

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """
        Verify XMSS signature
        """
        if len(public_key) != 2 * self.n or len(signature) < 4 + self.len * self.n:
            return False

        # Parse public key
        pub_seed = public_key[:self.n]
        root = public_key[self.n:]

        # Parse signature
        index = int.from_bytes(signature[:4], 'big')
        wots_sig = signature[4:4 + self.len * self.n]
        auth_path = signature[4 + self.len * self.n:]

        if index >= self.total_signatures:
            return False

        # Verify WOTS+ signature
        wots_pk = self._wots_verify(pub_seed, message, wots_sig)

        # Verify authentication path
        leaf_index = index
        current_hash = wots_pk

        for i in range(self.h):
            sibling_start = (i * self.n)
            sibling_end = (i + 1) * self.n
            if sibling_end > len(auth_path):
                return False
            sibling = auth_path[sibling_start:sibling_end]

            if leaf_index % 2 == 0:
                current_hash = self._hash(pub_seed, current_hash + sibling)
            else:
                current_hash = self._hash(pub_seed, sibling + current_hash)

            leaf_index //= 2

        return current_hash == root

    def _generate_wots_keypair(self, sk_seed: bytes, sk_prf: bytes, pub_seed: bytes, index: int) -> Tuple[bytes, bytes]:
        """Generate WOTS+ key pair"""
        # Simplified WOTS+ implementation
        sk = self._prf(sk_prf, index.to_bytes(4, 'big'))
        pk = self._hash(pub_seed, sk)
        return pk, sk

    def _wots_sign(self, sk_seed: bytes, sk_prf: bytes, pub_seed: bytes, index: int, message: bytes) -> bytes:
        """Generate WOTS+ signature"""
        # Simplified implementation
        msg_hash = self._hash(pub_seed, message)
        signature = b''
        for i in range(self.len):
            sk = self._prf(sk_prf, index.to_bytes(4, 'big') + i.to_bytes(4, 'big'))
            signature += sk
        return signature

    def _wots_verify(self, pub_seed: bytes, message: bytes, signature: bytes) -> bytes:
        """Verify WOTS+ signature"""
        # Simplified implementation
        msg_hash = self._hash(pub_seed, message)
        pk = b''
        for i in range(self.len):
            sig_part = signature[i*self.n:(i+1)*self.n]
            pk += self._hash(pub_seed, sig_part)
        return self._hash(pub_seed, pk)

    def _build_merkle_tree(self, leaves: List[bytes], pub_seed: bytes) -> bytes:
        """Build Merkle tree and return root"""
        if len(leaves) == 0:
            return b''

        current_level = leaves[:]
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]  # Duplicate last node
                next_level.append(self._hash(pub_seed, combined))
            current_level = next_level

        return current_level[0] if current_level else b''

    def _generate_auth_path(self, sk_seed: bytes, sk_prf: bytes, pub_seed: bytes, index: int) -> bytes:
        """Generate authentication path for leaf at given index"""
        # Simplified implementation - in practice, this would build the full tree
        auth_path = b''
        for i in range(self.h):
            # Generate sibling node (simplified)
            sibling = self._hash(pub_seed, f"sibling_{index}_{i}".encode())
            auth_path += sibling
        return auth_path

    def _hash(self, pub_seed: bytes, data: bytes) -> bytes:
        """Hash function with public seed"""
        return hashlib.sha256(pub_seed + data).digest()[:self.n]

    def _prf(self, key: bytes, data: bytes) -> bytes:
        """Pseudorandom function"""
        return hashlib.sha256(key + data).digest()[:self.n]

def generate_keys():
    """Generate XMSS key pair"""
    xmss = XMSS()
    return xmss.generate_keys()

def sign(private_key: bytes, message: bytes):
    """Sign message with XMSS private key"""
    xmss = XMSS()
    return xmss.sign(private_key, message)

def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Verify XMSS signature"""
    xmss = XMSS()
    return xmss.verify(public_key, message, signature)

# Example usage
if __name__ == "__main__":
    # Generate key pair
    public_key, private_key = generate_keys()
    print(f"Public key length: {len(public_key)}")
    print(f"Private key length: {len(private_key)}")

    # Sign message
    message = b"Hello, XMSS!"
    signature, updated_private_key = sign(private_key, message)
    print(f"Signature length: {len(signature)}")

    # Verify signature
    is_valid = verify(public_key, message, signature)
    print(f"Signature valid: {is_valid}")
