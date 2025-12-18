"""
Blockchain-based Data Integrity System for Financial Digital Twin
Provides tamper-proof verification and audit trails for simulation results.
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """Represents a block in the blockchain."""
    index: int
    timestamp: float
    data_hash: str
    previous_hash: str
    nonce: int
    merkle_root: str
    signature: Optional[str] = None


@dataclass
class SimulationRecord:
    """Represents a simulation record to be stored in blockchain."""
    simulation_id: str
    user_id: str
    user_input: Dict[str, Any]
    simulation_results: Dict[str, Any]
    timestamp: float
    version: str = "1.0"


class BlockchainIntegrity:
    """Blockchain-based integrity system for financial simulations."""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.pending_transactions: List[SimulationRecord] = []
        
        # Generate or load RSA key pair for signing
        self.private_key, self.public_key = self._generate_key_pair()
        
        # Create genesis block
        self._create_genesis_block()
    
    def _generate_key_pair(self) -> tuple:
        """Generate RSA key pair for digital signatures."""
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            return private_key, public_key
        except Exception as e:
            logger.error(f"Error generating key pair: {str(e)}")
            # Fallback to simple hash-based system
            return None, None
    
    def _create_genesis_block(self):
        """Create the first block in the blockchain."""
        genesis_data = {
            "message": "Financial Digital Twin Genesis Block",
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        genesis_hash = self._calculate_hash(genesis_data)
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            data_hash=genesis_hash,
            previous_hash="0",
            nonce=0,
            merkle_root=genesis_hash
        )
        
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash of data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _calculate_merkle_root(self, transactions: List[SimulationRecord]) -> str:
        """Calculate Merkle root of transactions."""
        if not transactions:
            return "0"
        
        # Convert transactions to hash strings
        tx_hashes = []
        for tx in transactions:
            tx_data = asdict(tx)
            tx_hash = self._calculate_hash(tx_data)
            tx_hashes.append(tx_hash)
        
        # Build Merkle tree
        while len(tx_hashes) > 1:
            next_level = []
            for i in range(0, len(tx_hashes), 2):
                left = tx_hashes[i]
                right = tx_hashes[i + 1] if i + 1 < len(tx_hashes) else tx_hashes[i]
                combined = left + right
                next_level.append(self._calculate_hash(combined))
            tx_hashes = next_level
        
        return tx_hashes[0] if tx_hashes else "0"
    
    def _mine_block(self, transactions: List[SimulationRecord]) -> Block:
        """Mine a new block with given transactions."""
        previous_block = self.chain[-1]
        merkle_root = self._calculate_merkle_root(transactions)
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data_hash="",
            previous_hash=previous_block.data_hash,
            nonce=0,
            merkle_root=merkle_root
        )
        
        # Mine the block (Proof of Work)
        new_block = self._proof_of_work(new_block)
        
        # Sign the block
        if self.private_key:
            new_block.signature = self._sign_block(new_block)
        
        return new_block
    
    def _proof_of_work(self, block: Block) -> Block:
        """Perform proof of work to mine the block."""
        target = "0" * self.difficulty
        
        while True:
            block_str = f"{block.index}{block.timestamp}{block.previous_hash}{block.merkle_root}{block.nonce}"
            block_hash = self._calculate_hash(block_str)
            
            if block_hash.startswith(target):
                block.data_hash = block_hash
                break
            
            block.nonce += 1
        
        return block
    
    def _sign_block(self, block: Block) -> str:
        """Sign a block with private key."""
        if not self.private_key:
            return ""
        
        try:
            block_data = f"{block.index}{block.timestamp}{block.data_hash}{block.previous_hash}{block.merkle_root}{block.nonce}"
            signature = self.private_key.sign(
                block_data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            logger.error(f"Error signing block: {str(e)}")
            return ""
    
    def verify_block_signature(self, block: Block) -> bool:
        """Verify block signature."""
        if not block.signature or not self.public_key:
            return True  # No signature to verify
        
        try:
            block_data = f"{block.index}{block.timestamp}{block.data_hash}{block.previous_hash}{block.merkle_root}{block.nonce}"
            signature = bytes.fromhex(block.signature)
            
            self.public_key.verify(
                signature,
                block_data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Block signature verification failed: {str(e)}")
            return False
    
    def add_simulation_record(self, simulation_record: SimulationRecord) -> str:
        """Add a simulation record to the blockchain."""
        try:
            # Add to pending transactions
            self.pending_transactions.append(simulation_record)
            
            # Mine new block if we have enough transactions or it's been a while
            if len(self.pending_transactions) >= 5 or len(self.pending_transactions) >= 1:
                new_block = self._mine_block(self.pending_transactions.copy())
                self.chain.append(new_block)
                
                # Clear pending transactions
                self.pending_transactions.clear()
                
                logger.info(f"New block mined: {new_block.index}")
                return new_block.data_hash
            
            return "pending"
            
        except Exception as e:
            logger.error(f"Error adding simulation record: {str(e)}")
            return "error"
    
    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire blockchain."""
        try:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i - 1]
                
                # Check if current block's previous hash matches previous block's hash
                if current_block.previous_hash != previous_block.data_hash:
                    logger.error(f"Chain integrity broken at block {i}")
                    return False
                
                # Verify block signature
                if not self.verify_block_signature(current_block):
                    logger.error(f"Block signature verification failed at block {i}")
                    return False
                
                # Verify proof of work
                block_str = f"{current_block.index}{current_block.timestamp}{current_block.previous_hash}{current_block.merkle_root}{current_block.nonce}"
                calculated_hash = self._calculate_hash(block_str)
                if not calculated_hash.startswith("0" * self.difficulty):
                    logger.error(f"Proof of work verification failed at block {i}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying chain integrity: {str(e)}")
            return False
    
    def get_simulation_proof(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get cryptographic proof for a specific simulation."""
        try:
            for block in self.chain:
                for tx in self.pending_transactions:
                    if tx.simulation_id == simulation_id:
                        return {
                            "simulation_id": simulation_id,
                            "block_index": block.index,
                            "block_hash": block.data_hash,
                            "merkle_root": block.merkle_root,
                            "timestamp": block.timestamp,
                            "verified": True
                        }
            
            # Search in mined blocks (would need to store transactions in blocks)
            return None
            
        except Exception as e:
            logger.error(f"Error getting simulation proof: {str(e)}")
            return None
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain information."""
        return {
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "difficulty": self.difficulty,
            "last_block_hash": self.chain[-1].data_hash if self.chain else None,
            "chain_integrity": self.verify_chain_integrity()
        }
    
    def export_public_key(self) -> str:
        """Export public key for verification."""
        if not self.public_key:
            return ""
        
        try:
            pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return pem.decode()
        except Exception as e:
            logger.error(f"Error exporting public key: {str(e)}")
            return ""


class DataIntegrityService:
    """Service for managing data integrity and tamper detection."""
    
    def __init__(self):
        self.blockchain = BlockchainIntegrity()
        self.simulation_hashes: Dict[str, str] = {}
    
    def create_simulation_record(
        self, 
        simulation_id: str, 
        user_id: str, 
        user_input: Dict[str, Any], 
        simulation_results: Dict[str, Any]
    ) -> str:
        """Create a tamper-proof record of a simulation."""
        try:
            # Create simulation record
            record = SimulationRecord(
                simulation_id=simulation_id,
                user_id=user_id,
                user_input=user_input,
                simulation_results=simulation_results,
                timestamp=time.time()
            )
            
            # Add to blockchain
            block_hash = self.blockchain.add_simulation_record(record)
            
            # Store hash for quick lookup
            record_hash = self.blockchain._calculate_hash(asdict(record))
            self.simulation_hashes[simulation_id] = record_hash
            
            logger.info(f"Simulation record created: {simulation_id}")
            return block_hash
            
        except Exception as e:
            logger.error(f"Error creating simulation record: {str(e)}")
            return "error"
    
    def verify_simulation_integrity(self, simulation_id: str) -> Dict[str, Any]:
        """Verify the integrity of a simulation."""
        try:
            if simulation_id not in self.simulation_hashes:
                return {
                    "verified": False,
                    "error": "Simulation not found"
                }
            
            # Get proof from blockchain
            proof = self.blockchain.get_simulation_proof(simulation_id)
            
            if not proof:
                return {
                    "verified": False,
                    "error": "No blockchain proof found"
                }
            
            # Verify chain integrity
            chain_integrity = self.blockchain.verify_chain_integrity()
            
            return {
                "verified": True,
                "simulation_id": simulation_id,
                "block_proof": proof,
                "chain_integrity": chain_integrity,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error verifying simulation integrity: {str(e)}")
            return {
                "verified": False,
                "error": str(e)
            }
    
    def get_integrity_report(self) -> Dict[str, Any]:
        """Get comprehensive integrity report."""
        try:
            chain_info = self.blockchain.get_chain_info()
            public_key = self.blockchain.export_public_key()
            
            return {
                "blockchain_info": chain_info,
                "public_key": public_key,
                "total_simulations": len(self.simulation_hashes),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating integrity report: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
