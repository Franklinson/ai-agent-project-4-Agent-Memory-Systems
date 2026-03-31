"""
Secure Profile Storage with Privacy and Security

Implements profile storage with encryption, access control, and audit logging.
"""

import json
import hashlib
import secrets
import sqlite3
from abc import ABC, abstractmethod
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Set
import base64
import os
from pathlib import Path

from user_profile import UserProfile, ProfileType


class AccessLevel(Enum):
    """Access levels for profile operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class AuditAction(Enum):
    """Audit actions for logging."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"


class SecurityError(Exception):
    """Base security error."""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed."""
    pass


class AuthorizationError(SecurityError):
    """Authorization failed."""
    pass


class EncryptionError(SecurityError):
    """Encryption/decryption failed."""
    pass


@dataclass
class AuditEntry:
    """Audit log entry."""
    timestamp: datetime
    user_id: str
    action: AuditAction
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    success: bool = True


@dataclass
class AccessToken:
    """Access token for authentication."""
    user_id: str
    permissions: Set[AccessLevel]
    expires_at: datetime
    token_hash: str


class EncryptionManager:
    """Manages encryption and decryption of sensitive data."""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize encryption manager."""
        if master_key:
            self._master_key = master_key.encode()
        else:
            self._master_key = self._generate_master_key()
        
        self._fernet = self._create_fernet()
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master key."""
        return secrets.token_bytes(32)
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # In production, use random salt per key
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_key))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")
    
    def get_master_key(self) -> str:
        """Get master key for storage (base64 encoded)."""
        return base64.urlsafe_b64encode(self._master_key).decode()


class AccessController:
    """Manages access control and permissions."""
    
    def __init__(self):
        self._tokens: Dict[str, AccessToken] = {}
        self._user_permissions: Dict[str, Set[AccessLevel]] = {}
    
    def authenticate(self, user_id: str, password: str) -> str:
        """Authenticate user and return token."""
        # Simple password hashing (in production, use proper password hashing)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # For demo, accept any user with password "password"
        if password != "password":
            raise AuthenticationError("Invalid credentials")
        
        # Generate access token
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # Default permissions based on user_id
        permissions = self._get_default_permissions(user_id)
        
        access_token = AccessToken(
            user_id=user_id,
            permissions=permissions,
            expires_at=datetime.now().replace(hour=23, minute=59, second=59),
            token_hash=token_hash
        )
        
        self._tokens[token] = access_token
        return token
    
    def _get_default_permissions(self, user_id: str) -> Set[AccessLevel]:
        """Get default permissions for user."""
        if user_id.startswith("admin"):
            return {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.DELETE, AccessLevel.ADMIN}
        else:
            return {AccessLevel.READ, AccessLevel.WRITE}
    
    def authorize(self, token: str, required_permission: AccessLevel, resource_user_id: Optional[str] = None) -> str:
        """Authorize access and return user_id."""
        if token not in self._tokens:
            raise AuthorizationError("Invalid token")
        
        access_token = self._tokens[token]
        
        # Check token expiration
        if datetime.now() > access_token.expires_at:
            del self._tokens[token]
            raise AuthorizationError("Token expired")
        
        # Check permission
        if required_permission not in access_token.permissions:
            raise AuthorizationError(f"Insufficient permissions: {required_permission.value}")
        
        # Check resource access (users can only access their own profiles unless admin)
        if resource_user_id and resource_user_id != access_token.user_id:
            if AccessLevel.ADMIN not in access_token.permissions:
                raise AuthorizationError("Cannot access other user's profile")
        
        return access_token.user_id
    
    def revoke_token(self, token: str) -> None:
        """Revoke access token."""
        if token in self._tokens:
            del self._tokens[token]
    
    def set_user_permissions(self, user_id: str, permissions: Set[AccessLevel]) -> None:
        """Set user permissions."""
        self._user_permissions[user_id] = permissions


class AuditLogger:
    """Manages audit logging."""
    
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize audit database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    success INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_log(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)")
    
    def log(self, entry: AuditEntry) -> None:
        """Log audit entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_log 
                (timestamp, user_id, action, resource_id, details, ip_address, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp.isoformat(),
                entry.user_id,
                entry.action.value,
                entry.resource_id,
                json.dumps(entry.details),
                entry.ip_address,
                1 if entry.success else 0
            ))
    
    def get_user_activity(self, user_id: str, limit: int = 100) -> List[AuditEntry]:
        """Get user activity log."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, user_id, action, resource_id, details, ip_address, success
                FROM audit_log
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            entries = []
            for row in cursor.fetchall():
                entries.append(AuditEntry(
                    timestamp=datetime.fromisoformat(row[0]),
                    user_id=row[1],
                    action=AuditAction(row[2]),
                    resource_id=row[3],
                    details=json.loads(row[4]) if row[4] else {},
                    ip_address=row[5],
                    success=bool(row[6])
                ))
            
            return entries
    
    def get_resource_activity(self, resource_id: str, limit: int = 100) -> List[AuditEntry]:
        """Get resource activity log."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, user_id, action, resource_id, details, ip_address, success
                FROM audit_log
                WHERE resource_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (resource_id, limit))
            
            entries = []
            for row in cursor.fetchall():
                entries.append(AuditEntry(
                    timestamp=datetime.fromisoformat(row[0]),
                    user_id=row[1],
                    action=AuditAction(row[2]),
                    resource_id=row[3],
                    details=json.loads(row[4]) if row[4] else {},
                    ip_address=row[5],
                    success=bool(row[6])
                ))
            
            return entries


class SecureProfileStorage:
    """Secure profile storage with encryption, access control, and audit logging."""
    
    def __init__(self, db_path: str = "profiles.db", master_key: Optional[str] = None):
        self.db_path = db_path
        self.encryption_manager = EncryptionManager(master_key)
        self.access_controller = AccessController()
        self.audit_logger = AuditLogger()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize profile database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_type TEXT NOT NULL,
                    encrypted_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL
                )
            """)
    
    def authenticate(self, user_id: str, password: str, ip_address: Optional[str] = None) -> str:
        """Authenticate user and return access token."""
        try:
            token = self.access_controller.authenticate(user_id, password)
            
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.LOGIN,
                ip_address=ip_address,
                success=True
            ))
            
            return token
        except AuthenticationError as e:
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.LOGIN,
                ip_address=ip_address,
                success=False,
                details={"error": str(e)}
            ))
            raise
    
    def logout(self, token: str, ip_address: Optional[str] = None) -> None:
        """Logout user and revoke token."""
        try:
            # Get user_id before revoking token
            user_id = self.access_controller.authorize(token, AccessLevel.READ)
            self.access_controller.revoke_token(token)
            
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.LOGOUT,
                ip_address=ip_address,
                success=True
            ))
        except AuthorizationError:
            pass  # Token already invalid
    
    def store_profile(self, token: str, profile: UserProfile, ip_address: Optional[str] = None) -> None:
        """Store encrypted profile."""
        try:
            user_id = self.access_controller.authorize(token, AccessLevel.WRITE, profile.user_id)
            
            # Encrypt sensitive data
            profile_data = profile.to_dict()
            encrypted_data = self.encryption_manager.encrypt(json.dumps(profile_data))
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO profiles 
                    (user_id, profile_type, encrypted_data, created_at, updated_at, version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    profile.profile_type.value,
                    encrypted_data,
                    profile.metadata.created_at.isoformat(),
                    profile.metadata.updated_at.isoformat(),
                    profile.metadata.version
                ))
            
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.CREATE,
                resource_id=profile.user_id,
                ip_address=ip_address,
                details={"profile_type": profile.profile_type.value},
                success=True
            ))
            
        except (AuthorizationError, EncryptionError) as e:
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id if 'user_id' in locals() else "unknown",
                action=AuditAction.CREATE,
                resource_id=profile.user_id,
                ip_address=ip_address,
                success=False,
                details={"error": str(e)}
            ))
            raise
    
    def retrieve_profile(self, token: str, profile_user_id: str, ip_address: Optional[str] = None) -> UserProfile:
        """Retrieve and decrypt profile."""
        try:
            user_id = self.access_controller.authorize(token, AccessLevel.READ, profile_user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT encrypted_data FROM profiles WHERE user_id = ?
                """, (profile_user_id,))
                
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Profile not found: {profile_user_id}")
                
                # Decrypt profile data
                decrypted_data = self.encryption_manager.decrypt(row[0])
                profile_data = json.loads(decrypted_data)
                profile = UserProfile.from_dict(profile_data)
            
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.READ,
                resource_id=profile_user_id,
                ip_address=ip_address,
                success=True
            ))
            
            return profile
            
        except (AuthorizationError, EncryptionError, ValueError) as e:
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id if 'user_id' in locals() else "unknown",
                action=AuditAction.READ,
                resource_id=profile_user_id,
                ip_address=ip_address,
                success=False,
                details={"error": str(e)}
            ))
            raise
    
    def update_profile(self, token: str, profile: UserProfile, ip_address: Optional[str] = None) -> None:
        """Update encrypted profile."""
        try:
            user_id = self.access_controller.authorize(token, AccessLevel.WRITE, profile.user_id)
            
            # Check if profile exists
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT version FROM profiles WHERE user_id = ?", (profile.user_id,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Profile not found: {profile.user_id}")
                
                # Encrypt updated data
                profile_data = profile.to_dict()
                encrypted_data = self.encryption_manager.encrypt(json.dumps(profile_data))
                
                conn.execute("""
                    UPDATE profiles 
                    SET encrypted_data = ?, updated_at = ?, version = ?
                    WHERE user_id = ?
                """, (
                    encrypted_data,
                    profile.metadata.updated_at.isoformat(),
                    profile.metadata.version,
                    profile.user_id
                ))
            
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.UPDATE,
                resource_id=profile.user_id,
                ip_address=ip_address,
                details={"new_version": profile.metadata.version},
                success=True
            ))
            
        except (AuthorizationError, EncryptionError, ValueError) as e:
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id if 'user_id' in locals() else "unknown",
                action=AuditAction.UPDATE,
                resource_id=profile.user_id,
                ip_address=ip_address,
                success=False,
                details={"error": str(e)}
            ))
            raise
    
    def delete_profile(self, token: str, profile_user_id: str, ip_address: Optional[str] = None) -> None:
        """Delete profile."""
        try:
            user_id = self.access_controller.authorize(token, AccessLevel.DELETE, profile_user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT user_id FROM profiles WHERE user_id = ?", (profile_user_id,))
                if not cursor.fetchone():
                    raise ValueError(f"Profile not found: {profile_user_id}")
                
                conn.execute("DELETE FROM profiles WHERE user_id = ?", (profile_user_id,))
            
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.DELETE,
                resource_id=profile_user_id,
                ip_address=ip_address,
                success=True
            ))
            
        except (AuthorizationError, ValueError) as e:
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id if 'user_id' in locals() else "unknown",
                action=AuditAction.DELETE,
                resource_id=profile_user_id,
                ip_address=ip_address,
                success=False,
                details={"error": str(e)}
            ))
            raise
    
    def list_profiles(self, token: str, ip_address: Optional[str] = None) -> List[Dict[str, Any]]:
        """List profiles (metadata only, admin required)."""
        try:
            user_id = self.access_controller.authorize(token, AccessLevel.ADMIN)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT user_id, profile_type, created_at, updated_at, version
                    FROM profiles
                    ORDER BY created_at DESC
                """)
                
                profiles = []
                for row in cursor.fetchall():
                    profiles.append({
                        "user_id": row[0],
                        "profile_type": row[1],
                        "created_at": row[2],
                        "updated_at": row[3],
                        "version": row[4]
                    })
            
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                action=AuditAction.READ,
                ip_address=ip_address,
                details={"operation": "list_profiles", "count": len(profiles)},
                success=True
            ))
            
            return profiles
            
        except AuthorizationError as e:
            self.audit_logger.log(AuditEntry(
                timestamp=datetime.now(),
                user_id=user_id if 'user_id' in locals() else "unknown",
                action=AuditAction.READ,
                ip_address=ip_address,
                success=False,
                details={"error": str(e), "operation": "list_profiles"}
            ))
            raise
    
    def get_audit_log(self, token: str, target_user_id: Optional[str] = None, limit: int = 100) -> List[AuditEntry]:
        """Get audit log (admin or own records)."""
        user_id = self.access_controller.authorize(token, AccessLevel.READ)
        
        # Users can only see their own audit log unless they're admin
        if target_user_id and target_user_id != user_id:
            self.access_controller.authorize(token, AccessLevel.ADMIN)
            return self.audit_logger.get_user_activity(target_user_id, limit)
        else:
            return self.audit_logger.get_user_activity(user_id, limit)
    
    def cleanup_expired_tokens(self) -> None:
        """Clean up expired tokens."""
        current_time = datetime.now()
        expired_tokens = [
            token for token, access_token in self.access_controller._tokens.items()
            if current_time > access_token.expires_at
        ]
        
        for token in expired_tokens:
            del self.access_controller._tokens[token]
    
    def get_master_key(self) -> str:
        """Get master key for backup (admin only)."""
        return self.encryption_manager.get_master_key()