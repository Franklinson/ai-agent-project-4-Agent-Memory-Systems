"""
User Profile System with Data Structures

Implements profile management with preferences, validation, and multiple profile types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Type
import json
import uuid


class ProfileType(Enum):
    """Profile types supported by the system."""
    BASIC = "basic"
    PREMIUM = "premium"
    ADMIN = "admin"
    GUEST = "guest"


class ValidationError(Exception):
    """Raised when profile validation fails."""
    pass


class ProfileNotFoundError(Exception):
    """Raised when profile is not found."""
    pass


@dataclass
class ProfileMetadata:
    """Metadata for profile tracking."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    last_login: Optional[datetime] = None
    login_count: int = 0


@dataclass
class UserPreferences:
    """User preferences with validation."""
    language: str = "en"
    timezone: str = "UTC"
    theme: str = "light"
    notifications: bool = True
    email_updates: bool = True
    privacy_level: str = "medium"
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate preferences."""
        valid_languages = ["en", "es", "fr", "de", "ja", "zh"]
        valid_themes = ["light", "dark", "auto"]
        valid_privacy = ["low", "medium", "high"]
        
        if self.language not in valid_languages:
            raise ValidationError(f"Invalid language: {self.language}")
        if self.theme not in valid_themes:
            raise ValidationError(f"Invalid theme: {self.theme}")
        if self.privacy_level not in valid_privacy:
            raise ValidationError(f"Invalid privacy level: {self.privacy_level}")


class ProfileValidator(ABC):
    """Abstract base class for profile validators."""
    
    @abstractmethod
    def validate(self, profile: 'UserProfile') -> None:
        """Validate a profile."""
        pass


class BasicProfileValidator(ProfileValidator):
    """Validator for basic profiles."""
    
    def validate(self, profile: 'UserProfile') -> None:
        """Validate basic profile requirements."""
        if not profile.user_id:
            raise ValidationError("User ID is required")
        if not profile.email or "@" not in profile.email:
            raise ValidationError("Valid email is required")
        if len(profile.username) < 3:
            raise ValidationError("Username must be at least 3 characters")


class PremiumProfileValidator(ProfileValidator):
    """Validator for premium profiles."""
    
    def validate(self, profile: 'UserProfile') -> None:
        """Validate premium profile requirements."""
        BasicProfileValidator().validate(profile)
        if not profile.full_name:
            raise ValidationError("Full name is required for premium profiles")
        if profile.age and profile.age < 13:
            raise ValidationError("Premium profiles require age 13+")


@dataclass
class UserProfile:
    """Main user profile data structure."""
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile_type: ProfileType = ProfileType.BASIC
    username: str = ""
    email: str = ""
    full_name: str = ""
    age: Optional[int] = None
    bio: str = ""
    avatar_url: str = ""
    preferences: UserPreferences = field(default_factory=UserPreferences)
    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)
    tags: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()

    def validate(self) -> None:
        """Validate the profile using appropriate validator."""
        validator = self._get_validator()
        validator.validate(self)
        self.preferences.validate()

    def _get_validator(self) -> ProfileValidator:
        """Get appropriate validator for profile type."""
        validators = {
            ProfileType.BASIC: BasicProfileValidator(),
            ProfileType.PREMIUM: PremiumProfileValidator(),
            ProfileType.ADMIN: PremiumProfileValidator(),
            ProfileType.GUEST: ProfileValidator.__new__(type('GuestValidator', (ProfileValidator,), {
                'validate': lambda self, profile: None
            }))
        }
        return validators.get(self.profile_type, BasicProfileValidator())

    def update_preferences(self, **kwargs) -> None:
        """Update user preferences."""
        for key, value in kwargs.items():
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
            else:
                self.preferences.custom_settings[key] = value
        
        self.preferences.validate()
        self._update_metadata()

    def update_profile(self, **kwargs) -> None:
        """Update profile data with validation."""
        old_version = self.metadata.version
        
        for key, value in kwargs.items():
            if hasattr(self, key) and key not in ['user_id', 'metadata']:
                setattr(self, key, value)
        
        try:
            self.validate()
            self._update_metadata()
        except ValidationError:
            # Rollback on validation failure
            self.metadata.version = old_version
            raise

    def _update_metadata(self) -> None:
        """Update metadata on profile changes."""
        self.metadata.updated_at = datetime.now()
        self.metadata.version += 1

    def record_login(self) -> None:
        """Record user login."""
        self.metadata.last_login = datetime.now()
        self.metadata.login_count += 1
        self._update_metadata()

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        data = asdict(self)
        data['profile_type'] = self.profile_type.value
        data['metadata']['created_at'] = self.metadata.created_at.isoformat()
        data['metadata']['updated_at'] = self.metadata.updated_at.isoformat()
        if self.metadata.last_login:
            data['metadata']['last_login'] = self.metadata.last_login.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create profile from dictionary."""
        # Handle enum conversion
        if 'profile_type' in data:
            data['profile_type'] = ProfileType(data['profile_type'])
        
        # Handle datetime conversion
        if 'metadata' in data:
            meta = data['metadata']
            if 'created_at' in meta:
                meta['created_at'] = datetime.fromisoformat(meta['created_at'])
            if 'updated_at' in meta:
                meta['updated_at'] = datetime.fromisoformat(meta['updated_at'])
            if 'last_login' in meta and meta['last_login']:
                meta['last_login'] = datetime.fromisoformat(meta['last_login'])
        
        # Handle nested dataclasses
        if 'preferences' in data:
            data['preferences'] = UserPreferences(**data['preferences'])
        if 'metadata' in data:
            data['metadata'] = ProfileMetadata(**data['metadata'])
        
        return cls(**data)

    def to_json(self) -> str:
        """Convert profile to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'UserProfile':
        """Create profile from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ProfileManager:
    """Manager for user profiles with CRUD operations."""
    
    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
    
    def create_profile(self, profile_type: ProfileType = ProfileType.BASIC, **kwargs) -> UserProfile:
        """Create a new user profile."""
        profile = UserProfile(profile_type=profile_type, **kwargs)
        self._profiles[profile.user_id] = profile
        return profile
    
    def get_profile(self, user_id: str) -> UserProfile:
        """Get profile by user ID."""
        if user_id not in self._profiles:
            raise ProfileNotFoundError(f"Profile not found: {user_id}")
        return self._profiles[user_id]
    
    def update_profile(self, user_id: str, **kwargs) -> UserProfile:
        """Update existing profile."""
        profile = self.get_profile(user_id)
        profile.update_profile(**kwargs)
        return profile
    
    def update_preferences(self, user_id: str, **kwargs) -> UserProfile:
        """Update profile preferences."""
        profile = self.get_profile(user_id)
        profile.update_preferences(**kwargs)
        return profile
    
    def delete_profile(self, user_id: str) -> None:
        """Delete profile."""
        if user_id not in self._profiles:
            raise ProfileNotFoundError(f"Profile not found: {user_id}")
        del self._profiles[user_id]
    
    def list_profiles(self, profile_type: Optional[ProfileType] = None) -> List[UserProfile]:
        """List profiles, optionally filtered by type."""
        profiles = list(self._profiles.values())
        if profile_type:
            profiles = [p for p in profiles if p.profile_type == profile_type]
        return profiles
    
    def search_profiles(self, **criteria) -> List[UserProfile]:
        """Search profiles by criteria."""
        results = []
        for profile in self._profiles.values():
            match = True
            for key, value in criteria.items():
                if not hasattr(profile, key) or getattr(profile, key) != value:
                    match = False
                    break
            if match:
                results.append(profile)
        return results
    
    def export_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Export all profiles to dictionary."""
        return {uid: profile.to_dict() for uid, profile in self._profiles.items()}
    
    def import_profiles(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Import profiles from dictionary."""
        for user_id, profile_data in data.items():
            profile = UserProfile.from_dict(profile_data)
            self._profiles[user_id] = profile
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profile statistics."""
        total = len(self._profiles)
        by_type = {}
        for profile in self._profiles.values():
            ptype = profile.profile_type.value
            by_type[ptype] = by_type.get(ptype, 0) + 1
        
        return {
            "total_profiles": total,
            "by_type": by_type,
            "active_users": sum(1 for p in self._profiles.values() if p.metadata.last_login)
        }