"""
Tests for User Profile System

Tests profile creation, validation, preferences, updates, and multiple profile types.
"""

import pytest
import json
from datetime import datetime, timedelta
from day_54.user_profile import (
    UserProfile, UserPreferences, ProfileMetadata, ProfileManager,
    ProfileType, ValidationError, ProfileNotFoundError,
    BasicProfileValidator, PremiumProfileValidator
)


class TestUserPreferences:
    """Test user preferences functionality."""
    
    def test_default_preferences(self):
        """Test default preference values."""
        prefs = UserPreferences()
        assert prefs.language == "en"
        assert prefs.timezone == "UTC"
        assert prefs.theme == "light"
        assert prefs.notifications is True
        assert prefs.email_updates is True
        assert prefs.privacy_level == "medium"
        assert prefs.custom_settings == {}
    
    def test_preference_validation(self):
        """Test preference validation."""
        prefs = UserPreferences()
        prefs.validate()  # Should not raise
        
        prefs.language = "invalid"
        with pytest.raises(ValidationError, match="Invalid language"):
            prefs.validate()
        
        prefs.language = "en"
        prefs.theme = "invalid"
        with pytest.raises(ValidationError, match="Invalid theme"):
            prefs.validate()
        
        prefs.theme = "dark"
        prefs.privacy_level = "invalid"
        with pytest.raises(ValidationError, match="Invalid privacy level"):
            prefs.validate()


class TestUserProfile:
    """Test user profile functionality."""
    
    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = UserProfile(
            username="testuser",
            email="test@example.com"
        )
        assert profile.user_id
        assert profile.username == "testuser"
        assert profile.email == "test@example.com"
        assert profile.profile_type == ProfileType.BASIC
        assert isinstance(profile.preferences, UserPreferences)
        assert isinstance(profile.metadata, ProfileMetadata)
    
    def test_profile_validation(self):
        """Test profile validation."""
        # Valid basic profile
        profile = UserProfile(
            username="testuser",
            email="test@example.com"
        )
        profile.validate()  # Should not raise
        
        # Invalid email
        with pytest.raises(ValidationError, match="Valid email is required"):
            UserProfile(username="testuser", email="invalid")
        
        # Short username
        with pytest.raises(ValidationError, match="Username must be at least 3 characters"):
            UserProfile(username="ab", email="test@example.com")
    
    def test_premium_profile_validation(self):
        """Test premium profile validation."""
        # Valid premium profile
        profile = UserProfile(
            profile_type=ProfileType.PREMIUM,
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            age=25
        )
        profile.validate()  # Should not raise
        
        # Missing full name
        with pytest.raises(ValidationError, match="Full name is required"):
            UserProfile(
                profile_type=ProfileType.PREMIUM,
                username="testuser",
                email="test@example.com"
            )
        
        # Age too young
        with pytest.raises(ValidationError, match="Premium profiles require age 13+"):
            UserProfile(
                profile_type=ProfileType.PREMIUM,
                username="testuser",
                email="test@example.com",
                full_name="Test User",
                age=10
            )
    
    def test_preference_updates(self):
        """Test preference updates."""
        profile = UserProfile(
            username="testuser",
            email="test@example.com"
        )
        
        old_version = profile.metadata.version
        profile.update_preferences(theme="dark", language="es")
        
        assert profile.preferences.theme == "dark"
        assert profile.preferences.language == "es"
        assert profile.metadata.version > old_version
        
        # Custom setting
        profile.update_preferences(custom_key="custom_value")
        assert profile.preferences.custom_settings["custom_key"] == "custom_value"
        
        # Invalid preference
        with pytest.raises(ValidationError):
            profile.update_preferences(theme="invalid")
    
    def test_profile_updates(self):
        """Test profile updates."""
        profile = UserProfile(
            username="testuser",
            email="test@example.com"
        )
        
        old_version = profile.metadata.version
        profile.update_profile(bio="New bio", age=30)
        
        assert profile.bio == "New bio"
        assert profile.age == 30
        assert profile.metadata.version > old_version
        
        # Invalid update should rollback
        old_version = profile.metadata.version
        with pytest.raises(ValidationError):
            profile.update_profile(email="invalid")
        assert profile.metadata.version == old_version
    
    def test_login_recording(self):
        """Test login recording."""
        profile = UserProfile(
            username="testuser",
            email="test@example.com"
        )
        
        assert profile.metadata.last_login is None
        assert profile.metadata.login_count == 0
        
        profile.record_login()
        
        assert profile.metadata.last_login is not None
        assert profile.metadata.login_count == 1
        
        profile.record_login()
        assert profile.metadata.login_count == 2
    
    def test_serialization(self):
        """Test profile serialization/deserialization."""
        original = UserProfile(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            age=25,
            bio="Test bio",
            tags=["tag1", "tag2"]
        )
        original.update_preferences(theme="dark", language="es")
        original.record_login()
        
        # To/from dict
        data = original.to_dict()
        restored = UserProfile.from_dict(data)
        
        assert restored.username == original.username
        assert restored.email == original.email
        assert restored.preferences.theme == original.preferences.theme
        assert restored.metadata.login_count == original.metadata.login_count
        
        # To/from JSON
        json_str = original.to_json()
        restored_json = UserProfile.from_json(json_str)
        
        assert restored_json.username == original.username
        assert restored_json.email == original.email


class TestProfileManager:
    """Test profile manager functionality."""
    
    def test_create_profile(self):
        """Test profile creation."""
        manager = ProfileManager()
        
        profile = manager.create_profile(
            username="testuser",
            email="test@example.com"
        )
        
        assert profile.user_id in manager._profiles
        assert profile.username == "testuser"
        assert profile.profile_type == ProfileType.BASIC
    
    def test_get_profile(self):
        """Test profile retrieval."""
        manager = ProfileManager()
        
        profile = manager.create_profile(
            username="testuser",
            email="test@example.com"
        )
        
        retrieved = manager.get_profile(profile.user_id)
        assert retrieved is profile
        
        with pytest.raises(ProfileNotFoundError):
            manager.get_profile("nonexistent")
    
    def test_update_profile(self):
        """Test profile updates through manager."""
        manager = ProfileManager()
        
        profile = manager.create_profile(
            username="testuser",
            email="test@example.com"
        )
        
        updated = manager.update_profile(profile.user_id, bio="New bio")
        assert updated.bio == "New bio"
        assert updated is profile
    
    def test_update_preferences(self):
        """Test preference updates through manager."""
        manager = ProfileManager()
        
        profile = manager.create_profile(
            username="testuser",
            email="test@example.com"
        )
        
        updated = manager.update_preferences(profile.user_id, theme="dark")
        assert updated.preferences.theme == "dark"
        assert updated is profile
    
    def test_delete_profile(self):
        """Test profile deletion."""
        manager = ProfileManager()
        
        profile = manager.create_profile(
            username="testuser",
            email="test@example.com"
        )
        
        manager.delete_profile(profile.user_id)
        
        with pytest.raises(ProfileNotFoundError):
            manager.get_profile(profile.user_id)
    
    def test_list_profiles(self):
        """Test profile listing."""
        manager = ProfileManager()
        
        basic = manager.create_profile(
            ProfileType.BASIC,
            username="basic",
            email="basic@example.com"
        )
        premium = manager.create_profile(
            ProfileType.PREMIUM,
            username="premium",
            email="premium@example.com",
            full_name="Premium User"
        )
        
        all_profiles = manager.list_profiles()
        assert len(all_profiles) == 2
        
        basic_profiles = manager.list_profiles(ProfileType.BASIC)
        assert len(basic_profiles) == 1
        assert basic_profiles[0] is basic
        
        premium_profiles = manager.list_profiles(ProfileType.PREMIUM)
        assert len(premium_profiles) == 1
        assert premium_profiles[0] is premium
    
    def test_search_profiles(self):
        """Test profile search."""
        manager = ProfileManager()
        
        profile1 = manager.create_profile(
            username="user1",
            email="user1@example.com",
            age=25
        )
        profile2 = manager.create_profile(
            username="user2",
            email="user2@example.com",
            age=30
        )
        
        results = manager.search_profiles(age=25)
        assert len(results) == 1
        assert results[0] is profile1
        
        results = manager.search_profiles(username="user2")
        assert len(results) == 1
        assert results[0] is profile2
    
    def test_export_import(self):
        """Test profile export/import."""
        manager = ProfileManager()
        
        profile = manager.create_profile(
            username="testuser",
            email="test@example.com"
        )
        profile.update_preferences(theme="dark")
        
        # Export
        data = manager.export_profiles()
        assert profile.user_id in data
        
        # Import to new manager
        new_manager = ProfileManager()
        new_manager.import_profiles(data)
        
        imported = new_manager.get_profile(profile.user_id)
        assert imported.username == profile.username
        assert imported.preferences.theme == profile.preferences.theme
    
    def test_stats(self):
        """Test profile statistics."""
        manager = ProfileManager()
        
        basic = manager.create_profile(
            ProfileType.BASIC,
            username="basic",
            email="basic@example.com"
        )
        premium = manager.create_profile(
            ProfileType.PREMIUM,
            username="premium",
            email="premium@example.com",
            full_name="Premium User"
        )
        
        basic.record_login()
        
        stats = manager.get_stats()
        assert stats["total_profiles"] == 2
        assert stats["by_type"]["basic"] == 1
        assert stats["by_type"]["premium"] == 1
        assert stats["active_users"] == 1


class TestValidators:
    """Test profile validators."""
    
    def test_basic_validator(self):
        """Test basic profile validator."""
        validator = BasicProfileValidator()
        
        # Valid profile
        profile = UserProfile(
            username="testuser",
            email="test@example.com"
        )
        validator.validate(profile)  # Should not raise
        
        # Invalid profiles
        profile.email = "invalid"
        with pytest.raises(ValidationError):
            validator.validate(profile)
    
    def test_premium_validator(self):
        """Test premium profile validator."""
        validator = PremiumProfileValidator()
        
        # Valid profile
        profile = UserProfile(
            profile_type=ProfileType.PREMIUM,
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            age=25
        )
        validator.validate(profile)  # Should not raise
        
        # Missing full name
        profile.full_name = ""
        with pytest.raises(ValidationError):
            validator.validate(profile)


if __name__ == "__main__":
    pytest.main([__file__])