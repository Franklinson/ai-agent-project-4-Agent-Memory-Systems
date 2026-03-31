"""
Tests for Efficient Profile Retrieval System

Tests caching, query optimization, performance monitoring, and retrieval patterns.
"""

import pytest
import tempfile
import os
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch

from profile_retrieval import (
    ProfileRetrieval, LRUCache, TTLCache, ConnectionPool, CacheStrategy,
    QueryType, CacheStats, QueryStats, PerformanceMetrics
)
from user_profile import UserProfile, ProfileType, UserPreferences


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3)
        
        # Test put and get
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lru_ordering(self):
        """Test LRU ordering with access."""
        cache = LRUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5
        assert stats.size == 1
    
    def test_thread_safety(self):
        """Test thread safety of LRU cache."""
        cache = LRUCache(max_size=100)
        results = []
        
        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                cache.put(key, value)
                retrieved = cache.get(key)
                results.append(retrieved == value)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        assert all(results)


class TestTTLCache:
    """Test TTL cache implementation."""
    
    def test_basic_operations(self):
        """Test basic TTL cache operations."""
        cache = TTLCache(max_size=10, default_ttl=1)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_custom_ttl(self):
        """Test custom TTL values."""
        cache = TTLCache(max_size=10, default_ttl=10)
        
        cache.put("key1", "value1", ttl=1)  # Short TTL
        cache.put("key2", "value2", ttl=10)  # Long TTL
        
        time.sleep(1.1)
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
    
    def test_ttl_stats(self):
        """Test TTL cache statistics."""
        cache = TTLCache(max_size=10, default_ttl=1)
        
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        
        time.sleep(1.1)
        cache.get("key1")  # Miss (expired)
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1


class TestConnectionPool:
    """Test database connection pool."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
    
    def teardown_method(self):
        """Clean up test database."""
        try:
            os.unlink(self.db_path)
        except FileNotFoundError:
            pass
    
    def test_connection_pool_basic(self):
        """Test basic connection pool operations."""
        pool = ConnectionPool(self.db_path, max_connections=2)
        
        with pool.get_connection() as conn:
            assert conn is not None
            cursor = conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
        
        stats = pool.get_stats()
        assert stats["total_connections"] >= 1
    
    def test_connection_pool_reuse(self):
        """Test connection reuse in pool."""
        pool = ConnectionPool(self.db_path, max_connections=2)
        
        # Use connection and return to pool
        with pool.get_connection() as conn1:
            conn1_id = id(conn1)
        
        # Get connection again, should reuse
        with pool.get_connection() as conn2:
            conn2_id = id(conn2)
        
        stats = pool.get_stats()
        assert stats["pool_hits"] >= 1
        
        pool.close_all()
    
    def test_connection_pool_limit(self):
        """Test connection pool size limit."""
        pool = ConnectionPool(self.db_path, max_connections=1)
        
        connections = []
        
        # Get multiple connections
        for _ in range(3):
            with pool.get_connection() as conn:
                connections.append(id(conn))
        
        stats = pool.get_stats()
        # Should have created connections but limited pool size
        assert stats["total_connections"] >= 1
        
        pool.close_all()


class TestProfileRetrieval:
    """Test profile retrieval system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        self.retrieval = ProfileRetrieval(self.db_path, {
            "profile_cache_size": 10,
            "preference_cache_size": 10,
            "query_cache_size": 5
        })
        
        # Create test profiles
        self.test_profiles = []
        for i in range(5):
            profile = UserProfile(
                user_id=f"user_{i}",
                username=f"testuser_{i}",
                email=f"test{i}@example.com",
                age=20 + i,
                profile_type=ProfileType.BASIC if i < 3 else ProfileType.PREMIUM,
                full_name=f"Test User {i}" if i >= 3 else ""  # Add full_name for premium profiles
            )
            profile.preferences.language = "en" if i < 3 else "es"
            profile.preferences.theme = "light" if i % 2 == 0 else "dark"
            
            self.retrieval.store_profile(profile)
            self.test_profiles.append(profile)
    
    def teardown_method(self):
        """Clean up test environment."""
        self.retrieval.close()
        try:
            os.unlink(self.db_path)
        except FileNotFoundError:
            pass
    
    def test_direct_profile_lookup(self):
        """Test direct profile lookup."""
        profile = self.retrieval.get_profile("user_0")
        
        assert profile is not None
        assert profile.user_id == "user_0"
        assert profile.username == "testuser_0"
        assert profile.email == "test0@example.com"
    
    def test_profile_caching(self):
        """Test profile caching functionality."""
        # First lookup (cache miss)
        profile1 = self.retrieval.get_profile("user_0", use_cache=True)
        
        # Second lookup (cache hit)
        profile2 = self.retrieval.get_profile("user_0", use_cache=True)
        
        assert profile1.user_id == profile2.user_id
        
        # Check cache stats
        metrics = self.retrieval.get_performance_metrics()
        cache_stats = metrics.cache_stats["profiles"]
        assert cache_stats.hits >= 1
    
    def test_batch_retrieval(self):
        """Test batch profile retrieval."""
        user_ids = ["user_0", "user_1", "user_2"]
        profiles = self.retrieval.get_profiles_batch(user_ids)
        
        assert len(profiles) == 3
        for user_id in user_ids:
            assert user_id in profiles
            assert profiles[user_id].user_id == user_id
    
    def test_batch_retrieval_with_cache(self):
        """Test batch retrieval with partial cache hits."""
        # Cache some profiles first
        self.retrieval.get_profile("user_0", use_cache=True)
        self.retrieval.get_profile("user_1", use_cache=True)
        
        # Batch retrieve including cached and uncached
        user_ids = ["user_0", "user_1", "user_2", "user_3"]
        profiles = self.retrieval.get_profiles_batch(user_ids, use_cache=True)
        
        assert len(profiles) == 4
        for user_id in user_ids:
            assert user_id in profiles
    
    def test_filtered_search(self):
        """Test filtered profile search."""
        # Search by profile type
        results = self.retrieval.search_profiles({"profile_type": "basic"})
        assert len(results) == 3
        
        # Search by age range
        results = self.retrieval.search_profiles({"age_min": 22, "age_max": 24})
        assert len(results) == 3
        
        # Search by language
        results = self.retrieval.search_profiles({"language": "es"})
        assert len(results) == 2
    
    def test_search_caching(self):
        """Test search result caching."""
        filters = {"profile_type": "basic"}
        
        # First search (cache miss)
        results1 = self.retrieval.search_profiles(filters, use_cache=True)
        
        # Second search (cache hit)
        results2 = self.retrieval.search_profiles(filters, use_cache=True)
        
        assert len(results1) == len(results2)
        assert results1[0].user_id == results2[0].user_id
    
    def test_preference_lookup(self):
        """Test preference lookup with caching."""
        prefs = self.retrieval.get_preferences("user_0")
        
        assert prefs is not None
        assert prefs["language"] == "en"
        assert prefs["theme"] == "light"
        assert isinstance(prefs["notifications"], bool)
    
    def test_metadata_query(self):
        """Test lightweight metadata queries."""
        metadata = self.retrieval.get_profile_metadata("user_0")
        
        assert metadata is not None
        assert metadata["user_id"] == "user_0"
        assert metadata["username"] == "testuser_0"
        assert "created_at" in metadata
        assert "version" in metadata
    
    def test_cache_invalidation(self):
        """Test cache invalidation on profile updates."""
        # Cache profile
        profile = self.retrieval.get_profile("user_0", use_cache=True)
        
        # Update profile
        profile.bio = "Updated bio"
        self.retrieval.store_profile(profile)
        
        # Retrieve again, should get updated version
        updated_profile = self.retrieval.get_profile("user_0", use_cache=True)
        assert updated_profile.bio == "Updated bio"
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Perform various operations
        self.retrieval.get_profile("user_0")
        self.retrieval.get_profiles_batch(["user_1", "user_2"])
        self.retrieval.search_profiles({"profile_type": "basic"})
        self.retrieval.get_preferences("user_0")
        self.retrieval.get_profile_metadata("user_0")
        
        # Check metrics
        metrics = self.retrieval.get_performance_metrics()
        
        assert QueryType.DIRECT_LOOKUP in metrics.query_stats
        assert QueryType.BATCH_RETRIEVAL in metrics.query_stats
        assert QueryType.FILTERED_SEARCH in metrics.query_stats
        assert QueryType.PREFERENCE_LOOKUP in metrics.query_stats
        assert QueryType.METADATA_QUERY in metrics.query_stats
        
        # Check that stats were recorded
        direct_stats = metrics.query_stats[QueryType.DIRECT_LOOKUP]
        assert direct_stats.count >= 1
        assert direct_stats.total_time > 0
    
    def test_query_optimization(self):
        """Test query optimization features."""
        # Test that indexed queries are fast
        start_time = time.time()
        
        # These should use indexes
        self.retrieval.search_profiles({"profile_type": "basic"})
        self.retrieval.search_profiles({"username_like": "testuser"})
        self.retrieval.search_profiles({"age_min": 20})
        
        end_time = time.time()
        
        # Should complete quickly with indexes
        assert (end_time - start_time) < 1.0
    
    def test_concurrent_access(self):
        """Test concurrent access to retrieval system."""
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(5):
                    user_id = f"user_{i % len(self.test_profiles)}"
                    profile = self.retrieval.get_profile(user_id)
                    if profile:
                        results.append(profile.user_id)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have no errors and successful retrievals
        assert len(errors) == 0
        assert len(results) > 0
    
    def test_database_optimization(self):
        """Test database optimization operations."""
        # Should not raise any errors
        self.retrieval.optimize_database()
        
        # Verify database is still functional
        profile = self.retrieval.get_profile("user_0")
        assert profile is not None
    
    def test_cache_strategies(self):
        """Test different cache strategies."""
        # Test LRU cache behavior
        lru_cache = LRUCache(max_size=2)
        lru_cache.put("a", 1)
        lru_cache.put("b", 2)
        lru_cache.put("c", 3)  # Should evict 'a'
        
        assert lru_cache.get("a") is None
        assert lru_cache.get("b") == 2
        assert lru_cache.get("c") == 3
        
        # Test TTL cache behavior
        ttl_cache = TTLCache(max_size=10, default_ttl=1)
        ttl_cache.put("key", "value")
        assert ttl_cache.get("key") == "value"
        
        time.sleep(1.1)
        assert ttl_cache.get("key") is None
    
    def test_performance_metrics_summary(self):
        """Test performance metrics summary."""
        # Perform operations
        self.retrieval.get_profile("user_0")
        self.retrieval.search_profiles({"profile_type": "basic"})
        
        metrics = self.retrieval.get_performance_metrics()
        summary = metrics.get_summary()
        
        assert "uptime_seconds" in summary
        assert "cache_hit_rates" in summary
        assert "query_avg_times" in summary
        assert "total_queries" in summary
        assert summary["total_queries"] >= 2
    
    def test_error_handling(self):
        """Test error handling in retrieval operations."""
        # Test with non-existent profile
        profile = self.retrieval.get_profile("nonexistent")
        assert profile is None
        
        # Test with invalid filters
        results = self.retrieval.search_profiles({"invalid_field": "value"})
        assert isinstance(results, list)  # Should return empty list, not error
    
    def test_cache_memory_management(self):
        """Test cache memory management."""
        # Fill cache beyond capacity
        cache = LRUCache(max_size=3)
        
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")
        
        stats = cache.get_stats()
        assert stats.size <= 3
        assert stats.evictions > 0


if __name__ == "__main__":
    pytest.main([__file__])