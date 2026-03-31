"""
Efficient Profile Retrieval with Caching

Implements optimized profile retrieval with multi-level caching, query optimization,
and comprehensive performance monitoring.
"""

import sqlite3
import json
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
import hashlib
import weakref
from functools import wraps

from user_profile import UserProfile, ProfileType, ProfileManager


class CacheStrategy(Enum):
    """Cache strategies for different data types."""
    LRU = "lru"
    TTL = "ttl"
    LFU = "lfu"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class QueryType(Enum):
    """Types of queries for optimization."""
    DIRECT_LOOKUP = "direct_lookup"
    FILTERED_SEARCH = "filtered_search"
    BATCH_RETRIEVAL = "batch_retrieval"
    PREFERENCE_LOOKUP = "preference_lookup"
    METADATA_QUERY = "metadata_query"


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class QueryStats:
    """Query performance statistics."""
    query_type: QueryType
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: int = 0
    
    @property
    def avg_time(self) -> float:
        """Calculate average query time."""
        return self.total_time / self.count if self.count > 0 else 0.0


@dataclass
class PerformanceMetrics:
    """Overall performance metrics."""
    cache_stats: Dict[str, CacheStats] = field(default_factory=dict)
    query_stats: Dict[QueryType, QueryStats] = field(default_factory=dict)
    connection_pool_stats: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "cache_hit_rates": {name: stats.hit_rate for name, stats in self.cache_stats.items()},
            "query_avg_times": {qtype.value: stats.avg_time for qtype, stats in self.query_stats.items()},
            "total_queries": sum(stats.count for stats in self.query_stats.values()),
            "total_errors": sum(stats.errors for stats in self.query_stats.values())
        }


class Cache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class LRUCache(Cache):
    """LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._stats.hits += 1
                return value
            else:
                self._stats.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Evict least recently used
                self._cache.popitem(last=False)
                self._stats.evictions += 1
            
            self._cache[key] = value
            self._stats.size = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats


class TTLCache(Cache):
    """TTL (Time To Live) cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._expiry = {}
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        return key in self._expiry and time.time() > self._expiry[key]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [k for k, exp_time in self._expiry.items() if current_time > exp_time]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                self._stats.hits += 1
                return self._cache[key]
            else:
                if key in self._cache:
                    # Remove expired entry
                    del self._cache[key]
                    self._expiry.pop(key, None)
                self._stats.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Cleanup expired entries periodically
            if len(self._cache) % 100 == 0:
                self._cleanup_expired()
            
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._expiry.pop(oldest_key, None)
                self._stats.evictions += 1
            
            self._cache[key] = value
            self._expiry[key] = time.time() + (ttl or self.default_ttl)
            self._stats.size = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._expiry.pop(key, None)
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._cleanup_expired()
            self._stats.size = len(self._cache)
            return self._stats


class ConnectionPool:
    """Database connection pool for optimized access."""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = []
        self._in_use = set()
        self._lock = threading.Lock()
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "pool_hits": 0,
            "pool_misses": 0
        }
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool."""
        conn = None
        try:
            with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    self._stats["pool_hits"] += 1
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    self._stats["total_connections"] += 1
                    self._stats["pool_misses"] += 1
                
                self._in_use.add(conn)
                self._stats["active_connections"] = len(self._in_use)
            
            yield conn
            
        finally:
            if conn:
                with self._lock:
                    self._in_use.discard(conn)
                    if len(self._pool) < self.max_connections:
                        self._pool.append(conn)
                    else:
                        conn.close()
                    self._stats["active_connections"] = len(self._in_use)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return self._stats.copy()
    
    def close_all(self) -> None:
        """Close all connections in pool."""
        with self._lock:
            for conn in self._pool:
                conn.close()
            for conn in self._in_use:
                conn.close()
            self._pool.clear()
            self._in_use.clear()


def performance_monitor(query_type: QueryType):
    """Decorator to monitor query performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                execution_time = time.time() - start_time
                self._update_query_stats(query_type, execution_time, success=True)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self._update_query_stats(query_type, execution_time, success=False)
                raise
        return wrapper
    return decorator


class ProfileRetrieval:
    """Efficient profile retrieval system with caching and optimization."""
    
    def __init__(self, db_path: str, cache_config: Optional[Dict[str, Any]] = None):
        self.db_path = db_path
        self.connection_pool = ConnectionPool(db_path)
        
        # Initialize caches
        cache_config = cache_config or {}
        self.profile_cache = LRUCache(cache_config.get("profile_cache_size", 1000))
        self.preference_cache = TTLCache(
            cache_config.get("preference_cache_size", 500),
            cache_config.get("preference_ttl", 1800)  # 30 minutes
        )
        self.query_cache = TTLCache(
            cache_config.get("query_cache_size", 200),
            cache_config.get("query_ttl", 300)  # 5 minutes
        )
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self._init_metrics()
        
        # Initialize database
        self._init_database()
    
    def _init_metrics(self) -> None:
        """Initialize performance metrics."""
        self.metrics.cache_stats["profiles"] = self.profile_cache.get_stats()
        self.metrics.cache_stats["preferences"] = self.preference_cache.get_stats()
        self.metrics.cache_stats["queries"] = self.query_cache.get_stats()
        
        for query_type in QueryType:
            self.metrics.query_stats[query_type] = QueryStats(query_type)
    
    def _update_query_stats(self, query_type: QueryType, execution_time: float, success: bool = True) -> None:
        """Update query performance statistics."""
        stats = self.metrics.query_stats[query_type]
        stats.count += 1
        stats.total_time += execution_time
        stats.min_time = min(stats.min_time, execution_time)
        stats.max_time = max(stats.max_time, execution_time)
        if not success:
            stats.errors += 1
    
    def _init_database(self) -> None:
        """Initialize database with optimized schema and indexes."""
        with self.connection_pool.get_connection() as conn:
            # Create profiles table with indexes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_type TEXT NOT NULL,
                    username TEXT NOT NULL,
                    email TEXT NOT NULL,
                    full_name TEXT,
                    age INTEGER,
                    bio TEXT,
                    avatar_url TEXT,
                    tags TEXT,  -- JSON array
                    custom_data TEXT,  -- JSON object
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    last_login TEXT
                )
            """)
            
            # Create preferences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT PRIMARY KEY,
                    language TEXT DEFAULT 'en',
                    timezone TEXT DEFAULT 'UTC',
                    theme TEXT DEFAULT 'light',
                    notifications INTEGER DEFAULT 1,
                    email_updates INTEGER DEFAULT 1,
                    privacy_level TEXT DEFAULT 'medium',
                    custom_settings TEXT,  -- JSON object
                    FOREIGN KEY (user_id) REFERENCES profiles (user_id)
                )
            """)
            
            # Create optimized indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_profiles_type ON profiles(profile_type)",
                "CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(username)",
                "CREATE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email)",
                "CREATE INDEX IF NOT EXISTS idx_profiles_age ON profiles(age)",
                "CREATE INDEX IF NOT EXISTS idx_profiles_updated ON profiles(updated_at)",
                "CREATE INDEX IF NOT EXISTS idx_preferences_language ON preferences(language)",
                "CREATE INDEX IF NOT EXISTS idx_preferences_theme ON preferences(theme)"
            ]
            
            for index_sql in indexes:
                conn.execute(index_sql)
            
            conn.commit()
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @performance_monitor(QueryType.DIRECT_LOOKUP)
    def get_profile(self, user_id: str, use_cache: bool = True) -> Optional[UserProfile]:
        """Get profile by user ID with caching."""
        if use_cache:
            cache_key = f"profile:{user_id}"
            cached_profile = self.profile_cache.get(cache_key)
            if cached_profile:
                return cached_profile
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT p.*, pr.language, pr.timezone, pr.theme, pr.notifications,
                       pr.email_updates, pr.privacy_level, pr.custom_settings
                FROM profiles p
                LEFT JOIN preferences pr ON p.user_id = pr.user_id
                WHERE p.user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            profile = self._row_to_profile(row)
            
            if use_cache:
                self.profile_cache.put(cache_key, profile)
            
            return profile
    
    @performance_monitor(QueryType.BATCH_RETRIEVAL)
    def get_profiles_batch(self, user_ids: List[str], use_cache: bool = True) -> Dict[str, UserProfile]:
        """Get multiple profiles efficiently."""
        results = {}
        uncached_ids = []
        
        if use_cache:
            # Check cache first
            for user_id in user_ids:
                cache_key = f"profile:{user_id}"
                cached_profile = self.profile_cache.get(cache_key)
                if cached_profile:
                    results[user_id] = cached_profile
                else:
                    uncached_ids.append(user_id)
        else:
            uncached_ids = user_ids
        
        if uncached_ids:
            # Batch query for uncached profiles
            placeholders = ",".join("?" * len(uncached_ids))
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(f"""
                    SELECT p.*, pr.language, pr.timezone, pr.theme, pr.notifications,
                           pr.email_updates, pr.privacy_level, pr.custom_settings
                    FROM profiles p
                    LEFT JOIN preferences pr ON p.user_id = pr.user_id
                    WHERE p.user_id IN ({placeholders})
                """, uncached_ids)
                
                for row in cursor.fetchall():
                    profile = self._row_to_profile(row)
                    results[profile.user_id] = profile
                    
                    if use_cache:
                        cache_key = f"profile:{profile.user_id}"
                        self.profile_cache.put(cache_key, profile)
        
        return results
    
    @performance_monitor(QueryType.FILTERED_SEARCH)
    def search_profiles(self, filters: Dict[str, Any], limit: int = 100, use_cache: bool = True) -> List[UserProfile]:
        """Search profiles with filters and caching."""
        cache_key = self._generate_cache_key("search", **filters, limit=limit)
        
        if use_cache:
            cached_results = self.query_cache.get(cache_key)
            if cached_results:
                return cached_results
        
        # Build dynamic query
        where_clauses = []
        params = []
        
        if "profile_type" in filters:
            where_clauses.append("p.profile_type = ?")
            params.append(filters["profile_type"])
        
        if "age_min" in filters:
            where_clauses.append("p.age >= ?")
            params.append(filters["age_min"])
        
        if "age_max" in filters:
            where_clauses.append("p.age <= ?")
            params.append(filters["age_max"])
        
        if "username_like" in filters:
            where_clauses.append("p.username LIKE ?")
            params.append(f"%{filters['username_like']}%")
        
        if "email_domain" in filters:
            where_clauses.append("p.email LIKE ?")
            params.append(f"%@{filters['email_domain']}")
        
        if "language" in filters:
            where_clauses.append("pr.language = ?")
            params.append(filters["language"])
        
        if "theme" in filters:
            where_clauses.append("pr.theme = ?")
            params.append(filters["theme"])
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
            SELECT p.*, pr.language, pr.timezone, pr.theme, pr.notifications,
                   pr.email_updates, pr.privacy_level, pr.custom_settings
            FROM profiles p
            LEFT JOIN preferences pr ON p.user_id = pr.user_id
            WHERE {where_clause}
            ORDER BY p.updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(query, params)
            results = [self._row_to_profile(row) for row in cursor.fetchall()]
        
        if use_cache:
            self.query_cache.put(cache_key, results)
        
        return results
    
    @performance_monitor(QueryType.PREFERENCE_LOOKUP)
    def get_preferences(self, user_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get user preferences with caching."""
        if use_cache:
            cache_key = f"prefs:{user_id}"
            cached_prefs = self.preference_cache.get(cache_key)
            if cached_prefs:
                return cached_prefs
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT language, timezone, theme, notifications, email_updates,
                       privacy_level, custom_settings
                FROM preferences
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            preferences = {
                "language": row["language"],
                "timezone": row["timezone"],
                "theme": row["theme"],
                "notifications": bool(row["notifications"]),
                "email_updates": bool(row["email_updates"]),
                "privacy_level": row["privacy_level"],
                "custom_settings": json.loads(row["custom_settings"]) if row["custom_settings"] else {}
            }
            
            if use_cache:
                self.preference_cache.put(cache_key, preferences)
            
            return preferences
    
    @performance_monitor(QueryType.METADATA_QUERY)
    def get_profile_metadata(self, user_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get profile metadata (lightweight query)."""
        cache_key = f"meta:{user_id}"
        
        if use_cache:
            cached_meta = self.query_cache.get(cache_key)
            if cached_meta:
                return cached_meta
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT user_id, profile_type, username, created_at, updated_at,
                       version, last_login
                FROM profiles
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            metadata = {
                "user_id": row["user_id"],
                "profile_type": row["profile_type"],
                "username": row["username"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "version": row["version"],
                "last_login": row["last_login"]
            }
            
            if use_cache:
                self.query_cache.put(cache_key, metadata, ttl=600)  # 10 minutes
            
            return metadata
    
    def _row_to_profile(self, row: sqlite3.Row) -> UserProfile:
        """Convert database row to UserProfile object."""
        from user_profile import UserPreferences, ProfileMetadata
        
        # Parse preferences
        preferences = UserPreferences(
            language=row["language"] or "en",
            timezone=row["timezone"] or "UTC",
            theme=row["theme"] or "light",
            notifications=bool(row["notifications"]) if row["notifications"] is not None else True,
            email_updates=bool(row["email_updates"]) if row["email_updates"] is not None else True,
            privacy_level=row["privacy_level"] or "medium",
            custom_settings=json.loads(row["custom_settings"]) if row["custom_settings"] else {}
        )
        
        # Parse metadata
        metadata = ProfileMetadata(
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            version=row["version"],
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None
        )
        
        # Create profile
        profile = UserProfile(
            user_id=row["user_id"],
            profile_type=ProfileType(row["profile_type"]),
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"] or "",
            age=row["age"],
            bio=row["bio"] or "",
            avatar_url=row["avatar_url"] or "",
            preferences=preferences,
            metadata=metadata,
            tags=json.loads(row["tags"]) if row["tags"] else [],
            custom_data=json.loads(row["custom_data"]) if row["custom_data"] else {}
        )
        
        return profile
    
    def store_profile(self, profile: UserProfile) -> None:
        """Store profile in database and invalidate cache."""
        with self.connection_pool.get_connection() as conn:
            # Insert/update profile
            conn.execute("""
                INSERT OR REPLACE INTO profiles
                (user_id, profile_type, username, email, full_name, age, bio,
                 avatar_url, tags, custom_data, created_at, updated_at, version, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.profile_type.value,
                profile.username,
                profile.email,
                profile.full_name,
                profile.age,
                profile.bio,
                profile.avatar_url,
                json.dumps(profile.tags),
                json.dumps(profile.custom_data),
                profile.metadata.created_at.isoformat(),
                profile.metadata.updated_at.isoformat(),
                profile.metadata.version,
                profile.metadata.last_login.isoformat() if profile.metadata.last_login else None
            ))
            
            # Insert/update preferences
            conn.execute("""
                INSERT OR REPLACE INTO preferences
                (user_id, language, timezone, theme, notifications, email_updates,
                 privacy_level, custom_settings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.preferences.language,
                profile.preferences.timezone,
                profile.preferences.theme,
                int(profile.preferences.notifications),
                int(profile.preferences.email_updates),
                profile.preferences.privacy_level,
                json.dumps(profile.preferences.custom_settings)
            ))
            
            conn.commit()
        
        # Invalidate caches
        self.invalidate_cache(profile.user_id)
    
    def invalidate_cache(self, user_id: str) -> None:
        """Invalidate all cached data for a user."""
        self.profile_cache.delete(f"profile:{user_id}")
        self.preference_cache.delete(f"prefs:{user_id}")
        self.query_cache.delete(f"meta:{user_id}")
        
        # Clear query cache (since search results might be affected)
        self.query_cache.clear()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Update cache stats
        self.metrics.cache_stats["profiles"] = self.profile_cache.get_stats()
        self.metrics.cache_stats["preferences"] = self.preference_cache.get_stats()
        self.metrics.cache_stats["queries"] = self.query_cache.get_stats()
        self.metrics.connection_pool_stats = self.connection_pool.get_stats()
        
        return self.metrics
    
    def optimize_database(self) -> None:
        """Run database optimization operations."""
        with self.connection_pool.get_connection() as conn:
            # Analyze tables for query optimization
            conn.execute("ANALYZE profiles")
            conn.execute("ANALYZE preferences")
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
            conn.commit()
    
    def close(self) -> None:
        """Close all resources."""
        self.connection_pool.close_all()
        self.profile_cache.clear()
        self.preference_cache.clear()
        self.query_cache.clear()