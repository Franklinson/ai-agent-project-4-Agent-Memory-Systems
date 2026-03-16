"""SQL-backed memory storage using SQLAlchemy with SQLite."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from memory_storage import (
    Memory, MemoryStorageInterface,
    MemoryStorageError, MemoryNotFoundError, MemoryConflictError,
)


class Base(DeclarativeBase):
    pass


class MemoryRow(Base):
    """SQLAlchemy model for the memories table."""
    __tablename__ = "memories"

    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=False, default="{}")
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    def to_memory(self) -> Memory:
        return Memory(
            id=self.id,
            content=self.content,
            metadata=json.loads(self.metadata_json),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class SQLStorage(MemoryStorageInterface):
    """SQL storage backend using SQLAlchemy (defaults to SQLite)."""

    def __init__(self, db_url: str = "sqlite:///memories.db", echo: bool = False):
        self._engine = create_engine(db_url, echo=echo)
        # Enable WAL mode for SQLite for better concurrent access
        if db_url.startswith("sqlite"):
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_conn, _):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def _get_session(self) -> Session:
        return self._Session()

    def save(self, content: Any, metadata: Optional[Dict[str, Any]] = None, memory_id: Optional[str] = None) -> Memory:
        now = datetime.now().isoformat()
        row = MemoryRow(
            id=memory_id or str(uuid.uuid4()),
            content=str(content),
            metadata_json=json.dumps(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        with self._get_session() as session:
            try:
                session.add(row)
                session.commit()
                return row.to_memory()
            except IntegrityError:
                session.rollback()
                raise MemoryConflictError(f"Memory '{row.id}' already exists")
            except SQLAlchemyError as e:
                session.rollback()
                raise MemoryStorageError(f"Failed to save memory: {e}")

    def retrieve(self, memory_id: str) -> Memory:
        with self._get_session() as session:
            try:
                row = session.get(MemoryRow, memory_id)
                if row is None:
                    raise MemoryNotFoundError(f"Memory '{memory_id}' not found")
                return row.to_memory()
            except MemoryNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise MemoryStorageError(f"Failed to retrieve memory: {e}")

    def update(self, memory_id: str, content: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Memory:
        with self._get_session() as session:
            try:
                row = session.get(MemoryRow, memory_id)
                if row is None:
                    raise MemoryNotFoundError(f"Memory '{memory_id}' not found")
                if content is not None:
                    row.content = str(content)
                if metadata is not None:
                    existing = json.loads(row.metadata_json)
                    existing.update(metadata)
                    row.metadata_json = json.dumps(existing)
                row.updated_at = datetime.now().isoformat()
                session.commit()
                return row.to_memory()
            except (MemoryNotFoundError, MemoryConflictError):
                raise
            except SQLAlchemyError as e:
                session.rollback()
                raise MemoryStorageError(f"Failed to update memory: {e}")

    def delete(self, memory_id: str) -> bool:
        with self._get_session() as session:
            try:
                row = session.get(MemoryRow, memory_id)
                if row is None:
                    raise MemoryNotFoundError(f"Memory '{memory_id}' not found")
                session.delete(row)
                session.commit()
                return True
            except MemoryNotFoundError:
                raise
            except SQLAlchemyError as e:
                session.rollback()
                raise MemoryStorageError(f"Failed to delete memory: {e}")

    def list_memories(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Memory]:
        with self._get_session() as session:
            try:
                rows = session.query(MemoryRow).all()
                memories = [r.to_memory() for r in rows]
                if not filter_metadata:
                    return memories
                return [
                    m for m in memories
                    if all(m.metadata.get(k) == v for k, v in filter_metadata.items())
                ]
            except SQLAlchemyError as e:
                raise MemoryStorageError(f"Failed to list memories: {e}")

    def query(self, **criteria) -> List[Memory]:
        """Query memories by column values (content, id) or metadata keys."""
        with self._get_session() as session:
            try:
                q = session.query(MemoryRow)
                if "content" in criteria:
                    q = q.filter(MemoryRow.content.contains(criteria["content"]))
                if "id" in criteria:
                    q = q.filter(MemoryRow.id == criteria["id"])
                rows = q.all()
                memories = [r.to_memory() for r in rows]
                meta_filters = {k: v for k, v in criteria.items() if k not in ("content", "id")}
                if meta_filters:
                    memories = [
                        m for m in memories
                        if all(m.metadata.get(k) == v for k, v in meta_filters.items())
                    ]
                return memories
            except SQLAlchemyError as e:
                raise MemoryStorageError(f"Query failed: {e}")

    def close(self):
        """Dispose of the engine and release connections."""
        self._engine.dispose()
