# -*- coding: utf-8 -*-
"""
AI 配置与报告历史存储
"""
import json
import sqlite3
from datetime import datetime

from config import DB_PATH


class AIStore:
    """管理 AI 接入配置和报告历史"""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._ensure_schema()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        # 每个新连接显式启用 FK 约束（SQLite 默认 OFF）
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        # 迁移期间临时关闭 FK，避免 DROP/RENAME 中间态触发约束失败
        cursor.execute("PRAGMA foreign_keys = OFF")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_settings (
                settings_id INTEGER PRIMARY KEY CHECK (settings_id = 1),
                enabled INTEGER NOT NULL DEFAULT 0,
                provider_name TEXT DEFAULT '自定义（兼容 OpenAI）',
                base_url TEXT DEFAULT '',
                api_key TEXT DEFAULT '',
                model_name TEXT DEFAULT '',
                system_prompt TEXT DEFAULT '',
                temperature REAL DEFAULT 0.2,
                timeout_sec INTEGER DEFAULT 60,
                updated_at TEXT DEFAULT ''
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM ai_settings")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO ai_settings
                (settings_id, enabled, provider_name, base_url, api_key, model_name,
                 system_prompt, temperature, timeout_sec, updated_at)
                VALUES (1, 0, '自定义（兼容 OpenAI）', '', '', '', '', 0.2, 60, ?)
            """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_report_history (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 0,
                username TEXT NOT NULL,
                report_time TEXT NOT NULL,
                provider_name TEXT DEFAULT '',
                model_name TEXT DEFAULT '',
                record_count INTEGER DEFAULT 0,
                source_summary TEXT DEFAULT '',
                report_json TEXT NOT NULL,
                raw_response TEXT DEFAULT ''
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_chat_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                user_id INTEGER DEFAULT 0,
                username TEXT NOT NULL,
                session_title TEXT DEFAULT '新聊天',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_report_chat_history (
                chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER DEFAULT 0,
                report_id INTEGER NOT NULL,
                user_id INTEGER DEFAULT 0,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        self._ensure_column(cursor, "ai_report_history", "user_id", "INTEGER DEFAULT 0")
        self._ensure_column(cursor, "ai_chat_sessions", "user_id", "INTEGER DEFAULT 0")
        self._ensure_column(cursor, "ai_report_chat_history", "user_id", "INTEGER DEFAULT 0")
        self._ensure_column(cursor, "ai_report_chat_history", "session_id", "INTEGER DEFAULT 0")
        self._backfill_user_id_columns(cursor)
        self._migrate_legacy_chat_sessions(cursor)

        # Batch 1: 升级 ai_report_history 为带 FK 的版本
        self._upgrade_ai_report_history_fk(cursor)
        # Batch 2: 升级 ai_chat_sessions (依赖 users + ai_report_history)
        self._upgrade_ai_chat_sessions_fk(cursor)
        # Batch 3: 升级 ai_report_chat_history (依赖 users + ai_report_history + ai_chat_sessions)
        self._upgrade_ai_report_chat_history_fk(cursor)

        # 迁移完成后校验完整性并重开 FK
        cursor.execute("PRAGMA foreign_key_check")
        violations = cursor.fetchall()
        if violations:
            conn.close()
            raise RuntimeError(f"ai_store 外键完整性校验失败: {violations}")
        cursor.execute("PRAGMA foreign_keys = ON")

        conn.commit()
        conn.close()

    def _upgrade_ai_report_history_fk(self, cursor):
        """将 ai_report_history 升级为 user_id NOT NULL + FK→users ON DELETE CASCADE"""
        cursor.execute("PRAGMA foreign_key_list(ai_report_history)")
        if any(fk[2] == 'users' for fk in cursor.fetchall()):
            return

        cursor.execute("SELECT COUNT(*) FROM ai_report_history WHERE user_id IS NULL OR user_id = 0")
        bad = cursor.fetchone()[0]
        if bad > 0:
            raise RuntimeError(
                f"ai_report_history 中有 {bad} 行 user_id 为 0 或 NULL，无法升级为 NOT NULL + FK。"
            )
        cursor.execute("""
            SELECT COUNT(*) FROM ai_report_history
            WHERE user_id NOT IN (SELECT user_id FROM users)
        """)
        orphan = cursor.fetchone()[0]
        if orphan > 0:
            raise RuntimeError(
                f"ai_report_history 中有 {orphan} 行 user_id 指向不存在的用户，无法升级为 FK。"
            )

        cursor.execute("""
            CREATE TABLE ai_report_history_fknew (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                report_time TEXT NOT NULL,
                provider_name TEXT DEFAULT '',
                model_name TEXT DEFAULT '',
                record_count INTEGER DEFAULT 0,
                source_summary TEXT DEFAULT '',
                report_json TEXT NOT NULL,
                raw_response TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            INSERT INTO ai_report_history_fknew
            (report_id, user_id, username, report_time, provider_name, model_name,
             record_count, source_summary, report_json, raw_response)
            SELECT report_id, user_id, username, report_time, provider_name, model_name,
                   record_count, source_summary, report_json, raw_response
            FROM ai_report_history
        """)
        cursor.execute("DROP TABLE ai_report_history")
        cursor.execute("ALTER TABLE ai_report_history_fknew RENAME TO ai_report_history")
        # 注：ALTER TABLE RENAME 已自动维护 sqlite_sequence
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_report_history_user_id ON ai_report_history(user_id)")
        print("[ai_store 升级] ai_report_history 已添加 FK 约束 + user_id NOT NULL")

    def _upgrade_ai_chat_sessions_fk(self, cursor):
        """
        ai_chat_sessions 升级：
          user_id   NOT NULL  → FK users(user_id)           ON DELETE CASCADE
          report_id NOT NULL  → FK ai_report_history(report_id) ON DELETE CASCADE
        """
        cursor.execute("PRAGMA foreign_key_list(ai_chat_sessions)")
        fks = cursor.fetchall()
        has_users_fk = any(fk[2] == 'users' for fk in fks)
        has_report_fk = any(fk[2] == 'ai_report_history' for fk in fks)
        if has_users_fk and has_report_fk:
            return

        # 防御检查
        cursor.execute("SELECT COUNT(*) FROM ai_chat_sessions WHERE user_id IS NULL OR user_id = 0")
        bad_user = cursor.fetchone()[0]
        if bad_user > 0:
            raise RuntimeError(f"ai_chat_sessions 有 {bad_user} 行 user_id 无效，无法升级。")

        cursor.execute("""
            SELECT COUNT(*) FROM ai_chat_sessions
            WHERE user_id NOT IN (SELECT user_id FROM users)
        """)
        orphan_user = cursor.fetchone()[0]
        if orphan_user > 0:
            raise RuntimeError(f"ai_chat_sessions 有 {orphan_user} 行 user_id 指向不存在用户。")

        cursor.execute("SELECT COUNT(*) FROM ai_chat_sessions WHERE report_id IS NULL OR report_id = 0")
        bad_report = cursor.fetchone()[0]
        if bad_report > 0:
            raise RuntimeError(f"ai_chat_sessions 有 {bad_report} 行 report_id 无效，无法升级。")

        cursor.execute("""
            SELECT COUNT(*) FROM ai_chat_sessions
            WHERE report_id NOT IN (SELECT report_id FROM ai_report_history)
        """)
        orphan_report = cursor.fetchone()[0]
        if orphan_report > 0:
            raise RuntimeError(f"ai_chat_sessions 有 {orphan_report} 行 report_id 指向不存在的报告。")

        cursor.execute("""
            CREATE TABLE ai_chat_sessions_fknew (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                session_title TEXT DEFAULT '新聊天',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (report_id) REFERENCES ai_report_history(report_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            INSERT INTO ai_chat_sessions_fknew
            (session_id, report_id, user_id, username, session_title, created_at, updated_at)
            SELECT session_id, report_id, user_id, username, session_title, created_at, updated_at
            FROM ai_chat_sessions
        """)
        cursor.execute("DROP TABLE ai_chat_sessions")
        cursor.execute("ALTER TABLE ai_chat_sessions_fknew RENAME TO ai_chat_sessions")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_chat_sessions_user_id ON ai_chat_sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_chat_sessions_report_id ON ai_chat_sessions(report_id)")
        print("[ai_store 升级] ai_chat_sessions 已添加 FK 约束 (user_id + report_id，均 ON DELETE CASCADE)")

    def _upgrade_ai_report_chat_history_fk(self, cursor):
        """
        ai_report_chat_history 升级（最复杂，3 个 FK）：
          user_id    NOT NULL → FK users(user_id)               ON DELETE CASCADE
          report_id  NOT NULL → FK ai_report_history(report_id) ON DELETE CASCADE
          session_id NOT NULL → FK ai_chat_sessions(session_id) ON DELETE CASCADE
        """
        cursor.execute("PRAGMA foreign_key_list(ai_report_chat_history)")
        fks = cursor.fetchall()
        has_users_fk = any(fk[2] == 'users' for fk in fks)
        has_report_fk = any(fk[2] == 'ai_report_history' for fk in fks)
        has_session_fk = any(fk[2] == 'ai_chat_sessions' for fk in fks)
        if has_users_fk and has_report_fk and has_session_fk:
            return

        # 防御检查 1: user_id
        cursor.execute("SELECT COUNT(*) FROM ai_report_chat_history WHERE user_id IS NULL OR user_id = 0")
        bad_user = cursor.fetchone()[0]
        if bad_user > 0:
            raise RuntimeError(
                f"ai_report_chat_history 有 {bad_user} 行 user_id 无效（NULL/0），无法升级为 NOT NULL + FK。"
            )
        cursor.execute("""
            SELECT COUNT(*) FROM ai_report_chat_history
            WHERE user_id NOT IN (SELECT user_id FROM users)
        """)
        orphan_user = cursor.fetchone()[0]
        if orphan_user > 0:
            raise RuntimeError(
                f"ai_report_chat_history 有 {orphan_user} 行 user_id 指向不存在的用户，无法升级为 FK。"
            )

        # 防御检查 2: report_id
        cursor.execute("SELECT COUNT(*) FROM ai_report_chat_history WHERE report_id IS NULL OR report_id = 0")
        bad_report = cursor.fetchone()[0]
        if bad_report > 0:
            raise RuntimeError(
                f"ai_report_chat_history 有 {bad_report} 行 report_id 无效（NULL/0），无法升级为 NOT NULL + FK。"
            )
        cursor.execute("""
            SELECT COUNT(*) FROM ai_report_chat_history
            WHERE report_id NOT IN (SELECT report_id FROM ai_report_history)
        """)
        orphan_report = cursor.fetchone()[0]
        if orphan_report > 0:
            raise RuntimeError(
                f"ai_report_chat_history 有 {orphan_report} 行 report_id 指向不存在的报告，无法升级为 FK。"
            )

        # 防御检查 3: session_id
        cursor.execute("SELECT COUNT(*) FROM ai_report_chat_history WHERE session_id IS NULL OR session_id = 0")
        bad_session = cursor.fetchone()[0]
        if bad_session > 0:
            raise RuntimeError(
                f"ai_report_chat_history 有 {bad_session} 行 session_id 无效（NULL/0）。"
                f" 这通常来自未经 _migrate_legacy_chat_sessions 处理的 legacy 数据。"
            )
        cursor.execute("""
            SELECT COUNT(*) FROM ai_report_chat_history
            WHERE session_id NOT IN (SELECT session_id FROM ai_chat_sessions)
        """)
        orphan_session = cursor.fetchone()[0]
        if orphan_session > 0:
            raise RuntimeError(
                f"ai_report_chat_history 有 {orphan_session} 行 session_id 指向不存在的会话，无法升级为 FK。"
            )

        cursor.execute("""
            CREATE TABLE ai_report_chat_history_fknew (
                chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                report_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (report_id) REFERENCES ai_report_history(report_id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES ai_chat_sessions(session_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            INSERT INTO ai_report_chat_history_fknew
            (chat_id, session_id, report_id, user_id, username, role, content, created_at)
            SELECT chat_id, session_id, report_id, user_id, username, role, content, created_at
            FROM ai_report_chat_history
        """)
        cursor.execute("DROP TABLE ai_report_chat_history")
        cursor.execute("ALTER TABLE ai_report_chat_history_fknew RENAME TO ai_report_chat_history")
        # 注：ALTER TABLE RENAME 已自动维护 sqlite_sequence
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_report_chat_history_user_id ON ai_report_chat_history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_report_chat_history_session_id ON ai_report_chat_history(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_report_chat_history_report_id ON ai_report_chat_history(report_id)")
        print("[ai_store 升级] ai_report_chat_history 已添加 3 个 FK 约束 (user_id + report_id + session_id，全部 ON DELETE CASCADE)")

    def _ensure_column(self, cursor, table_name, column_name, ddl):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")

    def _clean_user_id(self, user_id):
        try:
            value = int(user_id)
            return value if value > 0 else None
        except Exception:
            return None

    def _resolve_user_identity(self, cursor, user_id=None, username=None):
        clean_user_id = self._clean_user_id(user_id)
        clean_username = (username or "").strip() or None

        if clean_user_id is not None:
            cursor.execute("SELECT user_id, username FROM users WHERE user_id=?", (clean_user_id,))
            row = cursor.fetchone()
            if row:
                return int(row[0]), row[1]

        if clean_username:
            cursor.execute("SELECT user_id, username FROM users WHERE username=?", (clean_username,))
            row = cursor.fetchone()
            if row:
                return int(row[0]), row[1]

        return clean_user_id, clean_username

    def _backfill_user_id_columns(self, cursor):
        for table_name in ["ai_report_history", "ai_chat_sessions", "ai_report_chat_history"]:
            cursor.execute(f"""
                UPDATE {table_name}
                SET user_id = (
                    SELECT users.user_id
                    FROM users
                    WHERE users.username = {table_name}.username
                )
                WHERE (user_id IS NULL OR user_id = 0)
                  AND COALESCE(username, '') != ''
            """)

        cursor.execute("""
            UPDATE ai_chat_sessions
            SET user_id = (
                SELECT ai_report_history.user_id
                FROM ai_report_history
                WHERE ai_report_history.report_id = ai_chat_sessions.report_id
            )
            WHERE (user_id IS NULL OR user_id = 0)
        """)
        cursor.execute("""
            UPDATE ai_report_chat_history
            SET user_id = (
                SELECT ai_chat_sessions.user_id
                FROM ai_chat_sessions
                WHERE ai_chat_sessions.session_id = ai_report_chat_history.session_id
            )
            WHERE (user_id IS NULL OR user_id = 0)
              AND COALESCE(session_id, 0) > 0
        """)

    def _migrate_legacy_chat_sessions(self, cursor):
        cursor.execute("""
            SELECT report_id, COALESCE(user_id, 0), username, MIN(created_at), MAX(created_at)
            FROM ai_report_chat_history
            WHERE COALESCE(session_id, 0) = 0
            GROUP BY report_id, COALESCE(user_id, 0), username
        """)
        legacy_groups = cursor.fetchall()
        for report_id, user_id, username, min_created_at, max_created_at in legacy_groups:
            # 方案 A 严格模式：legacy 数据中 user_id=0 意味着无法归属到任何真实用户，
            # 启用 FK 后无法作为合法子行存在。跳过这些孤儿并打印警告，避免迁移崩溃。
            if not user_id or int(user_id) <= 0:
                print(f"[ai_store 迁移] 跳过 legacy 聊天组 (report_id={report_id}, username={username}): "
                      f"user_id=0，无法归属到真实用户")
                continue
            created_at = min_created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated_at = max_created_at or created_at
            cursor.execute("""
                INSERT INTO ai_chat_sessions
                (report_id, user_id, username, session_title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(report_id),
                int(user_id),
                username,
                "历史聊天",
                created_at,
                updated_at,
            ))
            session_id = cursor.lastrowid
            cursor.execute("""
                UPDATE ai_report_chat_history
                SET session_id=?
                WHERE report_id=? AND COALESCE(user_id, 0)=? AND COALESCE(username, '')=? AND COALESCE(session_id, 0) = 0
            """, (session_id, int(report_id), int(user_id), username or ""))

    def get_ai_settings(self):
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ai_settings WHERE settings_id = 1")
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return {
                "settings_id": 1,
                "enabled": 0,
                "provider_name": "自定义（兼容 OpenAI）",
                "base_url": "",
                "api_key": "",
                "model_name": "",
                "system_prompt": "",
                "temperature": 0.2,
                "timeout_sec": 60,
                "updated_at": "",
            }
        return dict(row)

    def save_ai_settings(self, enabled, provider_name, base_url, api_key,
                         model_name, system_prompt, temperature, timeout_sec):
        existing = self.get_ai_settings()
        final_api_key = api_key.strip() if api_key and api_key.strip() else existing.get("api_key", "")
        conn = self._get_conn()
        cursor = conn.cursor()
        payload = (
            int(bool(enabled)),
            provider_name.strip(),
            base_url.strip(),
            final_api_key,
            model_name.strip(),
            system_prompt.strip(),
            float(temperature),
            int(timeout_sec),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        cursor.execute("SELECT COUNT(*) FROM ai_settings WHERE settings_id = 1")
        exists = cursor.fetchone()[0] > 0
        if exists:
            cursor.execute("""
                UPDATE ai_settings
                SET enabled=?, provider_name=?, base_url=?, api_key=?, model_name=?,
                    system_prompt=?, temperature=?, timeout_sec=?, updated_at=?
                WHERE settings_id = 1
            """, payload)
        else:
            cursor.execute("""
                INSERT INTO ai_settings
                (enabled, provider_name, base_url, api_key, model_name,
                 system_prompt, temperature, timeout_sec, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, payload)
        conn.commit()
        conn.close()

    def save_ai_report(self, username=None, provider_name="", model_name="", record_count=0,
                       source_summary=None, report_json=None, raw_response="", user_id=None):
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is None:
            conn.close()
            raise ValueError("save_ai_report: 缺少有效的 user_id，调用方必须传入已登录用户 ID")
        cursor.execute("""
            INSERT INTO ai_report_history
            (user_id, username, report_time, provider_name, model_name, record_count,
             source_summary, report_json, raw_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            username,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            provider_name,
            model_name,
            int(record_count),
            json.dumps(source_summary, ensure_ascii=False),
            json.dumps(report_json, ensure_ascii=False),
            raw_response,
        ))
        report_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return report_id

    def get_ai_reports(self, username=None, limit=20, user_id=None):
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None:
            cursor.execute("""
                SELECT report_id, username, report_time, provider_name, model_name, record_count
                FROM ai_report_history
                WHERE user_id=?
                ORDER BY report_time DESC, report_id DESC
                LIMIT ?
            """, (user_id, limit))
        elif username:
            cursor.execute("""
                SELECT report_id, username, report_time, provider_name, model_name, record_count
                FROM ai_report_history
                WHERE username=?
                ORDER BY report_time DESC, report_id DESC
                LIMIT ?
            """, (username, limit))
        else:
            cursor.execute("""
                SELECT report_id, username, report_time, provider_name, model_name, record_count
                FROM ai_report_history
                ORDER BY report_time DESC, report_id DESC
                LIMIT ?
            """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_ai_report(self, report_id, username=None, user_id=None):
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None:
            cursor.execute("""
                SELECT * FROM ai_report_history
                WHERE report_id=? AND user_id=?
            """, (report_id, user_id))
        elif username:
            cursor.execute("""
                SELECT * FROM ai_report_history
                WHERE report_id=? AND username=?
            """, (report_id, username))
        else:
            cursor.execute("SELECT * FROM ai_report_history WHERE report_id=?", (report_id,))
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None

        data = dict(row)
        for key in ["source_summary", "report_json"]:
            raw = data.get(key, "")
            if raw:
                try:
                    data[key] = json.loads(raw)
                except Exception:
                    data[key] = {}
            else:
                data[key] = {}
        return data

    def create_chat_session(self, report_id, username=None, session_title=None, user_id=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = (session_title or "").strip() or f"新聊天 {datetime.now().strftime('%m-%d %H:%M')}"
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is None:
            conn.close()
            raise ValueError("create_chat_session: 缺少有效的 user_id，调用方必须传入已登录用户 ID")
        cursor.execute("""
            INSERT INTO ai_chat_sessions
            (report_id, user_id, username, session_title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(report_id),
            user_id,
            username,
            title,
            now,
            now,
        ))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id

    def get_chat_sessions(self, report_id, username=None, limit=30, user_id=None):
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None:
            cursor.execute("""
                SELECT s.*,
                       (SELECT COUNT(*) FROM ai_report_chat_history h WHERE h.session_id = s.session_id) AS message_count
                FROM ai_chat_sessions s
                WHERE s.report_id=? AND s.user_id=?
                ORDER BY s.updated_at DESC, s.session_id DESC
                LIMIT ?
            """, (int(report_id), user_id, int(limit)))
        elif username:
            cursor.execute("""
                SELECT s.*,
                       (SELECT COUNT(*) FROM ai_report_chat_history h WHERE h.session_id = s.session_id) AS message_count
                FROM ai_chat_sessions s
                WHERE s.report_id=? AND s.username=?
                ORDER BY s.updated_at DESC, s.session_id DESC
                LIMIT ?
            """, (int(report_id), username, int(limit)))
        else:
            cursor.execute("""
                SELECT s.*,
                       (SELECT COUNT(*) FROM ai_report_chat_history h WHERE h.session_id = s.session_id) AS message_count
                FROM ai_chat_sessions s
                WHERE s.report_id=?
                ORDER BY s.updated_at DESC, s.session_id DESC
                LIMIT ?
            """, (int(report_id), int(limit)))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def _resolve_chat_session_id(self, cursor, report_id, username=None, session_id=None, user_id=None):
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is None:
            # 由调用方 save_report_chat_message 在更早的位置已 guard；此处是防御性 assertion
            raise ValueError("_resolve_chat_session_id: 缺少有效的 user_id")
        if session_id:
            cursor.execute("""
                SELECT session_id
                FROM ai_chat_sessions
                WHERE session_id=? AND report_id=? AND user_id=?
            """, (int(session_id), int(report_id), user_id))
            row = cursor.fetchone()
            if row:
                return int(row[0])

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = f"新聊天 {datetime.now().strftime('%m-%d %H:%M')}"
        cursor.execute("""
            INSERT INTO ai_chat_sessions
            (report_id, user_id, username, session_title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(report_id),
            user_id,
            username,
            title,
            now,
            now,
        ))
        return int(cursor.lastrowid)

    def save_report_chat_message(self, report_id, username=None, role="", content="", session_id=None, user_id=None):
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is None:
            conn.close()
            raise ValueError("save_report_chat_message: 缺少有效的 user_id，调用方必须传入已登录用户 ID")
        final_session_id = self._resolve_chat_session_id(
            cursor, report_id, username=username, session_id=session_id, user_id=user_id
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_content = content.strip()
        cursor.execute("""
            INSERT INTO ai_report_chat_history
            (session_id, report_id, user_id, username, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            final_session_id,
            int(report_id),
            user_id,
            username,
            role.strip(),
            clean_content,
            now,
        ))
        chat_id = cursor.lastrowid
        cursor.execute("""
            UPDATE ai_chat_sessions
            SET updated_at=?
            WHERE session_id=?
        """, (now, final_session_id))

        if role.strip() == "user" and clean_content:
            preview = clean_content.replace("\n", " ").strip()
            if len(preview) > 20:
                preview = f"{preview[:20]}..."
            cursor.execute("SELECT session_title FROM ai_chat_sessions WHERE session_id=?", (final_session_id,))
            row = cursor.fetchone()
            current_title = row[0] if row else ""
            if not current_title or current_title.startswith("新聊天") or current_title == "历史聊天":
                cursor.execute("""
                    UPDATE ai_chat_sessions
                    SET session_title=?
                    WHERE session_id=?
                """, (preview or current_title or "新聊天", final_session_id))
        conn.commit()
        conn.close()
        return chat_id

    def get_report_chat_messages(self, report_id, username=None, limit=50, session_id=None, user_id=None):
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        params = [int(report_id)]
        filters = ["report_id=?"]
        if user_id is not None:
            filters.append("user_id=?")
            params.append(user_id)
        elif username:
            filters.append("username=?")
            params.append(username)
        if session_id:
            filters.append("session_id=?")
            params.append(int(session_id))
        params.append(int(limit))
        where_clause = " AND ".join(filters)
        cursor.execute("""
            SELECT chat_id, session_id, report_id, user_id, username, role, content, created_at
            FROM ai_report_chat_history
            WHERE """ + where_clause + """
            ORDER BY chat_id DESC
            LIMIT ?
        """, tuple(params))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return list(reversed(rows))

    def clear_report_chat_messages(self, report_id, username=None, session_id=None, user_id=None):
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        filters = ["report_id=?"]
        params = [int(report_id)]
        if user_id is not None:
            filters.append("user_id=?")
            params.append(user_id)
        elif username:
            filters.append("username=?")
            params.append(username)
        if session_id:
            filters.append("session_id=?")
            params.append(int(session_id))
        cursor.execute(
            "DELETE FROM ai_report_chat_history WHERE " + " AND ".join(filters),
            tuple(params),
        )
        conn.commit()
        conn.close()

    def delete_chat_session(self, session_id, username=None, user_id=None):
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None:
            cursor.execute("""
                SELECT report_id FROM ai_chat_sessions
                WHERE session_id=? AND user_id=?
            """, (int(session_id), user_id))
        elif username:
            cursor.execute("""
                SELECT report_id FROM ai_chat_sessions
                WHERE session_id=? AND username=?
            """, (int(session_id), username))
        else:
            cursor.execute("SELECT report_id FROM ai_chat_sessions WHERE session_id=?", (int(session_id),))
        row = cursor.fetchone()
        if row is None:
            conn.close()
            return False

        cursor.execute("DELETE FROM ai_report_chat_history WHERE session_id=?", (int(session_id),))
        if user_id is not None:
            cursor.execute("""
                DELETE FROM ai_chat_sessions
                WHERE session_id=? AND user_id=?
            """, (int(session_id), user_id))
        elif username:
            cursor.execute("""
                DELETE FROM ai_chat_sessions
                WHERE session_id=? AND username=?
            """, (int(session_id), username))
        else:
            cursor.execute("DELETE FROM ai_chat_sessions WHERE session_id=?", (int(session_id),))
        conn.commit()
        conn.close()
        return True

    def delete_ai_report(self, report_id, username=None, user_id=None):
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        filters = ["report_id=?"]
        params = [int(report_id)]
        if user_id is not None:
            filters.append("user_id=?")
            params.append(user_id)
        elif username:
            filters.append("username=?")
            params.append(username)

        where_clause = " AND ".join(filters)
        cursor.execute("DELETE FROM ai_report_chat_history WHERE " + where_clause, tuple(params))
        cursor.execute("DELETE FROM ai_chat_sessions WHERE " + where_clause, tuple(params))
        cursor.execute("DELETE FROM ai_report_history WHERE " + where_clause, tuple(params))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0
