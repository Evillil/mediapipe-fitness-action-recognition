# -*- coding: utf-8 -*-
"""
数据库管理模块
SQLite 本地存储训练记录与动作规则

表结构：
- training_records: record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark
- action_rules: rule_id, action_type, rule_name, rule_value, prompt_text
"""
import sqlite3
import os
import hashlib
import json
from datetime import datetime
from config import DB_PATH


class Database:
    """SQLite 数据库管理"""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        # 每个新连接都必须显式启用外键约束（SQLite 默认 OFF）
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        """初始化数据库表，含旧表迁移"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 迁移期间临时关闭 FK 约束：老迁移逻辑中存在 DROP/RENAME 等中间态，
        # 如果此时 FK 生效会导致迁移失败。迁移完成后再开启并校验完整性。
        cursor.execute("PRAGMA foreign_keys = OFF")

        # 检查是否存在旧版 training_records 表（含 rep_count 字段）
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_records'")
        old_table_exists = cursor.fetchone() is not None

        if old_table_exists:
            # 检查是否是旧版表结构（通过字段名判断）
            cursor.execute("PRAGMA table_info(training_records)")
            columns = [col[1] for col in cursor.fetchall()]
            if "rep_count" in columns:
                # 旧表存在，执行迁移
                self._migrate_old_tables(conn, cursor)

        # 创建训练记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                train_time TEXT NOT NULL,
                action_type TEXT NOT NULL,
                repetitions INTEGER NOT NULL DEFAULT 0,
                avg_score REAL DEFAULT 0.0,
                duration_sec REAL DEFAULT 0.0,
                remark TEXT DEFAULT '',
                username TEXT DEFAULT 'user'
            )
        """)

        # 迁移：给旧记录补上 username 字段
        cursor.execute("PRAGMA table_info(training_records)")
        col_names = [col[1] for col in cursor.fetchall()]
        if "username" not in col_names:
            cursor.execute("ALTER TABLE training_records ADD COLUMN username TEXT DEFAULT 'user'")
            cursor.execute("UPDATE training_records SET username='user' WHERE username IS NULL OR username=''")

        # 修复 blob 类型的数值字段（历史遗留脏数据）
        import struct
        cursor.execute("SELECT record_id, avg_score, repetitions, duration_sec FROM training_records")
        for row in cursor.fetchall():
            rid, score, reps, dur = row
            updates = {}
            for col, val in [("avg_score", score), ("repetitions", reps), ("duration_sec", dur)]:
                if isinstance(val, bytes):
                    if len(val) == 4:
                        try:
                            updates[col] = round(struct.unpack('<f', val)[0], 1)
                        except Exception:
                            updates[col] = 0
                    elif len(val) == 8:
                        try:
                            updates[col] = round(struct.unpack('<d', val)[0], 1)
                        except Exception:
                            updates[col] = 0
                    else:
                        updates[col] = 0
            if updates:
                set_clause = ", ".join(f"{k}=?" for k in updates)
                cursor.execute(f"UPDATE training_records SET {set_clause} WHERE record_id=?",
                               list(updates.values()) + [rid])

        # 创建动作规则表
        cursor.execute("DROP TABLE IF EXISTS action_rules")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                rule_value REAL NOT NULL,
                prompt_text TEXT DEFAULT ''
            )
        """)

        # 填充动作规则数据
        self._populate_action_rules(cursor)

        # 创建动作指南表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_guides (
                guide_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT '',
                key_points TEXT DEFAULT '',
                common_mistakes TEXT DEFAULT '',
                target_muscles TEXT DEFAULT '',
                difficulty TEXT DEFAULT '初级',
                calories_per_rep REAL DEFAULT 0.0
            )
        """)

        # 仅在表为空时填充初始数据
        cursor.execute("SELECT COUNT(*) FROM action_guides")
        if cursor.fetchone()[0] == 0:
            self._populate_action_guides(cursor)

        # 创建用户账号表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT DEFAULT ''
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            self._populate_default_users(cursor)

        # 创建用户信息表
        # 外键约束：user_id 必须对应 users 表中真实用户；删除用户时级联删除其个人资料
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id INTEGER PRIMARY KEY,
                username TEXT DEFAULT '' UNIQUE,
                nickname TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                age INTEGER DEFAULT 0,
                height_cm REAL DEFAULT 0.0,
                weight_kg REAL DEFAULT 0.0,
                fitness_goal TEXT DEFAULT '',
                created_at TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)

        # 迁移：给旧 user_profile 表补上 username 字段
        cursor.execute("PRAGMA table_info(user_profile)")
        profile_cols = [col[1] for col in cursor.fetchall()]
        if "username" not in profile_cols:
            cursor.execute("ALTER TABLE user_profile ADD COLUMN username TEXT DEFAULT 'user'")
            cursor.execute("UPDATE user_profile SET username='user' WHERE username IS NULL OR username=''")

        # 创建训练计划表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_plans (
                plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_reps INTEGER NOT NULL DEFAULT 0,
                target_score REAL NOT NULL DEFAULT 60.0,
                plan_date TEXT NOT NULL,
                is_completed INTEGER DEFAULT 0,
                actual_reps INTEGER DEFAULT 0,
                actual_score REAL DEFAULT 0.0
            )
        """)

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
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        self._migrate_to_user_id_relations(cursor)

        # 迁移完成后做一次 FK 完整性校验，确保所有业务行都能被约束接受
        cursor.execute("PRAGMA foreign_key_check")
        violations = cursor.fetchall()
        if violations:
            conn.close()
            raise RuntimeError(f"数据库外键完整性校验失败: {violations}")

        # 恢复 FK 开启状态（本连接马上关闭，后续 _get_conn 的连接默认就是 ON）
        cursor.execute("PRAGMA foreign_keys = ON")

        conn.commit()
        conn.close()

    def _clean_user_id(self, user_id):
        try:
            value = int(user_id)
            return value if value > 0 else None
        except Exception:
            return None

    def _ensure_column(self, cursor, table_name, column_name, ddl):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")

    def _migrate_user_profile_to_user_id(self, cursor):
        # 幂等守卫：如果 user_profile 已经声明了指向 users 的外键，说明迁移+FK升级已完成，直接跳过
        cursor.execute("PRAGMA foreign_key_list(user_profile)")
        existing_fks = cursor.fetchall()
        if any(fk[2] == 'users' for fk in existing_fks):
            return

        cursor.execute("PRAGMA table_info(user_profile)")
        columns = [row[1] for row in cursor.fetchall()]
        if "username" not in columns:
            cursor.execute("ALTER TABLE user_profile ADD COLUMN username TEXT DEFAULT 'user'")
            cursor.execute("UPDATE user_profile SET username='user' WHERE username IS NULL OR username=''")

        # 新表同时完成两件事：username→user_id 迁移 + FK 约束升级
        # ON DELETE CASCADE：删除 users 行时自动清理对应的 user_profile 行
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile_new (
                user_id INTEGER PRIMARY KEY,
                username TEXT DEFAULT '' UNIQUE,
                nickname TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                age INTEGER DEFAULT 0,
                height_cm REAL DEFAULT 0.0,
                weight_kg REAL DEFAULT 0.0,
                fitness_goal TEXT DEFAULT '',
                created_at TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("DELETE FROM user_profile_new")
        cursor.execute("""
            INSERT OR REPLACE INTO user_profile_new
            (user_id, username, nickname, gender, age, height_cm, weight_kg, fitness_goal, created_at)
            SELECT
                COALESCE(u.user_id, up.user_id),
                up.username,
                up.nickname,
                up.gender,
                up.age,
                up.height_cm,
                up.weight_kg,
                up.fitness_goal,
                up.created_at
            FROM user_profile up
            LEFT JOIN users u ON u.username = up.username
            WHERE COALESCE(u.user_id, up.user_id) IS NOT NULL
              AND COALESCE(u.user_id, up.user_id) IN (SELECT user_id FROM users)
        """)
        cursor.execute("DROP TABLE user_profile")
        cursor.execute("ALTER TABLE user_profile_new RENAME TO user_profile")
        print("[数据库升级] user_profile 已添加 FK 约束 (user_id → users ON DELETE CASCADE)")

    def _enforce_user_profile_user_id_notnull(self, cursor):
        """
        补丁：Phase B 遗漏了 user_profile.user_id 的显式 NOT NULL 声明。
        原用 `user_id INTEGER PRIMARY KEY`（ROWID 别名，语义上非空但 table_info 报告 notnull=0），
        改为 `user_id INTEGER NOT NULL PRIMARY KEY` 与其他 5 张子表保持一致。
        幂等：通过 PRAGMA table_info 判断 user_id.notnull 字段。
        """
        cursor.execute("PRAGMA table_info(user_profile)")
        for col in cursor.fetchall():
            if col[1] == 'user_id':
                if col[3] == 1:
                    return  # 已经 NOT NULL
                break

        cursor.execute("""
            CREATE TABLE user_profile_fknew (
                user_id INTEGER NOT NULL PRIMARY KEY,
                username TEXT DEFAULT '' UNIQUE,
                nickname TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                age INTEGER DEFAULT 0,
                height_cm REAL DEFAULT 0.0,
                weight_kg REAL DEFAULT 0.0,
                fitness_goal TEXT DEFAULT '',
                created_at TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            INSERT INTO user_profile_fknew
            (user_id, username, nickname, gender, age, height_cm, weight_kg, fitness_goal, created_at)
            SELECT user_id, username, nickname, gender, age, height_cm, weight_kg, fitness_goal, created_at
            FROM user_profile
        """)
        cursor.execute("DROP TABLE user_profile")
        cursor.execute("ALTER TABLE user_profile_fknew RENAME TO user_profile")
        print("[数据库升级] user_profile.user_id 已显式声明 NOT NULL")

    def _backfill_user_id_by_username(self, cursor, table_name):
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

    def _migrate_to_user_id_relations(self, cursor):
        self._migrate_user_profile_to_user_id(cursor)
        self._enforce_user_profile_user_id_notnull(cursor)

        for table_name in [
            "training_records",
            "training_plans",
            "ai_report_history",
            "ai_chat_sessions",
            "ai_report_chat_history",
        ]:
            self._ensure_column(cursor, table_name, "user_id", "INTEGER DEFAULT 0")

        self._backfill_user_id_by_username(cursor, "training_records")
        self._backfill_user_id_by_username(cursor, "training_plans")
        self._backfill_user_id_by_username(cursor, "ai_report_history")
        self._backfill_user_id_by_username(cursor, "ai_chat_sessions")
        self._backfill_user_id_by_username(cursor, "ai_report_chat_history")

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

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_records_user_id ON training_records(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_plans_user_id ON training_plans(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_report_history_user_id ON ai_report_history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_chat_sessions_user_id ON ai_chat_sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_report_chat_history_user_id ON ai_report_chat_history(user_id)")

        # Batch 1: 将 training_records / training_plans 升级为带 FK 约束的版本
        # （ai_report_history 由 ai_store.py 自行升级）
        self._upgrade_training_records_fk(cursor)
        self._upgrade_training_plans_fk(cursor)

    def _upgrade_training_records_fk(self, cursor):
        """将 training_records 升级为 user_id NOT NULL + FK→users ON DELETE CASCADE"""
        # 幂等守卫
        cursor.execute("PRAGMA foreign_key_list(training_records)")
        if any(fk[2] == 'users' for fk in cursor.fetchall()):
            return

        # 防御：所有行必须已经有有效 user_id
        cursor.execute("SELECT COUNT(*) FROM training_records WHERE user_id IS NULL OR user_id = 0")
        bad = cursor.fetchone()[0]
        if bad > 0:
            raise RuntimeError(
                f"training_records 中有 {bad} 行 user_id 为 0 或 NULL，无法升级为 NOT NULL + FK。"
                "请先通过 backfill 或手动清理修复这些行。"
            )
        cursor.execute("""
            SELECT COUNT(*) FROM training_records
            WHERE user_id NOT IN (SELECT user_id FROM users)
        """)
        orphan = cursor.fetchone()[0]
        if orphan > 0:
            raise RuntimeError(
                f"training_records 中有 {orphan} 行 user_id 指向不存在的用户，无法升级为 FK。"
            )

        cursor.execute("""
            CREATE TABLE training_records_fknew (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                train_time TEXT NOT NULL,
                action_type TEXT NOT NULL,
                repetitions INTEGER NOT NULL DEFAULT 0,
                avg_score REAL DEFAULT 0.0,
                duration_sec REAL DEFAULT 0.0,
                remark TEXT DEFAULT '',
                username TEXT DEFAULT 'user',
                user_id INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            INSERT INTO training_records_fknew
            (record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark, username, user_id)
            SELECT record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark,
                   COALESCE(username, 'user'), user_id
            FROM training_records
        """)
        cursor.execute("DROP TABLE training_records")
        cursor.execute("ALTER TABLE training_records_fknew RENAME TO training_records")
        # 注：ALTER TABLE RENAME 已自动维护 sqlite_sequence，不需要手动 INSERT OR REPLACE
        # (sqlite_sequence 无 UNIQUE 约束，手动 INSERT OR REPLACE 会产生重复行)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_records_user_id ON training_records(user_id)")
        print("[数据库升级] training_records 已添加 FK 约束 + user_id NOT NULL")

    def _upgrade_training_plans_fk(self, cursor):
        """将 training_plans 升级为 user_id NOT NULL + FK→users ON DELETE CASCADE"""
        cursor.execute("PRAGMA foreign_key_list(training_plans)")
        if any(fk[2] == 'users' for fk in cursor.fetchall()):
            return

        cursor.execute("SELECT COUNT(*) FROM training_plans WHERE user_id IS NULL OR user_id = 0")
        bad = cursor.fetchone()[0]
        if bad > 0:
            raise RuntimeError(
                f"training_plans 中有 {bad} 行 user_id 为 0 或 NULL，无法升级为 NOT NULL + FK。"
            )
        cursor.execute("""
            SELECT COUNT(*) FROM training_plans
            WHERE user_id NOT IN (SELECT user_id FROM users)
        """)
        orphan = cursor.fetchone()[0]
        if orphan > 0:
            raise RuntimeError(
                f"training_plans 中有 {orphan} 行 user_id 指向不存在的用户，无法升级为 FK。"
            )

        cursor.execute("""
            CREATE TABLE training_plans_fknew (
                plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_reps INTEGER NOT NULL DEFAULT 0,
                target_score REAL NOT NULL DEFAULT 60.0,
                plan_date TEXT NOT NULL,
                is_completed INTEGER DEFAULT 0,
                actual_reps INTEGER DEFAULT 0,
                actual_score REAL DEFAULT 0.0,
                user_id INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            INSERT INTO training_plans_fknew
            (plan_id, username, action_type, target_reps, target_score, plan_date,
             is_completed, actual_reps, actual_score, user_id)
            SELECT plan_id, username, action_type, target_reps, target_score, plan_date,
                   is_completed, actual_reps, actual_score, user_id
            FROM training_plans
        """)
        cursor.execute("DROP TABLE training_plans")
        cursor.execute("ALTER TABLE training_plans_fknew RENAME TO training_plans")
        # 注：ALTER TABLE RENAME 已自动维护 sqlite_sequence
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_plans_user_id ON training_plans(user_id)")
        print("[数据库升级] training_plans 已添加 FK 约束 + user_id NOT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_report_chat_history_session_id ON ai_report_chat_history(session_id)")

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

    def _populate_default_users(self, cursor):
        """填充默认账号"""
        defaults = [
            ("admin", hashlib.sha256("admin123".encode()).hexdigest(), "admin"),
            ("user", hashlib.sha256("user123".encode()).hexdigest(), "user"),
        ]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for username, pwd_hash, role in defaults:
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, pwd_hash, role, now))

    def _migrate_old_tables(self, conn, cursor):
        """从旧表迁移数据到新表结构"""
        try:
            # 读取旧数据
            cursor.execute("""
                SELECT id, date, action_type, rep_count, avg_score, duration, notes
                FROM training_records
            """)
            old_records = cursor.fetchall()

            # 删除旧表
            cursor.execute("DROP TABLE IF EXISTS training_records")

            # 创建新表
            cursor.execute("""
                CREATE TABLE training_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    train_time TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    repetitions INTEGER NOT NULL DEFAULT 0,
                    avg_score REAL DEFAULT 0.0,
                    duration_sec REAL DEFAULT 0.0,
                    remark TEXT DEFAULT ''
                )
            """)

            # 迁移数据
            for record in old_records:
                cursor.execute("""
                    INSERT INTO training_records
                    (record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, record)

            conn.commit()
            print(f"[数据库迁移] 成功迁移 {len(old_records)} 条训练记录到新表结构")
        except Exception as e:
            print(f"[数据库迁移] 迁移失败: {e}")

    def _populate_action_rules(self, cursor):
        """填充动作规则数据（从 config.py 中的阈值导入）"""
        # 清空旧数据后重新插入
        cursor.execute("DELETE FROM action_rules")

        rules = [
            # 深蹲规则
            ("深蹲", "下蹲膝角阈值", 120, "膝关节角度需低于此值才计为有效下蹲"),
            ("深蹲", "站立膝角阈值", 140, "膝关节角度需高于此值才计为站起完成"),
            ("深蹲", "最低膝角警告", 110, "下蹲深度不足"),
            ("深蹲", "躯干前倾上限", 35, "背部前倾过大"),

            # 俯卧撑规则
            ("俯卧撑", "下压肘角阈值", 100, "肘关节角度需低于此值才计为有效下压"),
            ("俯卧撑", "撑起肘角阈值", 150, "肘关节角度需高于此值才计为撑起完成"),
            ("俯卧撑", "最低肘角警告", 95, "下压不充分"),
            ("俯卧撑", "身体弯曲上限", 15, "核心不稳，注意收腹"),

            # 卷腹规则
            ("卷腹", "卷起躯干角阈值", 110, "躯干角度需低于此值才计为有效卷起"),
            ("卷腹", "平躺躯干角阈值", 120, "躯干角度需高于此值才计为回到平躺"),
            ("卷腹", "卷腹幅度下限", 30, "卷腹幅度不足"),

            # 弓步蹲规则
            ("弓步蹲", "下蹲膝角阈值", 110, "前腿膝关节角度需低于此值才计为有效弓步"),
            ("弓步蹲", "站立膝角阈值", 140, "膝关节角度需高于此值才计为站起完成"),
            ("弓步蹲", "前腿膝角下限", 80, "前腿膝盖弯曲过深"),
            ("弓步蹲", "前腿膝角上限", 100, "前腿角度不合适，下蹲不够"),
            ("弓步蹲", "躯干偏斜上限", 15, "身体重心不稳定"),
        ]

        cursor.executemany("""
            INSERT INTO action_rules (action_type, rule_name, rule_value, prompt_text)
            VALUES (?, ?, ?, ?)
        """, rules)

    def _populate_action_guides(self, cursor):
        """填充动作指南数据"""
        guides = [
            (
                "深蹲", "深蹲是下肢训练的王牌动作，通过屈膝屈髋下蹲再站起，全面锻炼下肢肌群和核心力量。",
                "双脚与肩同宽，脚尖略微外展；膝盖方向始终与脚尖一致；下蹲至大腿与地面平行或略低；背部全程保持挺直；重心落在全脚掌，避免脚跟离地",
                "膝盖内扣，增加膝关节损伤风险；弯腰驼背，腰椎压力过大；下蹲深度不足，训练效果打折；重心前移，脚跟离地失去平衡",
                "股四头肌、臀大肌、腘绳肌、核心肌群", "初级", 0.5
            ),
            (
                "俯卧撑", "俯卧撑是经典的上肢推类训练动作，通过撑起和下压身体锻炼胸部、肩部和手臂力量。",
                "双手略宽于肩，指尖朝前；从头到脚保持一条直线；下压至胸部接近地面；手肘约45度角展开，不过度外翻；核心全程收紧",
                "塌腰或拱背，身体未成一条直线；手肘过度外展成90度，肩关节压力大；下压幅度不足，未充分刺激胸肌；头部过度前伸或后仰",
                "胸大肌、三角肌前束、肱三头肌、核心肌群", "初级", 0.4
            ),
            (
                "卷腹", "卷腹是针对腹部核心的基础训练动作，通过卷起上半身集中锻炼腹直肌，比仰卧起坐更安全高效。",
                "仰卧屈膝，双脚平放于地面；双手轻放耳侧或交叉于胸前；用腹肌力量卷起上背部离地；下背部始终紧贴地面；全程匀速控制，避免借力",
                "双手抱头用力拉扯颈部，导致颈椎受伤；借助惯性快速起身，腹肌未充分发力；卷腹幅度不足，仅头部离地；全程憋气，应配合呼吸",
                "腹直肌、腹横肌", "初级", 0.3
            ),
            (
                "弓步蹲", "弓步蹲是单腿主导的下肢训练动作，通过前后脚交替下蹲锻炼下肢力量、平衡能力和协调性。",
                "前脚向前跨出一大步，步幅约为肩宽的1.5倍；下蹲至前腿大腿与地面平行；后膝接近地面但不触地；躯干全程保持直立；前膝不超过脚尖过多",
                "步幅过小导致膝盖大幅超过脚尖；身体前倾过多，重心不稳；左右晃动，核心力量不足；后腿膝盖直接触地撞击",
                "股四头肌、臀大肌、腘绳肌、小腿肌群", "中级", 0.6
            ),
        ]
        cursor.executemany("""
            INSERT INTO action_guides
            (action_type, description, key_points, common_mistakes, target_muscles, difficulty, calories_per_rep)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, guides)

    def get_action_guides(self, action_type=None):
        """获取动作指南"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if action_type:
            cursor.execute("SELECT * FROM action_guides WHERE action_type = ?", (action_type,))
        else:
            cursor.execute("SELECT * FROM action_guides")
        guides = cursor.fetchall()
        conn.close()
        return guides

    def save_training_record(self, action_type, repetitions, avg_score, duration_sec, remark="", username=None, user_id=None):
        """保存训练记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        # 强制要求有效 user_id（外键约束下 0/None 会被拒绝，必须由调用方提供真实用户）
        if user_id is None:
            conn.close()
            raise ValueError("save_training_record: 缺少有效的 user_id，调用方必须传入已登录用户 ID")
        train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO training_records
            (train_time, action_type, repetitions, avg_score, duration_sec, remark, username, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            train_time, action_type, repetitions, round(avg_score, 1),
            round(duration_sec, 1), remark, username or "user", user_id
        ))
        conn.commit()
        conn.close()

    def get_training_records(self, limit=50, username=None, user_id=None):
        """获取训练记录，可按用户筛选"""
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None:
            cursor.execute("""
                SELECT record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark
                FROM training_records WHERE user_id=?
                ORDER BY train_time DESC LIMIT ?
            """, (user_id, limit))
        elif username:
            cursor.execute("""
                SELECT record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark
                FROM training_records WHERE username=?
                ORDER BY train_time DESC LIMIT ?
            """, (username, limit))
        else:
            cursor.execute("""
                SELECT record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark
                FROM training_records
                ORDER BY train_time DESC LIMIT ?
            """, (limit,))
        records = cursor.fetchall()
        conn.close()
        return records

    def get_records_by_action(self, action_type, limit=20):
        """按动作类型获取记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT record_id, train_time, action_type, repetitions, avg_score, duration_sec, remark
            FROM training_records
            WHERE action_type = ?
            ORDER BY train_time DESC
            LIMIT ?
        """, (action_type, limit))
        records = cursor.fetchall()
        conn.close()
        return records

    def get_statistics(self, username=None, user_id=None):
        """获取训练统计，可按用户筛选"""
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None:
            where = "WHERE user_id=?"
            params = (user_id,)
        elif username:
            where = "WHERE username=?"
            params = (username,)
        else:
            where = ""
            params = ()

        cursor.execute(f"SELECT COUNT(*) FROM training_records {where}", params)
        total_sessions = cursor.fetchone()[0]

        cursor.execute(f"SELECT COALESCE(SUM(repetitions), 0) FROM training_records {where}", params)
        total_reps = cursor.fetchone()[0]

        cursor.execute(f"SELECT COALESCE(SUM(duration_sec), 0) FROM training_records {where}", params)
        total_duration = cursor.fetchone()[0]

        cursor.execute(f"SELECT COALESCE(AVG(avg_score), 0) FROM training_records {where}", params)
        avg_score = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT action_type,
                   COUNT(*) as sessions,
                   SUM(repetitions) as total_reps,
                   AVG(avg_score) as avg_score
            FROM training_records {where}
            GROUP BY action_type
        """, params)
        per_action = cursor.fetchall()

        conn.close()

        return {
            "total_sessions": total_sessions,
            "total_reps": total_reps,
            "total_duration": round(total_duration, 1),
            "avg_score": round(avg_score, 1),
            "per_action": per_action
        }

    def get_action_rules(self, action_type=None):
        """获取动作规则"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if action_type:
            cursor.execute("""
                SELECT rule_id, action_type, rule_name, rule_value, prompt_text
                FROM action_rules
                WHERE action_type = ?
            """, (action_type,))
        else:
            cursor.execute("""
                SELECT rule_id, action_type, rule_name, rule_value, prompt_text
                FROM action_rules
            """)
        rules = cursor.fetchall()
        conn.close()
        return rules

    def get_user_profile(self, username=None, user_id=None):
        """获取指定用户的个人信息，返回字典"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None:
            cursor.execute("SELECT * FROM user_profile WHERE user_id=?", (user_id,))
        elif username:
            cursor.execute("SELECT * FROM user_profile WHERE username=?", (username,))
        else:
            conn.close()
            return None
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        return dict(row)

    def save_user_profile(self, nickname, gender, age, height_cm, weight_kg, fitness_goal, username=None, user_id=None):
        """保存/更新指定用户的个人信息"""
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is None and not username:
            conn.close()
            raise ValueError("保存用户资料时缺少 user_id 或 username")

        if user_id is not None:
            cursor.execute("SELECT COUNT(*) FROM user_profile WHERE user_id=?", (user_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM user_profile WHERE username=?", (username,))
        exists = cursor.fetchone()[0] > 0
        if exists:
            if user_id is not None:
                cursor.execute("""
                    UPDATE user_profile SET username=?, nickname=?, gender=?, age=?, height_cm=?,
                    weight_kg=?, fitness_goal=? WHERE user_id=?
                """, (username or "", nickname, gender, age, height_cm, weight_kg, fitness_goal, user_id))
            else:
                cursor.execute("""
                    UPDATE user_profile SET nickname=?, gender=?, age=?, height_cm=?,
                    weight_kg=?, fitness_goal=? WHERE username=?
                """, (nickname, gender, age, height_cm, weight_kg, fitness_goal, username))
        else:
            cursor.execute("""
                INSERT INTO user_profile
                (user_id, username, nickname, gender, age, height_cm, weight_kg, fitness_goal, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                username or "",
                nickname,
                gender,
                age,
                height_cm,
                weight_kg,
                fitness_goal,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))
        conn.commit()
        conn.close()

    def get_user_by_id(self, user_id):
        """按 ID 获取用户"""
        clean_user_id = self._clean_user_id(user_id)
        if clean_user_id is None:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, role, created_at FROM users WHERE user_id=?", (clean_user_id,))
        row = cursor.fetchone()
        conn.close()
        return row

    def get_user_by_username(self, username):
        """按用户名获取用户"""
        clean_username = (username or "").strip()
        if not clean_username:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, role, created_at FROM users WHERE username=?", (clean_username,))
        row = cursor.fetchone()
        conn.close()
        return row

    # ==================== 用户账号管理 ====================
    def verify_user(self, username, password):
        """验证用户登录，返回 (user_id, username, role) 或 None"""
        conn = self._get_conn()
        cursor = conn.cursor()
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute(
            "SELECT user_id, username, role FROM users WHERE username=? AND password_hash=?",
            (username, pwd_hash))
        user = cursor.fetchone()
        conn.close()
        return user

    def get_all_users(self):
        """获取所有用户"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, role, created_at FROM users")
        users = cursor.fetchall()
        conn.close()
        return users

    def add_user(self, username, password, role="user"):
        """添加用户，成功返回 True，用户名重复返回 False"""
        conn = self._get_conn()
        cursor = conn.cursor()
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, pwd_hash, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False

    def delete_user(self, user_id):
        """删除用户（不允许删除管理员）"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_id = ? AND role != 'admin'", (user_id,))
        conn.commit()
        conn.close()

    def reset_password(self, user_id, new_password):
        """重置用户密码"""
        conn = self._get_conn()
        cursor = conn.cursor()
        pwd_hash = hashlib.sha256(new_password.encode()).hexdigest()
        cursor.execute("UPDATE users SET password_hash=? WHERE user_id=?", (pwd_hash, user_id))
        conn.commit()
        conn.close()

    def delete_record(self, record_id):
        """删除指定记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM training_records WHERE record_id = ?", (record_id,))
        conn.commit()
        conn.close()

    # ==================== 训练计划管理 ====================
    def add_plan(self, username=None, action_type=None, target_reps=0, target_score=60.0, plan_date=None, user_id=None):
        """添加训练计划"""
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is None:
            conn.close()
            raise ValueError("add_plan: 缺少有效的 user_id，调用方必须传入已登录用户 ID")
        cursor.execute("""
            INSERT INTO training_plans (username, user_id, action_type, target_reps, target_score, plan_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username or "user", user_id, action_type, target_reps, target_score, plan_date))
        conn.commit()
        conn.close()

    def get_plans(self, username=None, plan_date=None, user_id=None):
        """获取训练计划，可按日期筛选"""
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        if user_id is not None and plan_date:
            cursor.execute("""
                SELECT plan_id, username, action_type, target_reps, target_score,
                       plan_date, is_completed, actual_reps, actual_score
                FROM training_plans WHERE user_id=? AND plan_date=?
                ORDER BY plan_id
            """, (user_id, plan_date))
        elif plan_date and username:
            cursor.execute("""
                SELECT plan_id, username, action_type, target_reps, target_score,
                       plan_date, is_completed, actual_reps, actual_score
                FROM training_plans WHERE username=? AND plan_date=?
                ORDER BY plan_id
            """, (username, plan_date))
        elif user_id is not None:
            cursor.execute("""
                SELECT plan_id, username, action_type, target_reps, target_score,
                       plan_date, is_completed, actual_reps, actual_score
                FROM training_plans WHERE user_id=?
                ORDER BY plan_date DESC, plan_id
            """, (user_id,))
        elif username:
            cursor.execute("""
                SELECT plan_id, username, action_type, target_reps, target_score,
                       plan_date, is_completed, actual_reps, actual_score
                FROM training_plans WHERE username=?
                ORDER BY plan_date DESC, plan_id
            """, (username,))
        else:
            cursor.execute("""
                SELECT plan_id, username, action_type, target_reps, target_score,
                       plan_date, is_completed, actual_reps, actual_score
                FROM training_plans
                ORDER BY plan_date DESC, plan_id
            """)
        plans = cursor.fetchall()
        conn.close()
        return plans

    def update_plan_progress(self, plan_id, actual_reps, actual_score, is_completed):
        """更新计划进度"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE training_plans SET actual_reps=?, actual_score=?, is_completed=?
            WHERE plan_id=?
        """, (actual_reps, actual_score, is_completed, plan_id))
        conn.commit()
        conn.close()

    def sync_plans_with_records(self, username=None, plan_date=None, user_id=None):
        """根据训练记录自动更新当天计划完成情况"""
        conn = self._get_conn()
        cursor = conn.cursor()
        user_id, username = self._resolve_user_identity(cursor, user_id=user_id, username=username)
        # 获取当天的计划
        if user_id is not None:
            cursor.execute("""
                SELECT plan_id, action_type, target_reps, target_score
                FROM training_plans WHERE user_id=? AND plan_date=?
            """, (user_id, plan_date))
        else:
            cursor.execute("""
                SELECT plan_id, action_type, target_reps, target_score
                FROM training_plans WHERE username=? AND plan_date=?
            """, (username, plan_date))
        plans = cursor.fetchall()

        for plan_id, action_type, target_reps, target_score in plans:
            # 查询当天该动作的训练记录汇总
            if user_id is not None:
                cursor.execute("""
                    SELECT COALESCE(SUM(repetitions), 0), COALESCE(AVG(avg_score), 0)
                    FROM training_records
                    WHERE user_id=? AND action_type=? AND train_time LIKE ?
                """, (user_id, action_type, f"{plan_date}%"))
            else:
                cursor.execute("""
                    SELECT COALESCE(SUM(repetitions), 0), COALESCE(AVG(avg_score), 0)
                    FROM training_records
                    WHERE username=? AND action_type=? AND train_time LIKE ?
                """, (username, action_type, f"{plan_date}%"))
            row = cursor.fetchone()
            actual_reps = row[0]
            actual_score = round(row[1], 1)
            is_completed = 1 if actual_reps >= target_reps and actual_score >= target_score else 0
            cursor.execute("""
                UPDATE training_plans SET actual_reps=?, actual_score=?, is_completed=?
                WHERE plan_id=?
            """, (actual_reps, actual_score, is_completed, plan_id))

        conn.commit()
        conn.close()

    def delete_plan(self, plan_id):
        """删除训练计划"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM training_plans WHERE plan_id=?", (plan_id,))
        conn.commit()
        conn.close()
