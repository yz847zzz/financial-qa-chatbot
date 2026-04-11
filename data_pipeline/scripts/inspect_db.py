import sqlite3

DB_PATH = "data/financials.db"

def print_table_schema(cursor, table_name):
    print(f"\n=== Schema: {table_name} ===")
    cursor.execute(f"PRAGMA table_info({table_name});")
    cols = cursor.fetchall()
    for col in cols:
        # col = (cid, name, type, notnull, default_value, pk)
        print(f"{col[1]:20s} {col[2]:10s} {'PK' if col[5] else ''}")

def print_table_sample(cursor, table_name, limit=5):
    print(f"\n=== Sample rows: {table_name} (top {limit}) ===")
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
    rows = cursor.fetchall()

    if not rows:
        print("  (empty)")
        return

    for i, row in enumerate(rows, 1):
        print(f"{i}. {row}")

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    print("Tables found:", tables)

    for table in tables:
        print_table_schema(cursor, table)
        print_table_sample(cursor, table, limit=5)

    conn.close()


if __name__ == "__main__":
    main()