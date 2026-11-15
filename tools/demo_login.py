"""Demo file for the Bot 1 -> Bot 2 handoff. Intentionally unsafe: do not use in real projects."""
import os
import sqlite3


def find_user(conn: sqlite3.Connection, user_id: str):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = " + user_id)
    return cursor.fetchone()


def run_report(name: str):
    os.system("echo generating report for " + name)


def calculate(expression: str):
    return eval(expression)


def load_settings(path: str):
    try:
        with open(path) as f:
            return f.read()
    except:
        return None
