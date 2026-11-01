import sqlite3
conn = sqlite3.connect('dashboard/job_queue.db')
cursor = conn.cursor()
cursor.execute("SELECT error_message FROM jobs WHERE status='failed' ORDER BY created_at DESC LIMIT 1")
result = cursor.fetchone()
print("EXACT ERROR MESSAGE:", result[0] if result else "No failed jobs found")
