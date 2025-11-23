import sqlite3
import os
import sys

# get the data from csv and create business, users, reviews table in sqlite3
# --- Configuration ---
# Set paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#  UPDATE THESE PATHS
FULL_DB_PATH = os.path.join(SCRIPT_DIR, "data/yelp_all_fromcsv.db")
TARGET_USERS_FILE = os.path.join(SCRIPT_DIR, "data/users.txt")
SMALL_DB_PATH = os.path.join(SCRIPT_DIR, "data/filtered_yelp_subset.db")
TARGET_USERS_COUNT = 2466 # Maximum number of users to process


def load_target_users(file_path: str) -> list:
    """Reads user_ids from the file, ensures uniqueness, and limits the list."""
    if not os.path.exists(file_path):
        print(f"Error: Target user ID file not found at '{file_path}'")
        sys.exit(1)
        
    # Read, strip whitespace, and filter out duplicates and header row if present
    with open(file_path, 'r') as f:
        # Use set comprehension to efficiently handle duplicates
        ids = {line.strip() for line in f if line.strip() and line.strip() != 'user_id'}
        
    # Limit to the specified count
    return list(ids)[:TARGET_USERS_COUNT]


def run_sqlite_native_filter(source_db_path: str, target_db_path: str, user_ids: list):
    """
    Filters data to include only target users and all associated reviews/businesses,
    and inserts it into the target database (filtered_db).
    """
    if not user_ids:
        print("Error: No target user IDs provided. Aborting.")
        return

    print(f"1. Preparing to filter data for {len(user_ids)} users.")
    
    # Clean up any existing target file
    if os.path.exists(target_db_path):
        os.remove(target_db_path)
    
    source_conn = None
    try:
        source_conn = sqlite3.connect(source_db_path)
        source_cursor = source_conn.cursor()
        
        # 2. Attach the new, empty database file
        source_cursor.execute("ATTACH DATABASE ? AS filtered_db", (target_db_path,))
        
        # Create schema in the new database (important for clean inserts)
        print("2. Creating schema in the new filtered database...")
        source_conn.execute('''
        CREATE TABLE IF NOT EXISTS filtered_db.business (
            business_id TEXT PRIMARY KEY,
            name TEXT,
            state TEXT,
            city TEXT,
            postal_code TEXT,
            lat REAL,
            lon REAL,
            stars REAL,
            categories TEXT
        )
        ''')
        source_conn.execute('''
        CREATE TABLE IF NOT EXISTS filtered_db.reviews (
            review_id TEXT PRIMARY KEY,
            user_id TEXT,
            business_id TEXT,
            stars INTEGER,
            useful INTEGER,
            funny INTEGER,
            cool INTEGER,
            text TEXT,
            date TEXT
        )
        ''')
        source_conn.execute('''
        CREATE TABLE IF NOT EXISTS filtered_db.users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            review_count INTEGER,
            yelping_since TEXT,
            useful INTEGER,
            funny INTEGER,
            cool INTEGER,
            elite TEXT,
            friends TEXT,
            fans INTEGER,
            average_stars REAL,
            compliment_hot INTEGER,
            compliment_more INTEGER,
            compliment_profile INTEGER,
            compliment_cute INTEGER,
            compliment_list INTEGER,
            compliment_note INTEGER,
            compliment_plain INTEGER,
            compliment_cool INTEGER,
            compliment_funny INTEGER,
            compliment_writer INTEGER,
            compliment_photos INTEGER
        )
        ''')
        #source_conn.commit()


        # 3. Insert Target Users
        print("3. Filtering and inserting target users...")
        placeholders = ','.join(['?'] * len(user_ids))
        
        source_conn.execute(f'''
            INSERT INTO filtered_db.users
            SELECT * FROM main.users WHERE user_id IN ({placeholders})
        ''', user_ids)
        
        # 4. Filter Reviews (Reviews written by the target users)
        print("4. Filtering and inserting reviews written by target users...")
        source_conn.execute('''
            INSERT INTO filtered_db.reviews
            SELECT r.*
            FROM main.reviews r
            JOIN filtered_db.users u ON r.user_id = u.user_id
        ''')
        
        # 5. Filter Businesses (Businesses reviewed by the target users)
        #  LOGIC CHANGE: Filter businesses by the business_id column in the filtered reviews table.
        print("5. Filtering and inserting ALL businesses reviewed by target users...")
        source_conn.execute('''
            INSERT INTO filtered_db.business
            SELECT DISTINCT b.*
            FROM main.business b
            JOIN filtered_db.reviews r ON b.business_id = r.business_id
        ''')

        # 6. Commit changes and detach
        source_conn.commit()
        source_cursor.execute("DETACH DATABASE filtered_db")
        
        print(f"\n Success! Filtered database created at: {target_db_path}")

    except Exception as e:
        print(f"\n FATAL ERROR: {e}")
        if source_conn:
            # Clean up failed file attempt
            if os.path.exists(target_db_path):
                os.remove(target_db_path)
        raise
    finally:
        if source_conn:
            source_conn.close()

if __name__ == "__main__":
    # 1. Load the list of users
    target_users = load_target_users(TARGET_USERS_FILE)
    
    # 2. Run the native SQLite filter job
    run_sqlite_native_filter(FULL_DB_PATH, SMALL_DB_PATH, target_users)