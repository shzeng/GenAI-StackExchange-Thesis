import sqlite3
import os
import time
import gc
import sys

# --- CONFIGURATION ---
OUTPUT_DIR = r"E:\\"

# Update this to point to your massive file location
SITE_PATHS = {
    #'Ask Ubuntu':     r'E:\Stack Exchange Data Dump\askubuntu.com\Posts.xml',
    #'Stack Overflow': r'E:\Stack Exchange Data Dump\stackoverflow.com\Posts.xml'
    #'Mathematics':    r'F:\StackExchange Data Dump\stackexchange_20250930\math.stackexchange.com\Posts.xml',
}

FILES_TO_PARSE = [
    'Posts.xml', 'Users.xml', 'Votes.xml', 'Comments.xml', 
    'PostHistory.xml', 'PostLinks.xml', 'Tags.xml', 'Badges.xml'
]

# --- OPTIMIZED IMPORTS ---
try:
    from lxml import etree as ET
    print("SUCCESS: Using lxml")
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    print("WARNING: lxml not found")
    HAS_LXML = False

# --- STANDARD SCHEMAS ---
STANDARD_SCHEMAS = {
    'Posts': ['Id', 'PostTypeId', 'AcceptedAnswerId', 'ParentId', 'CreationDate', 'DeletionDate', 'Score', 'ViewCount', 'Body', 'OwnerUserId', 'OwnerDisplayName', 'LastEditorUserId', 'LastEditorDisplayName', 'LastEditDate', 'LastActivityDate', 'Title', 'Tags', 'AnswerCount', 'CommentCount', 'FavoriteCount', 'ClosedDate', 'CommunityOwnedDate', 'ContentLicense'],
    'Users': ['Id', 'Reputation', 'CreationDate', 'DisplayName', 'LastAccessDate', 'WebsiteUrl', 'Location', 'AboutMe', 'Views', 'UpVotes', 'DownVotes', 'ProfileImageUrl', 'EmailHash', 'AccountId'],
    'Votes': ['Id', 'PostId', 'VoteTypeId', 'UserId', 'CreationDate', 'BountyAmount'],
    'Comments': ['Id', 'PostId', 'Score', 'Text', 'CreationDate', 'UserDisplayName', 'UserId', 'ContentLicense'],
    'Tags': ['Id', 'TagName', 'Count', 'ExcerptPostId', 'WikiPostId'],
    'PostHistory': ['Id', 'PostHistoryTypeId', 'PostId', 'RevisionGUID', 'CreationDate', 'UserId', 'UserDisplayName', 'Comment', 'Text', 'ContentLicense', 'JEventId'],
    'PostLinks': ['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId'],
    'Badges': ['Id', 'UserId', 'Name', 'Date', 'Class', 'TagBased']
}

def get_file_size(path):
    return os.path.getsize(path)

def process_xml_file(xml_path, table_name, db_conn):
    if not os.path.exists(xml_path):
        return

    file_size = get_file_size(xml_path)
    print(f"  > Parsing {table_name} ({file_size / (1024**3):.2f} GB)...")
    
    start_time = time.time()
    
    # 1. OPTIMIZATION: Disable Garbage Collection
    # Python's GC is slow when creating millions of small list objects. 
    # We turn it off and manually collect later.
    gc.disable()

    # 2. Get Columns
    if table_name in STANDARD_SCHEMAS:
        columns = STANDARD_SCHEMAS[table_name]
    else:
        # Fallback for unknown tables
        context = ET.iterparse(xml_path, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        for event, elem in context:
            if event == 'end' and elem.tag == 'row':
                columns = list(elem.attrib.keys())
                break
        del context, root
    
    # 3. Prepare DB
    db_conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")
    col_defs = [f"[{c}] TEXT" for c in columns]
    if 'Id' in columns:
        col_defs = [c.replace("[Id] TEXT", "[Id] INTEGER PRIMARY KEY") for c in col_defs]
        
    db_conn.execute(f"CREATE TABLE [{table_name}] ({', '.join(col_defs)});")
    
    placeholders = ','.join(['?'] * len(columns))
    insert_query = f"INSERT INTO [{table_name}] ({','.join(columns)}) VALUES ({placeholders})"
    
    # 4. Stream and Insert
    context = ET.iterparse(xml_path, events=('end',))
    context = iter(context)
    
    # For lxml compatibility, we don't grab root immediately in the same way 
    # because lxml handles root clearing differently.
    
    cursor = db_conn.cursor()
    batch_data = []
    
    # TUNING: 
    # For 90GB file, 50k is good balance between RAM usage and Transaction speed.
    BATCH_SIZE = 50000 
    row_count = 0
    
    # Optimization: Bind attribute lookup to local variable to avoid dict overhead
    
    for event, elem in context:
        if elem.tag == 'row':
            # Fast list comprehension
            # .get() is safer than direct access in case of missing optional cols
            vals = [elem.attrib.get(c) for c in columns]
            batch_data.append(vals)
            row_count += 1
            
            # Flush Batch
            if len(batch_data) >= BATCH_SIZE:
                cursor.executemany(insert_query, batch_data)
                db_conn.commit()
                batch_data.clear() # Re-use list memory
                
                # Cleanup XML memory
                elem.clear()
                # Clear parents (crucial for lxml/standard lib to keep memory flat)
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                
                # Manual GC and Progress Bar every 200k rows
                if row_count % 200000 == 0:
                    # Manually run GC to prevent RAM spikes
                    gc.collect() 
                    
                    # Calculate progress
                    # Note: This is an estimation. ET does not give current byte pos easily.
                    elapsed = time.time() - start_time
                    rate = row_count / elapsed
                    print(f"    Rows: {row_count:,} | Rate: {int(rate)} rows/s", end='\r')
            else:
                # Still clear memory for non-batch rows
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
    
    # Insert remaining
    if batch_data:
        cursor.executemany(insert_query, batch_data)
        db_conn.commit()
    
    del context
    gc.enable() # Re-enable GC
    gc.collect()
    
    print(f"\n    Done. {row_count:,} rows in {int(time.time() - start_time)}s.")

def create_indices(db_conn, table_name, columns):
    print(f"    Indexing {table_name} (this may take a while)...")
    cursor = db_conn.cursor()
    
    # Note: Id is typically handled by PRIMARY KEY definition
    if 'PostId' in columns:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_PostId ON [{table_name}](PostId)")
    if 'UserId' in columns:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_UserId ON [{table_name}](UserId)")
    if 'ParentId' in columns:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ParentId ON [{table_name}](ParentId)")
        
    db_conn.commit()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for site_name, posts_path in SITE_PATHS.items():
        print(f"\n=== Processing Site: {site_name} ===")
        source_folder = os.path.dirname(posts_path)
        db_path = os.path.join(OUTPUT_DIR, f"{site_name}.db")
        
        conn = sqlite3.connect(db_path)
        
        # --- EXTREME PERFORMANCE PRAGMAS ---
        # RISK: If PC crashes, DB corrupts. Benefit: 2x speed.
        conn.execute("PRAGMA synchronous = OFF") 
        conn.execute("PRAGMA journal_mode = MEMORY") 
        conn.execute("PRAGMA cache_size = -2000000") # Use ~2GB RAM for Cache
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        for filename in FILES_TO_PARSE:
            xml_full_path = os.path.join(source_folder, filename)
            table_name = filename.replace('.xml', '') 
            
            process_xml_file(xml_full_path, table_name, conn)
            
            # Indexing (Only check schema for indices)
            if table_name in STANDARD_SCHEMAS:
                create_indices(conn, table_name, STANDARD_SCHEMAS[table_name])
            
        print(f"=== Completed {site_name} ===")
        conn.close()

if __name__ == "__main__":
    main()

# Previous
##
#=== Processing Site: Mathematics ===
#  > Parsing Posts.xml...
#    Inserted 3,919,717 rows in 296s. --> 174
#    Indexing Posts...
#  > Parsing Users.xml...
#    Inserted 1,534,179 rows in 26s. --> 28
#    Indexing Users...
#  > Parsing Votes.xml...
#    Inserted 13,276,643 rows in 121s. --> 164
#    Indexing Votes...
#  > Parsing Comments.xml...
#    Inserted 7,400,437 rows in 135s.* --> 195