#!/usr/bin/env python3
"""
Quick Vector Store Reset with Document Count Check
Usage: python quick_reset.py
"""

import os
import sys
import shutil
import logging
import sqlite3

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_chroma_document_count(db_path):
    """Check document count in Chroma database."""
    try:
        # Chroma stores data in SQLite files
        chroma_sqlite_path = os.path.join(db_path, "chroma.sqlite3")
        
        if not os.path.exists(chroma_sqlite_path):
            return 0
        
        # Connect to SQLite database
        conn = sqlite3.connect(chroma_sqlite_path)
        cursor = conn.cursor()
        
        # Query to count documents (embeddings table)
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
        
    except Exception as e:
        logger.warning(f"⚠️  Could not count documents in {db_path}: {e}")
        return "Unknown"

def get_folder_size(folder_path):
    """Get folder size in MB."""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except Exception as e:
        return "Unknown"

def quick_reset():
    """Quickly delete the Chroma database folder with document count check."""
    
    # Primary path to check
    primary_path = './data/chroma_db'
    
    # Additional paths to check if primary doesn't exist
    additional_paths = [
        './chroma_db',
        '../data/chroma_db',
        'data/chroma_db'
    ]
    
    all_paths = [primary_path] + additional_paths
    
    found_databases = []
    deleted = False
    
    logger.info("🔍 Checking for Chroma databases...")
    
    # Check all possible paths
    for db_path in all_paths:
        if os.path.exists(db_path):
            # Get document count and folder info
            doc_count = check_chroma_document_count(db_path)
            folder_size = get_folder_size(db_path)
            abs_path = os.path.abspath(db_path)
            
            found_databases.append({
                'path': db_path,
                'abs_path': abs_path,
                'doc_count': doc_count,
                'size': folder_size
            })
            
            logger.info(f"📊 Found database: {db_path}")
            logger.info(f"   📍 Full path: {abs_path}")
            logger.info(f"   📄 Documents: {doc_count}")
            logger.info(f"   💾 Size: {folder_size}")
            logger.info("")
    
    if not found_databases:
        logger.info("📭 No Chroma database folders found")
        logger.info("Searched in:")
        for path in all_paths:
            logger.info(f"   - {os.path.abspath(path)}")
        return
    
    # Show summary
    total_docs = 0
    for db in found_databases:
        if isinstance(db['doc_count'], int):
            total_docs += db['doc_count']
    
    logger.info(f"📋 SUMMARY:")
    logger.info(f"   🗂️  Databases found: {len(found_databases)}")
    logger.info(f"   📄 Total documents: {total_docs}")
    logger.info("")
    
    # Ask for confirmation
    try:
        confirm = input("⚠️  DELETE all databases above? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            logger.info("❌ Cancelled")
            return
    except KeyboardInterrupt:
        logger.info("\n❌ Cancelled")
        return
    
    # Delete databases
    logger.info("🗑️  Starting deletion...")
    
    for db in found_databases:
        try:
            logger.info(f"   Deleting: {db['path']} ({db['doc_count']} docs)")
            shutil.rmtree(db['path'])
            logger.info(f"   ✅ Deleted: {db['path']}")
            deleted = True
        except Exception as e:
            logger.error(f"   ❌ Failed to delete {db['path']}: {e}")
    
    if deleted:
        logger.info("\n✅ RESET COMPLETE!")
        logger.info("💡 Restart your application to rebuild the vector store")
    else:
        logger.info("\n❌ NO DATABASES WERE DELETED")

def check_only():
    """Just check document count without deleting."""
    
    primary_path = './data/chroma_db'
    
    if not os.path.exists(primary_path):
        logger.info(f"📭 Database not found: {primary_path}")
        return
    
    doc_count = check_chroma_document_count(primary_path)
    folder_size = get_folder_size(primary_path)
    abs_path = os.path.abspath(primary_path)
    
    logger.info(f"📊 CHROMA DATABASE STATUS:")
    logger.info(f"   📍 Path: {abs_path}")
    logger.info(f"   📄 Documents: {doc_count}")
    logger.info(f"   💾 Size: {folder_size}")
    logger.info(f"   ⚡ Status: {'Active' if doc_count > 0 else 'Empty'}")

if __name__ == "__main__":
    print("🚀 QUICK VECTOR STORE RESET")
    print("=" * 30)
    
    # Check if user wants to just check count
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--check', '-c', 'check']:
        check_only()
    else:
        quick_reset()