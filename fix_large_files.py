#!/usr/bin/env python3
"""
Fix Large Files in Git Repository
Removes large files from git history and prepares for safe push
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, check=True):
    """Run shell command and return result"""
    print(f"🔄 Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=check
        )
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False

def check_file_sizes():
    """Check for large files in repository"""
    print("\n📁 CHECKING FILE SIZES...")
    
    large_files = []
    for file_path in Path(".").rglob("*"):
        if file_path.is_file() and not any(part.startswith('.git') for part in file_path.parts):
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > 50:  # Files larger than 50MB
                    large_files.append((str(file_path), size_mb))
            except:
                pass
    
    if large_files:
        print("⚠️  LARGE FILES FOUND:")
        for file, size in sorted(large_files, key=lambda x: x[1], reverse=True):
            print(f"   - {file} ({size:.1f} MB)")
        return large_files
    else:
        print("✅ No large files found in working directory")
        return []

def check_git_history_size():
    """Check size of git objects"""
    print("\n📊 CHECKING GIT REPOSITORY SIZE...")
    
    # Check git objects
    if run_command("git count-objects -vH", check=False):
        pass
    
    # Check largest files in git history
    print("\n🔍 LARGEST FILES IN GIT HISTORY:")
    command = '''git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print substr($0,6)}' | sort --numeric-sort --key=2 | tail -10'''
    
    run_command(command, check=False)

def remove_large_files_from_history():
    """Remove large files from git history"""
    print("\n🧹 REMOVING LARGE FILES FROM GIT HISTORY...")
    
    # Files to remove
    files_to_remove = [
        "data/Reviews.csv",
        "data/*.csv",
        "*.db",
        "database/*.db",
        "*.sqlite",
        "*.sqlite3"
    ]
    
    success = True
    
    for file_pattern in files_to_remove:
        print(f"\n🗑️  Removing {file_pattern} from history...")
        
        command = f'''git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch {file_pattern}' \
--prune-empty --tag-name-filter cat -- --all'''
        
        if not run_command(command, check=False):
            print(f"⚠️  Could not remove {file_pattern} (may not exist)")
        else:
            print(f"✅ Removed {file_pattern} from history")
    
    # Clean up
    print("\n🧽 CLEANING UP GIT REFERENCES...")
    
    cleanup_commands = [
        "rm -rf .git/refs/original/",
        "git reflog expire --expire=now --all",
        "git gc --prune=now --aggressive"
    ]
    
    for cmd in cleanup_commands:
        run_command(cmd, check=False)
    
    return success

def verify_clean_repository():
    """Verify repository is clean for pushing"""
    print("\n✅ VERIFYING CLEAN REPOSITORY...")
    
    # Check git status
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    
    if result.stdout.strip():
        print("📝 Uncommitted changes found:")
        print(result.stdout)
    else:
        print("✅ Working directory clean")
    
    # Check what will be pushed
    print("\n📤 FILES TO BE PUSHED:")
    run_command("git ls-files", check=False)
    
    # Check repository size
    print("\n📊 FINAL REPOSITORY SIZE CHECK:")
    run_command("git count-objects -vH", check=False)

def main():
    """Main execution"""
    print("🔧 LARGE FILE REMOVAL TOOL")
    print("=" * 50)
    
    if not Path(".git").exists():
        print("❌ Not in a git repository!")
        sys.exit(1)
    
    # Step 1: Check current situation
    large_files = check_file_sizes()
    check_git_history_size()
    
    # Step 2: Confirm action
    if large_files:
        print(f"\n⚠️  Found {len(large_files)} large files in working directory")
        print("These should be in .gitignore to prevent future issues")
    
    print("\n🚨 This will remove large files from ENTIRE git history!")
    print("This rewrites history and requires force push.")
    
    confirm = input("\nProceed? (yes/no): ").lower().strip()
    
    if confirm != 'yes':
        print("❌ Operation cancelled")
        sys.exit(1)
    
    # Step 3: Remove files from history
    if remove_large_files_from_history():
        print("\n✅ Large files removed from history")
    else:
        print("\n⚠️  Some issues occurred during cleanup")
    
    # Step 4: Verify clean state
    verify_clean_repository()
    
    # Step 5: Instructions for push
    print("\n" + "=" * 50)
    print("🎯 NEXT STEPS:")
    print("1. Review the files listed above")
    print("2. Ensure .gitignore contains:")
    print("   - *.csv")
    print("   - *.db")
    print("   - data/")
    print("   - database/")
    print("3. Force push with: git push origin main --force")
    print("4. ⚠️  WARNING: This overwrites remote history!")
    print("=" * 50)

if __name__ == "__main__":
    main()
