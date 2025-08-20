#!/usr/bin/env python3
"""
Git Safety Check & Auto-Push Script
Comprehensive security scanner and safe git workflow automation
"""

import os
import re
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class GitSafetyChecker:
    """
    Comprehensive git safety checker and workflow automation
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.issues_found = []
        self.warnings = []
        
        # Security patterns to detect
        self.secret_patterns = {
            'api_keys': [
                r'api[_-]?key[s]?\s*[=:]\s*["\'][^"\']{10,}["\']',
                r'secret[_-]?key[s]?\s*[=:]\s*["\'][^"\']{10,}["\']',
                r'access[_-]?token[s]?\s*[=:]\s*["\'][^"\']{10,}["\']',
                r'auth[_-]?token[s]?\s*[=:]\s*["\'][^"\']{10,}["\']',
            ],
            'passwords': [
                r'password[s]?\s*[=:]\s*["\'][^"\']{6,}["\']',
                r'passwd[s]?\s*[=:]\s*["\'][^"\']{6,}["\']',
                r'pwd[s]?\s*[=:]\s*["\'][^"\']{6,}["\']',
            ],
            'database_urls': [
                r'database[_-]?url[s]?\s*[=:]\s*["\'][^"\']*://[^"\']+["\']',
                r'db[_-]?url[s]?\s*[=:]\s*["\'][^"\']*://[^"\']+["\']',
            ],
            'cloud_keys': [
                r'aws[_-]?access[_-]?key[_-]?id[s]?\s*[=:]\s*["\'][^"\']{16,}["\']',
                r'aws[_-]?secret[_-]?access[_-]?key[s]?\s*[=:]\s*["\'][^"\']{32,}["\']',
                r'google[_-]?api[_-]?key[s]?\s*[=:]\s*["\'][^"\']{32,}["\']',
            ],
            'openai_keys': [
                r'sk-[a-zA-Z0-9]{32,}',  # OpenAI API key pattern
                r'openai[_-]?api[_-]?key[s]?\s*[=:]\s*["\']sk-[^"\']+["\']',
            ]
        }
        
        # File size limits (in MB)
        self.size_limits = {
            'warning': 10,   # Warn for files > 10MB
            'error': 50,     # Block files > 50MB
            'critical': 100  # GitHub limit
        }
        
        # Sensitive file patterns
        self.sensitive_files = [
            r'\.env$', r'\.env\.',
            r'kaggle\.json$',
            r'.*secret.*', r'.*credential.*', r'.*password.*',
            r'.*\.key$', r'.*\.pem$', r'.*\.p12$',
            r'config\.json$', r'settings\.json$',
        ]
        
        # File extensions to exclude completely
        self.dangerous_extensions = [
            '.db', '.sqlite', '.sqlite3',
            '.log', '.dump', '.sql',
            '.csv', '.xlsx', '.json'  # Large data files
        ]
    
    def run_command(self, command: str, check: bool = True) -> Tuple[str, int]:
        """Run shell command safely"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root,
                check=check
            )
            return result.stdout.strip(), result.returncode
        except subprocess.CalledProcessError as e:
            return e.stderr.strip() if e.stderr else str(e), e.returncode
    
    def check_git_status(self) -> bool:
        """Check if we're in a git repository"""
        output, code = self.run_command("git status", check=False)
        
        if code != 0:
            self.issues_found.append("❌ Not in a git repository or git not available")
            return False
        
        return True
    
    def scan_for_secrets(self) -> List[Dict]:
        """Scan files for potential secrets"""
        print("🔍 Scanning for secrets and API keys...")
        
        secrets_found = []
        
        # Files to scan
        file_patterns = ['*.py', '*.js', '*.json', '*.yaml', '*.yml', '*.txt', '*.md']
        
        for pattern in file_patterns:
            for file_path in self.project_root.rglob(pattern):
                # Skip .git directory and __pycache__
                if any(part.startswith('.git') or part == '__pycache__' for part in file_path.parts):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Scan for each secret type
                    for secret_type, patterns in self.secret_patterns.items():
                        for pattern in patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                
                                secrets_found.append({
                                    'file': str(file_path.relative_to(self.project_root)),
                                    'line': line_num,
                                    'type': secret_type,
                                    'pattern': pattern,
                                    'match': match.group()[:50] + "..." if len(match.group()) > 50 else match.group()
                                })
                
                except Exception as e:
                    self.warnings.append(f"⚠️  Could not scan {file_path}: {e}")
        
        return secrets_found
    
    def check_file_sizes(self) -> List[Dict]:
        """Check for large files"""
        print("📊 Checking file sizes...")
        
        large_files = []
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not any(part.startswith('.git') for part in file_path.parts):
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    if size_mb > self.size_limits['warning']:
                        severity = 'warning'
                        if size_mb > self.size_limits['critical']:
                            severity = 'critical'
                        elif size_mb > self.size_limits['error']:
                            severity = 'error'
                        
                        large_files.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'size_mb': size_mb,
                            'severity': severity
                        })
                
                except Exception as e:
                    self.warnings.append(f"⚠️  Could not check size of {file_path}: {e}")
        
        return large_files
    
    def check_sensitive_files(self) -> List[str]:
        """Check for sensitive filename patterns"""
        print("🕵️ Checking for sensitive files...")
        
        sensitive_found = []
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(self.project_root))
                
                # Check against sensitive patterns
                for pattern in self.sensitive_files:
                    if re.search(pattern, relative_path, re.IGNORECASE):
                        sensitive_found.append(relative_path)
                        break
                
                # Check dangerous extensions
                if file_path.suffix.lower() in self.dangerous_extensions:
                    if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                        sensitive_found.append(relative_path)
        
        return sensitive_found
    
    def check_gitignore(self) -> Dict:
        """Check .gitignore configuration"""
        print("📋 Checking .gitignore configuration...")
        
        gitignore_path = self.project_root / '.gitignore'
        
        if not gitignore_path.exists():
            return {
                'exists': False,
                'recommendations': ['Create .gitignore file']
            }
        
        try:
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
        except Exception as e:
            return {
                'exists': True,
                'error': f"Could not read .gitignore: {e}",
                'recommendations': []
            }
        
        # Essential patterns that should be in .gitignore
        essential_patterns = [
            '.env', '*.db', '*.sqlite', '*.log', 
            'data/', 'database/', '__pycache__/', 
            '*.csv', 'kaggle.json'
        ]
        
        missing_patterns = []
        for pattern in essential_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
        
        return {
            'exists': True,
            'missing_patterns': missing_patterns,
            'recommendations': [f"Add '{pattern}' to .gitignore" for pattern in missing_patterns]
        }
    
    def check_tracked_vs_ignored(self) -> List[str]:
        """Check for files that are tracked but should be ignored"""
        print("🔄 Checking tracked vs ignored files...")
        
        # Get tracked files
        tracked_output, _ = self.run_command("git ls-files")
        tracked_files = tracked_output.split('\n') if tracked_output else []
        
        problems = []
        
        for file in tracked_files:
            if file:  # Skip empty lines
                file_path = Path(file)
                
                # Check if tracked file should be ignored
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    # Large files shouldn't be tracked
                    if size_mb > self.size_limits['error']:
                        problems.append(f"{file} ({size_mb:.1f}MB) - too large for git")
                    
                    # Sensitive files shouldn't be tracked
                    for pattern in self.sensitive_files:
                        if re.search(pattern, file, re.IGNORECASE):
                            problems.append(f"{file} - sensitive file pattern")
                            break
                    
                    # Dangerous extensions
                    if file_path.suffix.lower() in self.dangerous_extensions and size_mb > 1:
                        problems.append(f"{file} - data file shouldn't be tracked")
        
        return problems
    
    def generate_safety_report(self) -> Dict:
        """Generate comprehensive safety report"""
        print("\n" + "=" * 60)
        print("🔒 GIT SAFETY ANALYSIS")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'secrets': [],
            'large_files': [],
            'sensitive_files': [],
            'gitignore_status': {},
            'tracking_issues': [],
            'recommendations': []
        }
        
        # Run all checks
        if not self.check_git_status():
            report['status'] = 'error'
            return report
        
        report['secrets'] = self.scan_for_secrets()
        report['large_files'] = self.check_file_sizes()
        report['sensitive_files'] = self.check_sensitive_files()
        report['gitignore_status'] = self.check_gitignore()
        report['tracking_issues'] = self.check_tracked_vs_ignored()
        
        # Determine overall status
        has_critical_issues = (
            len(report['secrets']) > 0 or
            any(f['severity'] == 'critical' for f in report['large_files']) or
            len(report['tracking_issues']) > 0
        )
        
        has_warnings = (
            len(report['sensitive_files']) > 0 or
            any(f['severity'] in ['warning', 'error'] for f in report['large_files']) or
            len(report['gitignore_status'].get('missing_patterns', [])) > 0
        )
        
        if has_critical_issues:
            report['status'] = 'critical'
        elif has_warnings:
            report['status'] = 'warning'
        else:
            report['status'] = 'safe'
        
        return report
    
    def display_report(self, report: Dict):
        """Display safety report"""
        
        # Status indicator
        status_icons = {
            'safe': '✅',
            'warning': '⚠️ ',
            'critical': '❌',
            'error': '💥'
        }
        
        icon = status_icons.get(report['status'], '❓')
        print(f"\n{icon} OVERALL STATUS: {report['status'].upper()}")
        
        # Secrets found
        if report['secrets']:
            print(f"\n🚨 SECRETS DETECTED ({len(report['secrets'])}):")
            for secret in report['secrets']:
                print(f"   ❌ {secret['file']}:{secret['line']} - {secret['type']}")
                print(f"      Match: {secret['match']}")
        else:
            print("\n✅ No secrets detected")
        
        # Large files
        critical_files = [f for f in report['large_files'] if f['severity'] == 'critical']
        large_files = [f for f in report['large_files'] if f['severity'] in ['warning', 'error']]
        
        if critical_files:
            print(f"\n❌ CRITICAL: Files too large for GitHub ({len(critical_files)}):")
            for file in critical_files:
                print(f"   - {file['file']} ({file['size_mb']:.1f}MB)")
        
        if large_files:
            print(f"\n⚠️  LARGE FILES ({len(large_files)}):")
            for file in large_files:
                print(f"   - {file['file']} ({file['size_mb']:.1f}MB)")
        
        if not critical_files and not large_files:
            print("\n✅ No problematic large files")
        
        # Sensitive files
        if report['sensitive_files']:
            print(f"\n🕵️ SENSITIVE FILES ({len(report['sensitive_files'])}):")
            for file in report['sensitive_files']:
                print(f"   - {file}")
        else:
            print("\n✅ No sensitive files in working directory")
        
        # Tracking issues
        if report['tracking_issues']:
            print(f"\n🔄 TRACKING ISSUES ({len(report['tracking_issues'])}):")
            for issue in report['tracking_issues']:
                print(f"   - {issue}")
        else:
            print("\n✅ No tracking issues detected")
        
        # .gitignore status
        gitignore = report['gitignore_status']
        if not gitignore.get('exists', False):
            print("\n❌ .gitignore file missing!")
        elif gitignore.get('missing_patterns'):
            print(f"\n⚠️  .gitignore missing patterns ({len(gitignore['missing_patterns'])}):")
            for pattern in gitignore['missing_patterns']:
                print(f"   - {pattern}")
        else:
            print("\n✅ .gitignore properly configured")
    
    def provide_recommendations(self, report: Dict):
        """Provide actionable recommendations"""
        print(f"\n" + "=" * 60)
        print("💡 RECOMMENDATIONS")
        print("=" * 60)
        
        if report['status'] == 'safe':
            print("✅ Repository is safe for git operations!")
            return
        
        # Critical fixes
        if report['secrets']:
            print("\n🚨 IMMEDIATE ACTIONS (CRITICAL):")
            print("1. Remove or encrypt all detected secrets")
            print("2. Move API keys to .env file")
            print("3. Add .env to .gitignore")
            print("4. Revoke and regenerate any exposed API keys")
        
        if any(f['severity'] == 'critical' for f in report['large_files']):
            print("\n💾 LARGE FILE FIXES:")
            print("1. Add large files to .gitignore:")
            for file in report['large_files']:
                if file['severity'] == 'critical':
                    print(f"   echo '{file['file']}' >> .gitignore")
            print("2. Remove from git tracking:")
            for file in report['large_files']:
                if file['severity'] == 'critical':
                    print(f"   git rm --cached '{file['file']}'")
        
        if report['tracking_issues']:
            print("\n🔄 TRACKING FIXES:")
            print("git rm --cached <file>  # for each problematic file")
        
        # .gitignore fixes
        gitignore = report['gitignore_status']
        if not gitignore.get('exists') or gitignore.get('missing_patterns'):
            print("\n📋 .gitignore FIXES:")
            if not gitignore.get('exists'):
                print("1. Create .gitignore file")
            if gitignore.get('missing_patterns'):
                print("2. Add missing patterns:")
                for pattern in gitignore['missing_patterns']:
                    print(f"   echo '{pattern}' >> .gitignore")
        
        print(f"\n🔧 AUTOMATED FIX:")
        print("python git_safety_check.py --fix  # Run with --fix flag")
    
    def auto_fix_issues(self, report: Dict) -> bool:
        """Automatically fix common issues"""
        print("\n🔧 ATTEMPTING AUTOMATIC FIXES...")
        
        fixed_something = False
        
        # Fix .gitignore
        gitignore = report['gitignore_status']
        if not gitignore.get('exists') or gitignore.get('missing_patterns'):
            self.fix_gitignore(gitignore)
            fixed_something = True
        
        # Remove large files from tracking
        critical_files = [f for f in report['large_files'] if f['severity'] == 'critical']
        if critical_files:
            for file_info in critical_files:
                file_path = file_info['file']
                print(f"📤 Removing {file_path} from git tracking...")
                self.run_command(f"git rm --cached '{file_path}'", check=False)
                fixed_something = True
        
        # Fix tracking issues
        if report['tracking_issues']:
            print("🔄 Fixing tracking issues...")
            for issue in report['tracking_issues']:
                if ' - ' in issue:
                    file_path = issue.split(' - ')[0]
                    self.run_command(f"git rm --cached '{file_path}'", check=False)
                    fixed_something = True
        
        return fixed_something
    
    def fix_gitignore(self, gitignore_status: Dict):
        """Fix .gitignore file"""
        gitignore_path = self.project_root / '.gitignore'
        
        essential_content = """# Sensitive Files
.env
.env.*
kaggle.json
*secret*
*credential*

# Database Files
*.db
*.sqlite
*.sqlite3
database/

# Data Files
*.csv
*.xlsx
data/
*.json

# Logs
*.log
logs/

# Python
__pycache__/
*.py[cod]
.pytest_cache/

# IDEs
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
"""
        
        if not gitignore_status.get('exists'):
            print("📋 Creating .gitignore file...")
            with open(gitignore_path, 'w') as f:
                f.write(essential_content)
        else:
            # Append missing patterns
            missing = gitignore_status.get('missing_patterns', [])
            if missing:
                print(f"📋 Adding {len(missing)} missing patterns to .gitignore...")
                with open(gitignore_path, 'a') as f:
                    f.write('\n# Auto-added by safety checker\n')
                    for pattern in missing:
                        f.write(f'{pattern}\n')
    
    def safe_git_workflow(self):
        """Execute safe git workflow"""
        print("\n🚀 SAFE GIT WORKFLOW")
        print("=" * 30)
        
        # Check status first
        output, _ = self.run_command("git status --porcelain")
        
        if not output:
            print("✅ No changes to commit")
            return
        
        print("📝 Changes detected:")
        print(output)
        
        # Ask for confirmation
        response = input("\n📤 Add all changes and commit? (y/N): ").lower().strip()
        
        if response != 'y':
            print("❌ Operation cancelled")
            return
        
        # Get commit message
        commit_msg = input("💬 Enter commit message: ").strip()
        if not commit_msg:
            commit_msg = f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Execute git workflow
        print("\n🔄 Executing git workflow...")
        
        commands = [
            "git add .",
            f'git commit -m "{commit_msg}"',
            "git push origin main"
        ]
        
        for cmd in commands:
            print(f"🔄 {cmd}")
            output, code = self.run_command(cmd, check=False)
            
            if code != 0:
                print(f"❌ Command failed: {output}")
                return False
            else:
                print(f"✅ Success")
        
        print("\n🎉 Successfully pushed to GitHub!")
        return True


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Git Safety Checker & Workflow Automation")
    parser.add_argument('--fix', action='store_true', help='Automatically fix issues')
    parser.add_argument('--push', action='store_true', help='Execute safe git push workflow')
    parser.add_argument('--report-only', action='store_true', help='Generate report only')
    
    args = parser.parse_args()
    
    checker = GitSafetyChecker()
    
    # Generate safety report
    report = checker.generate_safety_report()
    checker.display_report(report)
    
    if args.report_only:
        return
    
    # Provide recommendations
    checker.provide_recommendations(report)
    
    # Auto-fix if requested
    if args.fix and report['status'] != 'safe':
        fixed = checker.auto_fix_issues(report)
        if fixed:
            print("\n🔄 Re-running safety check after fixes...")
            report = checker.generate_safety_report()
            checker.display_report(report)
    
    # Safe git workflow
    if args.push or (report['status'] == 'safe' and not args.fix):
        if report['status'] == 'safe':
            checker.safe_git_workflow()
        else:
            print(f"\n❌ Cannot push - security issues detected!")
            print(f"Run with --fix flag to auto-fix issues")


if __name__ == "__main__":
    main()