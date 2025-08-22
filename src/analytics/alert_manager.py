"""
Alert Manager & Notification System - Day 3 Final Component
Comprehensive alert processing, notification, and escalation management
Completes the production monitoring infrastructure
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import time
import logging
import smtplib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alert_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: str  # 'system_threshold', 'model_performance', 'health_check'
    severity: str    # 'low', 'medium', 'high', 'critical'
    component: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    notification_sent: bool = False
    escalated: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if not self.alert_id:
            self.alert_id = f"{self.alert_type}_{self.component}_{int(self.timestamp.timestamp())}"

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_type: str  # 'email', 'console', 'log', 'webhook'
    enabled: bool
    config: Dict[str, Any]
    severity_filter: List[str]  # Which severities to notify for

@dataclass
class AlertRule:
    """Alert processing rule"""
    rule_id: str
    name: str
    condition: str
    severity: str
    cooldown_minutes: int
    escalation_minutes: int
    enabled: bool
    notification_channels: List[str]

class AlertManager:
    """
    Comprehensive alert management and notification system
    Processes alerts from monitoring system and manages notifications
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.alert_history = []
        self.active_alerts = {}  # component -> alert
        self.suppressed_alerts = {}  # For cooldown management
        
        # Alert configuration
        self.severity_levels = {
            'low': 1,
            'medium': 2, 
            'high': 3,
            'critical': 4
        }
        
        # Default alert rules
        self.alert_rules = {
            'cpu_high': AlertRule(
                rule_id='cpu_high',
                name='High CPU Usage',
                condition='cpu_usage > 85',
                severity='high',
                cooldown_minutes=15,
                escalation_minutes=60,
                enabled=True,
                notification_channels=['console', 'log']
            ),
            'memory_high': AlertRule(
                rule_id='memory_high',
                name='High Memory Usage',
                condition='memory_usage > 85',
                severity='high',
                cooldown_minutes=15,
                escalation_minutes=60,
                enabled=True,
                notification_channels=['console', 'log']
            ),
            'model_performance_low': AlertRule(
                rule_id='model_performance_low',
                name='Low Model Performance',
                condition='accuracy < 0.7',
                severity='medium',
                cooldown_minutes=30,
                escalation_minutes=120,
                enabled=True,
                notification_channels=['console', 'log', 'email']
            ),
            'health_check_failed': AlertRule(
                rule_id='health_check_failed',
                name='Health Check Failed',
                condition='status != healthy',
                severity='high',
                cooldown_minutes=10,
                escalation_minutes=30,
                enabled=True,
                notification_channels=['console', 'log']
            )
        }
        
        # Notification channels
        self.notification_channels = {
            'console': NotificationChannel(
                channel_type='console',
                enabled=True,
                config={},
                severity_filter=['medium', 'high', 'critical']
            ),
            'log': NotificationChannel(
                channel_type='log',
                enabled=True,
                config={'log_file': 'logs/alerts.log'},
                severity_filter=['low', 'medium', 'high', 'critical']
            ),
            'email': NotificationChannel(
                channel_type='email',
                enabled=False,  # Disabled by default (no SMTP config)
                config={
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipient': 'admin@company.com'
                },
                severity_filter=['high', 'critical']
            )
        }
        
        # Alert processing state
        self.processing_active = False
        self.processing_thread = None
        self.processing_interval = 30  # seconds
        
        # Statistics
        self.stats = {
            'total_alerts_processed': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'notifications_sent': 0,
            'alerts_resolved': 0
        }
        
        # Initialize database
        self._initialize_alert_database()
        
        logger.info("SUCCESS: AlertManager initialized successfully")
    
    def _initialize_alert_database(self):
        """Initialize alert database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL,
                    threshold REAL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT,
                    resolution_notes TEXT,
                    notification_sent BOOLEAN DEFAULT FALSE,
                    escalated BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create alert notifications table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    channel_type TEXT NOT NULL,
                    notification_status TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)
            
            # Create alert rules table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    condition_text TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    cooldown_minutes INTEGER,
                    escalation_minutes INTEGER,
                    enabled BOOLEAN DEFAULT TRUE,
                    notification_channels TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved)")
            
            conn.commit()
            conn.close()
            
            logger.info("SUCCESS: Alert database tables initialized")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize alert database: {e}")
    
    def process_monitoring_alerts(self) -> List[Alert]:
        """Process alerts from monitoring system data"""
        alerts = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent system metrics that might trigger alerts
            recent_time = (datetime.now() - timedelta(minutes=5)).isoformat()
            
            # Check system metrics for threshold violations
            system_query = """
                SELECT metric_name, metric_value, timestamp
                FROM system_metrics 
                WHERE timestamp > ? AND metric_name IN ('cpu_usage', 'memory_usage', 'disk_usage')
                ORDER BY timestamp DESC
            """
            
            system_df = pd.read_sql_query(system_query, conn, params=[recent_time])
            
            for _, row in system_df.iterrows():
                metric_name = row['metric_name']
                metric_value = row['metric_value']
                
                # Check against thresholds
                thresholds = {
                    'cpu_usage': 85.0,
                    'memory_usage': 85.0,
                    'disk_usage': 90.0
                }
                
                if metric_name in thresholds and metric_value > thresholds[metric_name]:
                    alert = Alert(
                        alert_id='',  # Will be auto-generated
                        alert_type='system_threshold',
                        severity='high' if metric_value > thresholds[metric_name] + 10 else 'medium',
                        component=metric_name,
                        message=f'{metric_name.replace("_", " ").title()} is {metric_value:.1f}% (threshold: {thresholds[metric_name]:.1f}%)',
                        metric_value=metric_value,
                        threshold=thresholds[metric_name],
                        timestamp=datetime.fromisoformat(row['timestamp'])
                    )
                    alerts.append(alert)
            
            # Check model performance metrics
            model_query = """
                SELECT model_name, metric_name, metric_value, timestamp
                FROM model_performance_metrics 
                WHERE timestamp > ? AND metric_name = 'accuracy'
                ORDER BY timestamp DESC
            """
            
            model_df = pd.read_sql_query(model_query, conn, params=[recent_time])
            
            for _, row in model_df.iterrows():
                if row['metric_value'] < 0.7:  # Performance threshold
                    alert = Alert(
                        alert_id='',
                        alert_type='model_performance',
                        severity='medium' if row['metric_value'] > 0.5 else 'high',
                        component=row['model_name'],
                        message=f'{row["model_name"]} accuracy dropped to {row["metric_value"]:.1%} (threshold: 70%)',
                        metric_value=row['metric_value'],
                        threshold=0.7,
                        timestamp=datetime.fromisoformat(row['timestamp'])
                    )
                    alerts.append(alert)
            
            # Check health check failures
            health_query = """
                SELECT component, status, response_time, error_message, timestamp
                FROM health_checks 
                WHERE timestamp > ? AND status IN ('warning', 'critical')
                ORDER BY timestamp DESC
            """
            
            health_df = pd.read_sql_query(health_query, conn, params=[recent_time])
            
            for _, row in health_df.iterrows():
                severity = 'critical' if row['status'] == 'critical' else 'medium'
                
                alert = Alert(
                    alert_id='',
                    alert_type='health_check',
                    severity=severity,
                    component=row['component'],
                    message=f'{row["component"]} health check {row["status"]}: {row.get("error_message", "Unknown issue")}',
                    metric_value=row.get('response_time'),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    metadata={'status': row['status'], 'error_message': row.get('error_message')}
                )
                alerts.append(alert)
            
            conn.close()
            
            # Apply deduplication and filtering
            filtered_alerts = self._deduplicate_and_filter_alerts(alerts)
            
            logger.info(f"SUCCESS: Processed {len(alerts)} potential alerts, {len(filtered_alerts)} after filtering")
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"ERROR: Failed to process monitoring alerts: {e}")
            return []
    
    def _deduplicate_and_filter_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Deduplicate and filter alerts based on rules and cooldowns"""
        filtered_alerts = []
        
        for alert in alerts:
            # Check if alert is in cooldown
            cooldown_key = f"{alert.alert_type}_{alert.component}"
            
            if cooldown_key in self.suppressed_alerts:
                last_alert_time = self.suppressed_alerts[cooldown_key]
                rule = self.alert_rules.get(alert.alert_type.replace('_threshold', '_high'), 
                                           self.alert_rules.get('cpu_high'))  # Default rule
                
                if datetime.now() - last_alert_time < timedelta(minutes=rule.cooldown_minutes):
                    continue  # Skip due to cooldown
            
            # Check if we already have an active alert for this component
            if alert.component in self.active_alerts:
                existing_alert = self.active_alerts[alert.component]
                
                # Only add if severity is higher or different type
                if (self.severity_levels[alert.severity] > self.severity_levels[existing_alert.severity] or
                    alert.alert_type != existing_alert.alert_type):
                    filtered_alerts.append(alert)
                    self.active_alerts[alert.component] = alert
            else:
                filtered_alerts.append(alert)
                self.active_alerts[alert.component] = alert
            
            # Update suppression tracking
            self.suppressed_alerts[cooldown_key] = datetime.now()
        
        return filtered_alerts
    
    def save_alerts_to_database(self, alerts: List[Alert]):
        """Save alerts to database"""
        if not alerts:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for alert in alerts:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, alert_type, severity, component, message, metric_value,
                     threshold, timestamp, resolved, notification_sent, escalated, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.alert_type,
                    alert.severity,
                    alert.component,
                    alert.message,
                    alert.metric_value,
                    alert.threshold,
                    alert.timestamp.isoformat(),
                    alert.resolved,
                    alert.notification_sent,
                    alert.escalated,
                    json.dumps(alert.metadata or {})
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"SUCCESS: Saved {len(alerts)} alerts to database")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save alerts to database: {e}")
    
    def send_notifications(self, alerts: List[Alert]):
        """Send notifications for alerts based on severity and channel configuration"""
        for alert in alerts:
            # Determine which channels should receive this alert
            channels_to_notify = []
            
            for channel_name, channel in self.notification_channels.items():
                if (channel.enabled and 
                    alert.severity in channel.severity_filter):
                    channels_to_notify.append(channel_name)
            
            # Send notifications
            for channel_name in channels_to_notify:
                try:
                    success = self._send_notification(alert, channel_name)
                    self._log_notification(alert, channel_name, success)
                    
                    if success:
                        alert.notification_sent = True
                        self.stats['notifications_sent'] += 1
                        
                except Exception as e:
                    logger.error(f"ERROR: Failed to send notification via {channel_name}: {e}")
                    self._log_notification(alert, channel_name, False, str(e))
    
    def _send_notification(self, alert: Alert, channel_name: str) -> bool:
        """Send notification via specific channel"""
        channel = self.notification_channels[channel_name]
        
        if channel.channel_type == 'console':
            return self._send_console_notification(alert)
        elif channel.channel_type == 'log':
            return self._send_log_notification(alert, channel.config)
        elif channel.channel_type == 'email':
            return self._send_email_notification(alert, channel.config)
        else:
            logger.warning(f"WARNING: Unknown notification channel type: {channel.channel_type}")
            return False
    
    def _send_console_notification(self, alert: Alert) -> bool:
        """Send console notification"""
        try:
            severity_icons = {
                'low': '🟢',
                'medium': '🟡', 
                'high': '🟠',
                'critical': '🔴'
            }
            
            icon = severity_icons.get(alert.severity, '⚪')
            
            print(f"\n{icon} ALERT [{alert.severity.upper()}] - {alert.component}")
            print(f"   Type: {alert.alert_type}")
            print(f"   Message: {alert.message}")
            print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if alert.metric_value is not None:
                print(f"   Value: {alert.metric_value:.2f}")
            if alert.threshold is not None:
                print(f"   Threshold: {alert.threshold:.2f}")
            print(f"   Alert ID: {alert.alert_id}")
            print("-" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Console notification failed: {e}")
            return False
    
    def _send_log_notification(self, alert: Alert, config: Dict) -> bool:
        """Send log notification"""
        try:
            log_message = (f"ALERT [{alert.severity.upper()}] {alert.component}: {alert.message} "
                          f"(ID: {alert.alert_id}, Time: {alert.timestamp.isoformat()})")
            
            # Log to both main logger and alert-specific logger
            if alert.severity in ['high', 'critical']:
                logger.error(log_message)
            elif alert.severity == 'medium':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # Also log to alerts file if specified
            if 'log_file' in config:
                alert_logger = logging.getLogger('alerts')
                if not alert_logger.handlers:
                    handler = logging.FileHandler(config['log_file'], encoding='utf-8')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    handler.setFormatter(formatter)
                    alert_logger.addHandler(handler)
                    alert_logger.setLevel(logging.INFO)
                
                alert_logger.info(log_message)
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Log notification failed: {e}")
            return False
    
    def _send_email_notification(self, alert: Alert, config: Dict) -> bool:
        """Send email notification (if configured)"""
        try:
            if not all(key in config for key in ['smtp_server', 'username', 'password', 'recipient']):
                logger.warning("WARNING: Email notification not configured properly")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = config['username']
            msg['To'] = config['recipient']
            msg['Subject'] = f"[{alert.severity.upper()}] Alert: {alert.component}"
            
            # Email body
            body = f"""
Alert Details:
- Component: {alert.component}
- Severity: {alert.severity.upper()}
- Type: {alert.alert_type}
- Message: {alert.message}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Alert ID: {alert.alert_id}

{f'Current Value: {alert.metric_value:.2f}' if alert.metric_value else ''}
{f'Threshold: {alert.threshold:.2f}' if alert.threshold else ''}

This alert was generated by the Product Review Intelligence System monitoring.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587))
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Email notification failed: {e}")
            return False
    
    def _log_notification(self, alert: Alert, channel: str, success: bool, error_message: str = None):
        """Log notification attempt to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute("""
                INSERT INTO alert_notifications 
                (alert_id, channel_type, notification_status, sent_at, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                channel,
                'success' if success else 'failed',
                datetime.now().isoformat(),
                error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"ERROR: Failed to log notification: {e}")
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = None) -> bool:
        """Mark an alert as resolved"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute("""
                UPDATE alerts 
                SET resolved = TRUE, resolved_at = ?, resolution_notes = ?
                WHERE alert_id = ?
            """, (datetime.now().isoformat(), resolution_notes, alert_id))
            
            conn.commit()
            conn.close()
            
            # Remove from active alerts
            for component, alert in list(self.active_alerts.items()):
                if alert.alert_id == alert_id:
                    del self.active_alerts[component]
                    break
            
            self.stats['alerts_resolved'] += 1
            logger.info(f"SUCCESS: Alert {alert_id} resolved")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to resolve alert {alert_id}: {e}")
            return False
    
    def check_escalations(self):
        """Check for alerts that need escalation"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Find unresolved alerts that haven't been escalated
            escalation_query = """
                SELECT alert_id, alert_type, severity, component, timestamp
                FROM alerts 
                WHERE resolved = FALSE AND escalated = FALSE
            """
            
            df = pd.read_sql_query(escalation_query, conn)
            
            for _, row in df.iterrows():
                alert_time = datetime.fromisoformat(row['timestamp'])
                
                # Check if alert is old enough for escalation
                rule = self.alert_rules.get(row['alert_type'].replace('_threshold', '_high'), 
                                           self.alert_rules.get('cpu_high'))
                
                if datetime.now() - alert_time > timedelta(minutes=rule.escalation_minutes):
                    # Escalate alert
                    conn.execute("""
                        UPDATE alerts SET escalated = TRUE WHERE alert_id = ?
                    """, (row['alert_id'],))
                    
                    logger.warning(f"ESCALATED: Alert {row['alert_id']} escalated after {rule.escalation_minutes} minutes")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"ERROR: Failed to check escalations: {e}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get alert counts by severity
            severity_query = """
                SELECT severity, COUNT(*) as count
                FROM alerts 
                GROUP BY severity
            """
            severity_df = pd.read_sql_query(severity_query, conn)
            severity_stats = dict(zip(severity_df['severity'], severity_df['count']))
            
            # Get alert counts by type
            type_query = """
                SELECT alert_type, COUNT(*) as count
                FROM alerts 
                GROUP BY alert_type
            """
            type_df = pd.read_sql_query(type_query, conn)
            type_stats = dict(zip(type_df['alert_type'], type_df['count']))
            
            # Get resolution statistics
            resolution_query = """
                SELECT 
                    COUNT(*) as total_alerts,
                    SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved_alerts,
                    COUNT(*) - SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as active_alerts
                FROM alerts
            """
            resolution_df = pd.read_sql_query(resolution_query, conn)
            resolution_stats = resolution_df.iloc[0].to_dict()
            
            # Get recent alert activity
            recent_query = """
                SELECT COUNT(*) as recent_alerts
                FROM alerts 
                WHERE timestamp > ?
            """
            recent_time = (datetime.now() - timedelta(hours=24)).isoformat()
            recent_df = pd.read_sql_query(recent_query, conn, params=[recent_time])
            recent_alerts = recent_df.iloc[0]['recent_alerts']
            
            conn.close()
            
            return {
                'severity_distribution': severity_stats,
                'type_distribution': type_stats,
                'resolution_stats': resolution_stats,
                'recent_24h_alerts': recent_alerts,
                'active_alerts_count': len(self.active_alerts),
                'current_stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get alert statistics: {e}")
            return {}
    
    def run_alert_processing_cycle(self):
        """Run one complete alert processing cycle"""
        logger.info("STARTING: Alert processing cycle")
        cycle_start = time.time()
        
        try:
            # Process new alerts from monitoring data
            new_alerts = self.process_monitoring_alerts()
            
            if new_alerts:
                # Save alerts to database
                self.save_alerts_to_database(new_alerts)
                
                # Send notifications
                self.send_notifications(new_alerts)
                
                # Update statistics
                self.stats['total_alerts_processed'] += len(new_alerts)
                for alert in new_alerts:
                    self.stats['alerts_by_severity'][alert.severity] += 1
                    self.stats['alerts_by_type'][alert.alert_type] += 1
                
                logger.info(f"PROCESSED: {len(new_alerts)} new alerts")
            
            # Check for escalations
            self.check_escalations()
            
            cycle_time = time.time() - cycle_start
            
            return {
                'success': True,
                'cycle_time': cycle_time,
                'new_alerts': len(new_alerts),
                'active_alerts': len(self.active_alerts)
            }
            
        except Exception as e:
            logger.error(f"ERROR: Alert processing cycle failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'cycle_time': time.time() - cycle_start
            }
    
    def start_alert_processing(self):
        """Start continuous alert processing"""
        if self.processing_active:
            logger.warning("WARNING: Alert processing already active")
            return
        
        logger.info("STARTING: Continuous alert processing")
        self.processing_active = True
        
        def processing_loop():
            while self.processing_active:
                try:
                    result = self.run_alert_processing_cycle()
                    
                    # Wait for next cycle
                    time.sleep(self.processing_interval)
                    
                except Exception as e:
                    logger.error(f"ERROR: Alert processing loop error: {e}")
                    time.sleep(10)  # Wait before retrying
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info(f"SUCCESS: Alert processing started (interval: {self.processing_interval}s)")
    
    def stop_alert_processing(self):
        """Stop continuous alert processing"""
        if not self.processing_active:
            logger.warning("WARNING: Alert processing not active")
            return
        
        logger.info("STOPPING: Continuous alert processing")
        self.processing_active = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        logger.info("SUCCESS: Alert processing stopped")

def main():
    """Main function for testing alert manager"""
    print("TESTING: Alert Manager & Notification System - Day 3 Final")
    print("=" * 60)
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    print("\nTESTING: Single alert processing cycle...")
    
    # Run single processing cycle
    result = alert_manager.run_alert_processing_cycle()
    
    if result['success']:
        print(f"✅ Alert processing cycle completed successfully")
        print(f"   Cycle time: {result['cycle_time']:.2f}s")
        print(f"   New alerts: {result['new_alerts']}")
        print(f"   Active alerts: {result['active_alerts']}")
    else:
        print(f"❌ Alert processing cycle failed: {result['error']}")
    
    # Get alert statistics
    print(f"\nTESTING: Alert statistics...")
    stats = alert_manager.get_alert_statistics()
    
    if stats:
        print(f"✅ Alert statistics retrieved:")
        print(f"   Total processed: {stats['current_stats']['total_alerts_processed']}")
        print(f"   Active alerts: {stats['active_alerts_count']}")
        print(f"   Recent (24h): {stats['recent_24h_alerts']}")
        print(f"   Notifications sent: {stats['current_stats']['notifications_sent']}")
        
        if stats['severity_distribution']:
            print(f"   Severity distribution: {dict(stats['severity_distribution'])}")
    else:
        print(f"❌ Failed to retrieve alert statistics")
    
    # Test short-term continuous processing
    print(f"\nTESTING: Short-term continuous alert processing (30 seconds)...")
    
    # Set short interval for testing
    alert_manager.processing_interval = 10  # 10 seconds for testing
    
    alert_manager.start_alert_processing()
    print("✅ Alert processing started")
    
    # Let it run for 30 seconds
    time.sleep(30)
    
    alert_manager.stop_alert_processing()
    print("✅ Alert processing stopped")
    
    # Final statistics
    final_stats = alert_manager.get_alert_statistics()
    print(f"\nFINAL STATISTICS:")
    print(f"Total alerts processed: {final_stats['current_stats']['total_alerts_processed']}")
    print(f"Notifications sent: {final_stats['current_stats']['notifications_sent']}")
    print(f"Alerts resolved: {final_stats['current_stats']['alerts_resolved']}")
    
    print(f"\n🎉 COMPLETE: Alert Manager test finished!")
    print(f"✅ DAY 3 COMPLETED: All 6 validation & analytics components implemented!")
    print(f"📊 Check database tables: alerts, alert_notifications, alert_rules")

if __name__ == "__main__":
    main()