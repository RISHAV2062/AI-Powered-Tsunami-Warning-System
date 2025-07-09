#!/usr/bin/env python3
"""
Seismic Data Processing Module
Advanced real-time seismic data processing system for tsunami detection
"""

import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
import websockets
import redis
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import requests
import sqlite3
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import seaborn as sns
from obspy import UTCDateTime, Stream, Trace
from obspy.signal import filter as obs_filter
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('seismic_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SeismicEvent:
    """Data class for seismic event information"""
    id: str
    timestamp: datetime
    magnitude: float
    depth: float
    latitude: float
    longitude: float
    location: str
    network: str
    station: str
    channel: str
    sampling_rate: float
    waveform_data: List[float]
    p_arrival: Optional[datetime] = None
    s_arrival: Optional[datetime] = None
    quality: str = "good"
    processed: bool = False
    tsunami_risk: str = "unknown"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessingConfig:
    """Configuration for seismic data processing"""
    sampling_rate: float = 100.0
    window_length: int = 3600  # 1 hour in seconds
    overlap: float = 0.5
    min_magnitude: float = 3.0
    max_magnitude: float = 9.5
    min_depth: float = 0.0
    max_depth: float = 700.0
    filter_freq_min: float = 0.1
    filter_freq_max: float = 50.0
    sta_length: float = 0.5  # seconds
    lta_length: float = 10.0  # seconds
    trigger_threshold: float = 3.0
    detrigger_threshold: float = 1.0
    max_processing_time: float = 30.0  # seconds
    batch_size: int = 100
    redis_host: str = "localhost"
    redis_port: int = 6379
    kafka_bootstrap_servers: str = "localhost:9092"
    websocket_url: str = "ws://localhost:8080/ws"
    database_path: str = "seismic_data.db"

class SeismicDataValidator:
    """Validator for seismic data quality and integrity"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def validate_event(self, event: SeismicEvent) -> Tuple[bool, List[str]]:
        """Validate seismic event data"""
        errors = []
        
        # Magnitude validation
        if not (self.config.min_magnitude <= event.magnitude <= self.config.max_magnitude):
            errors.append(f"Magnitude {event.magnitude} out of range")
        
        # Depth validation
        if not (self.config.min_depth <= event.depth <= self.config.max_depth):
            errors.append(f"Depth {event.depth} out of range")
        
        # Coordinate validation
        if not (-90 <= event.latitude <= 90):
            errors.append(f"Latitude {event.latitude} out of range")
        if not (-180 <= event.longitude <= 180):
            errors.append(f"Longitude {event.longitude} out of range")
        
        # Waveform validation
        if not event.waveform_data or len(event.waveform_data) == 0:
            errors.append("Empty waveform data")
        elif len(event.waveform_data) < 100:
            errors.append("Insufficient waveform data points")
        
        # Sampling rate validation
        if event.sampling_rate <= 0:
            errors.append("Invalid sampling rate")
        
        # Timestamp validation
        now = datetime.now()
        if event.timestamp > now:
            errors.append("Future timestamp")
        elif event.timestamp < now - timedelta(days=30):
            errors.append("Very old timestamp")
        
        return len(errors) == 0, errors
    
    def assess_data_quality(self, waveform: List[float]) -> Tuple[str, float]:
        """Assess waveform data quality"""
        try:
            data = np.array(waveform)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                return "poor", 0.0
            
            # Check for clipping
            max_val = np.max(np.abs(data))
            if max_val >= 0.99:  # Assuming normalized data
                return "poor", 0.3
            
            # Check for gaps (constant values)
            if np.std(data) < 1e-6:
                return "poor", 0.1
            
            # Check for spikes (outliers)
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            spike_ratio = np.sum(z_scores > 5) / len(data)
            
            if spike_ratio > 0.01:  # More than 1% spikes
                return "poor", 0.4
            
            # Signal-to-noise ratio estimation
            # Simple approach: assume first 10% is noise
            noise_samples = int(0.1 * len(data))
            noise_power = np.var(data[:noise_samples])
            signal_power = np.var(data[noise_samples:])
            
            if noise_power > 0:
                snr = signal_power / noise_power
                if snr > 100:
                    quality = "excellent"
                    score = 1.0
                elif snr > 50:
                    quality = "good"
                    score = 0.8
                elif snr > 10:
                    quality = "fair"
                    score = 0.6
                else:
                    quality = "poor"
                    score = 0.4
            else:
                quality = "good"
                score = 0.8
            
            return quality, score
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return "poor", 0.0

class WaveformProcessor:
    """Advanced waveform processing for seismic data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def preprocess_waveform(self, waveform: List[float], sampling_rate: float) -> np.ndarray:
        """Preprocess waveform data"""
        try:
            data = np.array(waveform)
            
            # Remove DC component
            data = data - np.mean(data)
            
            # Remove linear trend
            data = signal.detrend(data)
            
            # Apply bandpass filter
            nyquist = sampling_rate / 2
            low_freq = self.config.filter_freq_min / nyquist
            high_freq = min(self.config.filter_freq_max / nyquist, 0.95)
            
            if low_freq < high_freq:
                sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos')
                data = signal.sosfilt(sos, data)
            
            # Apply tapered window to reduce edge effects
            window = signal.windows.tukey(len(data), alpha=0.1)
            data = data * window
            
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing waveform: {str(e)}")
            return np.array(waveform)
    
    def detect_arrivals(self, waveform: np.ndarray, sampling_rate: float) -> Tuple[Optional[int], Optional[int]]:
        """Detect P and S wave arrivals using STA/LTA algorithm"""
        try:
            # STA/LTA parameters
            sta_samples = int(self.config.sta_length * sampling_rate)
            lta_samples = int(self.config.lta_length * sampling_rate)
            
            # Calculate STA/LTA
            sta_lta = classic_sta_lta(waveform, sta_samples, lta_samples)
            
            # Detect triggers
            triggers = trigger_onset(
                sta_lta, 
                self.config.trigger_threshold, 
                self.config.detrigger_threshold
            )
            
            p_arrival = None
            s_arrival = None
            
            if len(triggers) > 0:
                # First trigger is likely P-wave
                p_arrival = triggers[0][0]
                
                # Look for S-wave arrival (typically 1.5-2x P-wave time)
                if len(triggers) > 1:
                    for trigger in triggers[1:]:
                        if trigger[0] > p_arrival * 1.2:  # S-wave should be later
                            s_arrival = trigger[0]
                            break
            
            return p_arrival, s_arrival
            
        except Exception as e:
            logger.error(f"Error detecting arrivals: {str(e)}")
            return None, None
    
    def extract_features(self, waveform: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Extract features from waveform data"""
        try:
            features = {}
            
            # Time domain features
            features['max_amplitude'] = float(np.max(np.abs(waveform)))
            features['mean_amplitude'] = float(np.mean(np.abs(waveform)))
            features['std_amplitude'] = float(np.std(waveform))
            features['rms_amplitude'] = float(np.sqrt(np.mean(waveform**2)))
            features['skewness'] = float(self._calculate_skewness(waveform))
            features['kurtosis'] = float(self._calculate_kurtosis(waveform))
            
            # Frequency domain features
            fft_data = fft(waveform)
            freqs = np.fft.fftfreq(len(waveform), 1/sampling_rate)
            
            # Power spectral density
            psd = np.abs(fft_data)**2
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd[:len(psd)//2])
            features['dominant_frequency'] = float(freqs[dominant_freq_idx])
            
            # Spectral centroid
            features['spectral_centroid'] = float(
                np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
            )
            
            # Spectral rolloff (95% of energy)
            cumulative_energy = np.cumsum(psd[:len(psd)//2])
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = float(freqs[rolloff_idx[0]])
            else:
                features['spectral_rolloff'] = 0.0
            
            # Zero crossing rate
            features['zero_crossing_rate'] = float(
                np.sum(np.abs(np.diff(np.sign(waveform)))) / (2 * len(waveform))
            )
            
            # Energy in different frequency bands
            low_freq_energy = np.sum(psd[(freqs >= 0.1) & (freqs <= 1.0)])
            mid_freq_energy = np.sum(psd[(freqs >= 1.0) & (freqs <= 10.0)])
            high_freq_energy = np.sum(psd[(freqs >= 10.0) & (freqs <= 50.0)])
            
            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
            if total_energy > 0:
                features['low_freq_ratio'] = float(low_freq_energy / total_energy)
                features['mid_freq_ratio'] = float(mid_freq_energy / total_energy)
                features['high_freq_ratio'] = float(high_freq_energy / total_energy)
            else:
                features['low_freq_ratio'] = 0.0
                features['mid_freq_ratio'] = 0.0
                features['high_freq_ratio'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std)**3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std)**4) - 3

class DatabaseManager:
    """Database manager for seismic data storage"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.connection = None
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        try:
            self.connection = sqlite3.connect(self.config.database_path)
            cursor = self.connection.cursor()
            
            # Create seismic events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS seismic_events (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    magnitude REAL,
                    depth REAL,
                    latitude REAL,
                    longitude REAL,
                    location TEXT,
                    network TEXT,
                    station TEXT,
                    channel TEXT,
                    sampling_rate REAL,
                    p_arrival DATETIME,
                    s_arrival DATETIME,
                    quality TEXT,
                    processed BOOLEAN,
                    tsunami_risk TEXT,
                    confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create waveform data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS waveform_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    waveform_data BLOB,
                    features TEXT,
                    FOREIGN KEY (event_id) REFERENCES seismic_events (id)
                )
            ''')
            
            # Create processing logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    processing_time REAL,
                    status TEXT,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES seismic_events (id)
                )
            ''')
            
            self.connection.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def save_event(self, event: SeismicEvent, features: Dict[str, float] = None):
        """Save seismic event to database"""
        try:
            cursor = self.connection.cursor()
            
            # Insert event data
            cursor.execute('''
                INSERT OR REPLACE INTO seismic_events 
                (id, timestamp, magnitude, depth, latitude, longitude, location, 
                 network, station, channel, sampling_rate, p_arrival, s_arrival, 
                 quality, processed, tsunami_risk, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.id, event.timestamp, event.magnitude, event.depth,
                event.latitude, event.longitude, event.location,
                event.network, event.station, event.channel,
                event.sampling_rate, event.p_arrival, event.s_arrival,
                event.quality, event.processed, event.tsunami_risk,
                event.confidence
            ))
            
            # Insert waveform data
            waveform_blob = json.dumps(event.waveform_data).encode('utf-8')
            features_json = json.dumps(features) if features else None
            
            cursor.execute('''
                INSERT INTO waveform_data (event_id, waveform_data, features)
                VALUES (?, ?, ?)
            ''', (event.id, waveform_blob, features_json))
            
            self.connection.commit()
            logger.debug(f"Event {event.id} saved to database")
            
        except Exception as e:
            logger.error(f"Error saving event to database: {str(e)}")
            raise
    
    def get_event(self, event_id: str) -> Optional[SeismicEvent]:
        """Retrieve seismic event from database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT e.*, w.waveform_data, w.features
                FROM seismic_events e
                LEFT JOIN waveform_data w ON e.id = w.event_id
                WHERE e.id = ?
            ''', (event_id,))
            
            row = cursor.fetchone()
            if row:
                # Reconstruct event object
                waveform_data = json.loads(row[17].decode('utf-8')) if row[17] else []
                
                event = SeismicEvent(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    magnitude=row[2],
                    depth=row[3],
                    latitude=row[4],
                    longitude=row[5],
                    location=row[6],
                    network=row[7],
                    station=row[8],
                    channel=row[9],
                    sampling_rate=row[10],
                    waveform_data=waveform_data,
                    p_arrival=datetime.fromisoformat(row[11]) if row[11] else None,
                    s_arrival=datetime.fromisoformat(row[12]) if row[12] else None,
                    quality=row[13],
                    processed=bool(row[14]),
                    tsunami_risk=row[15],
                    confidence=row[16]
                )
                
                return event
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving event from database: {str(e)}")
            return None
    
    def get_recent_events(self, hours: int = 24) -> List[SeismicEvent]:
        """Get recent seismic events"""
        try:
            cursor = self.connection.cursor()
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT e.*, w.waveform_data
                FROM seismic_events e
                LEFT JOIN waveform_data w ON e.id = w.event_id
                WHERE e.timestamp > ?
                ORDER BY e.timestamp DESC
            ''', (cutoff_time,))
            
            events = []
            for row in cursor.fetchall():
                waveform_data = json.loads(row[17].decode('utf-8')) if row[17] else []
                
                event = SeismicEvent(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    magnitude=row[2],
                    depth=row[3],
                    latitude=row[4],
                    longitude=row[5],
                    location=row[6],
                    network=row[7],
                    station=row[8],
                    channel=row[9],
                    sampling_rate=row[10],
                    waveform_data=waveform_data,
                    p_arrival=datetime.fromisoformat(row[11]) if row[11] else None,
                    s_arrival=datetime.fromisoformat(row[12]) if row[12] else None,
                    quality=row[13],
                    processed=bool(row[14]),
                    tsunami_risk=row[15],
                    confidence=row[16]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving recent events: {str(e)}")
            return []
    
    def log_processing(self, event_id: str, processing_time: float, status: str, error_message: str = None):
        """Log processing information"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO processing_logs 
                (event_id, processing_time, status, error_message)
                VALUES (?, ?, ?, ?)
            ''', (event_id, processing_time, status, error_message))
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error logging processing: {str(e)}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

class SeismicDataProcessor:
    """Main seismic data processor"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.validator = SeismicDataValidator(config)
        self.waveform_processor = WaveformProcessor(config)
        self.database = DatabaseManager(config)
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.websocket = None
        self.processing_stats = {
            'events_processed': 0,
            'events_failed': 0,
            'average_processing_time': 0.0,
            'last_processing_time': 0.0
        }
        
        # Initialize connections
        self.init_connections()
    
    def init_connections(self):
        """Initialize external connections"""
        try:
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info("Kafka producer initialized")
            
            # Kafka consumer
            self.kafka_consumer = KafkaConsumer(
                'seismic-events',
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info("Kafka consumer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing connections: {str(e)}")
    
    async def process_event(self, event: SeismicEvent) -> Dict[str, Any]:
        """Process a single seismic event"""
        start_time = time.time()
        processing_result = {
            'event_id': event.id,
            'status': 'failed',
            'error': None,
            'processing_time': 0.0,
            'features': {},
            'arrivals': {},
            'quality_assessment': {}
        }
        
        try:
            # Validate event
            is_valid, errors = self.validator.validate_event(event)
            if not is_valid:
                processing_result['error'] = f"Validation failed: {'; '.join(errors)}"
                return processing_result
            
            # Preprocess waveform
            processed_waveform = self.waveform_processor.preprocess_waveform(
                event.waveform_data, event.sampling_rate
            )
            
            # Assess data quality
            quality, quality_score = self.validator.assess_data_quality(event.waveform_data)
            event.quality = quality
            processing_result['quality_assessment'] = {
                'quality': quality,
                'score': quality_score
            }
            
            # Detect P and S wave arrivals
            p_arrival, s_arrival = self.waveform_processor.detect_arrivals(
                processed_waveform, event.sampling_rate
            )
            
            if p_arrival is not None:
                event.p_arrival = event.timestamp + timedelta(seconds=p_arrival/event.sampling_rate)
            if s_arrival is not None:
                event.s_arrival = event.timestamp + timedelta(seconds=s_arrival/event.sampling_rate)
            
            processing_result['arrivals'] = {
                'p_arrival': p_arrival,
                's_arrival': s_arrival
            }
            
            # Extract features
            features = self.waveform_processor.extract_features(
                processed_waveform, event.sampling_rate
            )
            processing_result['features'] = features
            
            # Calculate tsunami risk (simplified)
            tsunami_risk, confidence = self.calculate_tsunami_risk(event, features)
            event.tsunami_risk = tsunami_risk
            event.confidence = confidence
            
            # Mark as processed
            event.processed = True
            
            # Save to database
            self.database.save_event(event, features)
            
            # Cache in Redis
            await self.cache_event(event)
            
            # Send to Kafka
            await self.send_to_kafka(event)
            
            # Send WebSocket notification
            await self.send_websocket_notification(event)
            
            processing_result['status'] = 'success'
            processing_result['processing_time'] = time.time() - start_time
            
            # Update statistics
            self.processing_stats['events_processed'] += 1
            self.processing_stats['last_processing_time'] = processing_result['processing_time']
            self.processing_stats['average_processing_time'] = (
                (self.processing_stats['average_processing_time'] * (self.processing_stats['events_processed'] - 1) +
                 processing_result['processing_time']) / self.processing_stats['events_processed']
            )
            
            logger.info(f"Successfully processed event {event.id} in {processing_result['processing_time']:.2f}s")
            
        except Exception as e:
            processing_result['error'] = str(e)
            self.processing_stats['events_failed'] += 1
            logger.error(f"Error processing event {event.id}: {str(e)}")
        
        finally:
            # Log processing
            self.database.log_processing(
                event.id,
                processing_result['processing_time'],
                processing_result['status'],
                processing_result.get('error')
            )
        
        return processing_result
    
    def calculate_tsunami_risk(self, event: SeismicEvent, features: Dict[str, float]) -> Tuple[str, float]:
        """Calculate tsunami risk based on event characteristics"""
        try:
            risk_score = 0.0
            
            # Magnitude contribution (40%)
            if event.magnitude >= 8.0:
                risk_score += 40
            elif event.magnitude >= 7.0:
                risk_score += 30
            elif event.magnitude >= 6.0:
                risk_score += 20
            elif event.magnitude >= 5.0:
                risk_score += 10
            
            # Depth contribution (25%)
            if event.depth <= 35:
                risk_score += 25
            elif event.depth <= 70:
                risk_score += 15
            elif event.depth <= 150:
                risk_score += 5
            
            # Location contribution (20%)
            if self.is_coastal_location(event.latitude, event.longitude):
                risk_score += 20
            
            # Waveform characteristics (15%)
            if features.get('max_amplitude', 0) > 0.5:
                risk_score += 15
            elif features.get('max_amplitude', 0) > 0.3:
                risk_score += 10
            elif features.get('max_amplitude', 0) > 0.1:
                risk_score += 5
            
            # Normalize to 0-100 range
            risk_score = min(100, max(0, risk_score))
            
            # Convert to risk level
            if risk_score >= 80:
                risk_level = "critical"
            elif risk_score >= 60:
                risk_level = "high"
            elif risk_score >= 40:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            confidence = risk_score / 100.0
            
            return risk_level, confidence
            
        except Exception as e:
            logger.error(f"Error calculating tsunami risk: {str(e)}")
            return "unknown", 0.0
    
    def is_coastal_location(self, lat: float, lng: float) -> bool:
        """Check if location is in coastal area"""
        # Simplified coastal detection
        coastal_regions = [
            (35.0, 139.0, 300),  # Japan
            (40.0, -125.0, 200), # US West Coast
            (-10.0, 110.0, 250), # Indonesia
            (36.0, 28.0, 150),   # Greece/Turkey
            (-40.0, 175.0, 200), # New Zealand
        ]
        
        for region_lat, region_lng, radius_km in coastal_regions:
            distance = np.sqrt((lat - region_lat)**2 + (lng - region_lng)**2) * 111
            if distance <= radius_km:
                return True
        
        return False
    
    async def cache_event(self, event: SeismicEvent):
        """Cache event in Redis"""
        try:
            event_data = asdict(event)
            event_data['timestamp'] = event.timestamp.isoformat()
            if event.p_arrival:
                event_data['p_arrival'] = event.p_arrival.isoformat()
            if event.s_arrival:
                event_data['s_arrival'] = event.s_arrival.isoformat()
            
            # Cache with expiration
            self.redis_client.setex(
                f"event:{event.id}",
                3600,  # 1 hour expiration
                json.dumps(event_data)
            )
            
            # Add to recent events list
            self.redis_client.lpush("recent_events", event.id)
            self.redis_client.ltrim("recent_events", 0, 999)  # Keep last 1000 events
            
        except Exception as e:
            logger.error(f"Error caching event: {str(e)}")
    
    async def send_to_kafka(self, event: SeismicEvent):
        """Send event to Kafka"""
        try:
            event_data = asdict(event)
            event_data['timestamp'] = event.timestamp.isoformat()
            if event.p_arrival:
                event_data['p_arrival'] = event.p_arrival.isoformat()
            if event.s_arrival:
                event_data['s_arrival'] = event.s_arrival.isoformat()
            
            # Send to appropriate topic based on risk level
            topic = f"seismic-events-{event.tsunami_risk}"
            
            self.kafka_producer.send(topic, event_data)
            self.kafka_producer.flush()
            
        except Exception as e:
            logger.error(f"Error sending to Kafka: {str(e)}")
    
    async def send_websocket_notification(self, event: SeismicEvent):
        """Send WebSocket notification"""
        try:
            if self.websocket is None:
                self.websocket = await websockets.connect(self.config.websocket_url)
            
            notification = {
                'type': 'seismic_data',
                'payload': {
                    'id': event.id,
                    'timestamp': event.timestamp.isoformat(),
                    'magnitude': event.magnitude,
                    'depth': event.depth,
                    'latitude': event.latitude,
                    'longitude': event.longitude,
                    'location': event.location,
                    'tsunami_risk': event.tsunami_risk,
                    'confidence': event.confidence,
                    'processed': event.processed
                }
            }
            
            await self.websocket.send(json.dumps(notification))
            
        except Exception as e:
            logger.error(f"Error sending WebSocket notification: {str(e)}")
            self.websocket = None
    
    async def process_batch(self, events: List[SeismicEvent]) -> List[Dict[str, Any]]:
        """Process a batch of events"""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [
                asyncio.create_task(self.process_event(event))
                for event in events
            ]
            
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_kafka_consumer(self):
        """Start Kafka consumer for incoming events"""
        try:
            logger.info("Starting Kafka consumer...")
            
            async for message in self.kafka_consumer:
                try:
                    event_data = message.value
                    
                    # Convert to SeismicEvent
                    event = SeismicEvent(
                        id=event_data['id'],
                        timestamp=datetime.fromisoformat(event_data['timestamp']),
                        magnitude=event_data['magnitude'],
                        depth=event_data['depth'],
                        latitude=event_data['latitude'],
                        longitude=event_data['longitude'],
                        location=event_data['location'],
                        network=event_data['network'],
                        station=event_data['station'],
                        channel=event_data['channel'],
                        sampling_rate=event_data['sampling_rate'],
                        waveform_data=event_data['waveform_data']
                    )
                    
                    # Process event
                    await self.process_event(event)
                    
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in Kafka consumer: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.kafka_producer:
                self.kafka_producer.close()
            if self.kafka_consumer:
                self.kafka_consumer.close()
            if self.redis_client:
                self.redis_client.close()
            if self.database:
                self.database.close()
            if self.websocket:
                asyncio.create_task(self.websocket.close())
                
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

async def main():
    """Main function to run the seismic processor"""
    config = ProcessingConfig()
    processor = SeismicDataProcessor(config)
    
    try:
        # Generate sample data for testing
        sample_events = []
        for i in range(10):
            event = SeismicEvent(
                id=f"event_{i:03d}",
                timestamp=datetime.now() - timedelta(minutes=i*5),
                magnitude=5.0 + np.random.uniform(-1, 3),
                depth=50 + np.random.uniform(-30, 200),
                latitude=35.0 + np.random.uniform(-5, 5),
                longitude=139.0 + np.random.uniform(-5, 5),
                location=f"Test Location {i}",
                network="TEST",
                station=f"STA{i:02d}",
                channel="HHZ",
                sampling_rate=100.0,
                waveform_data=np.random.normal(0, 0.1, 6000).tolist()
            )
            sample_events.append(event)
        
        # Process events
        logger.info(f"Processing {len(sample_events)} sample events...")
        results = await processor.process_batch(sample_events)
        
        # Print results
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] == 'failed'])
        
        logger.info(f"Processing completed: {successful} successful, {failed} failed")
        
        # Print statistics
        stats = processor.get_processing_stats()
        logger.info(f"Processing statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())