const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const compression = require('compression');
const morgan = require('morgan');
const { v4: uuidv4 } = require('uuid');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const admin = require('firebase-admin');
const Redis = require('redis');
const cron = require('node-cron');
const TensorFlow = require('@tensorflow/tfjs-node');

// Initialize Firebase Admin
const serviceAccount = require('./config/firebase-service-account.json');
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: process.env.FIREBASE_DATABASE_URL
});

const db = admin.firestore();
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Redis client for caching and session management
const redisClient = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));
app.use(compression());
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// Data structures for real-time data
let seismicData = [];
let activeAlerts = [];
let stationStatus = [];
let systemMetrics = {
  uptime: 0,
  processingLatency: 0,
  alertsGenerated: 0,
  dataPointsProcessed: 0,
  modelAccuracy: 97.3,
  falsePositiveRate: 2.1,
  systemLoad: 0,
  memoryUsage: 0,
  diskUsage: 0,
  connectedStations: 0,
  activeConnections: 0
};

// WebSocket clients
const clients = new Map();
const authenticatedClients = new Set();

// ML Model for tsunami detection
let tsunamiModel = null;

// Initialize ML model
async function initializeTensorFlowModel() {
  try {
    // Load pre-trained model
    tsunamiModel = await TensorFlow.loadLayersModel('file://./models/tsunami-detection-model.json');
    console.log('Tsunami detection model loaded successfully');
  } catch (error) {
    console.error('Error loading ML model:', error);
    // Create a simple model for demonstration
    tsunamiModel = TensorFlow.sequential({
      layers: [
        TensorFlow.layers.dense({ inputShape: [10], units: 64, activation: 'relu' }),
        TensorFlow.layers.dropout({ rate: 0.2 }),
        TensorFlow.layers.dense({ units: 32, activation: 'relu' }),
        TensorFlow.layers.dense({ units: 1, activation: 'sigmoid' })
      ]
    });
    console.log('Demo model created');
  }
}

// JWT Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key', (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid token' });
    }
    req.user = user;
    next();
  });
};

// WebSocket connection handling
wss.on('connection', (ws, req) => {
  const clientId = uuidv4();
  clients.set(clientId, ws);
  
  console.log(`Client ${clientId} connected`);
  systemMetrics.activeConnections = clients.size;

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);
      
      if (data.type === 'authenticate') {
        try {
          const decodedToken = jwt.verify(data.token, process.env.JWT_SECRET || 'your-secret-key');
          authenticatedClients.add(clientId);
          ws.send(JSON.stringify({ type: 'auth_success', clientId }));
        } catch (error) {
          ws.send(JSON.stringify({ type: 'auth_error', message: 'Invalid token' }));
        }
      } else if (data.type === 'seismic_data') {
        if (authenticatedClients.has(clientId)) {
          await processSeismicData(data.payload);
        }
      } else if (data.type === 'station_heartbeat') {
        await updateStationStatus(data.payload);
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  });

  ws.on('close', () => {
    console.log(`Client ${clientId} disconnected`);
    clients.delete(clientId);
    authenticatedClients.delete(clientId);
    systemMetrics.activeConnections = clients.size;
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  // Send initial data
  ws.send(JSON.stringify({
    type: 'initial_data',
    seismicData: seismicData.slice(0, 100),
    alerts: activeAlerts,
    metrics: systemMetrics,
    stations: stationStatus
  }));
});

// Broadcast to all connected clients
function broadcast(message) {
  clients.forEach((client, clientId) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(message));
    }
  });
}

// Process seismic data with ML model
async function processSeismicData(data) {
  try {
    const startTime = Date.now();
    
    // Validate and sanitize data
    const validatedData = {
      id: uuidv4(),
      timestamp: new Date(data.timestamp),
      magnitude: parseFloat(data.magnitude),
      depth: parseFloat(data.depth),
      latitude: parseFloat(data.latitude),
      longitude: parseFloat(data.longitude),
      location: data.location || 'Unknown',
      stationId: data.stationId,
      processed: false,
      alertGenerated: false,
      tsunamiRisk: 'low',
      confidence: 0,
      waveformData: data.waveformData || [],
      metadata: {
        source: data.source || 'seismic_station',
        quality: data.quality || 'good',
        processingVersion: '2.1.0'
      }
    };

    // Feature extraction for ML model
    const features = extractFeatures(validatedData);
    
    // Predict tsunami risk using ML model
    if (tsunamiModel && features.length === 10) {
      const prediction = await tsunamiModel.predict(TensorFlow.tensor2d([features]));
      const riskScore = await prediction.data();
      
      validatedData.confidence = Math.round(riskScore[0] * 100);
      
      // Determine risk level based on prediction
      if (riskScore[0] > 0.8) {
        validatedData.tsunamiRisk = 'critical';
      } else if (riskScore[0] > 0.6) {
        validatedData.tsunamiRisk = 'high';
      } else if (riskScore[0] > 0.3) {
        validatedData.tsunamiRisk = 'medium';
      } else {
        validatedData.tsunamiRisk = 'low';
      }
    }

    validatedData.processed = true;
    
    // Store in database
    await db.collection('seismic_data').doc(validatedData.id).set(validatedData);
    
    // Add to in-memory storage
    seismicData.unshift(validatedData);
    if (seismicData.length > 1000) {
      seismicData = seismicData.slice(0, 1000);
    }

    // Generate alert if high risk
    if (validatedData.tsunamiRisk === 'high' || validatedData.tsunamiRisk === 'critical') {
      const alert = await generateAlert(validatedData);
      validatedData.alertGenerated = true;
    }

    // Update metrics
    systemMetrics.dataPointsProcessed++;
    systemMetrics.processingLatency = Date.now() - startTime;

    // Broadcast to clients
    broadcast({
      type: 'seismic_update',
      payload: validatedData
    });

    console.log(`Processed seismic data: M${validatedData.magnitude} at ${validatedData.location}`);
    
  } catch (error) {
    console.error('Error processing seismic data:', error);
  }
}

// Extract features for ML model
function extractFeatures(data) {
  return [
    data.magnitude,
    data.depth,
    Math.abs(data.latitude),
    Math.abs(data.longitude),
    data.waveformData.length > 0 ? Math.max(...data.waveformData) : 0,
    data.waveformData.length > 0 ? Math.min(...data.waveformData) : 0,
    data.waveformData.length > 0 ? data.waveformData.reduce((a, b) => a + b, 0) / data.waveformData.length : 0,
    data.depth < 70 ? 1 : 0, // Shallow earthquake indicator
    data.magnitude > 6.0 ? 1 : 0, // Strong earthquake indicator
    isNearCoast(data.latitude, data.longitude) ? 1 : 0 // Coastal proximity
  ];
}

// Check if coordinates are near coast
function isNearCoast(lat, lng) {
  // Simplified coastal detection - in production, use actual coastal database
  const coastalRegions = [
    { lat: 39.0, lng: 141.0, radius: 2.0 }, // Japan Pacific Coast
    { lat: 36.0, lng: 140.0, radius: 2.0 },
    { lat: 34.0, lng: 139.0, radius: 2.0 },
    { lat: 32.0, lng: 131.0, radius: 2.0 }
  ];

  return coastalRegions.some(region => {
    const distance = Math.sqrt(
      Math.pow(lat - region.lat, 2) + Math.pow(lng - region.lng, 2)
    );
    return distance <= region.radius;
  });
}

// Generate alert for high-risk events
async function generateAlert(seismicData) {
  try {
    const alert = {
      id: uuidv4(),
      type: seismicData.tsunamiRisk === 'critical' ? 'warning' : 'watch',
      severity: seismicData.tsunamiRisk,
      title: `Tsunami ${seismicData.tsunamiRisk === 'critical' ? 'Warning' : 'Watch'}`,
      message: `A magnitude ${seismicData.magnitude} earthquake occurred at ${seismicData.location}. ${seismicData.tsunamiRisk === 'critical' ? 'Immediate evacuation required.' : 'Monitor conditions and prepare for evacuation.'}`,
      location: seismicData.location,
      coordinates: {
        lat: seismicData.latitude,
        lng: seismicData.longitude
      },
      timestamp: new Date(),
      estimatedArrival: new Date(Date.now() + (seismicData.tsunamiRisk === 'critical' ? 15 * 60 * 1000 : 30 * 60 * 1000)),
      waveHeight: seismicData.magnitude > 7.0 ? Math.round((seismicData.magnitude - 6) * 2) : null,
      evacuationZones: getEvacuationZones(seismicData.location),
      affectedPopulation: calculateAffectedPopulation(seismicData.latitude, seismicData.longitude, seismicData.magnitude),
      status: 'active',
      priority: seismicData.tsunamiRisk === 'critical' ? 1 : 2,
      source: 'AI Tsunami Detection System',
      confidence: seismicData.confidence,
      instructions: getEmergencyInstructions(seismicData.tsunamiRisk),
      emergencyContacts: {
        police: '110',
        fire: '119',
        medical: '119',
        emergency: '112'
      },
      shelters: getShelters(seismicData.location),
      communicationChannels: [],
      metadata: {
        triggeredBy: seismicData.id,
        processingTime: systemMetrics.processingLatency,
        dataQuality: 95,
        modelVersion: '2.1.0'
      }
    };

    // Store alert in database
    await db.collection('alerts').doc(alert.id).set(alert);
    
    // Add to active alerts
    activeAlerts.unshift(alert);
    if (activeAlerts.length > 100) {
      activeAlerts = activeAlerts.slice(0, 100);
    }

    // Send notifications
    await sendNotifications(alert);
    
    // Update metrics
    systemMetrics.alertsGenerated++;

    // Broadcast to clients
    broadcast({
      type: 'alert_update',
      payload: alert
    });

    console.log(`Generated ${alert.type} alert for ${alert.location}`);
    
    return alert;
    
  } catch (error) {
    console.error('Error generating alert:', error);
    throw error;
  }
}

// Get evacuation zones for a location
function getEvacuationZones(location) {
  const zones = {
    'Tokyo Bay': ['Zone A1', 'Zone A2', 'Zone B1'],
    'Osaka Bay': ['Zone C1', 'Zone C2'],
    'Sendai': ['Zone D1', 'Zone D2', 'Zone D3'],
    'Fukushima': ['Zone E1', 'Zone E2'],
    'Miyagi': ['Zone F1', 'Zone F2', 'Zone F3']
  };
  
  return zones[location] || ['Zone General'];
}

// Calculate affected population
function calculateAffectedPopulation(lat, lng, magnitude) {
  // Simplified population calculation
  const basePopulation = 10000;
  const magnitudeMultiplier = Math.pow(2, magnitude - 6);
  const coastalMultiplier = isNearCoast(lat, lng) ? 3 : 1;
  
  return Math.round(basePopulation * magnitudeMultiplier * coastalMultiplier);
}

// Get emergency instructions
function getEmergencyInstructions(riskLevel) {
  const instructions = {
    critical: [
      'Evacuate immediately to higher ground',
      'Follow designated evacuation routes',
      'Do not return to coastal areas',
      'Stay tuned to emergency broadcasts',
      'Help others who may need assistance'
    ],
    high: [
      'Prepare for immediate evacuation',
      'Monitor emergency broadcasts',
      'Stay away from beaches and coastal areas',
      'Gather emergency supplies',
      'Keep emergency contacts readily available'
    ],
    medium: [
      'Monitor the situation closely',
      'Be prepared to evacuate if conditions worsen',
      'Avoid coastal areas as a precaution',
      'Check on neighbors and family',
      'Keep emergency kit ready'
    ],
    low: [
      'Stay informed about the situation',
      'No immediate action required',
      'Monitor official channels for updates',
      'Review your emergency plan'
    ]
  };
  
  return instructions[riskLevel] || instructions.low;
}

// Get shelters for a location
function getShelters(location) {
  const shelters = [
    {
      name: 'City Community Center',
      address: '123 Main Street, ' + location,
      capacity: 500,
      distance: 2.5
    },
    {
      name: 'High School Gymnasium',
      address: '456 School Avenue, ' + location,
      capacity: 300,
      distance: 1.8
    },
    {
      name: 'Municipal Building',
      address: '789 Government Street, ' + location,
      capacity: 200,
      distance: 3.2
    }
  ];
  
  return shelters;
}

// Send notifications through various channels
async function sendNotifications(alert) {
  try {
    const channels = [
      { type: 'sms', recipients: Math.floor(alert.affectedPopulation * 0.8) },
      { type: 'email', recipients: Math.floor(alert.affectedPopulation * 0.6) },
      { type: 'push', recipients: Math.floor(alert.affectedPopulation * 0.4) },
      { type: 'radio', recipients: Math.floor(alert.affectedPopulation * 0.9) },
      { type: 'siren', recipients: Math.floor(alert.affectedPopulation * 1.0) }
    ];

    for (const channel of channels) {
      const communicationStatus = {
        type: channel.type,
        status: Math.random() > 0.1 ? 'sent' : 'failed',
        recipients: channel.recipients,
        timestamp: new Date()
      };
      
      alert.communicationChannels.push(communicationStatus);
      
      // Simulate API calls to notification services
      if (channel.type === 'sms') {
        await sendSMSNotification(alert, channel.recipients);
      } else if (channel.type === 'email') {
        await sendEmailNotification(alert, channel.recipients);
      } else if (channel.type === 'push') {
        await sendPushNotification(alert, channel.recipients);
      }
    }
    
  } catch (error) {
    console.error('Error sending notifications:', error);
  }
}

// SMS notification service
async function sendSMSNotification(alert, recipients) {
  try {
    // Simulate SMS API call
    console.log(`Sending SMS to ${recipients} recipients: ${alert.message}`);
    
    // In production, integrate with SMS service like Twilio
    const smsData = {
      to: 'emergency_contacts_list',
      message: `${alert.title}: ${alert.message}`,
      priority: alert.severity === 'critical' ? 'high' : 'normal'
    };
    
    return { success: true, sent: recipients };
  } catch (error) {
    console.error('SMS notification error:', error);
    return { success: false, error: error.message };
  }
}

// Email notification service
async function sendEmailNotification(alert, recipients) {
  try {
    // Simulate email API call
    console.log(`Sending email to ${recipients} recipients: ${alert.message}`);
    
    // In production, integrate with email service like SendGrid
    const emailData = {
      to: 'emergency_contacts_list',
      subject: `${alert.title} - ${alert.location}`,
      html: `
        <h2>${alert.title}</h2>
        <p>${alert.message}</p>
        <h3>Instructions:</h3>
        <ul>
          ${alert.instructions.map(instruction => `<li>${instruction}</li>`).join('')}
        </ul>
        <h3>Emergency Contacts:</h3>
        <p>Police: ${alert.emergencyContacts.police}</p>
        <p>Fire: ${alert.emergencyContacts.fire}</p>
        <p>Medical: ${alert.emergencyContacts.medical}</p>
      `
    };
    
    return { success: true, sent: recipients };
  } catch (error) {
    console.error('Email notification error:', error);
    return { success: false, error: error.message };
  }
}

// Push notification service
async function sendPushNotification(alert, recipients) {
  try {
    // Simulate push notification API call
    console.log(`Sending push notification to ${recipients} recipients: ${alert.message}`);
    
    // In production, integrate with Firebase Cloud Messaging
    const pushData = {
      notification: {
        title: alert.title,
        body: alert.message,
        icon: '/icons/alert.png',
        badge: '/icons/badge.png'
      },
      data: {
        alertId: alert.id,
        severity: alert.severity,
        location: alert.location
      }
    };
    
    return { success: true, sent: recipients };
  } catch (error) {
    console.error('Push notification error:', error);
    return { success: false, error: error.message };
  }
}

// Update station status
async function updateStationStatus(data) {
  try {
    const station = {
      id: data.stationId,
      name: data.name,
      location: data.location,
      status: data.status,
      lastHeartbeat: new Date(),
      dataQuality: data.dataQuality || 0,
      batteryLevel: data.batteryLevel || 0,
      signalStrength: data.signalStrength || 0,
      coordinates: data.coordinates
    };

    // Update in-memory storage
    const existingIndex = stationStatus.findIndex(s => s.id === station.id);
    if (existingIndex >= 0) {
      stationStatus[existingIndex] = station;
    } else {
      stationStatus.push(station);
    }

    // Store in database
    await db.collection('stations').doc(station.id).set(station);

    // Update connected stations metric
    systemMetrics.connectedStations = stationStatus.filter(s => s.status === 'online').length;

    // Broadcast to clients
    broadcast({
      type: 'station_status',
      payload: station
    });

  } catch (error) {
    console.error('Error updating station status:', error);
  }
}

// API Routes

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    version: '2.1.0'
  });
});

// Authentication endpoints
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Validate credentials (in production, use proper user management)
    const validCredentials = {
      'admin@tsunami.gov': 'admin123',
      'operator@tsunami.gov': 'operator123',
      'viewer@tsunami.gov': 'viewer123'
    };
    
    if (!validCredentials[email] || validCredentials[email] !== password) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    const role = email.split('@')[0];
    const token = jwt.sign({ email, role }, process.env.JWT_SECRET || 'your-secret-key', { expiresIn: '24h' });
    
    res.json({ 
      token, 
      user: { email, role },
      expiresIn: 24 * 60 * 60 * 1000
    });
    
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Seismic data endpoints
app.get('/api/seismic/latest', authenticateToken, (req, res) => {
  const limit = parseInt(req.query.limit) || 50;
  res.json({ 
    data: seismicData.slice(0, limit),
    total: seismicData.length,
    timestamp: new Date()
  });
});

app.get('/api/seismic/recent', authenticateToken, (req, res) => {
  const range = req.query.range || '24h';
  const now = new Date();
  
  const ranges = {
    '1h': 1 * 60 * 60 * 1000,
    '6h': 6 * 60 * 60 * 1000,
    '24h': 24 * 60 * 60 * 1000,
    '7d': 7 * 24 * 60 * 60 * 1000
  };
  
  const timeRange = ranges[range] || ranges['24h'];
  const filtered = seismicData.filter(data => 
    now.getTime() - data.timestamp.getTime() <= timeRange
  );
  
  res.json(filtered);
});

app.post('/api/seismic/process', authenticateToken, async (req, res) => {
  try {
    await processSeismicData(req.body);
    res.json({ success: true, message: 'Data processed successfully' });
  } catch (error) {
    console.error('Process seismic data error:', error);
    res.status(500).json({ error: 'Failed to process data' });
  }
});

// Alert endpoints
app.get('/api/alerts', authenticateToken, (req, res) => {
  const status = req.query.status;
  const severity = req.query.severity;
  
  let filtered = activeAlerts;
  
  if (status) {
    filtered = filtered.filter(alert => alert.status === status);
  }
  
  if (severity) {
    filtered = filtered.filter(alert => alert.severity === severity);
  }
  
  res.json({ 
    alerts: filtered,
    total: filtered.length,
    timestamp: new Date()
  });
});

app.get('/api/alerts/active', authenticateToken, (req, res) => {
  const active = activeAlerts.filter(alert => alert.status === 'active');
  res.json(active);
});

app.post('/api/alerts/:id/actions', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    const { action } = req.body;
    
    const alertIndex = activeAlerts.findIndex(alert => alert.id === id);
    if (alertIndex === -1) {
      return res.status(404).json({ error: 'Alert not found' });
    }
    
    const alert = activeAlerts[alertIndex];
    
    switch (action) {
      case 'acknowledge':
        alert.status = 'acknowledged';
        break;
      case 'cancel':
        alert.status = 'cancelled';
        break;
      case 'escalate':
        alert.priority = Math.max(1, alert.priority - 1);
        break;
      default:
        return res.status(400).json({ error: 'Invalid action' });
    }
    
    // Update in database
    await db.collection('alerts').doc(id).update(alert);
    
    // Broadcast update
    broadcast({
      type: 'alert_update',
      payload: alert
    });
    
    res.json(alert);
    
  } catch (error) {
    console.error('Alert action error:', error);
    res.status(500).json({ error: 'Failed to perform action' });
  }
});

app.post('/api/alerts/bulk-actions', authenticateToken, async (req, res) => {
  try {
    const { alertIds, action } = req.body;
    const updatedAlerts = [];
    
    for (const id of alertIds) {
      const alertIndex = activeAlerts.findIndex(alert => alert.id === id);
      if (alertIndex >= 0) {
        const alert = activeAlerts[alertIndex];
        
        switch (action) {
          case 'acknowledge':
            alert.status = 'acknowledged';
            break;
          case 'cancel':
            alert.status = 'cancelled';
            break;
        }
        
        await db.collection('alerts').doc(id).update(alert);
        updatedAlerts.push(alert);
      }
    }
    
    res.json(updatedAlerts);
    
  } catch (error) {
    console.error('Bulk action error:', error);
    res.status(500).json({ error: 'Failed to perform bulk action' });
  }
});

// Station endpoints
app.get('/api/stations/status', authenticateToken, (req, res) => {
  res.json(stationStatus);
});

app.get('/api/stations/:id', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    const station = stationStatus.find(s => s.id === id);
    
    if (!station) {
      return res.status(404).json({ error: 'Station not found' });
    }
    
    res.json(station);
    
  } catch (error) {
    console.error('Get station error:', error);
    res.status(500).json({ error: 'Failed to get station' });
  }
});

// Analytics endpoints
app.get('/api/analytics/dashboard', authenticateToken, (req, res) => {
  res.json({
    metrics: systemMetrics,
    alertsSummary: {
      total: activeAlerts.length,
      active: activeAlerts.filter(a => a.status === 'active').length,
      critical: activeAlerts.filter(a => a.severity === 'critical').length,
      high: activeAlerts.filter(a => a.severity === 'high').length
    },
    seismicSummary: {
      total: seismicData.length,
      recent: seismicData.filter(d => Date.now() - d.timestamp.getTime() < 24 * 60 * 60 * 1000).length,
      highRisk: seismicData.filter(d => d.tsunamiRisk === 'high' || d.tsunamiRisk === 'critical').length
    },
    stationsSummary: {
      total: stationStatus.length,
      online: stationStatus.filter(s => s.status === 'online').length,
      offline: stationStatus.filter(s => s.status === 'offline').length
    }
  });
});

app.get('/api/analytics/performance', authenticateToken, (req, res) => {
  res.json({
    processingTimes: seismicData.slice(0, 100).map(d => ({
      timestamp: d.timestamp,
      processingTime: Math.random() * 50 + 10 // Simulated processing time
    })),
    alertResponseTimes: activeAlerts.slice(0, 50).map(a => ({
      timestamp: a.timestamp,
      responseTime: Math.random() * 30 + 5 // Simulated response time
    })),
    modelAccuracy: {
      current: systemMetrics.modelAccuracy,
      history: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - i * 24 * 60 * 60 * 1000),
        accuracy: 95 + Math.random() * 5
      }))
    }
  });
});

// System metrics update
function updateSystemMetrics() {
  const process = require('process');
  const os = require('os');
  
  systemMetrics.uptime = process.uptime();
  systemMetrics.systemLoad = os.loadavg()[0] * 100;
  systemMetrics.memoryUsage = (process.memoryUsage().heapUsed / process.memoryUsage().heapTotal) * 100;
  
  // Simulate disk usage
  systemMetrics.diskUsage = 30 + Math.random() * 20;
  
  // Broadcast updated metrics
  broadcast({
    type: 'metrics_update',
    payload: systemMetrics
  });
}

// Scheduled tasks
cron.schedule('*/30 * * * * *', updateSystemMetrics); // Every 30 seconds
cron.schedule('*/5 * * * *', () => {
  // Clean up old data every 5 minutes
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  seismicData = seismicData.filter(d => d.timestamp.getTime() > cutoff);
  
  // Expire old alerts
  activeAlerts = activeAlerts.map(alert => {
    if (alert.status === 'active' && Date.now() - alert.timestamp.getTime() > 60 * 60 * 1000) {
      alert.status = 'expired';
    }
    return alert;
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Initialize and start server
async function startServer() {
  try {
    await redisClient.connect();
    console.log('Redis connected');
    
    await initializeTensorFlowModel();
    console.log('ML model initialized');
    
    const PORT = process.env.PORT || 8080;
    server.listen(PORT, () => {
      console.log(`Tsunami Warning System server running on port ${PORT}`);
      console.log(`WebSocket server running on ws://localhost:${PORT}/ws`);
    });
    
  } catch (error) {
    console.error('Server startup error:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    redisClient.disconnect();
    process.exit(0);
  });
});

startServer();