const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const jwt = require('jsonwebtoken');
const EventEmitter = require('events');

class WebSocketHandler extends EventEmitter {
  constructor() {
    super();
    this.clients = new Map();
    this.authenticatedClients = new Set();
    this.subscriptions = new Map();
    this.messageQueue = new Map();
    this.rateLimiter = new Map();
    this.heartbeatInterval = 30000; // 30 seconds
    this.maxMessageSize = 1024 * 1024; // 1MB
    this.maxConnections = 1000;
    this.connectionStats = {
      totalConnections: 0,
      activeConnections: 0,
      authenticatedConnections: 0,
      messagesSent: 0,
      messagesReceived: 0,
      errors: 0
    };
  }

  initialize(server) {
    this.wss = new WebSocket.Server({ 
      server,
      verifyClient: (info) => {
        // Check connection limits
        if (this.clients.size >= this.maxConnections) {
          return false;
        }
        
        // Basic IP rate limiting
        const clientIp = info.req.connection.remoteAddress;
        const now = Date.now();
        const windowMs = 60000; // 1 minute
        const maxConnections = 5;
        
        if (!this.rateLimiter.has(clientIp)) {
          this.rateLimiter.set(clientIp, []);
        }
        
        const connections = this.rateLimiter.get(clientIp);
        const recentConnections = connections.filter(time => now - time < windowMs);
        
        if (recentConnections.length >= maxConnections) {
          return false;
        }
        
        connections.push(now);
        this.rateLimiter.set(clientIp, connections);
        
        return true;
      }
    });

    this.wss.on('connection', (ws, req) => {
      this.handleConnection(ws, req);
    });

    // Start heartbeat interval
    this.startHeartbeat();
    
    // Clean up rate limiter periodically
    setInterval(() => {
      this.cleanupRateLimiter();
    }, 60000);

    console.log('WebSocket handler initialized');
  }

  handleConnection(ws, req) {
    const clientId = uuidv4();
    const clientIp = req.connection.remoteAddress;
    
    const client = {
      id: clientId,
      ws: ws,
      ip: clientIp,
      connectedAt: new Date(),
      lastPing: Date.now(),
      authenticated: false,
      subscriptions: new Set(),
      messageCount: 0,
      user: null,
      isAlive: true
    };

    this.clients.set(clientId, client);
    this.connectionStats.totalConnections++;
    this.connectionStats.activeConnections++;

    console.log(`Client ${clientId} connected from ${clientIp}`);

    // Set up message handling
    ws.on('message', (message) => {
      this.handleMessage(clientId, message);
    });

    ws.on('close', (code, reason) => {
      this.handleDisconnection(clientId, code, reason);
    });

    ws.on('error', (error) => {
      this.handleError(clientId, error);
    });

    ws.on('pong', () => {
      if (this.clients.has(clientId)) {
        this.clients.get(clientId).isAlive = true;
        this.clients.get(clientId).lastPing = Date.now();
      }
    });

    // Send initial connection message
    this.sendToClient(clientId, {
      type: 'connection_established',
      clientId: clientId,
      timestamp: new Date(),
      message: 'Welcome to Tsunami Warning System WebSocket'
    });

    // Emit connection event
    this.emit('client_connected', { clientId, client });
  }

  handleMessage(clientId, message) {
    try {
      const client = this.clients.get(clientId);
      if (!client) return;

      // Check message size
      if (message.length > this.maxMessageSize) {
        this.sendError(clientId, 'Message too large');
        return;
      }

      // Rate limiting per client
      const now = Date.now();
      const windowMs = 60000; // 1 minute
      const maxMessages = 100;
      
      if (!this.messageQueue.has(clientId)) {
        this.messageQueue.set(clientId, []);
      }
      
      const messages = this.messageQueue.get(clientId);
      const recentMessages = messages.filter(time => now - time < windowMs);
      
      if (recentMessages.length >= maxMessages) {
        this.sendError(clientId, 'Rate limit exceeded');
        return;
      }
      
      messages.push(now);
      this.messageQueue.set(clientId, messages);

      // Parse message
      let data;
      try {
        data = JSON.parse(message);
      } catch (error) {
        this.sendError(clientId, 'Invalid JSON');
        return;
      }

      // Validate message structure
      if (!data.type) {
        this.sendError(clientId, 'Message type required');
        return;
      }

      client.messageCount++;
      this.connectionStats.messagesReceived++;

      // Handle different message types
      switch (data.type) {
        case 'authenticate':
          this.handleAuthentication(clientId, data);
          break;
        case 'subscribe':
          this.handleSubscription(clientId, data);
          break;
        case 'unsubscribe':
          this.handleUnsubscription(clientId, data);
          break;
        case 'ping':
          this.handlePing(clientId, data);
          break;
        case 'seismic_data':
          this.handleSeismicData(clientId, data);
          break;
        case 'station_heartbeat':
          this.handleStationHeartbeat(clientId, data);
          break;
        case 'alert_action':
          this.handleAlertAction(clientId, data);
          break;
        case 'get_status':
          this.handleStatusRequest(clientId, data);
          break;
        default:
          this.sendError(clientId, 'Unknown message type');
      }

    } catch (error) {
      console.error('Message handling error:', error);
      this.sendError(clientId, 'Internal server error');
      this.connectionStats.errors++;
    }
  }

  handleAuthentication(clientId, data) {
    try {
      const client = this.clients.get(clientId);
      if (!client) return;

      if (!data.token) {
        this.sendError(clientId, 'Authentication token required');
        return;
      }

      // Verify JWT token
      const decoded = jwt.verify(data.token, process.env.JWT_SECRET || 'your-secret-key');
      
      client.authenticated = true;
      client.user = decoded;
      this.authenticatedClients.add(clientId);
      this.connectionStats.authenticatedConnections++;

      this.sendToClient(clientId, {
        type: 'auth_success',
        user: decoded,
        timestamp: new Date()
      });

      console.log(`Client ${clientId} authenticated as ${decoded.email}`);
      
      // Emit authentication event
      this.emit('client_authenticated', { clientId, user: decoded });

    } catch (error) {
      console.error('Authentication error:', error);
      this.sendError(clientId, 'Authentication failed');
    }
  }

  handleSubscription(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client || !client.authenticated) {
      this.sendError(clientId, 'Authentication required');
      return;
    }

    const { channels } = data;
    if (!Array.isArray(channels)) {
      this.sendError(clientId, 'Channels must be an array');
      return;
    }

    const validChannels = [
      'seismic_updates',
      'alert_updates',
      'station_status',
      'system_metrics',
      'emergency_broadcasts'
    ];

    channels.forEach(channel => {
      if (validChannels.includes(channel)) {
        client.subscriptions.add(channel);
        
        // Add to subscription map
        if (!this.subscriptions.has(channel)) {
          this.subscriptions.set(channel, new Set());
        }
        this.subscriptions.get(channel).add(clientId);
      }
    });

    this.sendToClient(clientId, {
      type: 'subscription_success',
      channels: Array.from(client.subscriptions),
      timestamp: new Date()
    });

    console.log(`Client ${clientId} subscribed to channels: ${channels.join(', ')}`);
  }

  handleUnsubscription(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client) return;

    const { channels } = data;
    if (!Array.isArray(channels)) {
      this.sendError(clientId, 'Channels must be an array');
      return;
    }

    channels.forEach(channel => {
      client.subscriptions.delete(channel);
      
      // Remove from subscription map
      if (this.subscriptions.has(channel)) {
        this.subscriptions.get(channel).delete(clientId);
      }
    });

    this.sendToClient(clientId, {
      type: 'unsubscription_success',
      channels: channels,
      timestamp: new Date()
    });
  }

  handlePing(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.lastPing = Date.now();
    
    this.sendToClient(clientId, {
      type: 'pong',
      timestamp: new Date(),
      serverTime: Date.now()
    });
  }

  handleSeismicData(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client || !client.authenticated) {
      this.sendError(clientId, 'Authentication required');
      return;
    }

    // Check if client has permission to send seismic data
    if (!client.user || !['admin', 'operator'].includes(client.user.role)) {
      this.sendError(clientId, 'Insufficient permissions');
      return;
    }

    // Validate seismic data
    const { payload } = data;
    if (!payload || typeof payload !== 'object') {
      this.sendError(clientId, 'Invalid seismic data payload');
      return;
    }

    // Required fields
    const requiredFields = ['magnitude', 'depth', 'latitude', 'longitude', 'timestamp'];
    const missingFields = requiredFields.filter(field => !(field in payload));
    
    if (missingFields.length > 0) {
      this.sendError(clientId, `Missing required fields: ${missingFields.join(', ')}`);
      return;
    }

    // Emit seismic data event
    this.emit('seismic_data', { clientId, payload });

    this.sendToClient(clientId, {
      type: 'seismic_data_received',
      timestamp: new Date()
    });
  }

  handleStationHeartbeat(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client || !client.authenticated) {
      this.sendError(clientId, 'Authentication required');
      return;
    }

    const { payload } = data;
    if (!payload || !payload.stationId) {
      this.sendError(clientId, 'Station ID required');
      return;
    }

    // Emit station heartbeat event
    this.emit('station_heartbeat', { clientId, payload });

    this.sendToClient(clientId, {
      type: 'heartbeat_received',
      stationId: payload.stationId,
      timestamp: new Date()
    });
  }

  handleAlertAction(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client || !client.authenticated) {
      this.sendError(clientId, 'Authentication required');
      return;
    }

    // Check permissions
    if (!client.user || !['admin', 'operator'].includes(client.user.role)) {
      this.sendError(clientId, 'Insufficient permissions');
      return;
    }

    const { alertId, action } = data;
    if (!alertId || !action) {
      this.sendError(clientId, 'Alert ID and action required');
      return;
    }

    // Emit alert action event
    this.emit('alert_action', { clientId, alertId, action, user: client.user });

    this.sendToClient(clientId, {
      type: 'alert_action_received',
      alertId,
      action,
      timestamp: new Date()
    });
  }

  handleStatusRequest(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client || !client.authenticated) {
      this.sendError(clientId, 'Authentication required');
      return;
    }

    this.sendToClient(clientId, {
      type: 'status_response',
      status: {
        clientId: clientId,
        connectedAt: client.connectedAt,
        messageCount: client.messageCount,
        subscriptions: Array.from(client.subscriptions),
        authenticated: client.authenticated,
        user: client.user,
        serverStats: this.connectionStats
      },
      timestamp: new Date()
    });
  }

  handleDisconnection(clientId, code, reason) {
    const client = this.clients.get(clientId);
    if (!client) return;

    console.log(`Client ${clientId} disconnected (code: ${code}, reason: ${reason})`);

    // Clean up subscriptions
    client.subscriptions.forEach(channel => {
      if (this.subscriptions.has(channel)) {
        this.subscriptions.get(channel).delete(clientId);
      }
    });

    // Remove from authenticated clients
    this.authenticatedClients.delete(clientId);
    
    // Remove from clients map
    this.clients.delete(clientId);
    
    // Update stats
    this.connectionStats.activeConnections--;
    if (client.authenticated) {
      this.connectionStats.authenticatedConnections--;
    }

    // Clean up message queue
    this.messageQueue.delete(clientId);

    // Emit disconnection event
    this.emit('client_disconnected', { clientId, client, code, reason });
  }

  handleError(clientId, error) {
    console.error(`WebSocket error for client ${clientId}:`, error);
    this.connectionStats.errors++;
    
    // Emit error event
    this.emit('client_error', { clientId, error });
  }

  sendToClient(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client || client.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    try {
      client.ws.send(JSON.stringify(message));
      this.connectionStats.messagesSent++;
      return true;
    } catch (error) {
      console.error('Error sending message to client:', error);
      this.connectionStats.errors++;
      return false;
    }
  }

  sendError(clientId, message) {
    this.sendToClient(clientId, {
      type: 'error',
      message: message,
      timestamp: new Date()
    });
  }

  broadcast(message, channel = null) {
    let targetClients = [];
    
    if (channel) {
      // Send to subscribed clients only
      const subscribers = this.subscriptions.get(channel);
      if (subscribers) {
        targetClients = Array.from(subscribers);
      }
    } else {
      // Send to all authenticated clients
      targetClients = Array.from(this.authenticatedClients);
    }

    const messageStr = JSON.stringify(message);
    let sentCount = 0;

    targetClients.forEach(clientId => {
      const client = this.clients.get(clientId);
      if (client && client.ws.readyState === WebSocket.OPEN) {
        try {
          client.ws.send(messageStr);
          sentCount++;
        } catch (error) {
          console.error('Broadcast error:', error);
        }
      }
    });

    this.connectionStats.messagesSent += sentCount;
    return sentCount;
  }

  broadcastToChannel(channel, message) {
    const subscribers = this.subscriptions.get(channel);
    if (!subscribers) return 0;

    const messageStr = JSON.stringify({
      ...message,
      channel: channel,
      timestamp: new Date()
    });

    let sentCount = 0;
    subscribers.forEach(clientId => {
      const client = this.clients.get(clientId);
      if (client && client.ws.readyState === WebSocket.OPEN) {
        try {
          client.ws.send(messageStr);
          sentCount++;
        } catch (error) {
          console.error('Channel broadcast error:', error);
        }
      }
    });

    this.connectionStats.messagesSent += sentCount;
    return sentCount;
  }

  startHeartbeat() {
    setInterval(() => {
      this.clients.forEach((client, clientId) => {
        if (client.ws.readyState === WebSocket.OPEN) {
          if (client.isAlive === false) {
            client.ws.terminate();
            return;
          }
          
          client.isAlive = false;
          client.ws.ping();
        }
      });
    }, this.heartbeatInterval);
  }

  cleanupRateLimiter() {
    const now = Date.now();
    const windowMs = 60000; // 1 minute
    
    this.rateLimiter.forEach((connections, ip) => {
      const recentConnections = connections.filter(time => now - time < windowMs);
      if (recentConnections.length === 0) {
        this.rateLimiter.delete(ip);
      } else {
        this.rateLimiter.set(ip, recentConnections);
      }
    });

    this.messageQueue.forEach((messages, clientId) => {
      const recentMessages = messages.filter(time => now - time < windowMs);
      if (recentMessages.length === 0) {
        this.messageQueue.delete(clientId);
      } else {
        this.messageQueue.set(clientId, recentMessages);
      }
    });
  }

  getStats() {
    return {
      ...this.connectionStats,
      clientsConnected: this.clients.size,
      authenticatedClients: this.authenticatedClients.size,
      subscriptions: Object.fromEntries(
        Array.from(this.subscriptions.entries()).map(([channel, clients]) => [
          channel,
          clients.size
        ])
      )
    };
  }

  getClientInfo(clientId) {
    const client = this.clients.get(clientId);
    if (!client) return null;

    return {
      id: clientId,
      ip: client.ip,
      connectedAt: client.connectedAt,
      authenticated: client.authenticated,
      user: client.user,
      subscriptions: Array.from(client.subscriptions),
      messageCount: client.messageCount,
      lastPing: client.lastPing,
      isAlive: client.isAlive
    };
  }

  closeConnection(clientId, code = 1000, reason = 'Server initiated close') {
    const client = this.clients.get(clientId);
    if (client && client.ws.readyState === WebSocket.OPEN) {
      client.ws.close(code, reason);
    }
  }

  shutdown() {
    console.log('Shutting down WebSocket handler...');
    
    // Close all connections
    this.clients.forEach((client, clientId) => {
      if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.close(1001, 'Server shutting down');
      }
    });

    // Close WebSocket server
    if (this.wss) {
      this.wss.close();
    }

    // Clear all data structures
    this.clients.clear();
    this.authenticatedClients.clear();
    this.subscriptions.clear();
    this.messageQueue.clear();
    this.rateLimiter.clear();

    console.log('WebSocket handler shutdown complete');
  }
}

module.exports = WebSocketHandler;