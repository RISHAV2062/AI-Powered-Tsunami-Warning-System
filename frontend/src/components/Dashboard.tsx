import React, { useState, useEffect, useCallback } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  MapPin, 
  TrendingUp, 
  Shield, 
  Bell, 
  Eye,
  Settings,
  Users,
  Database,
  Zap,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Info
} from 'lucide-react';

interface SeismicData {
  id: string;
  timestamp: Date;
  magnitude: number;
  depth: number;
  latitude: number;
  longitude: number;
  location: string;
  tsunamiRisk: 'low' | 'medium' | 'high' | 'critical';
  processed: boolean;
  alertGenerated: boolean;
}

interface AlertData {
  id: string;
  type: 'warning' | 'watch' | 'advisory' | 'info';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  location: string;
  timestamp: Date;
  estimatedArrival?: Date;
  waveHeight?: number;
  status: 'active' | 'expired' | 'cancelled';
  affectedPopulation: number;
}

interface SystemMetrics {
  uptime: number;
  processingLatency: number;
  alertsGenerated: number;
  dataPointsProcessed: number;
  modelAccuracy: number;
  falsePositiveRate: number;
  systemLoad: number;
  memoryUsage: number;
  diskUsage: number;
}

interface StationStatus {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'maintenance';
  lastHeartbeat: Date;
  dataQuality: number;
  batteryLevel: number;
  signalStrength: number;
}

const Dashboard: React.FC = () => {
  const [seismicData, setSeismicData] = useState<SeismicData[]>([]);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics>({
    uptime: 99.9,
    processingLatency: 28,
    alertsGenerated: 156,
    dataPointsProcessed: 2847563,
    modelAccuracy: 97.3,
    falsePositiveRate: 2.1,
    systemLoad: 45.2,
    memoryUsage: 67.8,
    diskUsage: 34.1
  });
  const [stations, setStations] = useState<StationStatus[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('24h');
  const [isRealTimeMode, setIsRealTimeMode] = useState(true);
  const [webSocketConnection, setWebSocketConnection] = useState<WebSocket | null>(null);

  // Initialize WebSocket connection for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8080/ws');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWebSocketConnection(ws);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'seismic_update':
          setSeismicData(prev => [data.payload, ...prev.slice(0, 99)]);
          break;
        case 'alert_update':
          setAlerts(prev => [data.payload, ...prev.slice(0, 49)]);
          break;
        case 'metrics_update':
          setMetrics(data.payload);
          break;
        case 'station_status':
          setStations(prev => {
            const updated = [...prev];
            const index = updated.findIndex(s => s.id === data.payload.id);
            if (index >= 0) {
              updated[index] = data.payload;
            } else {
              updated.push(data.payload);
            }
            return updated;
          });
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWebSocketConnection(null);
    };

    return () => {
      ws.close();
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    fetchInitialData();
  }, [selectedTimeRange]);

  const fetchInitialData = async () => {
    try {
      const [seismicResponse, alertResponse, stationResponse] = await Promise.all([
        fetch(`/api/seismic/recent?range=${selectedTimeRange}`),
        fetch(`/api/alerts/active`),
        fetch(`/api/stations/status`)
      ]);

      const seismicData = await seismicResponse.json();
      const alertData = await alertResponse.json();
      const stationData = await stationResponse.json();

      setSeismicData(seismicData);
      setAlerts(alertData);
      setStations(stationData);
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getAlertColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'border-l-green-500 bg-green-50';
      case 'medium': return 'border-l-yellow-500 bg-yellow-50';
      case 'high': return 'border-l-orange-500 bg-orange-50';
      case 'critical': return 'border-l-red-500 bg-red-50';
      default: return 'border-l-gray-500 bg-gray-50';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-600';
      case 'offline': return 'text-red-600';
      case 'maintenance': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const onlineStations = stations.filter(s => s.status === 'online').length;
  const offlineStations = stations.filter(s => s.status === 'offline').length;
  const maintenanceStations = stations.filter(s => s.status === 'maintenance').length;

  const activeAlerts = alerts.filter(a => a.status === 'active');
  const criticalAlerts = activeAlerts.filter(a => a.severity === 'critical');
  const highAlerts = activeAlerts.filter(a => a.severity === 'high');

  const recentSeismicEvents = seismicData.slice(0, 10);
  const highRiskEvents = seismicData.filter(s => s.tsunamiRisk === 'high' || s.tsunamiRisk === 'critical');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Tsunami Warning System</h1>
                <p className="text-gray-600">Real-time seismic monitoring and alert platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${webSocketConnection ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm text-gray-600">
                  {webSocketConnection ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value as any)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="1h">Last Hour</option>
                <option value="6h">Last 6 Hours</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
              </select>
              
              <button
                onClick={() => setIsRealTimeMode(!isRealTimeMode)}
                className={`px-4 py-2 rounded-md font-medium ${
                  isRealTimeMode 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {isRealTimeMode ? 'Real-time ON' : 'Real-time OFF'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {/* Critical Alerts Bar */}
        {criticalAlerts.length > 0 && (
          <div className="mb-6 bg-red-100 border border-red-300 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <h3 className="text-lg font-semibold text-red-800">Critical Alerts Active</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {criticalAlerts.map((alert) => (
                <div key={alert.id} className="bg-white rounded-lg p-3 border border-red-200">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-red-800">{alert.type.toUpperCase()}</span>
                    <span className="text-sm text-red-600">
                      {alert.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 mb-2">{alert.message}</p>
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <span>{alert.location}</span>
                    <span>{formatNumber(alert.affectedPopulation)} people affected</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Processing Latency</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.processingLatency}s</p>
                <p className="text-xs text-green-600">▼ 5% from yesterday</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <Clock className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">System Uptime</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.uptime}%</p>
                <p className="text-xs text-green-600">▲ 0.1% from last week</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Model Accuracy</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.modelAccuracy}%</p>
                <p className="text-xs text-blue-600">▲ 2.1% from last month</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Active Alerts</p>
                <p className="text-2xl font-bold text-gray-900">{activeAlerts.length}</p>
                <p className="text-xs text-yellow-600">
                  {criticalAlerts.length} critical, {highAlerts.length} high
                </p>
              </div>
              <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                <Bell className="w-6 h-6 text-yellow-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Station Status and Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Station Status */}
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900">Station Network</h3>
                <div className="flex items-center space-x-4 text-sm">
                  <span className="flex items-center">
                    <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                    Online: {onlineStations}
                  </span>
                  <span className="flex items-center">
                    <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                    Offline: {offlineStations}
                  </span>
                  <span className="flex items-center">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
                    Maintenance: {maintenanceStations}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="p-6">
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {stations.map((station) => (
                  <div key={station.id} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        station.status === 'online' ? 'bg-green-500' : 
                        station.status === 'offline' ? 'bg-red-500' : 
                        'bg-yellow-500'
                      }`}></div>
                      <div>
                        <p className="font-medium text-gray-900">{station.name}</p>
                        <p className="text-sm text-gray-600">{station.location}</p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <p className={`text-sm font-medium ${getStatusColor(station.status)}`}>
                        {station.status.toUpperCase()}
                      </p>
                      <p className="text-xs text-gray-600">
                        Quality: {station.dataQuality}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Recent Seismic Activity */}
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b">
              <h3 className="text-lg font-semibold text-gray-900">Recent Seismic Activity</h3>
            </div>
            
            <div className="p-6">
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {recentSeismicEvents.map((event) => (
                  <div key={event.id} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0">
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <div>
                        <p className="font-medium text-gray-900">
                          M{event.magnitude} - {event.location}
                        </p>
                        <p className="text-sm text-gray-600">
                          Depth: {event.depth}km • {event.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRiskColor(event.tsunamiRisk)}`}>
                        {event.tsunamiRisk.toUpperCase()}
                      </span>
                      <div className="flex items-center space-x-2 mt-1">
                        {event.processed && (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        )}
                        {event.alertGenerated && (
                          <Bell className="w-4 h-4 text-yellow-500" />
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Active Alerts */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-6 border-b">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Active Alerts</h3>
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-yellow-600" />
                <span className="text-sm text-gray-600">{activeAlerts.length} active</span>
              </div>
            </div>
          </div>
          
          <div className="p-6">
            {activeAlerts.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Shield className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No active alerts. System is operating normally.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {activeAlerts.map((alert) => (
                  <div key={alert.id} className={`border-l-4 rounded-lg p-4 ${getAlertColor(alert.severity)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className="w-5 h-5 text-gray-600" />
                        <span className="font-semibold text-gray-900">
                          {alert.type.toUpperCase()} - {alert.severity.toUpperCase()}
                        </span>
                      </div>
                      <span className="text-sm text-gray-600">
                        {alert.timestamp.toLocaleString()}
                      </span>
                    </div>
                    
                    <p className="text-gray-700 mb-3">{alert.message}</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Location:</span>
                        <p className="font-medium">{alert.location}</p>
                      </div>
                      <div>
                        <span className="text-gray-600">Affected Population:</span>
                        <p className="font-medium">{formatNumber(alert.affectedPopulation)}</p>
                      </div>
                      {alert.estimatedArrival && (
                        <div>
                          <span className="text-gray-600">Est. Arrival:</span>
                          <p className="font-medium">{alert.estimatedArrival.toLocaleTimeString()}</p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* System Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">System Load</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>CPU Usage</span>
                  <span>{metrics.systemLoad}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.systemLoad}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Memory Usage</span>
                  <span>{metrics.memoryUsage}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.memoryUsage}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Disk Usage</span>
                  <span>{metrics.diskUsage}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.diskUsage}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Processing Stats</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Data Points Processed</span>
                <span className="font-semibold">{formatNumber(metrics.dataPointsProcessed)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Alerts Generated</span>
                <span className="font-semibold">{metrics.alertsGenerated}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">False Positive Rate</span>
                <span className="font-semibold">{metrics.falsePositiveRate}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">High Risk Events</span>
                <span className="font-semibold">{highRiskEvents.length}</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors">
                Generate Test Alert
              </button>
              <button className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors">
                Run Diagnostics
              </button>
              <button className="w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors">
                Export Data
              </button>
              <button className="w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors">
                System Settings
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;