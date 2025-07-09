import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  AlertTriangle, 
  Bell, 
  Volume2, 
  VolumeX, 
  MapPin, 
  Clock, 
  Users, 
  Waves,
  Phone,
  Mail,
  MessageSquare,
  Shield,
  X,
  CheckCircle,
  AlertCircle,
  Info,
  ExternalLink,
  Settings,
  Download,
  Share2,
  Calendar,
  Filter,
  Search,
  Zap,
  Eye,
  EyeOff,
  RefreshCw,
  Pause,
  Play
} from 'lucide-react';

interface Alert {
  id: string;
  type: 'warning' | 'watch' | 'advisory' | 'info' | 'test';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  location: string;
  coordinates: {
    lat: number;
    lng: number;
  };
  timestamp: Date;
  estimatedArrival?: Date;
  waveHeight?: number;
  evacuationZones: string[];
  affectedPopulation: number;
  status: 'active' | 'expired' | 'cancelled' | 'testing';
  priority: number;
  source: string;
  confidence: number;
  instructions: string[];
  emergencyContacts: {
    police: string;
    fire: string;
    medical: string;
    emergency: string;
  };
  shelters: Array<{
    name: string;
    address: string;
    capacity: number;
    distance: number;
  }>;
  communicationChannels: Array<{
    type: 'sms' | 'email' | 'push' | 'radio' | 'siren';
    status: 'sent' | 'pending' | 'failed';
    recipients: number;
    timestamp: Date;
  }>;
  metadata: {
    triggeredBy: string;
    processingTime: number;
    dataQuality: number;
    modelVersion: string;
  };
}

interface AlertSystemProps {
  onAlertAction?: (alertId: string, action: string) => void;
  onAlertFilter?: (filters: any) => void;
  userRole?: 'admin' | 'operator' | 'viewer';
}

const AlertSystem: React.FC<AlertSystemProps> = ({
  onAlertAction,
  onAlertFilter,
  userRole = 'viewer'
}) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filteredAlerts, setFilteredAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(false);
  const [filters, setFilters] = useState({
    severity: 'all',
    type: 'all',
    status: 'all',
    location: '',
    dateRange: '24h'
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [notificationQueue, setNotificationQueue] = useState<Alert[]>([]);
  const [selectedAlerts, setSelectedAlerts] = useState<string[]>([]);
  const [bulkActions, setBulkActions] = useState(false);
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8080/alerts');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Alert WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'new_alert') {
        handleNewAlert(data.alert);
      } else if (data.type === 'alert_update') {
        handleAlertUpdate(data.alert);
      } else if (data.type === 'alert_cancelled') {
        handleAlertCancellation(data.alertId);
      }
    };

    ws.onerror = (error) => {
      console.error('Alert WebSocket error:', error);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  // Auto-refresh functionality
  useEffect(() => {
    if (autoRefresh) {
      refreshIntervalRef.current = setInterval(() => {
        fetchAlerts();
      }, 30000); // Refresh every 30 seconds
    } else {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [autoRefresh]);

  // Filter alerts based on current filters
  useEffect(() => {
    let filtered = alerts;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(alert => 
        alert.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.location.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply severity filter
    if (filters.severity !== 'all') {
      filtered = filtered.filter(alert => alert.severity === filters.severity);
    }

    // Apply type filter
    if (filters.type !== 'all') {
      filtered = filtered.filter(alert => alert.type === filters.type);
    }

    // Apply status filter
    if (filters.status !== 'all') {
      filtered = filtered.filter(alert => alert.status === filters.status);
    }

    // Apply location filter
    if (filters.location) {
      filtered = filtered.filter(alert => 
        alert.location.toLowerCase().includes(filters.location.toLowerCase())
      );
    }

    // Apply date range filter
    const now = new Date();
    const ranges = {
      '1h': 1 * 60 * 60 * 1000,
      '6h': 6 * 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000
    };

    if (filters.dateRange !== 'all') {
      const range = ranges[filters.dateRange as keyof typeof ranges];
      if (range) {
        filtered = filtered.filter(alert => 
          now.getTime() - alert.timestamp.getTime() <= range
        );
      }
    }

    // Sort by priority and timestamp
    filtered.sort((a, b) => {
      if (a.priority !== b.priority) {
        return b.priority - a.priority;
      }
      return b.timestamp.getTime() - a.timestamp.getTime();
    });

    setFilteredAlerts(filtered);
  }, [alerts, filters, searchQuery]);

  const fetchAlerts = async () => {
    try {
      const response = await fetch('/api/alerts');
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  const handleNewAlert = (alert: Alert) => {
    setAlerts(prev => [alert, ...prev]);
    setNotificationQueue(prev => [...prev, alert]);
    
    if (soundEnabled && !isMuted && alert.severity !== 'low') {
      playAlertSound(alert.severity);
    }
  };

  const handleAlertUpdate = (updatedAlert: Alert) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === updatedAlert.id ? updatedAlert : alert
    ));
  };

  const handleAlertCancellation = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, status: 'cancelled' } : alert
    ));
  };

  const playAlertSound = (severity: string) => {
    if (audioRef.current) {
      // Different sounds for different severities
      const soundMap = {
        'critical': '/sounds/critical-alert.mp3',
        'high': '/sounds/high-alert.mp3',
        'medium': '/sounds/medium-alert.mp3',
        'low': '/sounds/low-alert.mp3'
      };
      
      audioRef.current.src = soundMap[severity as keyof typeof soundMap] || soundMap.medium;
      audioRef.current.play().catch(console.error);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'expired': return 'bg-gray-100 text-gray-800';
      case 'cancelled': return 'bg-red-100 text-red-800';
      case 'testing': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'warning': return <AlertTriangle className="w-4 h-4" />;
      case 'watch': return <Eye className="w-4 h-4" />;
      case 'advisory': return <Info className="w-4 h-4" />;
      case 'info': return <Info className="w-4 h-4" />;
      case 'test': return <Settings className="w-4 h-4" />;
      default: return <Bell className="w-4 h-4" />;
    }
  };

  const handleAlertAction = async (alertId: string, action: string) => {
    if (onAlertAction) {
      onAlertAction(alertId, action);
    }

    try {
      const response = await fetch(`/api/alerts/${alertId}/actions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action }),
      });

      if (response.ok) {
        const updatedAlert = await response.json();
        handleAlertUpdate(updatedAlert);
      }
    } catch (error) {
      console.error('Error performing alert action:', error);
    }
  };

  const handleBulkAction = async (action: string) => {
    if (selectedAlerts.length === 0) return;

    try {
      const response = await fetch('/api/alerts/bulk-actions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          alertIds: selectedAlerts, 
          action 
        }),
      });

      if (response.ok) {
        const updatedAlerts = await response.json();
        updatedAlerts.forEach(handleAlertUpdate);
        setSelectedAlerts([]);
        setBulkActions(false);
      }
    } catch (error) {
      console.error('Error performing bulk action:', error);
    }
  };

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
  };

  const exportAlerts = () => {
    const csv = filteredAlerts.map(alert => ({
      id: alert.id,
      type: alert.type,
      severity: alert.severity,
      title: alert.title,
      location: alert.location,
      timestamp: alert.timestamp.toISOString(),
      status: alert.status,
      affectedPopulation: alert.affectedPopulation
    }));

    const csvContent = "data:text/csv;charset=utf-8," + 
      Object.keys(csv[0]).join(",") + "\n" +
      csv.map(row => Object.values(row).join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `alerts_${new Date().toISOString().split('T')[0]}.csv`);
    link.click();
  };

  const activeAlertsCount = filteredAlerts.filter(a => a.status === 'active').length;
  const criticalAlertsCount = filteredAlerts.filter(a => a.severity === 'critical').length;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Audio element for alert sounds */}
      <audio ref={audioRef} preload="auto" />
      
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-red-600 rounded-lg flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Alert System</h1>
                  <p className="text-gray-600">
                    {activeAlertsCount} active alerts
                    {criticalAlertsCount > 0 && (
                      <span className="ml-2 text-red-600 font-semibold">
                        ({criticalAlertsCount} critical)
                      </span>
                    )}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className={`p-2 rounded-lg ${isMuted ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-600'}`}
                >
                  {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                </button>
                
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`p-2 rounded-lg ${autoRefresh ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-600'}`}
                >
                  {autoRefresh ? <RefreshCw className="w-5 h-5" /> : <Pause className="w-5 h-5" />}
                </button>
                
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="p-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200"
                >
                  <Filter className="w-5 h-5" />
                </button>
                
                <button
                  onClick={exportAlerts}
                  className="p-2 rounded-lg bg-blue-100 text-blue-600 hover:bg-blue-200"
                >
                  <Download className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="bg-white border-b p-4">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Severity</label>
              <select
                value={filters.severity}
                onChange={(e) => setFilters({...filters, severity: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
              <select
                value={filters.type}
                onChange={(e) => setFilters({...filters, type: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Types</option>
                <option value="warning">Warning</option>
                <option value="watch">Watch</option>
                <option value="advisory">Advisory</option>
                <option value="info">Info</option>
                <option value="test">Test</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
              <select
                value={filters.status}
                onChange={(e) => setFilters({...filters, status: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="expired">Expired</option>
                <option value="cancelled">Cancelled</option>
                <option value="testing">Testing</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Time Range</label>
              <select
                value={filters.dateRange}
                onChange={(e) => setFilters({...filters, dateRange: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="1h">Last Hour</option>
                <option value="6h">Last 6 Hours</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="all">All Time</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
              <div className="relative">
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search alerts..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bulk Actions */}
      {userRole === 'admin' && (
        <div className="bg-white border-b p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setBulkActions(!bulkActions)}
                className={`px-4 py-2 rounded-lg font-medium ${
                  bulkActions 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {bulkActions ? 'Exit Bulk Mode' : 'Bulk Actions'}
              </button>
              
              {bulkActions && selectedAlerts.length > 0 && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">
                    {selectedAlerts.length} selected
                  </span>
                  <button
                    onClick={() => handleBulkAction('acknowledge')}
                    className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                  >
                    Acknowledge
                  </button>
                  <button
                    onClick={() => handleBulkAction('cancel')}
                    className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="p-6">
        {/* Alert List */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900">
              Alert History ({filteredAlerts.length})
            </h3>
          </div>
          
          <div className="divide-y divide-gray-200">
            {filteredAlerts.length === 0 ? (
              <div className="text-center py-12">
                <Bell className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No alerts match your current filters</p>
              </div>
            ) : (
              filteredAlerts.map((alert) => (
                <div key={alert.id} className="p-6 hover:bg-gray-50">
                  <div className="flex items-start space-x-4">
                    {bulkActions && (
                      <input
                        type="checkbox"
                        checked={selectedAlerts.includes(alert.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedAlerts([...selectedAlerts, alert.id]);
                          } else {
                            setSelectedAlerts(selectedAlerts.filter(id => id !== alert.id));
                          }
                        }}
                        className="mt-1 w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                      />
                    )}
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-3">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getSeverityColor(alert.severity)}`}>
                            {getTypeIcon(alert.type)}
                            <span className="ml-1">{alert.severity.toUpperCase()}</span>
                          </span>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(alert.status)}`}>
                            {alert.status.toUpperCase()}
                          </span>
                          <span className="text-sm text-gray-500">
                            {formatTimeAgo(alert.timestamp)}
                          </span>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => setSelectedAlert(alert)}
                            className="p-1 text-gray-400 hover:text-gray-600"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          
                          {userRole === 'admin' && alert.status === 'active' && (
                            <button
                              onClick={() => handleAlertAction(alert.id, 'cancel')}
                              className="p-1 text-red-400 hover:text-red-600"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </div>
                      
                      <h4 className="text-lg font-medium text-gray-900 mb-2">{alert.title}</h4>
                      <p className="text-gray-600 mb-3">{alert.message}</p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div className="flex items-center space-x-2">
                          <MapPin className="w-4 h-4 text-gray-400" />
                          <span>{alert.location}</span>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <Users className="w-4 h-4 text-gray-400" />
                          <span>{alert.affectedPopulation.toLocaleString()} affected</span>
                        </div>
                        
                        {alert.estimatedArrival && (
                          <div className="flex items-center space-x-2">
                            <Clock className="w-4 h-4 text-gray-400" />
                            <span>ETA: {alert.estimatedArrival.toLocaleTimeString()}</span>
                          </div>
                        )}
                      </div>
                      
                      {alert.waveHeight && (
                        <div className="mt-2 flex items-center space-x-2 text-sm">
                          <Waves className="w-4 h-4 text-blue-400" />
                          <span>Wave Height: {alert.waveHeight}m</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-900">{selectedAlert.title}</h2>
                <button
                  onClick={() => setSelectedAlert(null)}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
            
            <div className="p-6 space-y-6">
              {/* Alert Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Alert Information</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">ID:</span>
                      <span className="font-mono">{selectedAlert.id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Type:</span>
                      <span className="capitalize">{selectedAlert.type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Severity:</span>
                      <span className={`px-2 py-1 rounded text-xs ${getSeverityColor(selectedAlert.severity)}`}>
                        {selectedAlert.severity.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Status:</span>
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(selectedAlert.status)}`}>
                        {selectedAlert.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Confidence:</span>
                      <span>{selectedAlert.confidence}%</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Location & Impact</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Location:</span>
                      <span>{selectedAlert.location}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Coordinates:</span>
                      <span>{selectedAlert.coordinates.lat}, {selectedAlert.coordinates.lng}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Affected Population:</span>
                      <span>{selectedAlert.affectedPopulation.toLocaleString()}</span>
                    </div>
                    {selectedAlert.waveHeight && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">Wave Height:</span>
                        <span>{selectedAlert.waveHeight}m</span>
                      </div>
                    )}
                    {selectedAlert.estimatedArrival && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">Est. Arrival:</span>
                        <span>{selectedAlert.estimatedArrival.toLocaleString()}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Message */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Alert Message</h3>
                <p className="text-gray-700 bg-gray-50 p-4 rounded-lg">{selectedAlert.message}</p>
              </div>
              
              {/* Instructions */}
              {selectedAlert.instructions.length > 0 && (
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Emergency Instructions</h3>
                  <ul className="space-y-2">
                    {selectedAlert.instructions.map((instruction, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <span className="text-blue-600 font-bold">{index + 1}.</span>
                        <span className="text-gray-700">{instruction}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Evacuation Zones */}
              {selectedAlert.evacuationZones.length > 0 && (
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Evacuation Zones</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedAlert.evacuationZones.map((zone, index) => (
                      <span key={index} className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm">
                        {zone}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Emergency Contacts */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Emergency Contacts</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="flex items-center space-x-2 mb-1">
                      <Phone className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium">Emergency</span>
                    </div>
                    <p className="text-sm text-gray-700">{selectedAlert.emergencyContacts.emergency}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="flex items-center space-x-2 mb-1">
                      <Shield className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium">Police</span>
                    </div>
                    <p className="text-sm text-gray-700">{selectedAlert.emergencyContacts.police}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="flex items-center space-x-2 mb-1">
                      <Zap className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium">Fire</span>
                    </div>
                    <p className="text-sm text-gray-700">{selectedAlert.emergencyContacts.fire}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="flex items-center space-x-2 mb-1">
                      <AlertCircle className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium">Medical</span>
                    </div>
                    <p className="text-sm text-gray-700">{selectedAlert.emergencyContacts.medical}</p>
                  </div>
                </div>
              </div>
              
              {/* Shelters */}
              {selectedAlert.shelters.length > 0 && (
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Emergency Shelters</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {selectedAlert.shelters.map((shelter, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-2">{shelter.name}</h4>
                        <p className="text-sm text-gray-600 mb-2">{shelter.address}</p>
                        <div className="flex justify-between text-sm">
                          <span>Capacity: {shelter.capacity}</span>
                          <span>Distance: {shelter.distance}km</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Communication Status */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Communication Status</h3>
                <div className="space-y-2">
                  {selectedAlert.communicationChannels.map((channel, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="flex items-center space-x-2">
                          {channel.type === 'sms' && <MessageSquare className="w-4 h-4" />}
                          {channel.type === 'email' && <Mail className="w-4 h-4" />}
                          {channel.type === 'push' && <Bell className="w-4 h-4" />}
                          {channel.type === 'radio' && <Volume2 className="w-4 h-4" />}
                          {channel.type === 'siren' && <AlertTriangle className="w-4 h-4" />}
                          <span className="text-sm font-medium capitalize">{channel.type}</span>
                        </div>
                        <span className="text-sm text-gray-600">
                          {channel.recipients.toLocaleString()} recipients
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 rounded text-xs ${
                          channel.status === 'sent' ? 'bg-green-100 text-green-800' :
                          channel.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {channel.status.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-500">
                          {channel.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertSystem;