import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  Animated, Alert, Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card, OutcomeChip, ConfBadge } from '../components';
import { saveShot } from '../utils/storage';

// Shot zones matching your 9-zone Ballogy layout
const ZONES = [
  'Left corner 3', 'Top of key 3', 'Right corner 3',
  'Left wing', 'Paint', 'Right wing',
  'Left elbow', 'Mid-range', 'Right elbow',
];

// ── Prediction banner ────────────────────────────────────────────────────────
function PredictionBanner({ prediction }) {
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (prediction) {
      Animated.timing(fadeAnim, { toValue: 1, duration: 250, useNativeDriver: true }).start();
    } else {
      fadeAnim.setValue(0);
    }
  }, [prediction]);

  if (!prediction) {
    return (
      <View style={[styles.predBanner, { opacity: 0.5 }]}>
        <View style={styles.predIcon}>
          <Ionicons name="help" size={20} color={colors.textGhost} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.predLabel}>Prediction</Text>
          <Text style={styles.predValue}>Waiting…</Text>
        </View>
      </View>
    );
  }

  const isMade = prediction.outcome === 'made';
  const borderColor = isMade ? colors.greenBorder : colors.orangeBorder;
  const bgColor     = isMade ? 'rgba(76,175,125,0.06)' : 'rgba(232,93,38,0.05)';

  return (
    <Animated.View style={[styles.predBanner, { borderColor, backgroundColor: bgColor, opacity: fadeAnim }]}>
      <View style={[styles.predIcon, { backgroundColor: isMade ? colors.greenFade : colors.orangeFade }]}>
        <Ionicons name={isMade ? 'checkmark' : 'close'} size={20} color={isMade ? colors.green : colors.orange} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.predLabel}>Prediction · {prediction.zone}</Text>
        <Text style={[styles.predValue, { color: isMade ? colors.green : colors.orange }]}>
          {isMade ? 'MADE' : 'MISSED'}
        </Text>
      </View>
      <ConfBadge conf={prediction.confidence} />
    </Animated.View>
  );
}

// ── Feedback buttons ─────────────────────────────────────────────────────────
function FeedbackRow({ active, onFeedback }) {
  return (
    <View style={[styles.fbRow, { opacity: active ? 1 : 0.3 }]}>
      <TouchableOpacity
        style={[styles.fbBtn, active && { borderColor: colors.greenBorder }]}
        onPress={() => active && onFeedback('made')}
        activeOpacity={0.8}
      >
        <Ionicons name="checkmark" size={16} color={active ? colors.green : colors.textGhost} />
        <Text style={[styles.fbLabel, active && { color: colors.green }]}>Made (M)</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.fbBtn, active && { borderColor: colors.orangeBorder }]}
        onPress={() => active && onFeedback('missed')}
        activeOpacity={0.8}
      >
        <Ionicons name="close" size={16} color={active ? colors.orange : colors.textGhost} />
        <Text style={[styles.fbLabel, active && { color: colors.orange }]}>Miss (X)</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.fbBtn}
        onPress={() => active && onFeedback('skip')}
        activeOpacity={0.8}
      >
        <Ionicons name="arrow-forward" size={16} color={colors.textGhost} />
      </TouchableOpacity>
    </View>
  );
}

// ── Record button ────────────────────────────────────────────────────────────
function RecordButton({ recording, onPress }) {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (recording) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(scaleAnim, { toValue: 1.06, duration: 800, useNativeDriver: true }),
          Animated.timing(scaleAnim, { toValue: 1,    duration: 800, useNativeDriver: true }),
        ])
      ).start();
    } else {
      scaleAnim.stopAnimation();
      scaleAnim.setValue(1);
    }
  }, [recording]);

  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.85}>
      <Animated.View style={[styles.recOuter, { transform: [{ scale: scaleAnim }] }]}>
        <View style={[styles.recInner, recording && styles.recInnerStop]} />
      </Animated.View>
    </TouchableOpacity>
  );
}

// ── Main screen ──────────────────────────────────────────────────────────────
export default function RecordScreen() {
  const insets = useSafeAreaInsets();
  const [permission, requestPermission] = useCameraPermissions();
  const [recording, setRecording]       = useState(false);
  const [prediction, setPrediction]     = useState(null);
  const [fbActive, setFbActive]         = useState(false);
  const [liveTotal, setLiveTotal]       = useState(0);
  const [liveMade, setLiveMade]         = useState(0);
  const [status, setStatus]             = useState('Waiting for shot…');
  const simTimer = useRef(null);
  const shotIdx  = useRef(0);

  // Demo shots — replace this section with your real YOLO+LSTM output
  const DEMO_SHOTS = [
    { outcome: 'made',   confidence: 82, zone: 'Mid-range' },
    { outcome: 'missed', confidence: 61, zone: 'Left corner 3' },
    { outcome: 'made',   confidence: 91, zone: 'Paint' },
    { outcome: 'made',   confidence: 73, zone: 'Top of key 3' },
    { outcome: 'missed', confidence: 55, zone: 'Left elbow' },
    { outcome: 'made',   confidence: 88, zone: 'Right wing' },
  ];

  useFocusEffect(useCallback(() => {
    return () => {
      stopSession();
    };
  }, []));

  const stopSession = () => {
    if (simTimer.current) clearInterval(simTimer.current);
    setRecording(false);
    setPrediction(null);
    setFbActive(false);
    setStatus('Waiting for shot…');
  };

  const toggleRecording = async () => {
    if (!permission?.granted) {
      const { granted } = await requestPermission();
      if (!granted) {
        Alert.alert('Camera required', 'Allow camera access in Settings to record sessions.');
        return;
      }
    }

    if (recording) {
      stopSession();
    } else {
      setRecording(true);
      setLiveTotal(0);
      setLiveMade(0);
      shotIdx.current = 0;
      setStatus('Tracking…');
      // Simulate shots every 3.5s.
      // Replace simTimer with your real inference callback from lstm_infer.py output.
      simTimer.current = setInterval(simulateShot, 3500);
    }
  };

  const simulateShot = () => {
    const shot = DEMO_SHOTS[shotIdx.current % DEMO_SHOTS.length];
    shotIdx.current++;

    setStatus('Ball detected');
    setTimeout(() => setStatus('Running model…'), 800);
    setTimeout(() => {
      setPrediction(shot);
      setFbActive(true);
      setLiveTotal(t => t + 1);
      if (shot.outcome === 'made') setLiveMade(m => m + 1);
      setStatus(shot.outcome === 'made' ? '✓ Predicted: MADE' : '✕ Predicted: MISSED');
    }, 1600);
  };

  const handleFeedback = async (type) => {
    setFbActive(false);
    if (!prediction || type === 'skip') return;

    // Save to local storage — the actual label that gets written to feedback_labels.csv
    // on the backend should match this structure
    await saveShot({
      outcome:      type,
      predicted:    prediction.outcome,
      confidence:   prediction.confidence,
      zone:         prediction.zone,
      correct:      type === prediction.outcome,
    });

    setStatus(type === 'made' ? '✓ Labeled: made' : '✕ Labeled: missed');
    setTimeout(() => setStatus(recording ? 'Tracking…' : 'Waiting for shot…'), 1200);
  };

  const livePct = liveTotal > 0 ? Math.round((liveMade / liveTotal) * 100) : null;

  if (!permission) return <View style={styles.container} />;

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Live session</Text>
        {recording && (
          <View style={styles.recIndicator}>
            <View style={styles.recDot} />
            <Text style={styles.recLabel}>REC</Text>
          </View>
        )}
      </View>

      {/* Viewfinder */}
      <View style={styles.viewfinder}>
        {permission?.granted ? (
          <CameraView style={StyleSheet.absoluteFill} facing="back" />
        ) : (
          <View style={styles.noCam}>
            <Ionicons name="camera-outline" size={48} color={colors.textGhost} />
            <Text style={styles.noCamText}>Camera permission needed</Text>
            <TouchableOpacity onPress={requestPermission} style={styles.permBtn}>
              <Text style={styles.permBtnText}>Allow camera</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Status overlay */}
        <View style={[
          styles.statusBadge,
          recording && (status.includes('detected') || status.includes('MADE') || status.includes('MISSED'))
            && styles.statusBadgeActive,
        ]}>
          <Text style={styles.statusText}>{status}</Text>
        </View>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        {/* Live stats */}
        <View style={styles.liveRow}>
          <Card style={styles.liveCard}>
            <Text style={styles.liveVal}>{liveTotal}</Text>
            <Text style={styles.liveLbl}>shots</Text>
          </Card>
          <Card style={styles.liveCard}>
            <Text style={[styles.liveVal, { color: colors.green }]}>{liveMade}</Text>
            <Text style={styles.liveLbl}>made</Text>
          </Card>
          <Card style={styles.liveCard}>
            <Text style={[styles.liveVal, livePct != null && { color: livePct >= 50 ? colors.green : colors.orange }]}>
              {livePct != null ? `${livePct}%` : '—'}
            </Text>
            <Text style={styles.liveLbl}>pct</Text>
          </Card>
        </View>

        {/* Prediction */}
        <PredictionBanner prediction={prediction} />

        {/* Feedback */}
        <FeedbackRow active={fbActive} onFeedback={handleFeedback} />

        {/* Record button */}
        <View style={styles.recRow}>
          <Text style={styles.recHint}>{recording ? 'tap to stop' : 'tap to start'}</Text>
          <RecordButton recording={recording} onPress={toggleRecording} />
          <Text style={styles.recHint}>{liveTotal > 0 ? `${liveTotal} shots` : ''}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.md,
    borderBottomWidth: 0.5,
    borderBottomColor: '#1e1e1e',
  },
  title: { flex: 1, fontSize: font.md, fontWeight: '500', color: colors.white },
  recIndicator: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  recDot: {
    width: 7, height: 7,
    borderRadius: 3.5,
    backgroundColor: colors.orange,
  },
  recLabel: { fontSize: font.sm, color: colors.orange },

  viewfinder: {
    height: 230,
    backgroundColor: '#050505',
    position: 'relative',
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  noCam: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md },
  noCamText: { fontSize: font.base, color: colors.textMuted },
  permBtn: {
    backgroundColor: colors.orange,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
  },
  permBtnText: { color: colors.white, fontSize: font.base, fontWeight: '500' },

  statusBadge: {
    position: 'absolute',
    bottom: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderWidth: 0.5,
    borderColor: '#333',
    borderRadius: radius.full,
    paddingHorizontal: spacing.md,
    paddingVertical: 4,
  },
  statusBadgeActive: {
    borderColor: colors.orangeBorder,
  },
  statusText: { fontSize: font.sm, color: colors.textMuted },

  controls: {
    flex: 1,
    padding: spacing.lg,
    gap: spacing.md,
  },

  liveRow: { flexDirection: 'row', gap: spacing.sm },
  liveCard: {
    flex: 1,
    padding: spacing.md,
    alignItems: 'center',
  },
  liveVal: { fontSize: 20, fontWeight: '500', color: colors.white },
  liveLbl: { fontSize: font.xs, color: colors.textDim, marginTop: 2 },

  predBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.bgCard,
    borderRadius: radius.lg,
    padding: spacing.md,
    borderWidth: 0.5,
    borderColor: colors.border,
  },
  predIcon: {
    width: 38, height: 38,
    borderRadius: 19,
    backgroundColor: '#1e1e1e',
    alignItems: 'center',
    justifyContent: 'center',
  },
  predLabel: { fontSize: font.sm, color: colors.textDim, marginBottom: 2 },
  predValue: { fontSize: font.md, fontWeight: '500', color: colors.textMuted },

  fbRow: { flexDirection: 'row', gap: spacing.sm },
  fbBtn: {
    flex: 1,
    backgroundColor: colors.bgCard,
    borderWidth: 0.5,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingVertical: spacing.sm,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
  },
  fbLabel: { fontSize: font.sm, color: colors.textGhost },

  recRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xl,
    paddingTop: spacing.sm,
  },
  recHint: { fontSize: font.xs, color: colors.textGhost, width: 56, textAlign: 'center' },
  recOuter: {
    width: 64, height: 64,
    borderRadius: 32,
    borderWidth: 3,
    borderColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recInner: {
    width: 48, height: 48,
    borderRadius: 24,
    backgroundColor: colors.orange,
  },
  recInnerStop: {
    width: 22, height: 22,
    borderRadius: 4,
  },
});
