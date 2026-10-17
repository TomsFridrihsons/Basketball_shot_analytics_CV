import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, Animated, Alert, FlatList, ActivityIndicator,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card } from '../components';
import { saveSession, labelShot } from '../utils/storage';
import * as api from '../services/api';

// Flow: idle → recording → ready (have a clip) → uploading → analyzing → results
// The video is uploaded in original quality; only results come back.

const fmtTime = (s) => (s == null ? '' : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);
const pct = (x) => `${Math.round(x * 100)}%`;

// ── Record button ────────────────────────────────────────────────────────────
function RecordButton({ recording, disabled, onPress }) {
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
    <TouchableOpacity onPress={onPress} activeOpacity={0.85} disabled={disabled}
      accessibilityLabel={recording ? 'Stop recording' : 'Start recording'}>
      <Animated.View style={[styles.recOuter, disabled && { opacity: 0.35 }, { transform: [{ scale: scaleAnim }] }]}>
        <View style={[styles.recInner, recording && styles.recInnerStop]} />
      </Animated.View>
    </TouchableOpacity>
  );
}

// ── Progress bar ─────────────────────────────────────────────────────────────
function Progress({ label, value }) {
  return (
    <Card style={styles.progressCard}>
      <View style={styles.progressTop}>
        <ActivityIndicator size="small" color={colors.orange} />
        <Text style={styles.progressLabel}>{label}</Text>
        <Text style={styles.progressPct}>{value != null ? pct(value) : ''}</Text>
      </View>
      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${Math.round((value || 0) * 100)}%` }]} />
      </View>
    </Card>
  );
}

// ── Shot row with Made / Missed labeling ─────────────────────────────────────
function ShotRow({ shot, fps, onLabel }) {
  const made = shot.prediction === 1;
  const certainty = Math.max(shot.probability, 1 - shot.probability);
  const verdict = shot.label == null ? null : shot.label === shot.prediction;
  return (
    <View style={styles.shotRow}>
      <View style={styles.shotTop}>
        <Text style={styles.shotNum}>#{shot.shot_number}</Text>
        <Text style={styles.shotTime}>{fmtTime((shot.start_frame - 1) / fps)}</Text>
        <View style={[styles.chip, made ? styles.chipMade : styles.chipMissed]}>
          <Text style={[styles.chipText, { color: made ? colors.green : colors.orange }]}>
            {made ? 'MADE' : 'MISSED'}
          </Text>
        </View>
        <Text style={styles.shotConf}>{pct(certainty)}</Text>
      </View>
      <View style={styles.fbRow}>
        <TouchableOpacity
          style={[styles.fbBtn, shot.label === 1 && styles.fbMadeOn]}
          onPress={() => onLabel(shot, 1)} activeOpacity={0.8}
        >
          <Ionicons name="checkmark" size={15} color={shot.label === 1 ? colors.green : colors.textMuted} />
          <Text style={[styles.fbLabel, shot.label === 1 && { color: colors.green }]}>Made</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.fbBtn, shot.label === 0 && styles.fbMissOn]}
          onPress={() => onLabel(shot, 0)} activeOpacity={0.8}
        >
          <Ionicons name="close" size={15} color={shot.label === 0 ? colors.orange : colors.textMuted} />
          <Text style={[styles.fbLabel, shot.label === 0 && { color: colors.orange }]}>Missed</Text>
        </TouchableOpacity>
        <Text style={[styles.verdict, { color: verdict ? colors.green : colors.yellow }]}>
          {verdict == null ? '' : verdict ? '✓ model right' : '✗ model wrong'}
        </Text>
      </View>
    </View>
  );
}

// ── Main screen ──────────────────────────────────────────────────────────────
export default function RecordScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const cancelled = useRef(false);

  const [stage, setStage]       = useState('idle');
  const [clip, setClip]         = useState(null);     // { uri, durationS, source }
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [result, setResult]     = useState(null);
  const [jobId, setJobId]       = useState(null);
  const [server, setServer]     = useState(null);     // null = checking, true/false
  const [elapsed, setElapsed]   = useState(0);
  const stageRef = useRef(stage);
  useEffect(() => { stageRef.current = stage; }, [stage]);

  const checkServer = useCallback(async () => {
    setServer(null);
    try { await api.health(); setServer(true); } catch (_) { setServer(false); }
  }, []);

  useFocusEffect(useCallback(() => {
    cancelled.current = false;
    checkServer();
    return () => {
      cancelled.current = true;
      if (cameraRef.current && stageRef.current === 'recording') cameraRef.current.stopRecording();
    };
  }, [checkServer]));

  // recording timer
  useEffect(() => {
    if (stage !== 'recording') return undefined;
    const started = Date.now();
    const t = setInterval(() => setElapsed((Date.now() - started) / 1000), 500);
    return () => clearInterval(t);
  }, [stage]);

  const reset = () => {
    setStage('idle');
    setClip(null);
    setResult(null);
    setJobId(null);
    setProgress(0);
  };

  const toggleRecording = async () => {
    if (!permission?.granted) {
      const { granted } = await requestPermission();
      if (!granted) {
        Alert.alert('Camera required', 'Allow camera access in Settings to record sessions.');
        return;
      }
    }
    if (stage === 'recording') {
      cameraRef.current?.stopRecording();   // recordAsync below resolves with the file
      return;
    }
    try {
      setElapsed(0);
      setStage('recording');
      const video = await cameraRef.current.recordAsync({ maxDuration: 60 * 60 });
      if (video?.uri) {
        setClip({ uri: video.uri, durationS: null, source: 'camera' });
        setStage('ready');
      } else {
        setStage('idle');
      }
    } catch (e) {
      setStage('idle');
      Alert.alert('Recording failed', e.message);
    }
  };

  const pickVideo = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      // keep the original file (no re-encode on iOS)
      videoExportPreset: ImagePicker.VideoExportPreset.Passthrough,
    });
    if (res.canceled || !res.assets?.length) return;
    const a = res.assets[0];
    setClip({ uri: a.uri, durationS: a.duration ? a.duration / 1000 : null, source: 'library',
              sizeMb: a.fileSize ? a.fileSize / 1e6 : null });
    setStage('ready');
  };

  const runAnalysis = async () => {
    try {
      setStage('uploading');
      setProgress(0);
      setProgressText('Uploading video…');
      const { name } = await api.uploadVideo(clip.uri, setProgress);
      if (cancelled.current) return;

      setStage('analyzing');
      setProgress(0);
      setProgressText('Waiting for the laptop…');
      const { job_id } = await api.analyze(name);
      setJobId(job_id);
      const res = await api.waitForJob(job_id, (p, st) => {
        setProgress(p);
        setProgressText(st === 'analyzing' ? 'Detecting shots…' : `${st[0].toUpperCase()}${st.slice(1)}…`);
      }, () => cancelled.current);

      setResult(res);
      setStage('results');
      await saveSession(job_id, res);
    } catch (e) {
      if (e.message === 'cancelled') return;
      setStage('ready');
      Alert.alert('Analysis failed', e.message);
    }
  };

  const onLabel = async (shot, label) => {
    const before = shot.label;
    // optimistic update
    setResult(r => ({ ...r, shots: r.shots.map(s => s.shot_number === shot.shot_number ? { ...s, label } : s) }));
    try {
      await api.feedback(jobId, shot.shot_number, label);
      await labelShot(jobId, shot.shot_number, label);
    } catch (e) {
      setResult(r => ({ ...r, shots: r.shots.map(s => s.shot_number === shot.shot_number ? { ...s, label: before } : s) }));
      Alert.alert('Label not saved', e.message);
    }
  };

  if (!permission) return <View style={styles.container} />;

  const shots = result?.shots || [];
  const made = shots.filter(s => (s.label ?? s.prediction) === 1).length;
  const busy = stage === 'uploading' || stage === 'analyzing';

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Session</Text>
        <TouchableOpacity onPress={checkServer} style={styles.serverPill} accessibilityLabel="Server status">
          <View style={[styles.serverDot, {
            backgroundColor: server == null ? colors.textGhost : server ? colors.green : colors.orange,
          }]} />
          <Text style={styles.serverText}>
            {server == null ? 'checking…' : server ? 'laptop online' : 'laptop offline'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => navigation.navigate('Settings')} style={styles.gear}
          accessibilityLabel="Settings">
          <Ionicons name="settings-outline" size={20} color={colors.textMuted} />
        </TouchableOpacity>
      </View>

      {stage === 'results' ? (
        <View style={{ flex: 1 }}>
          <View style={styles.summaryRow}>
            <Card style={styles.liveCard}>
              <Text style={styles.liveVal}>{shots.length}</Text>
              <Text style={styles.liveLbl}>shots</Text>
            </Card>
            <Card style={styles.liveCard}>
              <Text style={[styles.liveVal, { color: colors.green }]}>{made}</Text>
              <Text style={styles.liveLbl}>made</Text>
            </Card>
            <Card style={styles.liveCard}>
              <Text style={[styles.liveVal, shots.length > 0 && { color: made / shots.length >= 0.5 ? colors.green : colors.orange }]}>
                {shots.length ? pct(made / shots.length) : '—'}
              </Text>
              <Text style={styles.liveLbl}>FG%</Text>
            </Card>
          </View>
          <Text style={styles.hint}>Check each shot – your labels train the next model.</Text>
          <FlatList
            data={shots}
            keyExtractor={s => String(s.shot_number)}
            renderItem={({ item }) => <ShotRow shot={item} fps={result.fps} onLabel={onLabel} />}
            contentContainerStyle={{ padding: spacing.lg, paddingTop: 0, gap: spacing.sm }}
            ListEmptyComponent={
              <Card><Text style={styles.emptyText}>No shots were detected in this clip.</Text></Card>
            }
          />
          <View style={styles.bottomBar}>
            <TouchableOpacity style={styles.primaryBtn} onPress={reset} activeOpacity={0.85}>
              <Text style={styles.primaryText}>New session</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <>
          {/* Viewfinder */}
          <View style={styles.viewfinder}>
            {permission.granted ? (
              <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back"
                mode="video" videoQuality="1080p" mute />
            ) : (
              <View style={styles.noCam}>
                <Ionicons name="camera-outline" size={48} color={colors.textGhost} />
                <Text style={styles.noCamText}>Camera permission needed</Text>
                <TouchableOpacity onPress={requestPermission} style={styles.permBtn}>
                  <Text style={styles.permBtnText}>Allow camera</Text>
                </TouchableOpacity>
              </View>
            )}
            {stage === 'recording' && (
              <View style={styles.recIndicator}>
                <View style={styles.recDot} />
                <Text style={styles.recLabel}>REC {fmtTime(elapsed)}</Text>
              </View>
            )}
            <View style={styles.statusBadge}>
              <Text style={styles.statusText}>
                {stage === 'recording' ? 'Point at the hoop – keep the ball and rim in frame'
                  : stage === 'ready' ? 'Clip ready'
                  : busy ? progressText
                  : 'Record a session or pick a video'}
              </Text>
            </View>
          </View>

          <View style={styles.controls}>
            {stage === 'ready' && clip && (
              <Card style={styles.clipCard}>
                <Ionicons name="film-outline" size={22} color={colors.textMuted} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.clipTitle}>{clip.source === 'camera' ? 'Recorded clip' : 'Video from library'}</Text>
                  <Text style={styles.clipMeta}>
                    {[clip.durationS && fmtTime(clip.durationS), clip.sizeMb && `${clip.sizeMb.toFixed(0)} MB`,
                      'original quality'].filter(Boolean).join(' · ')}
                  </Text>
                </View>
                <TouchableOpacity onPress={reset} accessibilityLabel="Discard clip">
                  <Ionicons name="trash-outline" size={20} color={colors.textDim} />
                </TouchableOpacity>
              </Card>
            )}

            {busy && <Progress label={progressText} value={progress} />}

            {stage === 'ready' && (
              <TouchableOpacity
                style={[styles.primaryBtn, server === false && { opacity: 0.5 }]}
                onPress={runAnalysis} activeOpacity={0.85}
              >
                <Ionicons name="analytics-outline" size={18} color={colors.white} />
                <Text style={styles.primaryText}>Analyze on laptop</Text>
              </TouchableOpacity>
            )}
            {server === false && !busy && (
              <Text style={styles.warn}>
                The laptop server is not reachable. Start it with HOOPER_HOST=0.0.0.0 and check the address in Settings.
              </Text>
            )}

            {/* Record / pick */}
            <View style={styles.recRow}>
              <TouchableOpacity style={styles.sideBtn} onPress={pickVideo} disabled={busy || stage === 'recording'}
                accessibilityLabel="Pick a video">
                <Ionicons name="images-outline" size={22} color={busy ? colors.textGhost : colors.text} />
                <Text style={styles.sideLbl}>Library</Text>
              </TouchableOpacity>
              <RecordButton recording={stage === 'recording'} disabled={busy} onPress={toggleRecording} />
              <View style={styles.sideBtn}>
                <Text style={styles.sideLbl}>{stage === 'recording' ? 'tap to stop' : 'tap to record'}</Text>
              </View>
            </View>
          </View>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.md,
    borderBottomWidth: 0.5,
    borderBottomColor: '#1e1e1e',
  },
  title: { flex: 1, fontSize: font.md, fontWeight: '500', color: colors.white },
  serverPill: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    borderWidth: 0.5, borderColor: colors.border, borderRadius: radius.full,
    paddingHorizontal: spacing.sm, paddingVertical: 3,
  },
  serverDot: { width: 7, height: 7, borderRadius: 3.5 },
  serverText: { fontSize: font.xs, color: colors.textMuted },
  gear: { padding: 4 },

  viewfinder: {
    height: 300,
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
  recIndicator: {
    position: 'absolute', top: 10, right: 12,
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: radius.full, paddingHorizontal: 8, paddingVertical: 3,
  },
  recDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: colors.orange },
  recLabel: { fontSize: font.sm, color: colors.orange, fontVariant: ['tabular-nums'] },
  statusBadge: {
    position: 'absolute',
    bottom: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderWidth: 0.5,
    borderColor: '#333',
    borderRadius: radius.full,
    paddingHorizontal: spacing.md,
    paddingVertical: 4,
    maxWidth: '92%',
  },
  statusText: { fontSize: font.sm, color: colors.textMuted },

  controls: { flex: 1, padding: spacing.lg, gap: spacing.md, justifyContent: 'flex-end' },

  clipCard: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.md },
  clipTitle: { fontSize: font.base, color: colors.white },
  clipMeta: { fontSize: font.xs, color: colors.textDim, marginTop: 2 },

  progressCard: { padding: spacing.md, gap: spacing.sm },
  progressTop: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  progressLabel: { flex: 1, fontSize: font.base, color: colors.text },
  progressPct: { fontSize: font.sm, color: colors.textMuted, fontVariant: ['tabular-nums'] },
  progressTrack: { height: 5, backgroundColor: colors.bgCardDeep, borderRadius: radius.full, overflow: 'hidden' },
  progressFill: { height: '100%', backgroundColor: colors.orange },

  primaryBtn: {
    backgroundColor: colors.orange,
    borderRadius: radius.md,
    paddingVertical: 13,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
  },
  primaryText: { color: colors.white, fontSize: font.md, fontWeight: '500' },
  warn: { fontSize: font.sm, color: colors.yellow, textAlign: 'center' },

  recRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
  },
  sideBtn: { width: 72, alignItems: 'center', gap: 4 },
  sideLbl: { fontSize: font.xs, color: colors.textGhost, textAlign: 'center' },
  recOuter: {
    width: 64, height: 64,
    borderRadius: 32,
    borderWidth: 3,
    borderColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recInner: { width: 48, height: 48, borderRadius: 24, backgroundColor: colors.orange },
  recInnerStop: { width: 22, height: 22, borderRadius: 4 },

  summaryRow: { flexDirection: 'row', gap: spacing.sm, padding: spacing.lg },
  liveCard: { flex: 1, padding: spacing.md, alignItems: 'center' },
  liveVal: { fontSize: 20, fontWeight: '500', color: colors.white },
  liveLbl: { fontSize: font.xs, color: colors.textDim, marginTop: 2 },
  hint: { fontSize: font.sm, color: colors.textDim, paddingHorizontal: spacing.lg, marginBottom: spacing.sm },
  emptyText: { fontSize: font.base, color: colors.textMuted, textAlign: 'center' },

  shotRow: {
    backgroundColor: colors.bgCard, borderRadius: radius.lg, padding: spacing.md,
    borderWidth: 0.5, borderColor: colors.border, gap: spacing.sm,
  },
  shotTop: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  shotNum: { fontSize: font.base, color: colors.textMuted, minWidth: 30 },
  shotTime: { fontSize: font.sm, color: colors.textDim, flex: 1 },
  shotConf: { fontSize: font.sm, color: colors.textMuted, minWidth: 36, textAlign: 'right' },
  chip: { borderRadius: radius.full, paddingHorizontal: spacing.sm, paddingVertical: 2, borderWidth: 0.5 },
  chipMade: { backgroundColor: colors.greenFade, borderColor: colors.greenBorder },
  chipMissed: { backgroundColor: colors.orangeFade, borderColor: colors.orangeBorder },
  chipText: { fontSize: font.xs, fontWeight: '700', letterSpacing: 0.5 },

  fbRow: { flexDirection: 'row', gap: spacing.sm, alignItems: 'center' },
  fbBtn: {
    flex: 1,
    backgroundColor: colors.bgCardDeep,
    borderWidth: 0.5,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingVertical: spacing.sm,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
  },
  fbMadeOn: { borderColor: colors.greenBorder, backgroundColor: colors.greenFade },
  fbMissOn: { borderColor: colors.orangeBorder, backgroundColor: colors.orangeFade },
  fbLabel: { fontSize: font.sm, color: colors.textMuted },
  verdict: { fontSize: font.xs, minWidth: 84, textAlign: 'right' },

  bottomBar: { padding: spacing.lg, paddingTop: spacing.sm, borderTopWidth: 0.5, borderTopColor: '#1e1e1e' },
});
