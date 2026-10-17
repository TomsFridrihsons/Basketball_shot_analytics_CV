import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, Animated, Alert, ActivityIndicator,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card } from '../components';
import SessionReview from '../components/SessionReview';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { saveSession } from '../utils/storage';
import * as api from '../services/api';

// Flow: idle → recording → ready (have a clip) → uploading → analyzing → results
// The video is uploaded in original quality; only results come back.

const PENDING_KEY = 'hooperai_pending_job';   // analysis still running on the laptop

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
  const [session, setSession]   = useState(null);   // { id, result, video, localUri }
  const [server, setServer]     = useState(null);     // null = checking, true/false
  const [elapsed, setElapsed]   = useState(0);
  const stageRef = useRef(stage);
  useEffect(() => { stageRef.current = stage; }, [stage]);

  const checkServer = useCallback(async () => {
    setServer(null);
    try { await api.health(); setServer(true); } catch (_) { setServer(false); }
  }, []);

  useFocusEffect(useCallback(() => {
    checkServer();
    return () => {
      if (cameraRef.current && stageRef.current === 'recording') cameraRef.current.stopRecording();
    };
  }, [checkServer]));

  // Keep following a job while the screen exists (tabs stay mounted); stop only on unmount.
  useEffect(() => () => { cancelled.current = true; }, []);

  // Resume an analysis that was still running when the screen/app was closed
  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem(PENDING_KEY);
        if (raw && stageRef.current === 'idle') {
          const p = JSON.parse(raw);
          setClip({ uri: p.localUri, durationS: null, source: 'resumed' });
          followJob(p.jobId, p.name, p.localUri);
        }
      } catch (_) { /* nothing to resume */ }
    })();
  }, []);

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
    setSession(null);
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
    // Default videoExportPreset is Passthrough: the original file, no re-encode
    const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['videos'], quality: 1 });
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

      const { job_id } = await api.analyze(name);
      await AsyncStorage.setItem(PENDING_KEY, JSON.stringify({ jobId: job_id, name, localUri: clip.uri }));
      await followJob(job_id, name, clip.uri);
    } catch (e) {
      if (e.message === 'cancelled') return;
      setStage('ready');
      Alert.alert('Analysis failed', e.message);
    }
  };

  /** Poll the laptop until the job is done, then open the review. */
  const followJob = async (jobId, name, localUri) => {
    setStage('analyzing');
    setProgress(0);
    setProgressText('Waiting for the laptop…');
    try {
      const res = await api.waitForJob(jobId, (p, st) => {
        if (p != null) setProgress(p);
        setProgressText(
          st === 'reconnecting' ? 'Connection lost – reconnecting (the laptop keeps working)…'
          : st === 'analyzing' ? 'Detecting shots…'
          : `${st[0].toUpperCase()}${st.slice(1)}…`);
      }, () => cancelled.current);
      await AsyncStorage.removeItem(PENDING_KEY);
      await saveSession(jobId, res, { localUri });
      setSession({ id: jobId, result: res, video: name, localUri });
      setStage('results');
    } catch (e) {
      if (e.message === 'cancelled') return;   // unmounted – resumes next time
      await AsyncStorage.removeItem(PENDING_KEY);
      setStage(localUri ? 'ready' : 'idle');
      Alert.alert('Analysis failed', e.status === 404
        ? 'The laptop no longer has this job (was the server restarted?). Analyze the clip again.'
        : e.message);
    }
  };

  if (!permission) return <View style={styles.container} />;

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

      {stage === 'results' && session?.result ? (
        <SessionReview
          session={session}
          footer={
            <TouchableOpacity style={[styles.primaryBtn, { marginTop: spacing.md }]} onPress={reset} activeOpacity={0.85}>
              <Ionicons name="add" size={18} color={colors.white} />
              <Text style={styles.primaryText}>New session</Text>
            </TouchableOpacity>
          }
        />
      ) : (
        <>
          {/* Viewfinder */}
          <View style={styles.viewfinder}>
            {busy ? (
              <View style={styles.noCam}>
                <Ionicons name="hardware-chip-outline" size={44} color={colors.textGhost} />
                <Text style={styles.noCamText}>Your laptop is analyzing the video</Text>
                <Text style={styles.busyHint}>You can leave the app – it keeps going and resumes here.</Text>
              </View>
            ) : permission.granted ? (
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
  busyHint: { fontSize: font.sm, color: colors.textDim, textAlign: 'center', paddingHorizontal: spacing.xl },
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




});
