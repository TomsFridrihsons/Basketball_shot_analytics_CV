import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, Animated, Alert, ActivityIndicator, Linking,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { saveSession, summarizeShots } from '../utils/storage';
import * as api from '../services/api';

// Flow: idle → recording → ready (have a clip) → uploading → analyzing → done
//                                  ↑ error ←───────────┘
// The video is uploaded in original quality; only results come back.
// Review happens on the Session screen, the same one Home opens.

const PENDING_KEY = 'hooperai_pending_job';   // analysis still running on the laptop

const fmtTime = (s) => (s == null ? '' : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);

// ── Shutter: red means recording, and only that ─────────────────────────────
function Shutter({ recording, disabled, onPress }) {
  const scale = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.spring(scale, { toValue: recording ? 0.42 : 1, useNativeDriver: true, speed: 20 }).start();
  }, [recording]);
  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.85} disabled={disabled}
      style={[styles.shutter, disabled && { opacity: 0.4 }]}
      accessibilityRole="button" accessibilityLabel={recording ? 'Stop recording' : 'Start recording'}>
      <Animated.View style={[styles.shutterInner, recording && styles.shutterStop, { transform: [{ scale }] }]} />
    </TouchableOpacity>
  );
}

// ── Determinate bar; the step text says what is happening, not "loading" ─────
function ProgressBar({ value }) {
  return (
    <View style={styles.track}
      accessibilityRole="progressbar"
      accessibilityValue={{ min: 0, max: 100, now: Math.round((value || 0) * 100) }}>
      <View style={[styles.fill, { width: `${Math.round((value || 0) * 100)}%` }]} />
    </View>
  );
}

function Secondary({ label, icon, onPress, a11y }) {
  return (
    <TouchableOpacity style={styles.secondaryBtn} onPress={onPress} activeOpacity={0.7}
      accessibilityRole="button" accessibilityLabel={a11y || label}>
      {icon && <Ionicons name={icon} size={18} color={colors.text} />}
      <Text style={styles.secondaryText}>{label}</Text>
    </TouchableOpacity>
  );
}

function Primary({ label, icon, onPress, disabled }) {
  return (
    <TouchableOpacity style={[styles.primaryBtn, disabled && styles.primaryOff]} onPress={onPress}
      disabled={disabled} activeOpacity={0.85}
      accessibilityRole="button" accessibilityState={{ disabled: !!disabled }}>
      {icon && <Ionicons name={icon} size={20} color={disabled ? colors.textDim : colors.white} />}
      <Text style={[styles.primaryText, disabled && { color: colors.textDim }]}>{label}</Text>
    </TouchableOpacity>
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
  const [clip, setClip]         = useState(null);     // { uri, durationS, source, sizeMb }
  const [progress, setProgress] = useState(0);
  const [step, setStep]         = useState('');       // what the laptop is doing now
  const [reconnecting, setReconnecting] = useState(false);
  const [done, setDone]         = useState(null);     // { id, total, made }
  const [error, setError]       = useState(null);
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
    setDone(null);
    setError(null);
    setProgress(0);
  };

  const toggleRecording = async () => {
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
      Alert.alert('Recording stopped', e.message);
    }
  };

  const pickVideo = async () => {
    // Default videoExportPreset is Passthrough: the original file, no re-encode
    const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['videos'], quality: 1 });
    if (res.canceled || !res.assets?.length) return;
    const a = res.assets[0];
    setClip({ uri: a.uri, durationS: a.duration ? a.duration / 1000 : null, source: 'library',
              sizeMb: a.fileSize ? a.fileSize / 1e6 : null });
    setError(null);
    setStage('ready');
  };

  const runAnalysis = async () => {
    try {
      setError(null);
      setStage('uploading');
      setProgress(0);
      setStep('Sending the video to your laptop');
      const { name } = await api.uploadVideo(clip.uri, setProgress);
      if (cancelled.current) return;

      const { job_id } = await api.analyze(name);
      await AsyncStorage.setItem(PENDING_KEY, JSON.stringify({ jobId: job_id, name, localUri: clip.uri }));
      await followJob(job_id, name, clip.uri);
    } catch (e) {
      if (e.message === 'cancelled') return;
      setStage('ready');
      setError(`The upload didn't finish: ${e.message}`);
    }
  };

  /** Poll the laptop until the job is done, then show the result. */
  const followJob = async (jobId, name, localUri) => {
    setStage('analyzing');
    setProgress(0);
    setStep('Waiting for the laptop to start');
    try {
      const res = await api.waitForJob(jobId, (p, st) => {
        if (p != null) setProgress(p);
        setReconnecting(st === 'reconnecting');
        if (st === 'analyzing') setStep('Finding shots');
        else if (st === 'queued') setStep('Waiting for the laptop to start');
        else if (st !== 'reconnecting') setStep(`${st[0].toUpperCase()}${st.slice(1)}`);
      }, () => cancelled.current);
      await AsyncStorage.removeItem(PENDING_KEY);
      await saveSession(jobId, res, { localUri });
      const sum = summarizeShots(res.shots || []);
      setDone({ id: jobId, total: sum.total, made: sum.made });
      setReconnecting(false);
      setStage('done');
    } catch (e) {
      if (e.message === 'cancelled') return;   // unmounted – resumes next time
      await AsyncStorage.removeItem(PENDING_KEY);
      setReconnecting(false);
      setStage(localUri ? 'ready' : 'idle');
      setError(e.status === 404
        ? 'The laptop no longer has this job – it was probably restarted. Analyze the clip again.'
        : `The analysis stopped: ${e.message}`);
    }
  };

  if (!permission) return <View style={styles.container} />;

  const busy = stage === 'uploading' || stage === 'analyzing';
  const camera = (stage === 'idle' || stage === 'recording') && permission.granted;
  const offline = server === false;

  // ── top bar: laptop status (tap to check again) + settings ──
  const topBar = (
    <View style={[styles.topBar, { paddingTop: insets.top + spacing.xs }]}>
      {stage === 'recording' ? (
        <View style={styles.recPill} accessibilityLiveRegion="polite"
          accessibilityLabel={`Recording, ${fmtTime(elapsed)}`}>
          <View style={styles.recDot} />
          <Text style={styles.recTime}>{fmtTime(elapsed)}</Text>
        </View>
      ) : (
        <TouchableOpacity onPress={checkServer} style={styles.statusPill}
          accessibilityRole="button"
          accessibilityLabel={server == null ? 'Checking the laptop' : offline ? 'Laptop offline. Check again' : 'Laptop online'}>
          {server == null
            ? <ActivityIndicator size="small" color={colors.textMuted} style={{ transform: [{ scale: 0.7 }] }} />
            : <Ionicons name={offline ? 'cloud-offline-outline' : 'laptop-outline'} size={15}
                color={offline ? colors.yellow : colors.textMuted} />}
          <Text style={[styles.statusText, offline && { color: colors.yellow }]}>
            {server == null ? 'Checking laptop' : offline ? 'Laptop offline' : 'Laptop online'}
          </Text>
        </TouchableOpacity>
      )}
      <View style={{ flex: 1 }} />
      {stage !== 'recording' && (
        <TouchableOpacity onPress={() => navigation.navigate('Settings')} style={styles.iconBtn}
          accessibilityRole="button" accessibilityLabel="Settings">
          <Ionicons name="settings-outline" size={22} color={colors.text} />
        </TouchableOpacity>
      )}
    </View>
  );

  // ── capture: full-bleed camera, controls in thumb reach ──
  if (stage === 'idle' || stage === 'recording') {
    return (
      <View style={styles.container}>
        {camera ? (
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back"
            mode="video" videoQuality="1080p" mute />
        ) : (
          <View style={styles.centerPanel}>
            <View style={styles.bigIcon}><Ionicons name="videocam-outline" size={34} color={colors.text} /></View>
            <Text style={styles.panelTitle}>Film your shots</Text>
            <Text style={styles.panelBody}>
              HooperAI uses the camera to record your session. Videos go only to your own laptop for analysis.
            </Text>
            {permission.canAskAgain ? (
              <Primary label="Continue" onPress={requestPermission} />
            ) : (
              <Primary label="Open Settings" icon="settings-outline" onPress={() => Linking.openSettings()} />
            )}
          </View>
        )}
        {topBar}

        <View style={styles.captureBottom}>
          {error && (
            <Text style={styles.errorText} accessibilityLiveRegion="polite">{error}</Text>
          )}
          {camera && stage === 'idle' && (
            <Text style={styles.hint}>Keep the hoop and the ball in frame</Text>
          )}
          <View style={styles.shutterRow}>
            <TouchableOpacity style={styles.sideBtn} onPress={pickVideo} disabled={stage === 'recording'}
              accessibilityRole="button" accessibilityLabel="Choose a video from your library">
              <View style={[styles.sideIcon, stage === 'recording' && { opacity: 0.3 }]}>
                <Ionicons name="images-outline" size={22} color={colors.white} />
              </View>
              <Text style={[styles.sideLbl, stage === 'recording' && { opacity: 0.3 }]}>Library</Text>
            </TouchableOpacity>
            {camera ? <Shutter recording={stage === 'recording'} onPress={toggleRecording} /> : <View style={styles.shutter} />}
            <View style={styles.sideBtn} />
          </View>
        </View>
      </View>
    );
  }

  // ── after capture: one panel per state ──
  return (
    <View style={styles.container}>
      {topBar}
      <View style={styles.panelWrap}>

        {stage === 'ready' && clip && (
          <>
            <View style={styles.clipCard}>
              <View style={styles.clipIcon}><Ionicons name="film-outline" size={22} color={colors.text} /></View>
              <View style={{ flex: 1 }}>
                <Text style={styles.clipTitle}>
                  {clip.source === 'camera' ? 'New recording' : clip.source === 'library' ? 'Video from your library' : 'Your clip'}
                </Text>
                <Text style={styles.clipMeta}>
                  {[clip.durationS && fmtTime(clip.durationS), clip.sizeMb && `${clip.sizeMb.toFixed(0)} MB`,
                    'original quality'].filter(Boolean).join(' · ')}
                </Text>
              </View>
            </View>

            {error && (
              <View style={styles.notice} accessibilityLiveRegion="polite">
                <Ionicons name="alert-circle-outline" size={20} color={colors.yellow} />
                <Text style={styles.noticeText}>{error}</Text>
              </View>
            )}
            {offline && (
              <View style={styles.notice}>
                <Ionicons name="cloud-offline-outline" size={20} color={colors.yellow} />
                <View style={{ flex: 1, gap: spacing.sm }}>
                  <Text style={styles.noticeText}>
                    Your laptop can't be reached. Make sure the HooperAI server is running and the phone is on the same network.
                  </Text>
                  <View style={{ flexDirection: 'row', gap: spacing.lg }}>
                    <TouchableOpacity onPress={checkServer} hitSlop={10} accessibilityRole="button">
                      <Text style={styles.link}>Check again</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => navigation.navigate('Settings')} hitSlop={10} accessibilityRole="button">
                      <Text style={styles.link}>Server address</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              </View>
            )}

            <View style={styles.actions}>
              <Primary label="Analyze on laptop" icon="sparkles-outline" onPress={runAnalysis} disabled={server !== true} />
              <Secondary label="Discard" icon="trash-outline" onPress={reset} a11y="Discard this clip" />
            </View>
          </>
        )}

        {busy && (
          <View style={styles.progressPanel}>
            <Text style={styles.stepCount}>Step {stage === 'uploading' ? 1 : 2} of 2</Text>
            <Text style={styles.panelTitle} accessibilityLiveRegion="polite">{step}</Text>
            <View style={styles.progressRow}>
              <View style={{ flex: 1 }}><ProgressBar value={progress} /></View>
              <Text style={styles.pct}>{Math.round(progress * 100)}%</Text>
            </View>
            {reconnecting ? (
              <View style={styles.notice} accessibilityLiveRegion="polite">
                <ActivityIndicator size="small" color={colors.yellow} />
                <Text style={styles.noticeText}>
                  Connection lost – reconnecting. Your laptop keeps working in the meantime.
                </Text>
              </View>
            ) : (
              <Text style={styles.panelBody}>
                {stage === 'uploading'
                  ? 'Keep the app open until the upload finishes.'
                  : 'You can leave this screen or the app – the result will be waiting here.'}
              </Text>
            )}
            {stage === 'analyzing' && (
              <View style={styles.actions}>
                <Secondary label="Back to Home" icon="home-outline" onPress={() => navigation.navigate('Home')} />
              </View>
            )}
          </View>
        )}

        {stage === 'done' && done && (
          <View style={styles.centerPanel}>
            <View style={[styles.bigIcon, styles.bigIconDone]}>
              <Ionicons name="checkmark" size={36} color={colors.white} />
            </View>
            <Text style={styles.panelTitle} accessibilityRole="header">Analysis complete</Text>
            <Text style={styles.doneStat} maxFontSizeMultiplier={1.4}>
              {done.total ? `${done.made} / ${done.total}` : 'No shots'}
            </Text>
            <Text style={styles.panelBody}>
              {done.total
                ? `${done.total === 1 ? 'shot' : 'shots'} made. Watch each one and confirm it to improve detection.`
                : 'No shots were found. Open the session to add any the detector missed.'}
            </Text>
            <View style={[styles.actions, { alignSelf: 'stretch' }]}>
              <Primary label={done.total ? 'Review shots' : 'Open session'} icon="play-circle-outline"
                onPress={() => { const id = done.id; reset(); navigation.navigate('Session', { sessionId: id }); }} />
              <Secondary label="New session" icon="add" onPress={reset} />
            </View>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },

  topBar: {
    position: 'absolute', top: 0, left: 0, right: 0, zIndex: 2,
    flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.lg,
  },
  statusPill: {
    flexDirection: 'row', alignItems: 'center', gap: 6, minHeight: 36,
    backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: radius.full, paddingHorizontal: spacing.md,
  },
  statusText: { fontSize: font.sm, color: colors.text, fontWeight: '500' },
  recPill: {
    flexDirection: 'row', alignItems: 'center', gap: 6, minHeight: 36,
    backgroundColor: colors.red, borderRadius: radius.full, paddingHorizontal: spacing.md,
  },
  recDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.white },
  recTime: { fontSize: font.md, color: colors.white, fontWeight: '600', fontVariant: ['tabular-nums'] },
  iconBtn: {
    width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
  },

  captureBottom: {
    position: 'absolute', left: 0, right: 0, bottom: 0,
    paddingBottom: spacing.xl, paddingTop: spacing.lg, gap: spacing.md,
    backgroundColor: 'rgba(0,0,0,0.45)', alignItems: 'center',
  },
  hint: { fontSize: font.base, color: colors.text },
  errorText: { fontSize: font.base, color: colors.yellow, textAlign: 'center', paddingHorizontal: spacing.xl },
  shutterRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    alignSelf: 'stretch', paddingHorizontal: spacing.xxl,
  },
  sideBtn: { width: 64, alignItems: 'center', gap: 4 },
  sideIcon: {
    width: 48, height: 48, borderRadius: 24, alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.14)',
  },
  sideLbl: { fontSize: font.sm, color: colors.white },
  shutter: {
    width: 76, height: 76, borderRadius: 38, borderWidth: 4, borderColor: colors.white,
    alignItems: 'center', justifyContent: 'center',
  },
  shutterInner: { width: 60, height: 60, borderRadius: 30, backgroundColor: colors.red },
  shutterStop: { borderRadius: 10 },

  panelWrap: { flex: 1, justifyContent: 'center', padding: spacing.xl, gap: spacing.lg },
  centerPanel: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md, padding: spacing.xl },
  bigIcon: {
    width: 72, height: 72, borderRadius: 36, backgroundColor: colors.bgCard,
    alignItems: 'center', justifyContent: 'center', marginBottom: spacing.sm,
  },
  bigIconDone: { backgroundColor: colors.green },
  panelTitle: { fontSize: 22, fontWeight: '700', color: colors.white, textAlign: 'center' },
  panelBody: { fontSize: font.md, color: colors.textMuted, textAlign: 'center', lineHeight: 21 },
  doneStat: { fontSize: 48, fontWeight: '700', color: colors.white, fontVariant: ['tabular-nums'] },

  clipCard: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.md,
    backgroundColor: colors.bgCard, borderRadius: radius.lg, padding: spacing.lg,
  },
  clipIcon: {
    width: 44, height: 44, borderRadius: radius.md, backgroundColor: colors.bgCardDeep,
    alignItems: 'center', justifyContent: 'center',
  },
  clipTitle: { fontSize: font.md, fontWeight: '600', color: colors.white },
  clipMeta: { fontSize: font.base, color: colors.textMuted, marginTop: 2 },

  notice: {
    flexDirection: 'row', alignItems: 'flex-start', gap: spacing.md,
    backgroundColor: colors.yellowFade, borderRadius: radius.lg, padding: spacing.lg,
  },
  noticeText: { flex: 1, fontSize: font.base, color: colors.text, lineHeight: 19 },
  link: { fontSize: font.base, color: colors.white, fontWeight: '600', textDecorationLine: 'underline' },

  progressPanel: { gap: spacing.md },
  stepCount: { fontSize: font.sm, color: colors.textDim, textAlign: 'center', textTransform: 'uppercase', letterSpacing: 0.8 },
  progressRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, marginVertical: spacing.sm },
  track: { height: 6, backgroundColor: colors.bgCard, borderRadius: radius.full, overflow: 'hidden' },
  fill: { height: '100%', backgroundColor: colors.orange, borderRadius: radius.full },
  pct: { width: 44, textAlign: 'right', fontSize: font.md, color: colors.text, fontVariant: ['tabular-nums'] },

  actions: { gap: spacing.sm, marginTop: spacing.sm },
  primaryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm,
    backgroundColor: colors.orange, borderRadius: radius.lg, minHeight: 50, paddingHorizontal: spacing.xl,
  },
  primaryOff: { backgroundColor: colors.bgCard },
  primaryText: { fontSize: 17, fontWeight: '700', color: colors.white },
  secondaryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm,
    minHeight: 50, borderRadius: radius.lg,
  },
  secondaryText: { fontSize: 17, fontWeight: '500', color: colors.text },
});
