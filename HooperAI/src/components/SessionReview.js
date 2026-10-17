import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, FlatList, StyleSheet, Alert, ActivityIndicator } from 'react-native';
import { useVideoPlayer, VideoView } from 'expo-video';
import { useEventListener } from 'expo';
import * as FileSystem from 'expo-file-system/legacy';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { labelShot } from '../utils/storage';
import * as api from '../services/api';

// Review an analyzed session: watch each shot and label it made / missed.
// Plays the original clip from this phone when it is still there, otherwise
// streams it back from the laptop.

const PRE_ROLL = 1.0;    // seconds shown before a shot starts
const POST_ROLL = 0.8;   // and after it ends

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const CONF = { HIGH: 'high confidence', MEDIUM: 'medium confidence', LOW: 'low confidence' };

function shotTimes(shot, fps) {
  return { start: Math.max(0, (shot.start_frame - 1) / fps), end: (shot.end_frame - 1) / fps };
}

// ── one shot ─────────────────────────────────────────────────────────────────
function ShotRow({ shot, fps, active, playing, onPlay, onLabel }) {
  const made = shot.prediction === 1;
  const { start, end } = shotTimes(shot, fps);
  const verdict = shot.label == null ? null : shot.label === shot.prediction;
  return (
    <View style={[styles.row, active && styles.rowActive]}>
      <TouchableOpacity style={styles.rowTop} onPress={onPlay} activeOpacity={0.7}
        accessibilityLabel={`Play shot ${shot.shot_number}`}>
        <View style={[styles.playBtn, active && playing && styles.playBtnOn]}>
          <Ionicons name={active && playing ? 'pause' : 'play'} size={14}
            color={active && playing ? colors.white : colors.text} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.rowTitle}>Shot {shot.shot_number}</Text>
          <Text style={styles.rowMeta}>{fmt(start)} – {fmt(end)} · {CONF[shot.confidence] || 'model'}</Text>
        </View>
        <View style={styles.predCol}>
          <Text style={styles.predLabel}>model says</Text>
          <View style={[styles.chip, made ? styles.chipMade : styles.chipMissed]}>
            <Text style={[styles.chipText, { color: made ? colors.green : colors.orange }]}>
              {made ? 'MADE' : 'MISSED'}
            </Text>
          </View>
        </View>
      </TouchableOpacity>

      <View style={styles.labelRow}>
        <Text style={styles.youSay}>You:</Text>
        <TouchableOpacity style={[styles.labelBtn, shot.label === 1 && styles.labelMadeOn]}
          onPress={() => onLabel(1)} activeOpacity={0.8}>
          <Ionicons name="checkmark" size={15} color={shot.label === 1 ? colors.green : colors.textMuted} />
          <Text style={[styles.labelText, shot.label === 1 && { color: colors.green }]}>Made</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.labelBtn, shot.label === 0 && styles.labelMissOn]}
          onPress={() => onLabel(0)} activeOpacity={0.8}>
          <Ionicons name="close" size={15} color={shot.label === 0 ? colors.orange : colors.textMuted} />
          <Text style={[styles.labelText, shot.label === 0 && { color: colors.orange }]}>Missed</Text>
        </TouchableOpacity>
      </View>
      {verdict != null && (
        <Text style={[styles.verdict, { color: verdict ? colors.green : colors.yellow }]}>
          {verdict ? '✓ Model was right' : '✗ Model was wrong – thanks, this helps the most'}
        </Text>
      )}
    </View>
  );
}

// ── review ───────────────────────────────────────────────────────────────────
export default function SessionReview({ session, footer }) {
  const result = session?.result ?? { shots: [], fps: 30 };   // never crash on a malformed session
  const fps = result.fps || 30;
  const [shots, setShots] = useState(result.shots || []);
  const [current, setCurrent] = useState(null);       // index of the selected shot
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoState, setVideoState] = useState('loading');   // loading | ready | missing
  const stopAt = useRef(null);
  const listRef = useRef(null);

  const player = useVideoPlayer(null, (p) => {
    p.loop = false;
    p.timeUpdateEventInterval = 0.25;
  });

  // Local original if it still exists on the phone, else stream from the laptop
  useEffect(() => {
    let cancelled = false;
    (async () => {
      let source = null;
      if (session.localUri) {
        try {
          const info = await FileSystem.getInfoAsync(session.localUri);
          if (info.exists) source = { uri: session.localUri };
        } catch (_) { /* fall back to the laptop */ }
      }
      if (!source && session.video) {
        try {
          await api.health();
          source = await api.videoSource(session.video);
        } catch (_) { /* laptop offline */ }
      }
      if (cancelled) return;
      if (source) {
        setVideoState('ready');
        player.replaceAsync(source).catch(() => !cancelled && setVideoState('missing'));
      } else {
        setVideoState('missing');
      }
    })();
    return () => { cancelled = true; };
  }, [session.id]);

  useEventListener(player, 'timeUpdate', ({ currentTime }) => {
    setTime(currentTime);
    if (stopAt.current != null && currentTime >= stopAt.current) {
      stopAt.current = null;
      player.pause();
    }
  });
  useEventListener(player, 'playingChange', ({ isPlaying }) => setPlaying(isPlaying));

  const playShot = (i) => {
    if (i == null || i < 0 || i >= shots.length) return;
    setCurrent(i);
    listRef.current?.scrollToIndex({ index: i, viewPosition: 0.1, animated: true });
    if (videoState !== 'ready') return;
    if (i === current && playing) { player.pause(); return; }
    const { start, end } = shotTimes(shots[i], fps);
    stopAt.current = end + POST_ROLL;
    player.currentTime = Math.max(0, start - PRE_ROLL);
    player.play();
  };

  const onLabel = async (i, label) => {
    const shot = shots[i];
    const before = shot.label;
    const next = shots.map((s, k) => (k === i ? { ...s, label } : s));
    setShots(next);
    try {
      await api.feedback(session.id, shot.shot_number, label);
      await labelShot(session.id, shot.shot_number, label);
    } catch (e) {
      setShots(shots.map((s, k) => (k === i ? { ...s, label: before } : s)));
      Alert.alert('Label not saved', e.message);
      return;
    }
    // jump to the next shot that still needs a label
    const after = next.findIndex((s, k) => k > i && s.label == null);
    if (after >= 0) setTimeout(() => playShot(after), 350);
  };

  const active = useMemo(() => shots.findIndex((s) => {
    const { start, end } = shotTimes(s, fps);
    return time >= start - 0.05 && time <= end + POST_ROLL;
  }), [time, shots, fps]);

  const made = shots.filter((s) => (s.label ?? s.prediction) === 1).length;
  const labeled = shots.filter((s) => s.label != null).length;
  const activeShot = active >= 0 ? shots[active] : null;

  return (
    <View style={{ flex: 1 }}>
      {/* Video */}
      <View style={styles.video}>
        {videoState === 'ready' && (
          <VideoView player={player} style={StyleSheet.absoluteFill} contentFit="contain"
            nativeControls allowsFullscreen />
        )}
        {videoState === 'loading' && <ActivityIndicator color={colors.orange} />}
        {videoState === 'missing' && (
          <View style={styles.noVideo}>
            <Ionicons name="videocam-off-outline" size={28} color={colors.textGhost} />
            <Text style={styles.noVideoText}>
              Video not available – it is no longer on this phone and the laptop is offline.
            </Text>
          </View>
        )}
        {activeShot && (
          <View style={styles.banner} pointerEvents="none">
            <Text style={styles.bannerText}>Shot {activeShot.shot_number}</Text>
          </View>
        )}
      </View>

      {/* Summary */}
      <View style={styles.summary}>
        <Stat value={shots.length} label="shots" />
        <Stat value={made} label="made" color={colors.green} />
        <Stat value={shots.length ? `${Math.round((made / shots.length) * 100)}%` : '—'} label="FG%"
          color={shots.length ? (made / shots.length >= 0.5 ? colors.green : colors.orange) : undefined} />
        <Stat value={`${labeled}/${shots.length}`} label="labeled" />
      </View>

      <FlatList
        ref={listRef}
        data={shots}
        keyExtractor={(s) => String(s.shot_number)}
        onScrollToIndexFailed={() => {}}
        contentContainerStyle={styles.list}
        ListHeaderComponent={shots.length > 0 ? (
          <Text style={styles.hint}>
            {labeled === shots.length
              ? 'All shots labeled – these labels train the next model.'
              : 'Tap a shot to watch it, then tell us if it went in.'}
          </Text>
        ) : null}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>No shots were detected in this clip.</Text>
            <Text style={styles.emptyHint}>Keep the hoop and the ball's flight in frame.</Text>
          </View>
        }
        renderItem={({ item, index }) => (
          <ShotRow shot={item} fps={fps} active={index === (active >= 0 ? active : current)}
            playing={playing} onPlay={() => playShot(index)} onLabel={(l) => onLabel(index, l)} />
        )}
        ListFooterComponent={footer}
      />
    </View>
  );
}

function Stat({ value, label, color }) {
  return (
    <View style={styles.stat}>
      <Text style={[styles.statVal, color && { color }]}>{value}</Text>
      <Text style={styles.statLbl}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  video: {
    width: '100%', aspectRatio: 16 / 9, backgroundColor: '#000',
    alignItems: 'center', justifyContent: 'center',
  },
  noVideo: { alignItems: 'center', gap: spacing.sm, paddingHorizontal: spacing.xl },
  noVideoText: { color: colors.textMuted, fontSize: font.sm, textAlign: 'center' },
  banner: {
    position: 'absolute', top: 10, left: 10, backgroundColor: 'rgba(0,0,0,0.65)',
    borderRadius: radius.sm, paddingHorizontal: spacing.sm, paddingVertical: 4,
    borderWidth: 0.5, borderColor: 'rgba(200,169,0,0.5)',
  },
  bannerText: { color: colors.yellow, fontSize: font.sm, fontWeight: '700' },

  summary: {
    flexDirection: 'row', paddingVertical: spacing.md, paddingHorizontal: spacing.lg,
    borderBottomWidth: 0.5, borderBottomColor: '#1e1e1e',
  },
  stat: { flex: 1, alignItems: 'center' },
  statVal: { fontSize: font.lg, fontWeight: '600', color: colors.white },
  statLbl: { fontSize: font.xs, color: colors.textDim, marginTop: 2 },

  list: { padding: spacing.lg, gap: spacing.sm, paddingBottom: spacing.xxl },
  hint: { color: colors.textDim, fontSize: font.sm, marginBottom: spacing.xs },
  empty: { alignItems: 'center', paddingVertical: spacing.xxl, gap: spacing.xs },
  emptyText: { color: colors.text, fontSize: font.base },
  emptyHint: { color: colors.textDim, fontSize: font.sm },

  row: {
    backgroundColor: colors.bgCard, borderRadius: radius.lg, padding: spacing.md,
    borderWidth: 0.5, borderColor: colors.border, gap: spacing.md,
  },
  rowActive: { borderColor: 'rgba(255,255,255,0.5)' },
  rowTop: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  playBtn: {
    width: 32, height: 32, borderRadius: 16, backgroundColor: colors.bgCardDeep,
    borderWidth: 0.5, borderColor: colors.border, alignItems: 'center', justifyContent: 'center',
  },
  playBtnOn: { backgroundColor: colors.orange, borderColor: colors.orange },
  rowTitle: { color: colors.white, fontSize: font.base, fontWeight: '500' },
  rowMeta: { color: colors.textDim, fontSize: font.xs, marginTop: 2 },
  predCol: { alignItems: 'flex-end', gap: 3 },
  predLabel: { color: colors.textGhost, fontSize: 9, textTransform: 'uppercase', letterSpacing: 0.6 },
  chip: { borderRadius: radius.full, paddingHorizontal: spacing.sm, paddingVertical: 2, borderWidth: 0.5 },
  chipMade: { backgroundColor: colors.greenFade, borderColor: colors.greenBorder },
  chipMissed: { backgroundColor: colors.orangeFade, borderColor: colors.orangeBorder },
  chipText: { fontSize: font.xs, fontWeight: '700', letterSpacing: 0.5 },

  labelRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  youSay: { color: colors.textDim, fontSize: font.sm, width: 32 },
  labelBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5,
    backgroundColor: colors.bgCardDeep, borderWidth: 0.5, borderColor: colors.border,
    borderRadius: radius.md, paddingVertical: 9,
  },
  labelMadeOn: { borderColor: colors.greenBorder, backgroundColor: colors.greenFade },
  labelMissOn: { borderColor: colors.orangeBorder, backgroundColor: colors.orangeFade },
  labelText: { color: colors.textMuted, fontSize: font.sm },
  verdict: { fontSize: font.xs, marginTop: -4 },
});
