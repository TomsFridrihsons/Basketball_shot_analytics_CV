import React, { useState, useCallback } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, RefreshControl,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card, SectionLabel, Divider } from '../components';
import { getSessions, sessionSummary, formatTimeAgo, NOT_A_SHOT } from '../utils/storage';
import { syncSessions } from '../services/sync';

const MAX_DOTS = 40;

const isToday = ts => new Date(ts).toDateString() === new Date().toDateString();

/** Today's shots across analyzed sessions, oldest first: true = made, false = missed. */
function todaysShots(sessions) {
  return sessions
    .filter(s => isToday(s.timestamp))
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    .flatMap(s => [...(s.result?.shots || [])]
      .sort((a, b) => (a.start_frame ?? 0) - (b.start_frame ?? 0))
      .filter(sh => sh.label !== NOT_A_SHOT)
      .map(sh => (sh.label ?? sh.prediction) === 1));
}

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [sessions, setSessions] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  // sessions saved before review support have no result to open
  const readLocal = useCallback(async () => {
    setSessions((await getSessions()).filter(s => s.result));
  }, []);

  // Show the local cache at once, then pull new/changed sessions from the laptop
  const load = useCallback(async () => {
    await readLocal();
    if (await syncSessions()) await readLocal();
  }, [readLocal]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  const today = todaysShots(sessions);
  const made = today.filter(Boolean).length;
  const pct = today.length ? Math.round((made / today.length) * 100) : null;
  const dots = today.slice(-MAX_DOTS);

  const summaries = sessions.map(s => ({ s, sum: sessionSummary(s) }));
  const toReview = summaries.filter(({ sum }) => sum.total - sum.labeled > 0);
  const toReviewShots = toReview.reduce((n, { sum }) => n + sum.total - sum.labeled, 0);

  const openSession = id => navigation.navigate('Session', { sessionId: id });

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.orange} />}
      >
        {/* Large title: the tab says where you are; the date says when */}
        <View style={styles.header}>
          <View style={{ flex: 1 }}>
            <Text style={styles.title} accessibilityRole="header">Today</Text>
            <Text style={styles.date}>
              {new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })}
            </Text>
          </View>
          <TouchableOpacity onPress={() => navigation.navigate('Settings')} style={styles.iconBtn}
            accessibilityRole="button" accessibilityLabel="Settings">
            <Ionicons name="settings-outline" size={22} color={colors.textMuted} />
          </TouchableOpacity>
        </View>

        {/* Hero: makes / attempts and the shot log, in the order you shot them */}
        <Card style={styles.hero}>
          <View
            accessible
            accessibilityLabel={today.length
              ? `${made} of ${today.length} shots made today, ${pct} percent`
              : 'No shots today'}
          >
            <View style={styles.tallyRow}>
              <Text style={styles.tally} maxFontSizeMultiplier={1.4}>
                {made}
                <Text style={styles.tallyOf}> / {today.length}</Text>
              </Text>
              {pct != null && (
                <Text style={styles.pct} maxFontSizeMultiplier={1.4}>{pct}%</Text>
              )}
            </View>
            <Text style={styles.tallyLbl}>
              {today.length ? 'makes today' : 'No shots yet today'}
            </Text>

            {dots.length > 0 && (
              <View style={styles.dots}>
                {dots.map((m, i) => (
                  <View key={i} style={[styles.dot, m ? styles.dotMade : styles.dotMissed]} />
                ))}
              </View>
            )}
          </View>

          <TouchableOpacity style={styles.recordBtn} activeOpacity={0.85}
            onPress={() => navigation.navigate('Record')}
            accessibilityRole="button" accessibilityLabel="Record a session">
            <Ionicons name="videocam" size={20} color={colors.white} />
            <Text style={styles.recordText}>Record a session</Text>
          </TouchableOpacity>
        </Card>

        {/* Review prompt, only when there is something to review */}
        {toReviewShots > 0 && (
          <TouchableOpacity style={styles.reviewRow} activeOpacity={0.7}
            onPress={() => openSession(toReview[0].s.id)}
            accessibilityRole="button"
            accessibilityLabel={`${toReviewShots} shots to review. Opens the latest session.`}>
            <Ionicons name="checkmark-done-outline" size={20} color={colors.yellow} />
            <View style={{ flex: 1 }}>
              <Text style={styles.reviewTitle}>
                {toReviewShots} {toReviewShots === 1 ? 'shot' : 'shots'} to review
              </Text>
              <Text style={styles.reviewSub}>Your labels make the next model better</Text>
            </View>
            <Ionicons name="chevron-forward" size={18} color={colors.textDim} />
          </TouchableOpacity>
        )}

        {/* Sessions */}
        <View style={styles.listHeader}>
          <SectionLabel style={{ marginBottom: 0 }}>Sessions</SectionLabel>
          {sessions.length > 0 && <Text style={styles.listSub}>{sessions.length}</Text>}
        </View>

        <Card style={{ padding: 0, overflow: 'hidden' }}>
          {sessions.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="basketball-outline" size={36} color={colors.textGhost} />
              <Text style={styles.emptyText}>No sessions yet</Text>
              <Text style={styles.emptyHint}>Record or pick a video to analyze your shots</Text>
            </View>
          ) : (
            summaries.slice(0, 30).map(({ s, sum }, i, list) => {
              const left = sum.total - sum.labeled;
              const when = new Date(s.timestamp).toLocaleString('en-GB',
                { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
              const stat = sum.total
                ? `${sum.made} / ${sum.total} made${sum.pct != null ? ` · ${sum.pct}%` : ''}`
                : 'No shots found';
              return (
                <View key={s.id}>
                  <TouchableOpacity style={styles.sessionRow} activeOpacity={0.7}
                    onPress={() => openSession(s.id)}
                    accessibilityRole="button"
                    accessibilityLabel={`${when}. ${stat}. ${left > 0 ? `${left} to review` : sum.total ? 'Reviewed' : ''}`}>
                    <View style={{ flex: 1, gap: 3 }}>
                      <Text style={styles.sessionTitle}>{when}</Text>
                      <Text style={styles.sessionMeta}>{stat} · {formatTimeAgo(s.timestamp)}</Text>
                    </View>
                    {left > 0 ? (
                      <View style={styles.badge}>
                        <Text style={styles.badgeText}>{left} to review</Text>
                      </View>
                    ) : sum.total > 0 ? (
                      <Text style={styles.reviewed}>Reviewed</Text>
                    ) : null}
                    <Ionicons name="chevron-forward" size={16} color={colors.textGhost} />
                  </TouchableOpacity>
                  {i < list.length - 1 && <Divider />}
                </View>
              );
            })
          )}
        </Card>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  scroll: { flex: 1 },
  scrollContent: { padding: spacing.lg, gap: spacing.lg, paddingBottom: 40 },

  header: { flexDirection: 'row', alignItems: 'flex-start', paddingTop: spacing.sm, paddingHorizontal: 2 },
  title: { fontSize: 34, fontWeight: '700', color: colors.white, letterSpacing: 0.4 },
  date: { fontSize: font.md, color: colors.textMuted, marginTop: 2 },
  iconBtn: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center', marginRight: -10 },

  hero: { gap: spacing.lg, padding: spacing.xl },
  tallyRow: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between' },
  tally: { fontSize: 56, fontWeight: '700', color: colors.white, letterSpacing: -1, fontVariant: ['tabular-nums'] },
  tallyOf: { fontSize: 32, fontWeight: '600', color: colors.textDim },
  pct: { fontSize: 28, fontWeight: '600', color: colors.green, fontVariant: ['tabular-nums'] },
  tallyLbl: { fontSize: font.md, color: colors.textMuted, marginTop: -2 },
  dots: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: spacing.lg },
  dot: { width: 10, height: 10, borderRadius: 5 },
  dotMade: { backgroundColor: colors.green },
  dotMissed: { borderWidth: 1.5, borderColor: colors.textDim },

  recordBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm,
    backgroundColor: colors.orange, borderRadius: radius.lg, minHeight: 50,
  },
  recordText: { fontSize: 17, fontWeight: '700', color: colors.white },

  reviewRow: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.md,
    backgroundColor: colors.bgCard, borderRadius: radius.lg, padding: spacing.lg, minHeight: 44,
  },
  reviewTitle: { fontSize: font.md, fontWeight: '600', color: colors.white },
  reviewSub: { fontSize: font.base, color: colors.textMuted, marginTop: 2 },

  listHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 2 },
  listSub: { fontSize: font.base, color: colors.textDim },

  sessionRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.lg, minHeight: 56 },
  sessionTitle: { fontSize: font.md, color: colors.white },
  sessionMeta: { fontSize: font.base, color: colors.textMuted },
  badge: { backgroundColor: colors.yellowFade, borderRadius: radius.full, paddingHorizontal: spacing.sm, paddingVertical: 3 },
  badgeText: { fontSize: font.sm, color: colors.yellow, fontWeight: '600' },
  reviewed: { fontSize: font.sm, color: colors.textDim },

  emptyState: { alignItems: 'center', paddingVertical: 40, gap: spacing.sm },
  emptyText: { fontSize: font.md, color: colors.textMuted, marginTop: spacing.sm },
  emptyHint: { fontSize: font.base, color: colors.textDim, textAlign: 'center', paddingHorizontal: spacing.xl },
});
