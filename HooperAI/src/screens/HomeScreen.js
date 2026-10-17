import React, { useState, useCallback } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, RefreshControl,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card, SectionLabel, PrimaryButton, Divider } from '../components';
import { getShots, getSessions, sessionSummary, formatTimeAgo, calcStats } from '../utils/storage';
import * as api from '../services/api';

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [shots, setShots] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [stats, setStats] = useState({ total: 0, made: 0, missed: 0, pct: 0 });
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const data = await getShots();
    setShots(data);
    setStats(calcStats(data));
    // sessions saved before review support have no result to open
    const local = (await getSessions()).filter(s => s.result);
    setSessions(local);
    // Add sessions the laptop knows but this phone doesn't (older app version, other device)
    try {
      const known = new Set(local.map(s => s.id));
      const remote = (await api.sessions())
        .filter(s => !known.has(s.id))
        .map(s => ({ id: s.id, timestamp: s.created, summary: {
          total: s.total, made: s.made, labeled: s.labeled,
          pct: s.total ? Math.round((s.made / s.total) * 100) : null,
        } }));
      if (remote.length) {
        setSessions([...local, ...remote].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)));
      }
    } catch (_) { /* laptop offline: local sessions only */ }
  }, []);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  const todayShots = shots.filter(s => {
    const d = new Date(s.timestamp);
    const now = new Date();
    return d.toDateString() === now.toDateString();
  });
  const todayStats = calcStats(todayShots);

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.logoRow}>
          <View style={styles.logoIcon}><Text style={{ fontSize: 15 }}>🏀</Text></View>
          <Text style={styles.logoText}>Hooper<Text style={{ color: colors.orange }}>AI</Text></Text>
          <TouchableOpacity onPress={() => navigation.navigate('Settings')} style={{ marginLeft: 'auto', padding: 4 }}
            accessibilityLabel="Settings">
            <Ionicons name="settings-outline" size={20} color={colors.textMuted} />
          </TouchableOpacity>
        </View>
        <Text style={styles.headerSub}>
          {new Date().toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}
        </Text>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.orange} />}
      >
        {/* Today's session card */}
        <Card style={styles.sessionCard}>
          <SectionLabel>Today's session</SectionLabel>
          <View style={styles.statRow}>
            <View style={styles.statCell}>
              <Text style={styles.statVal}>{todayStats.total}</Text>
              <Text style={styles.statLbl}>shots</Text>
            </View>
            <View style={[styles.statCell, styles.statCellBorder]}>
              <Text style={[styles.statVal, { color: colors.green }]}>{todayStats.pct}%</Text>
              <Text style={styles.statLbl}>made</Text>
            </View>
            <View style={styles.statCell}>
              <Text style={[styles.statVal, { color: colors.orange }]}>{todayStats.missed}</Text>
              <Text style={styles.statLbl}>missed</Text>
            </View>
          </View>
          <PrimaryButton
            label="Start new session"
            onPress={() => navigation.navigate('Record')}
          />
        </Card>

        {/* Sessions */}
        <View style={styles.logHeader}>
          <SectionLabel style={{ marginBottom: 0 }}>Sessions</SectionLabel>
          <Text style={styles.logSub}>{sessions.length} analyzed</Text>
        </View>

        <Card style={{ padding: 0, overflow: 'hidden' }}>
          {sessions.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="basketball-outline" size={36} color={colors.textGhost} />
              <Text style={styles.emptyText}>No sessions yet</Text>
              <Text style={styles.emptyHint}>Record or pick a video to analyze your shots</Text>
            </View>
          ) : (
            sessions.slice(0, 30).map((s, i, list) => {
              const sum = s.result ? sessionSummary(s) : s.summary;
              const toReview = sum.total - sum.labeled;
              return (
                <View key={s.id}>
                  <TouchableOpacity style={styles.sessionRow} activeOpacity={0.7}
                    onPress={() => navigation.navigate('Session', { sessionId: s.id })}>
                    <View style={styles.sessionIcon}>
                      <Ionicons name="film-outline" size={18} color={colors.textMuted} />
                    </View>
                    <View style={{ flex: 1, gap: 3 }}>
                      <Text style={styles.sessionTitle}>
                        {new Date(s.timestamp).toLocaleString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}
                      </Text>
                      <Text style={styles.sessionMeta}>
                        {sum.total} {sum.total === 1 ? 'shot' : 'shots'} · {sum.made} made
                        {sum.pct != null ? ` · ${sum.pct}%` : ''} · {formatTimeAgo(s.timestamp)}
                      </Text>
                      {sum.total > 0 && (
                        <View style={styles.progressTrack}>
                          <View style={[styles.progressFill, { width: `${(sum.labeled / sum.total) * 100}%` }]} />
                        </View>
                      )}
                    </View>
                    {toReview > 0 ? (
                      <View style={styles.reviewBadge}>
                        <Text style={styles.reviewText}>{toReview} to review</Text>
                      </View>
                    ) : sum.total > 0 ? (
                      <Ionicons name="checkmark-circle" size={18} color={colors.green} />
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

  sessionRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.md },
  sessionIcon: {
    width: 36, height: 36, borderRadius: 18, backgroundColor: colors.bgCardDeep,
    alignItems: 'center', justifyContent: 'center',
  },
  sessionTitle: { fontSize: font.base, color: colors.white },
  sessionMeta: { fontSize: font.xs, color: colors.textDim },
  progressTrack: { height: 3, backgroundColor: colors.bgCardDeep, borderRadius: radius.full, overflow: 'hidden', marginTop: 2 },
  progressFill: { height: '100%', backgroundColor: colors.green },
  reviewBadge: {
    backgroundColor: colors.yellowFade, borderRadius: radius.full,
    paddingHorizontal: spacing.sm, paddingVertical: 2,
  },
  reviewText: { fontSize: font.xs, color: colors.yellow, fontWeight: '600' },

  header: {
    backgroundColor: colors.bgScreen,
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.md,
    borderBottomWidth: 0.5,
    borderBottomColor: '#1e1e1e',
  },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: 2 },
  logoIcon: {
    width: 28, height: 28,
    backgroundColor: colors.orange,
    borderRadius: radius.sm,
    alignItems: 'center', justifyContent: 'center',
  },
  logoText: { fontSize: font.lg, fontWeight: '500', color: colors.white, letterSpacing: -0.3 },
  headerSub: { fontSize: font.sm, color: colors.textDim },

  scroll: { flex: 1 },
  scrollContent: { padding: spacing.lg, gap: spacing.lg, paddingBottom: 40 },

  sessionCard: { gap: spacing.md },
  statRow: { flexDirection: 'row', marginBottom: spacing.sm },
  statCell: { flex: 1, alignItems: 'center', paddingVertical: spacing.sm },
  statCellBorder: {
    borderLeftWidth: 0.5, borderRightWidth: 0.5,
    borderColor: colors.border,
  },
  statVal: { fontSize: 22, fontWeight: '500', color: colors.white },
  statLbl: { fontSize: font.xs, color: colors.textDim, marginTop: 2 },

  logHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 2,
  },
  logSub: { fontSize: font.sm, color: colors.textGhost },

  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    gap: spacing.sm,
  },
  emptyText: { fontSize: font.md, color: colors.textMuted, marginTop: spacing.sm },
  emptyHint: { fontSize: font.sm, color: colors.textGhost },
});
