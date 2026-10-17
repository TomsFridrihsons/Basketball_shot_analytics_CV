import React, { useState, useCallback } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, RefreshControl,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card, SectionLabel, OutcomeChip, ConfBadge, PrimaryButton, Divider } from '../components';
import { getShots, formatTimeAgo, calcStats } from '../utils/storage';

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [shots, setShots] = useState([]);
  const [stats, setStats] = useState({ total: 0, made: 0, missed: 0, pct: 0 });
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const data = await getShots();
    setShots(data);
    setStats(calcStats(data));
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

        {/* Shot log */}
        <View style={styles.logHeader}>
          <SectionLabel style={{ marginBottom: 0 }}>Shot log</SectionLabel>
          <Text style={styles.logSub}>{shots.length} total</Text>
        </View>

        <Card style={{ padding: 0, overflow: 'hidden' }}>
          {shots.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="basketball-outline" size={36} color={colors.textGhost} />
              <Text style={styles.emptyText}>No shots recorded yet</Text>
              <Text style={styles.emptyHint}>Start a session to begin tracking</Text>
            </View>
          ) : (
            shots.slice(0, 30).map((shot, i) => (
              <View key={shot.id}>
                <View style={styles.shotRow}>
                  <Text style={styles.shotNum}>{i + 1}</Text>
                  <OutcomeChip outcome={shot.outcome} />
                  <View style={styles.shotInfo}>
                    <Text style={styles.shotZone}>
                      {shot.zone || (shot.videoTime != null
                        ? `Shot ${shot.shotNumber} · ${Math.floor(shot.videoTime / 60)}:${String(Math.floor(shot.videoTime % 60)).padStart(2, '0')}${shot.labeled ? '' : ' · unlabeled'}`
                        : 'Unknown zone')}
                    </Text>
                    <Text style={styles.shotTime}>{formatTimeAgo(shot.timestamp)}</Text>
                  </View>
                  <ConfBadge conf={shot.confidence || 0} />
                </View>
                {i < shots.length - 1 && <Divider />}
              </View>
            ))
          )}
        </Card>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },

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

  shotRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  shotNum: { fontSize: font.sm, color: colors.textGhost, width: 18, textAlign: 'right' },
  shotInfo: { flex: 1 },
  shotZone: { fontSize: font.base, color: colors.text },
  shotTime: { fontSize: font.xs, color: colors.textGhost, marginTop: 1 },

  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    gap: spacing.sm,
  },
  emptyText: { fontSize: font.md, color: colors.textMuted, marginTop: spacing.sm },
  emptyHint: { fontSize: font.sm, color: colors.textGhost },
});
