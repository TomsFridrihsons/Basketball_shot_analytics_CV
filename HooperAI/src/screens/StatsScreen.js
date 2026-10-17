import React, { useState, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, RefreshControl,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import Svg, { Circle } from 'react-native-svg';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card, SectionLabel, Divider } from '../components';
import { getShots, calcStats, calcZoneStats } from '../utils/storage';

// ── Ring chart ───────────────────────────────────────────────────────────────
function PctRing({ pct, size = 80 }) {
  const r = (size / 2) - 7;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;

  return (
    <View style={{ width: size, height: size, position: 'relative' }}>
      <Svg width={size} height={size} style={{ position: 'absolute', top: 0, left: 0, transform: [{ rotate: '-90deg' }] }}>
        <Circle cx={size/2} cy={size/2} r={r} stroke="#1e1e1e" strokeWidth={7} fill="none" />
        <Circle
          cx={size/2} cy={size/2} r={r}
          stroke={colors.orange}
          strokeWidth={7}
          fill="none"
          strokeDasharray={`${circ}`}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </Svg>
      <View style={[StyleSheet.absoluteFill, { alignItems: 'center', justifyContent: 'center' }]}>
        <Text style={styles.ringPct}>{pct}%</Text>
        <Text style={styles.ringLbl}>FG%</Text>
      </View>
    </View>
  );
}

// ── Zone cell ────────────────────────────────────────────────────────────────
function ZoneCell({ zone }) {
  const pctColor = zone.pct >= 60 ? colors.green : zone.pct >= 45 ? colors.yellow : colors.orange;
  return (
    <View style={styles.zoneCell}>
      <Text style={styles.zoneName}>{zone.name}</Text>
      <Text style={[styles.zonePct, { color: pctColor }]}>{zone.pct}%</Text>
      <Text style={styles.zoneAtt}>{zone.att} att</Text>
      <View style={styles.zoneBar}>
        <View style={[styles.zoneBarFill, { width: `${zone.pct}%`, backgroundColor: pctColor }]} />
      </View>
    </View>
  );
}

// ── Model info row ───────────────────────────────────────────────────────────
function ModelRow({ label, value, valueColor, last }) {
  return (
    <View>
      <View style={styles.modelRow}>
        <Text style={styles.modelKey}>{label}</Text>
        <Text style={[styles.modelVal, valueColor && { color: valueColor }]}>{value}</Text>
      </View>
      {!last && <Divider />}
    </View>
  );
}

export default function StatsScreen() {
  const insets = useSafeAreaInsets();
  const [shots, setShots]       = useState([]);
  const [stats, setStats]       = useState({ total: 0, made: 0, missed: 0, pct: 0 });
  const [zones, setZones]       = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const data = await getShots();
    setShots(data);
    setStats(calcStats(data));
    setZones(calcZoneStats(data));
  }, []);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <Text style={styles.title}>Analytics</Text>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.orange} />}
      >
        {/* FG% hero */}
        <Card style={styles.heroCard}>
          <PctRing pct={stats.pct} size={80} />
          <View style={styles.heroDetails}>
            <View style={styles.heroRow}>
              <Text style={styles.heroKey}>Total shots</Text>
              <Text style={styles.heroVal}>{stats.total}</Text>
            </View>
            <Divider style={{ marginVertical: 4 }} />
            <View style={styles.heroRow}>
              <Text style={styles.heroKey}>Made</Text>
              <Text style={[styles.heroVal, { color: colors.green }]}>{stats.made}</Text>
            </View>
            <Divider style={{ marginVertical: 4 }} />
            <View style={styles.heroRow}>
              <Text style={styles.heroKey}>Missed</Text>
              <Text style={[styles.heroVal, { color: colors.orange }]}>{stats.missed}</Text>
            </View>
          </View>
        </Card>

        {/* Zone breakdown */}
        <View>
          <SectionLabel>Shot zones</SectionLabel>
          {zones.length === 0 ? (
            <Card>
              <Text style={styles.emptyText}>No zone data yet — record some shots first.</Text>
            </Card>
          ) : (
            <View style={styles.zoneGrid}>
              {zones.map(z => <ZoneCell key={z.name} zone={z} />)}
            </View>
          )}
        </View>

        {/* Model info */}
        <View>
          <SectionLabel>Model</SectionLabel>
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <ModelRow label="Version"        value="lstm_v5" />
            <ModelRow label="Threshold"      value="0.50"              valueColor={colors.green} />
            <ModelRow label="Features"       value="18-dim LSTM" />
            <ModelRow label="NaN handling"   value="ffill → bfill"    valueColor={colors.green} />
            <ModelRow label="Class balance"  value="compute_class_weight" />
            <ModelRow label="Architecture"   value="BiLSTM + attention" last />
          </Card>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  header: {
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.md,
    borderBottomWidth: 0.5,
    borderBottomColor: '#1e1e1e',
  },
  title: { fontSize: font.md, fontWeight: '500', color: colors.white },
  scroll: { flex: 1 },
  scrollContent: { padding: spacing.lg, gap: spacing.lg, paddingBottom: 40 },

  heroCard: { flexDirection: 'row', gap: spacing.xl, alignItems: 'center' },
  heroDetails: { flex: 1 },
  heroRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 4 },
  heroKey: { fontSize: font.base, color: colors.textDim },
  heroVal: { fontSize: font.base, color: colors.text, fontWeight: '500' },

  ringPct: { fontSize: 17, fontWeight: '500', color: colors.white, lineHeight: 20 },
  ringLbl: { fontSize: 9, color: colors.textDim, marginTop: 1 },

  zoneGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  zoneCell: {
    width: '31.5%',
    backgroundColor: colors.bgCard,
    borderRadius: radius.md,
    borderWidth: 0.5,
    borderColor: colors.border,
    padding: spacing.sm,
  },
  zoneName: { fontSize: 9, color: colors.textGhost, marginBottom: 4 },
  zonePct:  { fontSize: 16, fontWeight: '500', color: colors.white },
  zoneAtt:  { fontSize: 9, color: colors.textGhost, marginTop: 1 },
  zoneBar: {
    height: 2,
    backgroundColor: colors.border,
    borderRadius: 1,
    marginTop: 5,
    overflow: 'hidden',
  },
  zoneBarFill: { height: '100%', borderRadius: 1 },

  modelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  modelKey: { fontSize: font.base, color: colors.textDim },
  modelVal: { fontSize: font.base, color: colors.text, fontFamily: 'monospace' },

  emptyText: { fontSize: font.base, color: colors.textDim, textAlign: 'center', padding: spacing.lg },
});
