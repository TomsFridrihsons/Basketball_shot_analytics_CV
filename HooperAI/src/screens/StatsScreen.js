import React, { useState, useCallback } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity, StyleSheet, RefreshControl,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Line, Rect } from 'react-native-svg';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card, SectionLabel, Divider } from '../components';
import { getShots, getSessions, summarizeShots, calcModelStats, NOT_A_SHOT } from '../utils/storage';
import { getSettings } from '../services/settings';
import { syncSessions } from '../services/sync';

// One question: is my shooting getting better? Period → FG% and its change →
// FG% per session → volume → how I shoot (metrics) → how far to trust the numbers.

const DAY = 24 * 3600 * 1000;
const PERIODS = [
  { key: '7', label: '7 days', days: 7 },
  { key: '30', label: '30 days', days: 30 },
  { key: 'all', label: 'All', days: null },
];
const CHART_H = 140;
const MAX_BARS = 20;

const pctOf = (made, total) => (total ? Math.round((made / total) * 100) : null);
const avg = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

function inWindow(ts, days, offset = 0) {
  if (days == null) return offset === 0;
  const age = Date.now() - new Date(ts).getTime();
  return age >= offset * days * DAY && age < (offset + 1) * days * DAY;
}

// ── Segmented control ────────────────────────────────────────────────────────
function Segmented({ value, onChange }) {
  return (
    <View style={styles.segment} accessibilityRole="radiogroup" accessibilityLabel="Time period">
      {PERIODS.map(p => {
        const on = p.key === value;
        return (
          <TouchableOpacity key={p.key} style={[styles.segBtn, on && styles.segOn]} onPress={() => onChange(p.key)}
            activeOpacity={0.8} accessibilityRole="radio" accessibilityState={{ checked: on }}>
            <Text style={[styles.segText, on && styles.segTextOn]}>{p.label}</Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

// ── FG% per session: fixed 0–100 axis, average line, latest session brightest ─
function SessionChart({ data, average, width }) {
  const plotW = Math.max(0, width - 36);          // room for the axis labels on the right
  const n = data.length;
  const gap = n > 12 ? 3 : 6;
  const barW = n ? Math.max(4, Math.min(28, (plotW - gap * (n - 1)) / n)) : 0;
  const y = (pct) => CHART_H - (pct / 100) * CHART_H;
  return (
    <View style={{ flexDirection: 'row' }}>
      <Svg width={plotW} height={CHART_H}>
        {[0, 50, 100].map(t => (
          <Line key={t} x1={0} x2={plotW} y1={y(t) + (t === 0 ? -0.5 : t === 100 ? 0.5 : 0)} y2={y(t) + (t === 0 ? -0.5 : t === 100 ? 0.5 : 0)}
            stroke={colors.border} strokeWidth={1} />
        ))}
        {data.map((d, i) => {
          const h = Math.max(2, (d.pct / 100) * CHART_H);
          return (
            <Rect key={d.id} x={i * (barW + gap)} y={CHART_H - h} width={barW} height={h} rx={Math.min(4, barW / 2)}
              fill={i === n - 1 ? colors.white : colors.textDim} />
          );
        })}
        {average != null && (
          <Line x1={0} x2={plotW} y1={y(average)} y2={y(average)}
            stroke={colors.green} strokeWidth={1.5} strokeDasharray="4 4" />
        )}
      </Svg>
      {/* tick labels are visual only; the chart's label carries the data for VoiceOver */}
      <View style={styles.axis} importantForAccessibility="no-hide-descendants" accessibilityElementsHidden>
        <Text style={styles.axisText}>100%</Text>
        <Text style={styles.axisText}>50%</Text>
        <Text style={styles.axisText}>0%</Text>
      </View>
    </View>
  );
}

function Metric({ label, value, unit }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricVal} maxFontSizeMultiplier={1.4}>
        {value ?? '–'}<Text style={styles.metricUnit}>{value != null ? unit : ''}</Text>
      </Text>
      <Text style={styles.metricLbl}>{label}</Text>
    </View>
  );
}

function Row({ label, value, last }) {
  return (
    <View>
      <View style={styles.row}>
        <Text style={styles.rowKey}>{label}</Text>
        <Text style={styles.rowVal}>{value}</Text>
      </View>
      {!last && <Divider />}
    </View>
  );
}

export default function StatsScreen() {
  const insets = useSafeAreaInsets();
  const [period, setPeriod]     = useState('30');
  const [shots, setShots]       = useState([]);
  const [sessions, setSessions] = useState([]);
  const [settings, setSettings] = useState({});
  const [chartW, setChartW]     = useState(0);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    await syncSessions();   // labels made on the laptop / other devices
    setShots(await getShots());
    setSessions((await getSessions()).filter(s => s.result));
    setSettings(await getSettings());
  }, []);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  // ── numbers for the selected period ──
  const days = PERIODS.find(p => p.key === period).days;
  const periodName = days == null ? 'all time' : `the last ${days} days`;

  const now = shots.filter(s => inWindow(s.timestamp, days));
  const before = days == null ? [] : shots.filter(s => inWindow(s.timestamp, days, 1));
  const made = now.filter(s => s.outcome === 'made').length;
  const fg = pctOf(made, now.length);
  const fgBefore = pctOf(before.filter(s => s.outcome === 'made').length, before.length);
  const delta = fg != null && fgBefore != null ? fg - fgBefore : null;

  const played = sessions
    .filter(s => inWindow(s.timestamp, days))
    .map(s => ({ id: s.id, timestamp: s.timestamp, ...summarizeShots(s.result.shots || []) }))
    .filter(s => s.total > 0)
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  const bars = played.slice(-MAX_BARS);
  const best = played.reduce((b, s) => (b == null || s.pct > b.pct ? s : b), null);

  // shot metrics, only where the camera angle allowed a reliable measurement
  const measured = sessions
    .filter(s => inWindow(s.timestamp, days))
    .flatMap(s => s.result.shots || [])
    .filter(sh => sh.label !== NOT_A_SHOT && sh.metrics?.reliable);
  const m = (key, digits = 0) => {
    const v = avg(measured.map(sh => sh.metrics[key]).filter(x => x != null));
    return v == null ? null : v.toFixed(digits);
  };

  const model = calcModelStats(now);
  const fmtDay = ts => new Date(ts).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });

  const chartTitle = bars.length < 2 ? null
    : bars[bars.length - 1].pct > bars[0].pct ? `Up from ${bars[0].pct}% to ${bars[bars.length - 1].pct}% since ${fmtDay(bars[0].timestamp)}`
    : bars[bars.length - 1].pct < bars[0].pct ? `Down from ${bars[0].pct}% to ${bars[bars.length - 1].pct}% since ${fmtDay(bars[0].timestamp)}`
    : `Steady at ${bars[0].pct}% since ${fmtDay(bars[0].timestamp)}`;

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.orange} />}
      >
        <Text style={styles.title} accessibilityRole="header">Stats</Text>
        <Segmented value={period} onChange={setPeriod} />

        {now.length === 0 ? (
          <Card style={styles.empty}>
            <Ionicons name="stats-chart-outline" size={34} color={colors.textMuted} />
            <Text style={styles.emptyTitle}>No shots in {periodName}</Text>
            <Text style={styles.emptyBody}>Record a session and your shooting trends show up here.</Text>
          </Card>
        ) : (
          <>
            {/* Headline: FG% and how it moved against the previous period */}
            <Card style={styles.hero}>
              <View accessible accessibilityLabel={
                `Field goal percentage ${fg} percent, ${made} of ${now.length} made in ${periodName}.`
                + (delta != null ? ` ${delta === 0 ? 'No change' : `${delta > 0 ? 'Up' : 'Down'} ${Math.abs(delta)} points`} from the previous ${days} days.` : '')
              }>
                <Text style={styles.heroLbl}>Field goal %</Text>
                <Text style={styles.heroPct} maxFontSizeMultiplier={1.4}>{fg}%</Text>
                <Text style={styles.heroSub}>{made} of {now.length} made</Text>
                {delta != null && (
                  <View style={styles.delta}>
                    <Ionicons name={delta > 0 ? 'arrow-up' : delta < 0 ? 'arrow-down' : 'remove'} size={14}
                      color={delta > 0 ? colors.green : colors.textMuted} />
                    <Text style={[styles.deltaText, delta > 0 && { color: colors.green }]}>
                      {delta === 0 ? 'Same as' : `${Math.abs(delta)} pts ${delta > 0 ? 'better than' : 'lower than'}`} the {days} days before
                    </Text>
                  </View>
                )}
              </View>
            </Card>

            {/* FG% per session */}
            {bars.length > 0 && (
              <View>
                <SectionLabel>FG% by session</SectionLabel>
                <Card style={{ gap: spacing.md }}>
                  {chartTitle && <Text style={styles.chartTitle}>{chartTitle}</Text>}
                  <View onLayout={e => setChartW(e.nativeEvent.layout.width)}
                    accessible accessibilityRole="image"
                    accessibilityLabel={`Bar chart of field goal percentage for ${bars.length} sessions. `
                      + bars.map(b => `${fmtDay(b.timestamp)}: ${b.pct} percent`).join(', ')}>
                    {chartW > 0 && <SessionChart data={bars} average={fg} width={chartW} />}
                  </View>
                  <View style={styles.legend} importantForAccessibility="no-hide-descendants" accessibilityElementsHidden>
                    <Text style={styles.legendText}>{fmtDay(bars[0].timestamp)}</Text>
                    <View style={styles.legendAvg}>
                      <View style={styles.dash} />
                      <Text style={styles.legendText}>average {fg}%</Text>
                    </View>
                    <Text style={styles.legendText}>{fmtDay(bars[bars.length - 1].timestamp)}</Text>
                  </View>
                </Card>
              </View>
            )}

            {/* Volume */}
            <Card style={{ padding: 0, overflow: 'hidden' }}>
              <Row label="Sessions" value={String(played.length)} />
              <Row label="Shots" value={String(now.length)} />
              <Row label="Shots per session" value={played.length ? String(Math.round(now.length / played.length)) : '–'} />
              <Row label="Best session" value={best ? `${best.pct}% · ${fmtDay(best.timestamp)}` : '–'} last />
            </Card>

            {/* How you shoot */}
            {measured.length > 0 && (
              <View>
                <SectionLabel>Your shot, on average</SectionLabel>
                <Card style={{ gap: spacing.md }}>
                  <View style={styles.metrics}>
                    <Metric label="Release angle" value={m('release_angle_deg')} unit="°" />
                    <Metric label="Arc height" value={m('arc_height_m', 1)} unit=" m" />
                    <Metric label="Entry angle" value={m('entry_angle_deg')} unit="°" />
                  </View>
                  <Text style={styles.note}>
                    From {measured.length} {measured.length === 1 ? 'shot' : 'shots'} filmed from the side, where the camera can measure the arc.
                  </Text>
                </Card>
              </View>
            )}

            {/* How far to trust the numbers */}
            <View>
              <SectionLabel>Detection</SectionLabel>
              <Card style={{ padding: 0, overflow: 'hidden' }}>
                <Row label="Made / missed accuracy on new videos" value={settings.predictorAccuracy != null
                  ? `${Math.round(settings.predictorAccuracy * 100)}%` : '97%'} />
                <Row label="Shots you reviewed" value={String(model.labeled)} />
                <Row label="Model agreed with you" value={model.pct != null ? `${model.pct}%` : '–'} last />
              </Card>
              <Text style={[styles.note, { marginTop: spacing.sm, paddingHorizontal: 2 }]}>
                {settings.predictorLabel || 'Rim-pass rule'} · reviewing shots makes these numbers more accurate.
              </Text>
            </View>
          </>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  scroll: { flex: 1 },
  scrollContent: { padding: spacing.lg, gap: spacing.lg, paddingBottom: 40 },
  title: { fontSize: 34, fontWeight: '700', color: colors.white, letterSpacing: 0.4, paddingTop: spacing.sm, paddingHorizontal: 2 },

  segment: { flexDirection: 'row', backgroundColor: colors.bgCard, borderRadius: radius.md, padding: 3 },
  segBtn: { flex: 1, minHeight: 34, borderRadius: radius.sm, alignItems: 'center', justifyContent: 'center' },
  segOn: { backgroundColor: '#2c2c2c' },
  segText: { fontSize: font.base, color: colors.textMuted, fontWeight: '500' },
  segTextOn: { color: colors.white, fontWeight: '600' },

  empty: { alignItems: 'center', gap: spacing.sm, paddingVertical: 40 },
  emptyTitle: { fontSize: 17, fontWeight: '600', color: colors.white, marginTop: spacing.sm },
  emptyBody: { fontSize: font.md, color: colors.textMuted, textAlign: 'center', paddingHorizontal: spacing.lg },

  hero: { padding: spacing.xl },
  heroLbl: { fontSize: font.md, color: colors.textMuted },
  heroPct: { fontSize: 56, fontWeight: '700', color: colors.white, letterSpacing: -1, fontVariant: ['tabular-nums'] },
  heroSub: { fontSize: font.md, color: colors.textMuted, marginTop: -2 },
  delta: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: spacing.md },
  deltaText: { fontSize: font.base, color: colors.textMuted, fontWeight: '500' },

  chartTitle: { fontSize: font.md, fontWeight: '600', color: colors.white },
  axis: { width: 36, height: CHART_H, justifyContent: 'space-between', alignItems: 'flex-end', marginTop: -6, marginBottom: -6 },
  axisText: { fontSize: font.xs, color: colors.textDim, fontVariant: ['tabular-nums'] },
  legend: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginRight: 36 },
  legendText: { fontSize: font.xs, color: colors.textDim },
  legendAvg: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  dash: { width: 14, borderTopWidth: 1.5, borderColor: colors.green, borderStyle: 'dashed' },

  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: spacing.md,
         paddingHorizontal: spacing.lg, minHeight: 48 },
  rowKey: { flex: 1, fontSize: font.md, color: colors.text },
  rowVal: { fontSize: font.md, color: colors.white, fontWeight: '600', fontVariant: ['tabular-nums'] },

  metrics: { flexDirection: 'row' },
  metric: { flex: 1, alignItems: 'center', gap: 2 },
  metricVal: { fontSize: 24, fontWeight: '700', color: colors.white, fontVariant: ['tabular-nums'] },
  metricUnit: { fontSize: font.md, fontWeight: '500', color: colors.textMuted },
  metricLbl: { fontSize: font.sm, color: colors.textMuted },
  note: { fontSize: font.sm, color: colors.textMuted, lineHeight: 17 },
});
