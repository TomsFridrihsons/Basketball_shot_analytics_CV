import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { colors, radius, spacing, font } from '../utils/theme';

// ── Section label ────────────────────────────────────────────────────────────
export function SectionLabel({ children, style }) {
  return (
    <Text style={[styles.sectionLabel, style]}>{children}</Text>
  );
}

// ── Stat card (dark bg, centered number + label) ─────────────────────────────
export function StatCard({ value, label, valueColor, style }) {
  return (
    <View style={[styles.statCard, style]}>
      <Text style={[styles.statValue, valueColor && { color: valueColor }]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

// ── Confidence badge (pill) ──────────────────────────────────────────────────
export function ConfBadge({ conf }) {
  const level = conf >= 75 ? 'high' : conf >= 60 ? 'med' : 'low';
  const badgeColors = {
    high: { bg: colors.greenFade,  text: colors.green },
    med:  { bg: colors.yellowFade, text: colors.yellow },
    low:  { bg: colors.orangeFade, text: colors.orange },
  };
  const c = badgeColors[level];
  return (
    <View style={[styles.badge, { backgroundColor: c.bg }]}>
      <Text style={[styles.badgeText, { color: c.text }]}>{conf}%</Text>
    </View>
  );
}

// ── Outcome chip (made / missed) ─────────────────────────────────────────────
export function OutcomeChip({ outcome }) {
  const isMade = outcome === 'made';
  return (
    <View style={[styles.outcomeChip, isMade ? styles.madeChip : styles.missedChip]}>
      <Text style={[styles.outcomeText, { color: isMade ? colors.green : colors.orange }]}>
        {isMade ? '✓' : '✕'}
      </Text>
    </View>
  );
}

// ── Card wrapper ─────────────────────────────────────────────────────────────
export function Card({ children, style }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

// ── Row divider ──────────────────────────────────────────────────────────────
export function Divider({ style }) {
  return <View style={[styles.divider, style]} />;
}

// ── Primary button ───────────────────────────────────────────────────────────
export function PrimaryButton({ label, onPress, icon, style }) {
  return (
    <TouchableOpacity style={[styles.primaryBtn, style]} onPress={onPress} activeOpacity={0.85}>
      {icon && <Text style={styles.btnIcon}>{icon}</Text>}
      <Text style={styles.primaryBtnText}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  sectionLabel: {
    fontSize: font.xs,
    color: colors.textGhost,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: spacing.sm,
  },
  statCard: {
    flex: 1,
    backgroundColor: colors.bgCard,
    borderRadius: radius.md,
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 0.5,
    borderColor: colors.border,
  },
  statValue: {
    fontSize: 22,
    fontWeight: '500',
    color: colors.white,
  },
  statLabel: {
    fontSize: font.xs,
    color: colors.textDim,
    marginTop: 2,
  },
  badge: {
    borderRadius: radius.full,
    paddingHorizontal: spacing.sm,
    paddingVertical: 3,
  },
  badgeText: {
    fontSize: font.sm,
    fontWeight: '500',
  },
  outcomeChip: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
  },
  madeChip:   { backgroundColor: colors.greenFade },
  missedChip: { backgroundColor: colors.orangeFade },
  outcomeText: {
    fontSize: 15,
    fontWeight: '600',
  },
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.lg,
    padding: spacing.lg,
    borderWidth: 0.5,
    borderColor: colors.border,
  },
  divider: {
    height: 0.5,
    backgroundColor: colors.borderSub,
  },
  primaryBtn: {
    backgroundColor: colors.orange,
    borderRadius: radius.md,
    paddingVertical: 13,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
  },
  btnIcon: {
    fontSize: 18,
  },
  primaryBtnText: {
    color: colors.white,
    fontSize: font.md,
    fontWeight: '500',
  },
});
