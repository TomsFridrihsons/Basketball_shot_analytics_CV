import React, { useCallback, useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation, useRoute } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import SessionReview from '../components/SessionReview';
import { getSession, saveSession } from '../utils/storage';
import * as api from '../services/api';

// Watch and label an analyzed session. Opened from Home and right after an analysis.
// States: loading → review | unavailable (not on this phone and the laptop is offline).
export default function SessionScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const { sessionId } = useRoute().params;
  const [session, setSession] = useState(undefined);   // undefined = loading, null = unavailable

  const load = useCallback(async () => {
    setSession(undefined);
    const local = await getSession(sessionId);
    if (local) { setSession(local); return; }
    // Not stored on this phone (older app version / other device): load from the laptop
    try {
      const job = await api.job(sessionId);
      if (!job.result) { setSession(null); return; }
      // keep a local copy so labels made here show up on Home / Stats
      await saveSession(sessionId, job.result, { timestamp: job.created });
      setSession(await getSession(sessionId));
    } catch (_) {
      setSession(null);
    }
  }, [sessionId]);

  useEffect(() => { load(); }, [load]);

  const date = session?.timestamp
    ? new Date(session.timestamp).toLocaleString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
    : 'Session';

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.back}
          accessibilityRole="button" accessibilityLabel="Back">
          <Ionicons name="chevron-back" size={26} color={colors.white} />
        </TouchableOpacity>
        <Text style={styles.title} numberOfLines={1} accessibilityRole="header">{date}</Text>
        <View style={styles.back} />
      </View>

      {session === undefined && (
        <View style={styles.center}>
          <ActivityIndicator color={colors.textMuted} />
          <Text style={styles.body}>Opening the session</Text>
        </View>
      )}
      {session === null && (
        <View style={styles.center}>
          <Ionicons name="cloud-offline-outline" size={34} color={colors.textMuted} />
          <Text style={styles.heading}>Session not available</Text>
          <Text style={styles.body}>
            It isn't stored on this phone and your laptop can't be reached. Start the HooperAI server and try again.
          </Text>
          <TouchableOpacity style={styles.retry} onPress={load} accessibilityRole="button">
            <Text style={styles.retryText}>Try again</Text>
          </TouchableOpacity>
        </View>
      )}
      {session && <SessionReview session={session} />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  header: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: spacing.xs, minHeight: 48,
  },
  back: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' },
  title: { flex: 1, fontSize: 17, fontWeight: '600', color: colors.white, textAlign: 'center' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md, padding: spacing.xxl },
  heading: { fontSize: 20, fontWeight: '700', color: colors.white },
  body: { fontSize: font.md, color: colors.textMuted, textAlign: 'center', lineHeight: 21 },
  retry: {
    minHeight: 44, paddingHorizontal: spacing.xl, borderRadius: radius.lg,
    backgroundColor: colors.bgCard, alignItems: 'center', justifyContent: 'center', marginTop: spacing.sm,
  },
  retryText: { fontSize: 17, fontWeight: '600', color: colors.white },
});
