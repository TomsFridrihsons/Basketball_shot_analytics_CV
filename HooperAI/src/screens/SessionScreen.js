import React, { useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation, useRoute } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, font } from '../utils/theme';
import SessionReview from '../components/SessionReview';
import { getSession, saveSession } from '../utils/storage';
import * as api from '../services/api';

// Reopen an analyzed session from Home to watch and label its shots.
export default function SessionScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const { sessionId } = useRoute().params;
  const [session, setSession] = useState(undefined);

  useEffect(() => {
    (async () => {
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
    })();
  }, [sessionId]);

  const date = session?.timestamp
    ? new Date(session.timestamp).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
    : '';

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} accessibilityLabel="Back" style={{ padding: 4 }}>
          <Ionicons name="chevron-back" size={22} color={colors.textMuted} />
        </TouchableOpacity>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>Session review</Text>
          {!!date && <Text style={styles.sub}>{date}</Text>}
        </View>
      </View>
      {session === undefined && <ActivityIndicator color={colors.orange} style={{ marginTop: 40 }} />}
      {session === null && <Text style={styles.missing}>This session is not on this phone and the laptop is offline.</Text>}
      {session && <SessionReview session={session} />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  header: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.sm,
    paddingHorizontal: spacing.lg, paddingBottom: spacing.md,
    borderBottomWidth: 0.5, borderBottomColor: '#1e1e1e',
  },
  title: { fontSize: font.md, fontWeight: '500', color: colors.white },
  sub: { fontSize: font.xs, color: colors.textDim, marginTop: 1 },
  missing: { color: colors.textMuted, textAlign: 'center', marginTop: 40 },
});
