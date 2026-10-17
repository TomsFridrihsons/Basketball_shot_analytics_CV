import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator, Alert,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import Constants from 'expo-constants';
import { colors, radius, spacing, font } from '../utils/theme';
import { Divider } from '../components';
import { getSettings, saveSettings } from '../services/settings';
import { isFirebaseConfigured } from '../config';
import { currentUser, signOut, deleteAccount, authErrorMessage } from '../services/auth';
import * as api from '../services/api';

// Grouped list, iOS style. Changes save as you make them – there is no Save button
// to forget. The laptop is checked on open, so the predictor list is always there.

function Group({ title, footer, children }) {
  return (
    <View>
      {title && <Text style={styles.groupTitle} accessibilityRole="header">{title}</Text>}
      <View style={styles.group}>{children}</View>
      {footer && <Text style={styles.groupFooter}>{footer}</Text>}
    </View>
  );
}

function Field({ label, ...input }) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput style={styles.input} placeholderTextColor={colors.textGhost}
        autoCapitalize="none" autoCorrect={false} clearButtonMode="while-editing"
        accessibilityLabel={label} {...input} />
    </View>
  );
}

function ActionRow({ label, icon, onPress, destructive, busy }) {
  const color = destructive ? colors.red : colors.white;
  return (
    <TouchableOpacity style={styles.actionRow} onPress={onPress} disabled={busy} activeOpacity={0.7}
      accessibilityRole="button">
      {busy ? <ActivityIndicator size="small" color={colors.textMuted} />
        : <Ionicons name={icon} size={20} color={color} />}
      <Text style={[styles.actionText, { color }]}>{label}</Text>
    </TouchableOpacity>
  );
}

export default function SettingsScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [serverUrl, setServerUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [predictor, setPredictor] = useState(null);
  const [conn, setConn] = useState({ state: 'checking' });   // checking | ok | error
  const [deleting, setDeleting] = useState(false);
  const loaded = useRef(false);
  const user = currentUser();

  const check = useCallback(async () => {
    setConn({ state: 'checking' });
    try {
      const h = await api.health();
      const st = await api.status();
      setConn({
        state: 'ok', status: st,
        detail: [st.gpu ? 'GPU' : 'CPU', h.firestore ? 'cloud sync on' : null].filter(Boolean).join(' · '),
      });
    } catch (e) {
      setConn({ state: 'error', detail: e.message });
    }
  }, []);

  useEffect(() => {
    getSettings().then(s => {
      setServerUrl(s.serverUrl);
      setApiKey(s.apiKey);
      setPredictor(s.predictor);
      loaded.current = true;
      check();
    });
  }, [check]);

  // save as you type; re-check the laptop once typing pauses
  useEffect(() => {
    if (!loaded.current) return undefined;
    const t = setTimeout(async () => { await saveSettings({ serverUrl, apiKey }); check(); }, 800);
    return () => clearTimeout(t);
  }, [serverUrl, apiKey, check]);

  const choosePredictor = async (p) => {
    setPredictor(p.config);
    await saveSettings({ predictor: p.config, predictorLabel: p.label, predictorAccuracy: p.unseen_video_accuracy });
  };

  const confirmSignOut = () => Alert.alert('Sign out of HooperAI?', 'Your sessions stay on your laptop.', [
    { text: 'Cancel', style: 'cancel' },
    { text: 'Sign Out', style: 'destructive', onPress: () => signOut() },
  ]);

  const confirmDelete = () => Alert.alert(
    'Delete your account?',
    'This permanently deletes your HooperAI account. Videos and sessions on your laptop are not removed – delete them there if you want them gone. This can\'t be undone.',
    [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete Account', style: 'destructive', onPress: async () => {
          setDeleting(true);
          try { await deleteAccount(); }   // RootNavigator returns to sign-in
          catch (e) { Alert.alert('Account not deleted', authErrorMessage(e)); }
          finally { setDeleting(false); }
        },
      },
    ]);

  const predictors = conn.status?.predictors || [];
  const selected = predictor || conn.status?.default_predictor;

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.back}
          accessibilityRole="button" accessibilityLabel="Back">
          <Ionicons name="chevron-back" size={26} color={colors.white} />
        </TouchableOpacity>
        <Text style={styles.title} accessibilityRole="header">Settings</Text>
        <View style={styles.back} />
      </View>

      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        {/* Account first: who you are */}
        {isFirebaseConfigured && (
          <Group title="Account">
            <View style={styles.accountRow}>
              <View style={styles.avatar}>
                <Text style={styles.avatarText}>{(user?.email || '?')[0].toUpperCase()}</Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.rowLabel} numberOfLines={1}>{user?.email || '–'}</Text>
                <Text style={styles.rowSub}>Signed in with email</Text>
              </View>
            </View>
            <Divider style={styles.inset} />
            <ActionRow label="Sign Out" icon="log-out-outline" onPress={confirmSignOut} />
          </Group>
        )}

        {/* Laptop: live status, then the two fields that change it */}
        <Group title="Laptop server"
          footer="On the same Wi-Fi use the laptop's address, e.g. http://192.168.1.100:8000. Away from home use its Tailscale address, e.g. http://100.x.y.z:8000.">
          <TouchableOpacity style={styles.statusRow} onPress={check} activeOpacity={0.7}
            accessibilityRole="button"
            accessibilityLabel={conn.state === 'ok' ? `Connected. ${conn.detail}` : conn.state === 'error' ? 'Not connected. Check again' : 'Checking the laptop'}>
            {conn.state === 'checking'
              ? <ActivityIndicator size="small" color={colors.textMuted} />
              : <Ionicons name={conn.state === 'ok' ? 'checkmark-circle' : 'alert-circle'} size={22}
                  color={conn.state === 'ok' ? colors.green : colors.yellow} />}
            <View style={{ flex: 1 }}>
              <Text style={styles.rowLabel}>
                {conn.state === 'ok' ? 'Connected' : conn.state === 'error' ? 'Not connected' : 'Checking…'}
              </Text>
              {!!conn.detail && <Text style={styles.rowSub} numberOfLines={2}>{conn.detail}</Text>}
            </View>
            {conn.state === 'error' && <Text style={styles.rowAction}>Check again</Text>}
          </TouchableOpacity>
          <Divider style={styles.inset} />
          <Field label="Address" value={serverUrl} onChangeText={setServerUrl}
            placeholder="http://192.168.1.100:8000" keyboardType="url" textContentType="URL" />
          {(!isFirebaseConfigured || apiKey) && (
            <>
              <Divider style={styles.inset} />
              <Field label="API key" value={apiKey} onChangeText={setApiKey}
                placeholder="HOOPER_API_KEY from the laptop" secureTextEntry />
            </>
          )}
        </Group>

        {/* Predictor: a choice, so a checkmark list, not a text field */}
        {predictors.length > 0 && (
          <Group title="Made / missed predictor"
            footer="Accuracy is measured on videos the predictor never saw. The default suits most sessions.">
            {predictors.map((p, i) => {
              const on = selected === p.config;
              return (
                <View key={p.config}>
                  <TouchableOpacity style={styles.optionRow} onPress={() => choosePredictor(p)} activeOpacity={0.7}
                    accessibilityRole="radio" accessibilityState={{ checked: on }}>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.rowLabel}>{p.label}</Text>
                      {p.unseen_video_accuracy != null && (
                        <Text style={styles.rowSub}>{Math.round(p.unseen_video_accuracy * 100)}% accurate</Text>
                      )}
                    </View>
                    {on && <Ionicons name="checkmark" size={22} color={colors.orange} />}
                  </TouchableOpacity>
                  {i < predictors.length - 1 && <Divider style={styles.inset} />}
                </View>
              );
            })}
          </Group>
        )}

        {isFirebaseConfigured ? (
          <Group footer="Deleting your account removes your sign-in. Sessions on your laptop stay until you delete them there.">
            <ActionRow label="Delete Account" icon="trash-outline" destructive busy={deleting} onPress={confirmDelete} />
          </Group>
        ) : (
          <Group title="Account"
            footer="Accounts are off because Firebase isn't configured. The app talks to the laptop with the API key above.">
            <View style={styles.optionRow}><Text style={styles.rowSub}>No account</Text></View>
          </Group>
        )}

        <Text style={styles.version}>HooperAI {Constants.expoConfig?.version || ''}</Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  header: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.xs, minHeight: 48 },
  back: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' },
  title: { flex: 1, fontSize: 17, fontWeight: '600', color: colors.white, textAlign: 'center' },
  content: { padding: spacing.lg, gap: spacing.xxl, paddingBottom: 48 },

  groupTitle: {
    fontSize: font.sm, color: colors.textMuted, textTransform: 'uppercase', letterSpacing: 0.6,
    marginBottom: spacing.sm, paddingHorizontal: spacing.lg,
  },
  group: { backgroundColor: colors.bgCard, borderRadius: radius.lg, overflow: 'hidden' },
  groupFooter: { fontSize: font.base, color: colors.textMuted, lineHeight: 18, marginTop: spacing.sm, paddingHorizontal: spacing.lg },
  inset: { marginLeft: spacing.lg },

  accountRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.lg },
  avatar: {
    width: 44, height: 44, borderRadius: 22, backgroundColor: colors.orange,
    alignItems: 'center', justifyContent: 'center',
  },
  avatarText: { fontSize: 19, fontWeight: '700', color: colors.white },

  statusRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, paddingHorizontal: spacing.lg, minHeight: 60 },
  rowLabel: { fontSize: 17, color: colors.white },
  rowSub: { fontSize: font.base, color: colors.textMuted, marginTop: 1 },
  rowAction: { fontSize: font.md, color: colors.white, fontWeight: '600' },

  field: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.xs },
  fieldLabel: { fontSize: font.sm, color: colors.textMuted },
  input: { fontSize: 17, color: colors.white, minHeight: 40, paddingVertical: spacing.xs },

  optionRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, paddingHorizontal: spacing.lg, minHeight: 56 },
  actionRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, paddingHorizontal: spacing.lg, minHeight: 50 },
  actionText: { fontSize: 17 },

  version: { fontSize: font.sm, color: colors.textDim, textAlign: 'center' },
});
