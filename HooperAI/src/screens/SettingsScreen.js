import React, { useEffect, useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator, Alert,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { Card, SectionLabel, Divider } from '../components';
import { getSettings, saveSettings } from '../services/settings';
import { isFirebaseConfigured } from '../config';
import { currentUser, signOut } from '../services/auth';
import * as api from '../services/api';

export default function SettingsScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [serverUrl, setServerUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [predictor, setPredictor] = useState(null);
  const [check, setCheck] = useState(null);       // {ok, text, status}
  const [testing, setTesting] = useState(false);
  const user = currentUser();

  useEffect(() => {
    getSettings().then(s => {
      setServerUrl(s.serverUrl);
      setApiKey(s.apiKey);
      setPredictor(s.predictor);
    });
  }, []);

  const testConnection = async () => {
    setTesting(true);
    setCheck(null);
    try {
      await saveSettings({ serverUrl, apiKey });
      const h = await api.health();
      const st = await api.status();
      setCheck({ ok: true, status: st,
                 text: `Connected · ${st.gpu ? 'GPU' : 'CPU'} · auth: ${h.auth_mode}${h.firestore ? ' · Firestore sync on' : ''}` });
    } catch (e) {
      setCheck({ ok: false, text: e.message });
    } finally {
      setTesting(false);
    }
  };

  const choosePredictor = async (p) => {
    setPredictor(p.config);
    await saveSettings({ predictor: p.config, predictorLabel: p.label, predictorAccuracy: p.unseen_video_accuracy });
  };

  const save = async () => {
    await saveSettings({ serverUrl, apiKey });
    navigation.goBack();
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} accessibilityLabel="Back">
          <Ionicons name="chevron-back" size={22} color={colors.textMuted} />
        </TouchableOpacity>
        <Text style={styles.title}>Settings</Text>
        <TouchableOpacity onPress={save}><Text style={styles.saveText}>Save</Text></TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <View>
          <SectionLabel>Laptop server</SectionLabel>
          <Card style={{ gap: spacing.md }}>
            <View>
              <Text style={styles.label}>Address</Text>
              <TextInput
                style={styles.input} value={serverUrl} onChangeText={setServerUrl}
                placeholder="http://192.168.1.100:8000" placeholderTextColor={colors.textGhost}
                autoCapitalize="none" autoCorrect={false} keyboardType="url"
              />
              <Text style={styles.help}>
                Same Wi-Fi: the laptop's IP. Away from home: its Tailscale address (free), e.g. http://100.x.y.z:8000.
              </Text>
            </View>
            <View>
              <Text style={styles.label}>API key</Text>
              <TextInput
                style={styles.input} value={apiKey} onChangeText={setApiKey}
                placeholder="value of HOOPER_API_KEY on the laptop" placeholderTextColor={colors.textGhost}
                autoCapitalize="none" autoCorrect={false} secureTextEntry
              />
              {isFirebaseConfigured && (
                <Text style={styles.help}>
                  Only needed while the laptop server has no Firebase service account (auth: api_key).
                </Text>
              )}
            </View>
            <TouchableOpacity style={styles.secondaryBtn} onPress={testConnection} disabled={testing}>
              {testing ? <ActivityIndicator color={colors.orange} size="small" />
                : <Ionicons name="pulse-outline" size={16} color={colors.text} />}
              <Text style={styles.secondaryText}>Test connection</Text>
            </TouchableOpacity>
            {check && (
              <Text style={[styles.checkText, { color: check.ok ? colors.green : colors.orange }]}>{check.text}</Text>
            )}
          </Card>
        </View>

        {check?.status?.predictors?.length > 0 && (
          <View>
            <SectionLabel>Predictor</SectionLabel>
            <Card style={{ padding: 0, overflow: 'hidden' }}>
              {check.status.predictors.map((p, i, all) => {
                const selected = (predictor || check.status.default_predictor) === p.config;
                return (
                  <View key={p.config}>
                    <TouchableOpacity style={styles.optionRow} onPress={() => choosePredictor(p)}>
                      <Ionicons name={selected ? 'radio-button-on' : 'radio-button-off'} size={18}
                        color={selected ? colors.orange : colors.textGhost} />
                      <Text style={styles.optionText}>{p.label}</Text>
                      {p.unseen_video_accuracy != null && (
                        <Text style={styles.optionMeta}>{Math.round(p.unseen_video_accuracy * 100)}% on new videos</Text>
                      )}
                    </TouchableOpacity>
                    {i < all.length - 1 && <Divider />}
                  </View>
                );
              })}
            </Card>
          </View>
        )}

        <View>
          <SectionLabel>Account</SectionLabel>
          <Card style={{ gap: spacing.sm }}>
            {isFirebaseConfigured ? (
              <>
                <Text style={styles.label}>Signed in as</Text>
                <Text style={styles.value}>{user?.email || '—'}</Text>
                <TouchableOpacity style={styles.secondaryBtn} onPress={() =>
                  Alert.alert('Sign out?', '', [
                    { text: 'Cancel', style: 'cancel' },
                    { text: 'Sign out', style: 'destructive', onPress: () => signOut() },
                  ])}>
                  <Ionicons name="log-out-outline" size={16} color={colors.text} />
                  <Text style={styles.secondaryText}>Sign out</Text>
                </TouchableOpacity>
              </>
            ) : (
              <Text style={styles.help}>
                Firebase is not configured yet, so there is no sign-in. The app uses the API key above.
                Add the Firebase web config to HooperAI/.env to turn accounts on.
              </Text>
            )}
          </Card>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  header: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.md,
    paddingHorizontal: spacing.lg, paddingBottom: spacing.md,
    borderBottomWidth: 0.5, borderBottomColor: '#1e1e1e',
  },
  title: { flex: 1, fontSize: font.md, fontWeight: '500', color: colors.white },
  saveText: { color: colors.orange, fontSize: font.md, fontWeight: '500' },
  content: { padding: spacing.lg, gap: spacing.xl },
  label: { fontSize: font.sm, color: colors.textMuted, marginBottom: 6 },
  value: { fontSize: font.base, color: colors.white },
  input: {
    backgroundColor: colors.bgCardDeep, borderWidth: 0.5, borderColor: colors.border, borderRadius: radius.md,
    color: colors.white, paddingHorizontal: spacing.md, paddingVertical: 10, fontSize: font.base,
  },
  help: { fontSize: font.xs, color: colors.textDim, marginTop: 6, lineHeight: 16 },
  secondaryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    borderWidth: 0.5, borderColor: colors.border, borderRadius: radius.md, paddingVertical: 10,
    backgroundColor: colors.bgCardDeep,
  },
  secondaryText: { color: colors.text, fontSize: font.base },
  checkText: { fontSize: font.sm },
  optionRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.md },
  optionText: { flex: 1, color: colors.white, fontSize: font.base },
  optionMeta: { color: colors.textDim, fontSize: font.xs },
});
