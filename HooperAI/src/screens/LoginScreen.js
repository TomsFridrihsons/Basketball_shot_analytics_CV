import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator,
  KeyboardAvoidingView, Platform, Alert,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors, radius, spacing, font } from '../utils/theme';
import { signIn, signUp, resetPassword, authErrorMessage } from '../services/auth';

// Shown only when Firebase is configured and nobody is signed in.
export default function LoginScreen() {
  const insets = useSafeAreaInsets();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const submit = async () => {
    setError('');
    if (!email || password.length < 6) {
      setError('Enter your email and a password of at least 6 characters.');
      return;
    }
    setBusy(true);
    try {
      await (creating ? signUp(email, password) : signIn(email, password));
      // RootNavigator switches to the app when the auth state changes
    } catch (e) {
      setError(authErrorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const forgot = async () => {
    if (!email) { setError('Enter your email first.'); return; }
    try {
      await resetPassword(email);
      Alert.alert('Check your email', 'We sent you a link to reset your password.');
    } catch (e) {
      setError(authErrorMessage(e));
    }
  };

  return (
    <KeyboardAvoidingView style={[styles.container, { paddingTop: insets.top + 40 }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <View style={styles.logoRow}>
        <View style={styles.logoIcon}><Text style={{ fontSize: 20 }}>🏀</Text></View>
        <Text style={styles.logoText}>Hooper<Text style={{ color: colors.orange }}>AI</Text></Text>
      </View>
      <Text style={styles.subtitle}>{creating ? 'Create your account' : 'Sign in to track your shots'}</Text>

      <View style={styles.form}>
        <TextInput style={styles.input} value={email} onChangeText={setEmail} placeholder="Email"
          placeholderTextColor={colors.textGhost} autoCapitalize="none" autoCorrect={false}
          keyboardType="email-address" textContentType="emailAddress" />
        <TextInput style={styles.input} value={password} onChangeText={setPassword} placeholder="Password"
          placeholderTextColor={colors.textGhost} secureTextEntry
          textContentType={creating ? 'newPassword' : 'password'} onSubmitEditing={submit} />
        {!!error && <Text style={styles.error}>{error}</Text>}

        <TouchableOpacity style={styles.primaryBtn} onPress={submit} disabled={busy} activeOpacity={0.85}>
          {busy ? <ActivityIndicator color={colors.white} />
            : <Text style={styles.primaryText}>{creating ? 'Create account' : 'Sign in'}</Text>}
        </TouchableOpacity>

        <TouchableOpacity onPress={() => { setCreating(!creating); setError(''); }}>
          <Text style={styles.link}>{creating ? 'I already have an account' : 'Create an account'}</Text>
        </TouchableOpacity>
        {!creating && (
          <TouchableOpacity onPress={forgot}><Text style={styles.linkDim}>Forgot password?</Text></TouchableOpacity>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen, paddingHorizontal: spacing.xxl },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, justifyContent: 'center' },
  logoIcon: {
    width: 38, height: 38, borderRadius: 10, backgroundColor: colors.orange,
    alignItems: 'center', justifyContent: 'center',
  },
  logoText: { fontSize: font.xxl, fontWeight: '600', color: colors.white },
  subtitle: { textAlign: 'center', color: colors.textMuted, fontSize: font.base, marginTop: spacing.md },
  form: { marginTop: spacing.xxl, gap: spacing.md },
  input: {
    backgroundColor: colors.bgCard, borderWidth: 0.5, borderColor: colors.border, borderRadius: radius.md,
    color: colors.white, paddingHorizontal: spacing.md, paddingVertical: 12, fontSize: font.md,
  },
  error: { color: colors.orange, fontSize: font.sm },
  primaryBtn: {
    backgroundColor: colors.orange, borderRadius: radius.md, paddingVertical: 13,
    alignItems: 'center', marginTop: spacing.sm,
  },
  primaryText: { color: colors.white, fontSize: font.md, fontWeight: '500' },
  link: { color: colors.orange, textAlign: 'center', fontSize: font.base, marginTop: spacing.sm },
  linkDim: { color: colors.textDim, textAlign: 'center', fontSize: font.sm },
});
