import React, { useRef, useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator,
  KeyboardAvoidingView, Platform, Alert, ScrollView,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { colors, radius, spacing, font } from '../utils/theme';
import { signIn, signUp, resetPassword, authErrorMessage } from '../services/auth';

const MIN_PASSWORD = 6;

// Shown only when Firebase is configured and nobody is signed in.
// Says why an account is needed, labels every field, and keeps the main button
// disabled until the form can be sent.
export default function LoginScreen() {
  const insets = useSafeAreaInsets();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const passwordRef = useRef(null);

  const emailOk = /\S+@\S+\.\S+/.test(email.trim());
  const ready = emailOk && password.length >= MIN_PASSWORD && !busy;

  const submit = async () => {
    if (!ready) return;
    setError('');
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
    if (!emailOk) { setError('Enter your email above, then tap “Forgot password?” again.'); return; }
    try {
      await resetPassword(email);
      Alert.alert('Check your email', `We sent a password reset link to ${email.trim()}.`);
    } catch (e) {
      setError(authErrorMessage(e));
    }
  };

  const switchMode = () => { setCreating(!creating); setError(''); };

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={[styles.content, { paddingTop: insets.top + 56, paddingBottom: insets.bottom + spacing.xl }]}
        keyboardShouldPersistTaps="handled">
        {/* Identity + why an account */}
        <View style={styles.brand}>
          <View style={styles.mark}><Ionicons name="basketball" size={34} color={colors.white} /></View>
          <Text style={styles.title} accessibilityRole="header">
            {creating ? 'Create your account' : 'Welcome back'}
          </Text>
          <Text style={styles.subtitle}>
            Your account keeps your sessions and shot labels in sync between your phone and your laptop.
          </Text>
        </View>

        <View style={styles.form}>
          <View style={styles.field}>
            <Text style={styles.label} nativeID="emailLabel">Email</Text>
            <TextInput style={styles.input} value={email} onChangeText={setEmail}
              placeholder="you@example.com" placeholderTextColor={colors.textGhost}
              autoCapitalize="none" autoCorrect={false} autoComplete="email"
              keyboardType="email-address" textContentType="username"
              returnKeyType="next" onSubmitEditing={() => passwordRef.current?.focus()}
              accessibilityLabelledBy="emailLabel" accessibilityLabel="Email" />
          </View>

          <View style={styles.field}>
            <Text style={styles.label} nativeID="passwordLabel">Password</Text>
            <View style={styles.passwordWrap}>
              <TextInput ref={passwordRef} style={[styles.input, { flex: 1, borderWidth: 0 }]}
                value={password} onChangeText={setPassword}
                placeholder={creating ? `At least ${MIN_PASSWORD} characters` : 'Your password'}
                placeholderTextColor={colors.textGhost} secureTextEntry={!showPassword}
                autoCapitalize="none" autoCorrect={false}
                autoComplete={creating ? 'new-password' : 'current-password'}
                textContentType={creating ? 'newPassword' : 'password'}
                returnKeyType="go" onSubmitEditing={submit}
                accessibilityLabelledBy="passwordLabel" accessibilityLabel="Password" />
              <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.eye}
                accessibilityRole="button" accessibilityLabel={showPassword ? 'Hide password' : 'Show password'}>
                <Ionicons name={showPassword ? 'eye-off-outline' : 'eye-outline'} size={22} color={colors.textMuted} />
              </TouchableOpacity>
            </View>
            {creating && (
              <Text style={[styles.help, password.length >= MIN_PASSWORD && { color: colors.green }]}>
                {password.length >= MIN_PASSWORD ? '✓ ' : ''}At least {MIN_PASSWORD} characters
              </Text>
            )}
          </View>

          {!!error && (
            <View style={styles.error} accessibilityLiveRegion="assertive" accessibilityRole="alert">
              <Ionicons name="alert-circle" size={18} color={colors.red} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          <TouchableOpacity style={[styles.primaryBtn, !ready && styles.primaryOff]} onPress={submit}
            disabled={!ready} activeOpacity={0.85}
            accessibilityRole="button" accessibilityState={{ disabled: !ready, busy }}>
            {busy ? <ActivityIndicator color={colors.white} />
              : <Text style={[styles.primaryText, !ready && { color: colors.textDim }]}>
                  {creating ? 'Create Account' : 'Sign In'}
                </Text>}
          </TouchableOpacity>

          {!creating && (
            <TouchableOpacity onPress={forgot} style={styles.textBtn} accessibilityRole="button">
              <Text style={styles.textBtnLabel}>Forgot password?</Text>
            </TouchableOpacity>
          )}
        </View>

        <View style={styles.switchRow}>
          <Text style={styles.switchText}>{creating ? 'Already have an account?' : 'New to HooperAI?'}</Text>
          <TouchableOpacity onPress={switchMode} style={styles.textBtn} accessibilityRole="button">
            <Text style={[styles.textBtnLabel, { fontWeight: '600', color: colors.white }]}>
              {creating ? 'Sign In' : 'Create Account'}
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bgScreen },
  content: { flexGrow: 1, paddingHorizontal: spacing.xxl },

  brand: { alignItems: 'center', gap: spacing.md },
  mark: {
    width: 72, height: 72, borderRadius: 20, backgroundColor: colors.orange,
    alignItems: 'center', justifyContent: 'center', marginBottom: spacing.sm,
  },
  title: { fontSize: 28, fontWeight: '700', color: colors.white, textAlign: 'center' },
  subtitle: { fontSize: font.md, color: colors.textMuted, textAlign: 'center', lineHeight: 21 },

  form: { marginTop: spacing.xxl + spacing.sm, gap: spacing.lg },
  field: { gap: 6 },
  label: { fontSize: font.base, color: colors.text, fontWeight: '500' },
  input: {
    backgroundColor: colors.bgCard, borderRadius: radius.md, borderWidth: 0.5, borderColor: colors.border,
    color: colors.white, paddingHorizontal: spacing.md, minHeight: 50, fontSize: 17,
  },
  passwordWrap: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: colors.bgCard, borderRadius: radius.md, borderWidth: 0.5, borderColor: colors.border,
  },
  eye: { width: 48, height: 50, alignItems: 'center', justifyContent: 'center' },
  help: { fontSize: font.sm, color: colors.textMuted },

  error: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.sm },
  errorText: { flex: 1, fontSize: font.base, color: colors.text, lineHeight: 19 },

  primaryBtn: {
    backgroundColor: colors.orange, borderRadius: radius.lg, minHeight: 50,
    alignItems: 'center', justifyContent: 'center', marginTop: spacing.xs,
  },
  primaryOff: { backgroundColor: colors.bgCard },
  primaryText: { color: colors.white, fontSize: 17, fontWeight: '700' },
  textBtn: { minHeight: 44, alignItems: 'center', justifyContent: 'center', paddingHorizontal: spacing.sm },
  textBtnLabel: { fontSize: font.md, color: colors.textMuted },

  switchRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginTop: 'auto', paddingTop: spacing.xxl },
  switchText: { fontSize: font.md, color: colors.textMuted },
});
