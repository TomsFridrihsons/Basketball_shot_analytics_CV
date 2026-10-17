import React, { useEffect, useState } from 'react';
import { View, ActivityIndicator } from 'react-native';
import { createStackNavigator } from '@react-navigation/stack';
import TabNavigator from './TabNavigator';
import SettingsScreen from '../screens/SettingsScreen';
import SessionScreen from '../screens/SessionScreen';
import LoginScreen from '../screens/LoginScreen';
import { isFirebaseConfigured } from '../config';
import { onUserChanged } from '../services/auth';
import { setStorageOwner } from '../utils/storage';
import { colors } from '../utils/theme';

const Stack = createStackNavigator();

export default function RootNavigator() {
  // Without Firebase there is no sign-in; the app goes straight to the tabs.
  const [user, setUser] = useState(undefined);   // undefined = still restoring the session
  // Point the local cache at this account before any screen reads it
  useEffect(() => onUserChanged(async (u) => {
    try { await setStorageOwner(u?.uid); }
    catch (e) { console.warn('setStorageOwner failed', e); }
    finally { setUser(u); }   // never leave the app stuck on the spinner
  }), []);

  if (isFirebaseConfigured && user === undefined) {
    return (
      <View style={{ flex: 1, backgroundColor: colors.bg, alignItems: 'center', justifyContent: 'center' }}>
        <ActivityIndicator color={colors.orange} />
      </View>
    );
  }

  return (
    <Stack.Navigator screenOptions={{ headerShown: false, cardStyle: { backgroundColor: colors.bgScreen } }}>
      {isFirebaseConfigured && !user ? (
        <Stack.Screen name="Login" component={LoginScreen} />
      ) : (
        <>
          {/* keyed by account so a different user never sees the previous one's screens */}
          <Stack.Screen name="Tabs" component={TabNavigator} key={user?.uid || 'local'} />
          <Stack.Screen name="Settings" component={SettingsScreen} />
          <Stack.Screen name="Session" component={SessionScreen} />
        </>
      )}
    </Stack.Navigator>
  );
}
