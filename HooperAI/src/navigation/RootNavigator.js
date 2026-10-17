import React, { useEffect, useState } from 'react';
import { View, ActivityIndicator } from 'react-native';
import { createStackNavigator } from '@react-navigation/stack';
import TabNavigator from './TabNavigator';
import SettingsScreen from '../screens/SettingsScreen';
import LoginScreen from '../screens/LoginScreen';
import { isFirebaseConfigured } from '../config';
import { onUserChanged } from '../services/auth';
import { colors } from '../utils/theme';

const Stack = createStackNavigator();

export default function RootNavigator() {
  // Without Firebase there is no sign-in; the app goes straight to the tabs.
  const [user, setUser] = useState(undefined);   // undefined = still restoring the session
  useEffect(() => onUserChanged(setUser), []);

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
          <Stack.Screen name="Tabs" component={TabNavigator} />
          <Stack.Screen name="Settings" component={SettingsScreen} />
        </>
      )}
    </Stack.Navigator>
  );
}
