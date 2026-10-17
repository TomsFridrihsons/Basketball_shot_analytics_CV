import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import HomeScreen from '../screens/HomeScreen';
import RecordScreen from '../screens/RecordScreen';
import StatsScreen from '../screens/StatsScreen';
import { colors } from '../utils/theme';

const Tab = createBottomTabNavigator();

function CustomTabBar({ state, descriptors, navigation }) {
  const insets = useSafeAreaInsets();
  const routes = state.routes;

  return (
    <View style={[styles.bar, { paddingBottom: insets.bottom || 10 }]}>
      {routes.map((route, index) => {
        const isFocused = state.index === index;
        const isCenter = route.name === 'Record';

        const onPress = () => {
          const event = navigation.emit({ type: 'tabPress', target: route.key, canPreventDefault: true });
          if (!isFocused && !event.defaultPrevented) navigation.navigate(route.name);
        };

        if (isCenter) {
          return (
            <TouchableOpacity key={route.key} onPress={onPress} style={styles.centerWrap} activeOpacity={0.85}>
              <View style={styles.centerBtn}>
                <Ionicons name="videocam" size={24} color={colors.white} />
              </View>
            </TouchableOpacity>
          );
        }

        const iconMap = { Home: 'home', Stats: 'bar-chart' };
        const iconName = isFocused ? iconMap[route.name] : iconMap[route.name] + '-outline';

        return (
          <TouchableOpacity key={route.key} onPress={onPress} style={styles.tabItem} activeOpacity={0.7}>
            <Ionicons name={iconName} size={24} color={isFocused ? colors.orange : colors.textGhost} />
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

export default function TabNavigator() {
  return (
    <Tab.Navigator
      tabBar={props => <CustomTabBar {...props} />}
      screenOptions={{ headerShown: false }}
    >
      <Tab.Screen name="Home"   component={HomeScreen} />
      <Tab.Screen name="Record" component={RecordScreen} />
      <Tab.Screen name="Stats"  component={StatsScreen} />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: 'row',
    backgroundColor: colors.bg,
    borderTopWidth: 0.5,
    borderTopColor: '#1e1e1e',
    paddingTop: 6,
    alignItems: 'center',
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 4,
  },
  centerWrap: {
    flex: 1,
    alignItems: 'center',
    marginTop: -20,
  },
  centerBtn: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: colors.orange,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: colors.bg,
  },
});
