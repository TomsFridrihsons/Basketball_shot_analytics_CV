export const colors = {
  bg:         '#0a0a0a',
  bgScreen:   '#0f0f0f',
  bgCard:     '#161616',
  bgCardDeep: '#111111',
  border:     '#222222',
  borderSub:  '#1a1a1a',

  orange:     '#e85d26',
  orangeFade: 'rgba(232,93,38,0.12)',
  orangeBorder:'rgba(232,93,38,0.4)',

  green:      '#4caf7d',
  greenFade:  'rgba(76,175,125,0.12)',
  greenBorder:'rgba(76,175,125,0.35)',

  yellow:     '#c8a900',
  yellowFade: 'rgba(232,210,38,0.12)',

  red:        '#ff453a',   // recording only

  white:      '#ffffff',
  text:       '#cccccc',
  // contrast on bgCard #161616 (WCAG): secondary 6.9:1, tertiary 5.2:1, ghost 3.4:1
  textMuted:  '#a0a0a0',   // secondary text
  textDim:    '#8a8a8a',   // tertiary text, smallest labels
  textGhost:  '#6b6b6b',   // decorative icons and disabled only – not for text
};

export const radius = {
  sm:  8,
  md:  10,
  lg:  14,
  xl:  20,
  full: 999,
};

export const spacing = {
  xs:  4,
  sm:  8,
  md:  12,
  lg:  16,
  xl:  20,
  xxl: 28,
};

export const font = {
  xs:   11,   // HIG minimum on phones
  sm:   12,
  base: 13,
  md:   15,
  lg:   18,
  xl:   22,
  xxl:  28,
};
