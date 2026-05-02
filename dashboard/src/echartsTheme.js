/**
 * ECharts theme & shared config for the gold accent design system
 */
export const chartAnim = {
  animationDuration: 800,
  animationEasing: 'cubicOut',
  animationDurationUpdate: 500,
  animationEasingUpdate: 'cubicInOut',
}

export const tooltipStyle = {
  backgroundColor: 'rgba(10, 11, 16, 0.96)',
  borderColor: 'rgba(200, 170, 110, 0.15)',
  borderWidth: 1,
  textStyle: { color: '#f1f5f9', fontFamily: 'Outfit', fontSize: 12 },
  extraCssText: 'box-shadow: 0 4px 20px rgba(0,0,0,0.4); border-radius: 8px; backdrop-filter: blur(12px);',
}

export const goldColors = ['#c8aa6e', '#3b82f6', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6']
export const warmGoldColors = ['#c8aa6e', '#e8c98a', '#a88c5a', '#f5e6c8', '#8b7355', '#d4b483']

export function applyTheme(option) {
  return {
    ...chartAnim,
    ...option,
  }
}
