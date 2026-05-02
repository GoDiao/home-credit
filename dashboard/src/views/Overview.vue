<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart, BarChart, RadarChart, LineChart, HeatmapChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, RadarComponent, MarkLineComponent, VisualMapComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { useRouter } from 'vue-router'
import api from '../api'
import { chartAnim, tooltipStyle } from '../echartsTheme'

const router = useRouter()

use([CanvasRenderer, PieChart, BarChart, RadarChart, LineChart, HeatmapChart, GridComponent, TooltipComponent, LegendComponent, RadarComponent, MarkLineComponent, VisualMapComponent])

const data = ref(null)
const modelRegistry = ref(null)
const briefing = ref(null)
const loading = ref(true)
const registryLoading = ref(true)

// Merge animation config into any chart option
function withAnim(opt) { return { ...chartAnim, ...opt } }

// Best model (highest AUC)
const bestModel = computed(() => {
  if (!data.value?.models?.length) return null
  return data.value.models.reduce((best, m) => (m.AUC > (best?.AUC || 0)) ? m : best, null)?.Model
})

const registryModels = computed(() => modelRegistry.value?.models || [])

const registrySummary = computed(() => {
  const models = registryModels.value
  if (!models.length) return null
  const latest = [...models].sort((a, b) => String(b.trained_at || '').localeCompare(String(a.trained_at || '')))[0] || models[0]
  return {
    count: models.length,
    latest,
    featureCount: latest?.feature_count ?? data.value?.n_features ?? '-',
    updatedAt: modelRegistry.value?.updated_at || '-',
  }
})

// Pipeline detail panel
const activeStage = ref(null)
const stageDetail = ref(null)
const stageLoading = ref(false)

const stageKeyMap = {
  '数据加载': 'data_loading',
  '数据处理': 'data_processing',
  '特征工程': 'feature_engineering',
  '模型训练': 'model_training',
  '模型评估': 'model_evaluation',
  '策略模拟': 'policy_simulation',
  '稳定性监控': 'monitoring',
}

async function openStageDetail(stage) {
  const key = stageKeyMap[stage.name]
  if (!key) return
  activeStage.value = stage
  stageLoading.value = true
  try {
    const res = await api.get(`/api/pipeline/${key}`)
    const d = res.data
    // Add _expanded reactivity for expandable items
    if (d.auxiliary_tables) d.auxiliary_tables.forEach(t => t._expanded = false)
    if (d.steps) d.steps.forEach(s => s._expanded = false)
    stageDetail.value = d
  } catch (e) {
    console.error(e)
    stageDetail.value = null
  } finally {
    stageLoading.value = false
  }
}

function closePanel() {
  activeStage.value = null
  stageDetail.value = null
}

// Missing values chart for data_loading detail
const missingChartOption = computed(() => {
  if (!stageDetail.value?.details?.top_missing?.length) return null
  const items = stageDetail.value.details.top_missing.slice(0, 8)
  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    grid: { left: 180, right: 30, top: 10, bottom: 20 },
    xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLabel: { color: '#64748b', fontSize: 10, formatter: (v) => v + '%' } },
    yAxis: { type: 'category', data: items.map(i => i.column).reverse(), axisLabel: { fontSize: 10, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{ type: 'bar', data: items.map(i => i.pct).reverse(), itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#f43f5e' }, { offset: 1, color: '#fb7185' }] }, borderRadius: [0, 4, 4, 0] }, barMaxWidth: 14 }],
  }
})

// Column completeness donut chart for data_loading detail
const columnDonutOption = computed(() => {
  if (!stageDetail.value?.output || stageDetail.value?.stage !== 'data_loading') return null
  const out = stageDetail.value.output
  const total = out.total_columns || 122
  const missing = stageDetail.value.details?.missing_columns || 0
  const complete = total - missing
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)', ...tooltipStyle },
    series: [{
      type: 'pie',
      radius: ['55%', '78%'],
      center: ['50%', '50%'],
      avoidLabelOverlap: false,
      itemStyle: { borderRadius: 5, borderColor: 'var(--bg-secondary)', borderWidth: 2 },
      label: {
        show: true, position: 'center',
        formatter: `{a|${complete}}\n{b|Complete}`,
        rich: { a: { fontSize: 18, fontWeight: 700, color: '#f1f5f9', fontFamily: 'JetBrains Mono', lineHeight: 24 }, b: { fontSize: 10, color: '#94a3b8', lineHeight: 16 } },
      },
      data: [
        { value: complete, name: 'Complete', itemStyle: { color: '#10b981' } },
        { value: missing, name: 'Has Missing', itemStyle: { color: '#f43f5e' } },
      ],
    }],
  }
})

// Feature source pie chart for feature_engineering detail
const featureSourceOption = computed(() => {
  if (!stageDetail.value?.steps || stageDetail.value?.stage !== 'feature_engineering') return null
  const steps = stageDetail.value.steps
  const colors = ['#c8aa6e', '#e8c98a', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e']
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)', ...tooltipStyle },
    legend: { orient: 'vertical', right: 10, top: 'center', textStyle: { color: '#94a3b8', fontSize: 11, fontFamily: 'Outfit' }, itemGap: 8 },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      center: ['35%', '50%'],
      avoidLabelOverlap: false,
      itemStyle: { borderRadius: 6, borderColor: 'var(--bg-secondary)', borderWidth: 2 },
      label: { show: true, position: 'center', formatter: `{a|${stageDetail.value.total_new_features}}\n{b|Features}`, rich: { a: { fontSize: 20, fontWeight: 700, color: '#f1f5f9', fontFamily: 'JetBrains Mono', lineHeight: 28 }, b: { fontSize: 11, color: '#94a3b8', lineHeight: 18 } } },
      emphasis: { label: { show: true } },
      data: steps.map((s, i) => ({ name: s.name, value: s.count, itemStyle: { color: colors[i % colors.length] } })),
    }],
  }
})

// Correlation heatmap for feature_engineering detail
const corrHeatmapOption = computed(() => {
  const hm = stageDetail.value?.corr_heatmap
  if (!hm?.data?.length || stageDetail.value?.stage !== 'feature_engineering') return null
  const features = hm.features
  const n = features.length
  return {
    tooltip: {
      ...tooltipStyle,
      formatter: (p) => {
        if (!p.data) return ''
        const [x, y, v] = p.data
        return `${features[y]} vs ${features[x]}<br/>Correlation: <b>${v}</b>`
      },
    },
    grid: { left: 120, right: 40, top: 10, bottom: 60 },
    xAxis: { type: 'category', data: features, axisLabel: { color: '#64748b', fontSize: 9, fontFamily: 'JetBrains Mono', rotate: 45 }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } }, axisTick: { show: false } },
    yAxis: { type: 'category', data: features, axisLabel: { color: '#64748b', fontSize: 9, fontFamily: 'JetBrains Mono' }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } }, axisTick: { show: false } },
    visualMap: {
      min: -1, max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      itemWidth: 12,
      itemHeight: 100,
      textStyle: { color: '#64748b', fontSize: 10 },
      inRange: { color: ['#3b82f6', '#1e293b', '#f43f5e'] },
    },
    series: [{
      type: 'heatmap',
      data: hm.data,
      label: { show: n <= 10, fontSize: 8, color: '#94a3b8', fontFamily: 'JetBrains Mono' },
      itemStyle: { borderColor: 'var(--bg-secondary)', borderWidth: 1 },
      emphasis: { itemStyle: { borderColor: '#fff', borderWidth: 2 } },
    }],
  }
})

// Strategy comparison bar chart for policy_simulation detail
const strategyChartOption = computed(() => {
  if (!stageDetail.value?.strategies || stageDetail.value?.stage !== 'policy_simulation') return null
  const strats = stageDetail.value.strategies
  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    grid: { left: 120, right: 30, top: 10, bottom: 30 },
    xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLabel: { color: '#64748b', fontSize: 10, formatter: (v) => v + '%' } },
    yAxis: { type: 'category', data: strats.map(s => s.name).reverse(), axisLabel: { fontSize: 10, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [
      { name: 'Approval', type: 'bar', data: strats.map(s => s.approval_rate).reverse(), itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#3b82f6' }, { offset: 1, color: '#06b6d4' }] }, borderRadius: [0, 4, 4, 0] }, barMaxWidth: 12 },
      { name: 'EL Rate', type: 'bar', data: strats.map(s => s.el_rate).reverse(), itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#f43f5e' }, { offset: 1, color: '#fb7185' }] }, borderRadius: [0, 4, 4, 0] }, barMaxWidth: 12 },
    ],
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 11 } },
  }
})

// Model evaluation radar chart
const evalRadarOption = computed(() => {
  if (!stageDetail.value?.models || stageDetail.value?.stage !== 'model_evaluation') return null
  const models = stageDetail.value.models
  const metrics = ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall']
  const colors = ['#c8aa6e', '#3b82f6', '#10b981', '#f59e0b']
  return {
    ...chartAnim,
    tooltip: { ...tooltipStyle },
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 11 } },
    radar: {
      indicator: metrics.map(m => ({ name: m, max: 1 })),
      shape: 'polygon',
      axisName: { color: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono' },
      splitArea: { areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.04)'] } },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
    },
    series: [{
      type: 'radar',
      data: models.map((m, i) => ({
        name: m.Model,
        value: metrics.map(k => m[k] || 0),
        lineStyle: { color: colors[i], width: 2 },
        itemStyle: { color: colors[i] },
        areaStyle: { color: colors[i], opacity: 0.1 },
      })),
    }],
  }
})

// PSI distribution bar chart for monitoring detail
const psiDistOption = computed(() => {
  if (!stageDetail.value?.top_unstable_features || stageDetail.value?.stage !== 'monitoring') return null
  const features = stageDetail.value.top_unstable_features
  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    grid: { left: 180, right: 30, top: 10, bottom: 20 },
    xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
    yAxis: { type: 'category', data: features.map(f => f.feature).reverse(), axisLabel: { fontSize: 10, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'bar',
      data: features.map(f => ({
        value: f.psi,
        itemStyle: { color: f.psi > 0.25 ? { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#f43f5e' }, { offset: 1, color: '#fb7185' }] } : { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#f59e0b' }, { offset: 1, color: '#fbbf24' }] }, borderRadius: [0, 4, 4, 0] },
      })).reverse(),
      barMaxWidth: 14,
      markLine: { silent: true, symbol: 'none', lineStyle: { type: 'dashed', color: 'rgba(244,63,94,0.5)' }, data: [{ xAxis: 0.25, label: { formatter: 'Threshold', color: '#f43f5e', fontSize: 10, fontFamily: 'JetBrains Mono' } }] },
    }],
  }
})

// Model trend line chart
const modelTrendOption = computed(() => {
  if (!data.value?.models?.length) return null
  const models = data.value.models
  const metrics = ['AUC', 'KS', 'Gini']
  const colors = ['#c8aa6e', '#10b981', '#f59e0b']
  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', ...tooltipStyle },
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      data: models.map(m => m.Model?.replace(' Regression', '') || '-'),
      axisLabel: { color: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
    },
    yAxis: {
      type: 'value',
      min: 0.4,
      max: 1,
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' },
    },
    series: metrics.map((k, i) => ({
      name: k,
      type: 'line',
      data: models.map(m => m[k] || 0),
      smooth: true,
      lineStyle: { width: 2.5, color: colors[i] },
      itemStyle: { color: colors[i] },
      symbolSize: 8,
      symbol: 'circle',
    })),
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 12 } },
  }
})

// Model AUC comparison horizontal bar chart
const aucCompareOption = computed(() => {
  if (!data.value?.models?.length) return null
  const sorted = [...data.value.models].sort((a, b) => a.AUC - b.AUC)
  const names = sorted.map(m => m.Model?.replace(' Regression', '').replace(' Ensemble', '') || '-')
  const colors = sorted.map((m, i) => {
    if (m.Model?.includes('Stack')) return '#c8aa6e'
    if (m.Model?.includes('XG')) return '#3b82f6'
    if (m.Model?.includes('Light')) return '#10b981'
    return '#8b5cf6'
  })
  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle, formatter: (p) => {
      const d = p[0]
      return `<b>${sorted[d.dataIndex]?.Model}</b><br/>AUC: ${d.value.toFixed(4)}<br/>KS: ${(sorted[d.dataIndex]?.KS || 0).toFixed(4)}`
    }},
    grid: { left: 140, right: 50, top: 10, bottom: 20 },
    xAxis: { type: 'value', min: 0.7, max: 0.82, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLabel: { color: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' } },
    yAxis: { type: 'category', data: names, axisLabel: { fontSize: 11, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'bar',
      data: sorted.map((m, i) => ({
        value: m.AUC,
        itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: colors[i] }, { offset: 1, color: colors[i] + '99' }] }, borderRadius: [0, 4, 4, 0] },
      })),
      barMaxWidth: 18,
      label: { show: true, position: 'right', color: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono', formatter: (p) => p.value.toFixed(4) },
    }],
  }
})

// Navigate to model detail page
function goToModel(modelType) {
  const typeMap = { 'Logistic Regression': 'logistic', 'XGBoost': 'xgboost', 'LightGBM': 'lightgbm', 'Stacking Ensemble': 'stacking' }
  router.push({ path: '/model', query: { type: typeMap[modelType] || 'xgboost' } })
}

async function loadOverviewData() {
  const [overviewRes, registryRes, briefingRes] = await Promise.all([
    api.get('/api/overview'),
    api.get('/api/models/registry').catch(() => ({ data: null })),
    api.get('/api/overview/briefing').catch(() => ({ data: null })),
  ])
  data.value = overviewRes.data
  modelRegistry.value = registryRes.data
  briefing.value = briefingRes.data
}

onMounted(async () => {
  try {
    await loadOverviewData()
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
    registryLoading.value = false
  }
})

async function refreshData() {
  loading.value = true
  registryLoading.value = true
  try {
    await loadOverviewData()
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
    registryLoading.value = false
  }
}

function formatTime(iso) {
  if (!iso || iso === '-') return '-'
  try {
    const d = new Date(iso)
    return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } catch { return '-' }
}

function exportCSV() {
  if (!data.value?.models?.length) return
  const headers = ['Model', 'AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
  const rows = data.value.models.map(m => headers.map(h => m[h] ?? '').join(','))
  const csv = [headers.join(','), ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'model_comparison.csv'
  a.click()
  URL.revokeObjectURL(url)
}

// Target distribution donut chart
const targetOption = ref({})
watch(data, (d) => {
  if (!d?.target_distribution) return
  targetOption.value = {
    ...chartAnim,
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)',
      ...tooltipStyle,
    },
    legend: {
      orient: 'vertical',
      right: 20,
      top: 'center',
      textStyle: { color: '#94a3b8', fontFamily: 'Outfit', fontSize: 13 },
      itemGap: 16,
    },
    series: [{
      type: 'pie',
      radius: ['55%', '80%'],
      center: ['40%', '50%'],
      avoidLabelOverlap: false,
      itemStyle: { borderRadius: 8, borderColor: 'var(--bg-card)', borderWidth: 3 },
      label: {
        show: true,
        position: 'center',
        formatter: '{a|违约率}\n{b|' + (d.default_rate * 100).toFixed(1) + '%}',
        rich: {
          a: { fontSize: 13, color: '#94a3b8', fontFamily: 'Outfit', lineHeight: 24 },
          b: { fontSize: 22, fontWeight: 700, color: '#f1f5f9', fontFamily: 'JetBrains Mono', lineHeight: 32 },
        },
      },
      emphasis: { label: { show: true } },
      data: d.target_distribution.map((item, i) => ({
        ...item,
        itemStyle: {
          color: i === 0 ? { type: 'linear', x: 0, y: 0, x2: 1, y2: 1, colorStops: [{ offset: 0, color: '#10b981' }, { offset: 1, color: '#34d399' }] }
            : { type: 'linear', x: 0, y: 0, x2: 1, y2: 1, colorStops: [{ offset: 0, color: '#f43f5e' }, { offset: 1, color: '#fb7185' }] },
        },
      })),
    }],
  }
})

// Feature type bar chart
const featureTypeOption = ref({})
watch(data, (d) => {
  if (!d?.dataset_info) return
  const info = d.dataset_info
  featureTypeOption.value = {
    ...chartAnim,
    tooltip: {
      trigger: 'axis',
      ...tooltipStyle,
    },
    grid: { left: 10, right: 10, top: 10, bottom: 30, containLabel: true },
    xAxis: {
      type: 'category',
      data: ['Train', 'Test'],
      axisLabel: { color: '#64748b', fontSize: 12, fontFamily: 'JetBrains Mono' },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
      axisLabel: { color: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' },
    },
    series: [{
      type: 'bar',
      data: [
        { value: info.train_rows, itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#3b82f6' }, { offset: 1, color: '#1d4ed8' }] }, borderRadius: [6, 6, 0, 0] } },
        { value: info.test_rows, itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#8b5cf6' }, { offset: 1, color: '#6d28d9' }] }, borderRadius: [6, 6, 0, 0] } },
      ],
      barWidth: 50,
      label: { show: true, position: 'top', color: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 12, formatter: (p) => p.value.toLocaleString() },
    }],
  }
})

const pipelineIconMap = {
  download: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3',
  filter: 'M22 3H2l8 9.46V19l4 2v-8.54L22 3z',
  layers: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  cpu: 'M4 4h16v16H4zM9 9h6v6H9zM9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3',
  'bar-chart': 'M18 20V10M12 20V4M6 20v-6',
  sliders: 'M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6',
  activity: 'M22 12h-4l-3 9L9 3l-3 9H2',
}
</script>

<template>
  <div>
    <div class="page-header animate-in">
      <div class="header-row">
        <div>
          <h2>数据概览</h2>
          <p class="page-desc">Home Credit Default Risk - Credit Risk Analytics Platform</p>
        </div>
        <button class="refresh-btn" @click="refreshData" :disabled="loading">
          <svg :class="['refresh-icon', { spinning: loading }]" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
          <span>Refresh</span>
        </button>
      </div>
    </div>

    <!-- Executive Summary / Risk Briefing -->
    <div class="briefing animate-in" v-if="briefing && !loading">
      <div class="briefing-left">
        <div class="briefing-risk-badge">
          <div :class="['risk-level-indicator', briefing.monitoring_status === 'critical' ? 'critical' : 'healthy']"></div>
          <div class="risk-level-text">
            <span class="risk-label">Portfolio Risk</span>
            <span :class="['risk-status', briefing.monitoring_status === 'critical' ? 'critical' : 'healthy']">{{ briefing.monitoring_label }}</span>
          </div>
        </div>
        <div class="briefing-default">
          <span class="briefing-default-rate">{{ (briefing.default_rate * 100).toFixed(1) }}%</span>
          <span class="briefing-default-label">Default Rate</span>
          <span class="briefing-default-sub">{{ briefing.n_default?.toLocaleString() }} / {{ briefing.n_total?.toLocaleString() }}</span>
        </div>
      </div>
      <div class="briefing-center">
        <div class="briefing-model" v-if="briefing.best_model">
          <div class="bm-header">
            <span class="bm-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg></span>
            <span class="bm-label">Best Model</span>
          </div>
          <div class="bm-name">{{ briefing.best_model.name }}</div>
          <div class="bm-metrics">
            <span class="bm-metric"><span class="bm-metric-val gold">{{ briefing.best_model.auc }}</span><span class="bm-metric-key">AUC</span></span>
            <span class="bm-sep"></span>
            <span class="bm-metric"><span class="bm-metric-val">{{ briefing.best_model.ks }}</span><span class="bm-metric-key">KS</span></span>
            <span class="bm-sep"></span>
            <span class="bm-metric"><span class="bm-metric-val">{{ briefing.best_model.gini }}</span><span class="bm-metric-key">Gini</span></span>
          </div>
        </div>
        <div class="briefing-monitoring">
          <span class="bm-label">Stability</span>
          <span :class="['bm-metric-val', briefing.monitoring_status === 'critical' ? 'rose' : 'emerald']">{{ briefing.unstable_features }}</span>
          <span class="bm-metric-key">unstable / {{ briefing.total_features_monitored }} features</span>
        </div>
      </div>
      <div class="briefing-right" v-if="briefing.strategies?.length">
        <div class="bs-header">
          <span class="bm-label">Recommended Strategies</span>
        </div>
        <div class="strategy-cards">
          <div v-for="s in briefing.strategies" :key="s.name" :class="['strat-card', s.name.toLowerCase()]">
            <div class="strat-name">{{ s.name }}</div>
            <div class="strat-metrics">
              <span><span class="strat-val">{{ (s.approval_rate * 100).toFixed(1) }}%</span> Approval</span>
              <span><span class="strat-val">{{ (s.el_rate * 100).toFixed(2) }}%</span> EL Rate</span>
              <span><span class="strat-val">{{ (s.bad_capture * 100).toFixed(1) }}%</span> Bad Capture</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Skeleton: Stats Grid -->
    <div class="stats-grid" v-if="loading">
      <div v-for="i in 4" :key="i" class="skeleton-card">
        <div class="skeleton-icon"></div>
        <div class="skeleton-content">
          <div class="skeleton-line short"></div>
          <div class="skeleton-line medium"></div>
        </div>
      </div>
    </div>

    <!-- Key Stats -->
    <div class="stats-grid" v-if="data && !loading">
      <div class="stat-card animate-in animate-in-delay-1">
        <div class="stat-top">
          <div class="stat-icon blue">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          </div>
          <span class="stat-tag">Train</span>
        </div>
        <div class="stat-number">{{ data.n_samples?.toLocaleString() }}</div>
        <div class="stat-label">Training Samples</div>
        <div class="stat-footer">
          <span class="stat-foot-item">{{ data.n_features }} features</span>
        </div>
      </div>

      <div class="stat-card animate-in animate-in-delay-2">
        <div class="stat-top">
          <div class="stat-icon violet">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>
          </div>
          <span class="stat-tag">Features</span>
        </div>
        <div class="stat-number">{{ data.n_features }}</div>
        <div class="stat-label">Engineered Features</div>
        <div class="stat-footer">
          <span class="stat-foot-item">from 122 base</span>
        </div>
      </div>

      <div class="stat-card animate-in animate-in-delay-3">
        <div class="stat-top">
          <div class="stat-icon emerald">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="20 6 9 17 4 12"/></svg>
          </div>
          <span class="stat-tag emerald">{{ data.n_samples ? ((data.n_good / data.n_samples) * 100).toFixed(1) + '%' : '' }}</span>
        </div>
        <div class="stat-number">{{ data.n_good?.toLocaleString() }}</div>
        <div class="stat-label">Non-Default</div>
        <div class="stat-footer">
          <div class="stat-bar"><div class="stat-bar-fill emerald" :style="{ width: ((data.n_good / data.n_samples) * 100) + '%' }"></div></div>
        </div>
      </div>

      <div class="stat-card animate-in animate-in-delay-4">
        <div class="stat-top">
          <div class="stat-icon rose">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          </div>
          <span class="stat-tag rose">{{ data.n_samples ? ((data.n_default / data.n_samples) * 100).toFixed(1) + '%' : '' }}</span>
        </div>
        <div class="stat-number">{{ data.n_default?.toLocaleString() }}</div>
        <div class="stat-label">Default</div>
        <div class="stat-footer">
          <div class="stat-bar"><div class="stat-bar-fill rose" :style="{ width: ((data.n_default / data.n_samples) * 100) + '%' }"></div></div>
        </div>
      </div>
    </div>

    <!-- Skeleton: Pipeline -->
    <div class="skeleton-section" v-if="loading">
      <div class="skeleton-header"></div>
      <div class="skeleton-pipeline">
        <div v-for="i in 7" :key="i" class="skeleton-stage">
          <div class="skeleton-node"></div>
          <div class="skeleton-line tiny"></div>
        </div>
      </div>
    </div>

    <!-- Skeleton: Charts -->
    <div class="skeleton-charts" v-if="loading">
      <div class="skeleton-chart"><div class="skeleton-header"></div><div class="skeleton-body"></div></div>
      <div class="skeleton-chart"><div class="skeleton-header"></div><div class="skeleton-body"></div></div>
    </div>

    <!-- Skeleton: Table -->
    <div class="skeleton-section" v-if="loading">
      <div class="skeleton-header"></div>
      <div class="skeleton-table">
        <div v-for="i in 3" :key="i" class="skeleton-row">
          <div class="skeleton-cell wide"></div>
          <div v-for="j in 7" :key="j" class="skeleton-cell"></div>
        </div>
      </div>
    </div>

    <!-- Pipeline Visualization -->
    <div class="pipeline-section animate-in animate-in-delay-5" v-if="data?.pipeline && !loading">
      <div class="section-header">
        <div class="section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          <span>Data Pipeline</span>
        </div>
        <span class="pipeline-badge">7 Stages</span>
      </div>
      <div class="pipeline-scroll">
        <template v-for="(stage, idx) in data.pipeline" :key="idx">
          <div
            :class="['pipe-card', stage.status, { active: activeStage?.name === stage.name }]"
            @click="openStageDetail(stage)">
            <!-- Header -->
            <div class="pipe-card-head">
              <div class="pipe-step-num">{{ idx + 1 }}</div>
              <div :class="['pipe-card-icon', stage.status]">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path :d="pipelineIconMap[stage.icon] || ''"/>
                </svg>
              </div>
              <div class="pipe-card-titles">
                <span class="pipe-card-name">{{ stage.name }}</span>
                <span class="pipe-card-en">{{ stage.name_en }}</span>
              </div>
              <div :class="['pipe-card-badge', stage.status]">{{ stage.status === 'done' ? 'Done' : 'Pending' }}</div>
            </div>
            <!-- Description -->
            <div class="pipe-card-desc">{{ stage.desc }}</div>
            <!-- Meta Info -->
            <div class="pipe-card-meta" v-if="stage.meta">
              <span class="meta-size" v-if="stage.meta.size_mb">{{ stage.meta.size_mb }} MB</span>
              <span class="meta-time" v-if="stage.meta.modified">{{ stage.meta.modified }}</span>
            </div>
            <!-- Preview Visualization -->
            <div class="pipe-card-preview" v-if="stage.preview">
              <!-- Bar Group: Data Loading (Numeric vs Categorical) -->
              <template v-if="stage.preview.type === 'bar_group'">
                <div class="preview-bars">
                  <div v-for="(v, vi) in stage.preview.values" :key="vi" class="preview-bar-row">
                    <span class="preview-bar-label">{{ stage.preview.labels[vi] }}</span>
                    <div class="preview-bar-track">
                      <div class="preview-bar-fill" :style="{ width: (v / (stage.preview.values[0] + stage.preview.values[1]) * 100) + '%', background: stage.preview.colors[vi] }"></div>
                    </div>
                    <span class="preview-bar-val">{{ v }}</span>
                  </div>
                </div>
                <div class="preview-extra" v-if="stage.preview.extra">{{ stage.preview.extra }}</div>
              </template>
              <!-- Before/After: Data Processing -->
              <template v-if="stage.preview.type === 'before_after'">
                <div class="preview-ba">
                  <div class="preview-ba-item before">
                    <span class="preview-ba-label">{{ stage.preview.before_label }}</span>
                    <span class="preview-ba-val">{{ stage.preview.before_val }}</span>
                  </div>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                  <div class="preview-ba-item after">
                    <span class="preview-ba-label">{{ stage.preview.after_label }}</span>
                    <span class="preview-ba-val">{{ stage.preview.after_val }}</span>
                  </div>
                </div>
                <div class="preview-extra">{{ stage.preview.detail }}</div>
              </template>
              <!-- Pie Mini: Feature Engineering -->
              <template v-if="stage.preview.type === 'pie_mini'">
                <div class="preview-pie-row">
                  <div class="preview-pie-bars">
                    <div v-for="d in stage.preview.data" :key="d.name" class="preview-mini-bar">
                      <div class="mini-bar-fill" :style="{ width: (d.value / stage.preview.total * 100) + '%', background: d.color }"></div>
                      <span class="mini-bar-name">{{ d.name }}</span>
                      <span class="mini-bar-val">+{{ d.value }}</span>
                    </div>
                  </div>
                  <div class="preview-pie-total">
                    <span class="pie-total-num">{{ stage.preview.total }}</span>
                    <span class="pie-total-label">Features</span>
                  </div>
                </div>
              </template>
              <!-- Bar Compare: Model Training -->
              <template v-if="stage.preview.type === 'bar_compare'">
                <div class="preview-model-bars">
                  <div v-for="m in stage.preview.items" :key="m.name" class="preview-model-bar">
                    <span class="model-bar-name">{{ m.name }}</span>
                    <div class="model-bar-track">
                      <div class="model-bar-fill" :style="{ width: (m.auc * 100) + '%' }"></div>
                    </div>
                    <span class="model-bar-val">{{ m.auc }}</span>
                  </div>
                </div>
              </template>
              <!-- Metrics Row: Model Evaluation -->
              <template v-if="stage.preview.type === 'metrics_row'">
                <div class="preview-metrics">
                  <div v-for="(v, k) in stage.preview.metrics" :key="k" class="preview-metric-item">
                    <span class="preview-metric-val">{{ v }}</span>
                    <span class="preview-metric-key">{{ k }}</span>
                  </div>
                </div>
              </template>
              <!-- Strategy Count: Policy Simulation -->
              <template v-if="stage.preview.type === 'strategy_count'">
                <div class="preview-strategy">
                  <span class="strategy-big-num">{{ stage.preview.count }}</span>
                  <span class="strategy-label">Strategies</span>
                  <span class="strategy-range">PD: {{ stage.preview.range }}</span>
                </div>
              </template>
              <!-- PSI Bars: Monitoring -->
              <template v-if="stage.preview.type === 'psi_bars'">
                <div class="preview-psi">
                  <div class="psi-bar-group">
                    <div class="psi-bar-item emerald">
                      <span class="psi-bar-num">{{ stage.preview.stable }}</span>
                      <span class="psi-bar-label">Stable</span>
                    </div>
                    <div class="psi-bar-item amber">
                      <span class="psi-bar-num">{{ stage.preview.marginal }}</span>
                      <span class="psi-bar-label">Marginal</span>
                    </div>
                    <div class="psi-bar-item rose">
                      <span class="psi-bar-num">{{ stage.preview.unstable }}</span>
                      <span class="psi-bar-label">Unstable</span>
                    </div>
                  </div>
                </div>
              </template>
            </div>
          </div>
          <!-- Connector: separate flex item between cards -->
          <div v-if="idx < data.pipeline.length - 1" class="pipe-connector">
            <div class="connector-line">
              <div class="connector-dot" :style="{ animationDelay: (idx * 0.3) + 's' }"></div>
            </div>
            <div class="connector-arrow-wrap">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M9 18l6-6-6-6"/></svg>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Pipeline Detail Panel -->
    <transition name="panel">
      <div class="detail-panel" v-if="activeStage" v-loading="stageLoading" element-loading-background="rgba(10, 14, 26, 0.9)">
        <div class="panel-header">
          <div class="panel-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path :d="pipelineIconMap[activeStage?.icon] || ''"/>
            </svg>
            <span>{{ stageDetail?.title || activeStage?.name }}</span>
          </div>
          <button class="panel-close" @click="closePanel">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        <div class="panel-body" v-if="stageDetail">
          <!-- Data Loading -->
          <template v-if="stageDetail.stage === 'data_loading'">
            <div class="detail-section">
              <h4>Input</h4>
              <div class="info-grid">
                <div class="info-item"><span class="info-label">Source</span><span class="info-val">{{ stageDetail.input?.source }}</span></div>
                <div class="info-item"><span class="info-label">Train File</span><span class="info-val mono">{{ stageDetail.input?.train_file }}</span></div>
                <div class="info-item"><span class="info-label">Test File</span><span class="info-val mono">{{ stageDetail.input?.test_file }}</span></div>
              </div>
            </div>
            <div class="detail-section">
              <h4>Output</h4>
              <div class="info-grid">
                <div class="info-item"><span class="info-label">Train Shape</span><span class="info-val mono">{{ stageDetail.output?.train_shape?.join(' x ') }}</span></div>
                <div class="info-item"><span class="info-label">Test Shape</span><span class="info-val mono">{{ stageDetail.output?.test_shape?.join(' x ') }}</span></div>
                <div class="info-item"><span class="info-label">Numeric Cols</span><span class="info-val mono">{{ stageDetail.output?.numeric_columns }}</span></div>
                <div class="info-item"><span class="info-label">Categorical Cols</span><span class="info-val mono">{{ stageDetail.output?.categorical_columns }}</span></div>
                <div class="info-item full-width" v-if="stageDetail.output?.note"><span class="info-note">{{ stageDetail.output.note }}</span></div>
              </div>
            </div>
            <!-- Sample Data Preview -->
            <div class="detail-section" v-if="stageDetail.sample_data">
              <h4>Sample Data (application_train.csv)</h4>
              <div class="sample-table-wrap">
                <table class="sample-table">
                  <thead>
                    <tr>
                      <th v-for="col in stageDetail.sample_data.columns" :key="col" :class="{ 'col-target': col === 'TARGET' }">{{ col }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, ri) in stageDetail.sample_data.rows" :key="ri">
                      <td v-for="(val, vi) in row" :key="vi" :class="{ 'val-null': val === null, 'val-default': stageDetail.sample_data.columns[vi] === 'TARGET' && val === 1 }">{{ val === null ? 'NaN' : val }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div class="sample-dtypes">
                <span v-for="col in stageDetail.sample_data.columns" :key="col" class="dtype-chip">
                  <span class="dtype-col">{{ col }}</span>
                  <span class="dtype-type">{{ stageDetail.sample_data.dtypes?.[col] }}</span>
                </span>
              </div>
            </div>
            <!-- Auxiliary Tables -->
            <div class="detail-section">
              <h4>Auxiliary Tables ({{ stageDetail.auxiliary_tables?.length || 6 }})</h4>
              <div class="aux-expand-list">
                <div v-for="t in stageDetail.auxiliary_tables" :key="t.name" class="aux-expand-item">
                  <div class="aux-expand-header" @click="t._expanded = !t._expanded">
                    <div class="aux-expand-left">
                      <span class="aux-expand-name mono">{{ t.name }}</span>
                      <span class="aux-expand-shape mono" v-if="t.shape">{{ t.shape[0]?.toLocaleString() }} x {{ t.shape[1] }}</span>
                      <span class="aux-expand-size mono" v-if="t.size_mb">{{ t.size_mb }} MB</span>
                    </div>
                    <div class="aux-expand-right">
                      <span class="aux-expand-purpose">{{ t.purpose }}</span>
                      <svg :class="['aux-chevron', { open: t._expanded }]" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg>
                    </div>
                  </div>
                  <div class="aux-expand-body" v-if="t._expanded">
                    <div class="aux-key-info">
                      <span class="aux-key-label">Join Key:</span>
                      <span class="aux-key-val mono">{{ t.key_col }}</span>
                    </div>
                    <div class="aux-columns" v-if="t.columns?.length">
                      <span class="aux-col-label">Columns:</span>
                      <div class="aux-col-tags">
                        <span v-for="col in t.columns" :key="col" class="aux-col-tag mono">{{ col }}</span>
                      </div>
                    </div>
                    <div class="aux-sample" v-if="t.sample_rows?.length">
                      <span class="aux-sample-label">Sample (first 3 rows):</span>
                      <div class="aux-sample-table-wrap">
                        <table class="aux-sample-table">
                          <thead>
                            <tr><th v-for="col in t.columns" :key="col">{{ col }}</th></tr>
                          </thead>
                          <tbody>
                            <tr v-for="(row, ri) in t.sample_rows" :key="ri">
                              <td v-for="(val, vi) in row" :key="vi">{{ val }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <!-- Missing Values -->
            <div class="detail-section" v-if="stageDetail.details">
              <h4>Missing Values</h4>
              <div class="missing-overview">
                <div class="missing-overview-left">
                  <div class="info-grid">
                    <div class="info-item"><span class="info-label">Columns with Missing</span><span class="info-val mono">{{ stageDetail.details.missing_columns }}</span></div>
                    <div class="info-item"><span class="info-label">Total Missing</span><span class="info-val mono">{{ stageDetail.details.missing_total?.toLocaleString() }}</span></div>
                  </div>
                  <div class="bar-list" v-if="stageDetail.details.top_missing?.length">
                    <div v-for="m in stageDetail.details.top_missing" :key="m.column" class="bar-item">
                      <div class="bar-label">{{ m.column }}</div>
                      <div class="bar-track"><div class="bar-fill" :style="{ width: m.pct + '%' }"></div></div>
                      <div class="bar-val">{{ m.pct }}%</div>
                    </div>
                  </div>
                </div>
                <div class="missing-overview-right" v-if="columnDonutOption">
                  <v-chart :option="columnDonutOption" style="height: 140px" autoresize />
                </div>
              </div>
              <v-chart v-if="missingChartOption" :option="missingChartOption" style="height: 200px; margin-top: 12px" autoresize />
            </div>
          </template>

          <!-- Data Processing -->
          <template v-if="stageDetail.stage === 'data_processing'">
            <div class="detail-section">
              <h4>Data Flow</h4>
              <div class="flow-visual">
                <div class="flow-node input">
                  <div class="flow-node-label">Input</div>
                  <div class="flow-node-shape mono">{{ stageDetail.input?.shape?.join(' x ') }}</div>
                  <div class="flow-node-sub">Raw CSV</div>
                </div>
                <div class="flow-arrow-wrap">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="1.5" stroke-linecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                  <div class="flow-delta" v-if="stageDetail.input?.shape && stageDetail.output?.shape">
                    <span class="delta-badge cols" v-if="stageDetail.output.shape[1] !== stageDetail.input.shape[1]">
                      {{ stageDetail.output.shape[1] - stageDetail.input.shape[1] > 0 ? '+' : '' }}{{ stageDetail.output.shape[1] - stageDetail.input.shape[1] }} cols
                    </span>
                  </div>
                </div>
                <div class="flow-node output">
                  <div class="flow-node-label">Output</div>
                  <div class="flow-node-shape mono">{{ stageDetail.output?.shape?.join(' x ') }}</div>
                  <div class="flow-node-sub">Processed</div>
                </div>
              </div>
            </div>
            <div class="detail-section">
              <h4>Processing Steps</h4>
              <div class="step-expand-list">
                <div v-for="(step, i) in stageDetail.steps" :key="i" class="step-expand-item">
                  <div class="step-expand-header" @click="step._expanded = !step._expanded">
                    <div class="step-expand-left">
                      <div class="step-num">{{ i + 1 }}</div>
                      <div>
                        <div class="step-expand-name">{{ step.name }}</div>
                        <div class="step-expand-desc">{{ step.description }}</div>
                      </div>
                    </div>
                    <svg :class="['aux-chevron', { open: step._expanded }]" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg>
                  </div>
                  <transition name="step-slide">
                    <div class="step-expand-body" v-if="step._expanded && step.example">

                      <!-- Step 1: Business Rule — Cell transformation animation -->
                      <div v-if="step.example.column" class="transform-viz">
                        <div class="transform-cell">
                          <div class="transform-label">Before</div>
                          <div class="transform-value before mono">
                            <span class="strikethrough">{{ step.example.before }}</span>
                          </div>
                        </div>
                        <div class="transform-arrow">
                          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                        </div>
                        <div class="transform-cell">
                          <div class="transform-label">After</div>
                          <div class="transform-value after mono">{{ step.example.after }}</div>
                        </div>
                        <div class="transform-meta" v-if="step.example.affected_rows">
                          <span class="meta-affected">{{ step.example.affected_rows?.toLocaleString() }} rows affected</span>
                        </div>
                      </div>
                      <div class="new-flag" v-if="step.example.new_flag">
                        <span class="flag-badge">+New</span>
                        <span class="flag-text mono">{{ step.example.new_flag }}</span>
                      </div>

                      <!-- Step 2: Missing Values — Animated counter + progress -->
                      <div v-if="step.example.method_numeric" class="missing-viz">
                        <div class="missing-counter">
                          <div class="counter-block before">
                            <span class="counter-num mono">{{ step.example.before_missing_cols }}</span>
                            <span class="counter-label">Cols with NaN</span>
                          </div>
                          <div class="counter-arrow">
                            <div class="arrow-line"></div>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-emerald)" stroke-width="2.5" stroke-linecap="round"><path d="M9 18l6-6-6-6"/></svg>
                          </div>
                          <div class="counter-block after">
                            <span class="counter-num mono emerald">{{ step.example.after_missing_cols }}</span>
                            <span class="counter-label">Cols with NaN</span>
                          </div>
                        </div>
                        <div class="missing-progress">
                          <div class="progress-labels">
                            <span class="progress-label-before">{{ step.example.before_missing_cols }} cols</span>
                            <span class="progress-label-after">{{ step.example.after_missing_cols }} cols</span>
                          </div>
                          <div class="progress-track">
                            <div class="progress-fill" :style="{ width: ((1 - step.example.after_missing_cols / step.example.before_missing_cols) * 100) + '%' }"></div>
                          </div>
                          <div class="progress-subtitle">{{ (100 - step.example.after_missing_cols / step.example.before_missing_cols * 100).toFixed(0) }}% fixed</div>
                        </div>
                        <div class="missing-methods">
                          <div class="method-chip numeric">
                            <span class="method-icon">N</span>
                            <span>Numeric: {{ step.example.method_numeric }}</span>
                          </div>
                          <div class="method-chip categorical">
                            <span class="method-icon">C</span>
                            <span>Categorical: {{ step.example.method_categorical }}</span>
                          </div>
                        </div>
                        <div class="missing-sample" v-if="step.example.sample">
                          <span class="sample-arrow">Example:</span>
                          <span class="sample-col mono">{{ step.example.sample.column }}</span>
                          <span class="sample-fill">{{ step.example.sample.fill_value }}</span>
                        </div>
                      </div>

                      <!-- Step 3: Encoding — Category to number transformation -->
                      <div v-if="step.example.one_hot" class="encoding-viz">
                        <div class="encode-section">
                          <div class="encode-header">
                            <span class="encode-badge onehot">One-Hot</span>
                            <span class="encode-label">Low cardinality columns</span>
                          </div>
                          <div class="encode-transform">
                            <div class="encode-cell before">
                              <span class="encode-col mono">CODE_GENDER</span>
                              <span class="encode-val">"M"</span>
                            </div>
                            <div class="encode-arrow-cell">
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                            </div>
                            <div class="encode-cell after">
                              <span class="encode-bit on">M=1</span>
                              <span class="encode-bit off">F=0</span>
                            </div>
                          </div>
                          <div class="encode-cols mono">{{ step.example.one_hot.columns.join(', ') }}</div>
                        </div>
                        <div class="encode-section">
                          <div class="encode-header">
                            <span class="encode-badge freq">Freq</span>
                            <span class="encode-label">High cardinality columns</span>
                          </div>
                          <div class="encode-transform">
                            <div class="encode-cell before">
                              <span class="encode-col mono">ORG_TYPE</span>
                              <span class="encode-val">"Business Entity Type 3"</span>
                            </div>
                            <div class="encode-arrow-cell">
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-violet)" stroke-width="2" stroke-linecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                            </div>
                            <div class="encode-cell after">
                              <span class="encode-num mono">0.22</span>
                            </div>
                          </div>
                          <div class="encode-cols mono">{{ step.example.frequency.columns.join(', ') }}</div>
                        </div>
                      </div>

                      <!-- Step 4: Outlier — Value clamp visualization -->
                      <div v-if="step.example.method?.includes('Winsorize')" class="outlier-viz">
                        <div class="outlier-title">
                          <span class="outlier-badge">Winsorize</span>
                          <span class="outlier-scope">{{ step.example.columns_affected }}</span>
                        </div>
                        <div class="clamp-visual">
                          <div class="clamp-axis">
                            <div class="clamp-label left">Q1 - 1.5*IQR</div>
                            <div class="clamp-range">
                              <div class="clamp-bar"></div>
                              <div class="clamp-marker outlier" style="left: 5%">
                                <span class="clamp-marker-val">{{ step.example.sample?.before_max?.toLocaleString() }}</span>
                              </div>
                              <div class="clamp-marker cutoff" style="right: 25%">
                                <span class="clamp-marker-label">Q3 + 1.5*IQR</span>
                              </div>
                            </div>
                            <div class="clamp-label right">Q3 + 1.5*IQR</div>
                          </div>
                          <div class="clamp-note">
                            <span class="clamp-col mono">{{ step.example.sample?.column }}</span>
                            <span class="clamp-from mono rose">max: {{ step.example.sample?.before_max?.toLocaleString() }}</span>
                            <span class="clamp-arrow-icon">→</span>
                            <span class="clamp-to mono emerald">max: {{ step.example.sample?.after_max }}</span>
                          </div>
                        </div>
                      </div>

                    </div>
                  </transition>
                </div>
              </div>
            </div>
          </template>

          <!-- Feature Engineering -->
          <template v-if="stageDetail.stage === 'feature_engineering'">
            <div class="detail-section">
              <h4>Data Flow</h4>
              <div class="flow-visual">
                <div class="flow-node input">
                  <div class="flow-node-label">Input</div>
                  <div class="flow-node-shape mono">{{ stageDetail.input?.shape?.join(' x ') }}</div>
                  <div class="flow-node-sub">Processed</div>
                </div>
                <div class="flow-arrow-wrap">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-emerald)" stroke-width="1.5" stroke-linecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                  <div class="flow-delta">
                    <span class="delta-badge cols emerald">+{{ stageDetail.total_new_features }} feats</span>
                    <span class="delta-badge cols rose" v-if="stageDetail.correlation_removed">-{{ stageDetail.correlation_removed }} corr</span>
                  </div>
                </div>
                <div class="flow-node output">
                  <div class="flow-node-label">Output</div>
                  <div class="flow-node-shape mono">{{ stageDetail.output?.shape?.join(' x ') }}</div>
                  <div class="flow-node-sub">Final Features</div>
                </div>
              </div>
              <div class="info-grid" style="margin-top: 12px">
                <div class="info-item"><span class="info-label">New Features</span><span class="info-val mono highlight">+{{ stageDetail.total_new_features }}</span></div>
                <div class="info-item"><span class="info-label">After Correlation Removal</span><span class="info-val mono">{{ stageDetail.after_correlation_removal }}</span></div>
                <div class="info-item" v-if="stageDetail.correlation_removed"><span class="info-label">Correlated Removed</span><span class="info-val mono rose">-{{ stageDetail.correlation_removed }}</span></div>
              </div>
            </div>
            <div class="detail-section">
              <h4>Feature Sources</h4>
              <v-chart v-if="featureSourceOption" :option="featureSourceOption" style="height: 220px" autoresize />
              <div class="source-expand-list" style="margin-top: 12px">
                <div v-for="(step, i) in stageDetail.steps" :key="i" class="source-expand-item">
                  <div class="source-expand-header" @click="step._expanded = !step._expanded">
                    <div class="source-expand-left">
                      <span class="source-expand-name">{{ step.name }}</span>
                      <span class="source-expand-count">+{{ step.count }}</span>
                    </div>
                    <div class="source-expand-right">
                      <span class="source-expand-desc">{{ step.description }}</span>
                      <svg :class="['aux-chevron', { open: step._expanded }]" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg>
                    </div>
                  </div>
                  <div class="source-expand-body" v-if="step._expanded">
                    <!-- Formulas for basic features -->
                    <div v-if="step.formulas" class="formula-list">
                      <div v-for="f in step.formulas" :key="f.name" class="formula-item">
                        <div class="formula-name mono">{{ f.name }}</div>
                        <div class="formula-expr"><span class="formula-eq">=</span> <span class="mono">{{ f.formula }}</span></div>
                        <div class="formula-example"><span class="formula-label">Example:</span> <span class="mono">{{ f.example }}</span></div>
                        <div class="formula-purpose">{{ f.purpose }}</div>
                      </div>
                    </div>
                    <!-- Aggregations for auxiliary table features -->
                    <div v-if="step.aggregations" class="agg-list">
                      <div class="agg-source mono" v-if="step.source">Source: {{ step.source }}</div>
                      <div v-for="a in step.aggregations" :key="a.column" class="agg-item">
                        <div class="agg-col mono">{{ a.column }}</div>
                        <div class="agg-funcs">
                          <span v-for="f in a.funcs" :key="f" class="agg-func-tag">{{ f }}</span>
                        </div>
                        <div class="agg-example">{{ a.example }}</div>
                      </div>
                    </div>
                    <!-- Features list -->
                    <div v-if="step.features" class="feat-tags">
                      <span v-for="f in step.features" :key="f" class="feat-tag mono">{{ f }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div class="detail-section" v-if="stageDetail.corr_heatmap?.data?.length">
              <h4>Feature Correlation Matrix (Top 15)</h4>
              <div class="corr-note">Showing correlation between top-variance features. Highly correlated pairs (&gt;0.95) were removed during feature selection.</div>
              <v-chart v-if="corrHeatmapOption" :option="corrHeatmapOption" style="height: 420px" autoresize />
            </div>
          </template>

          <!-- Model Training -->
          <template v-if="stageDetail.stage === 'model_training'">
            <!-- Data Split -->
            <div class="detail-section" v-if="stageDetail.data_split">
              <h4>Data Split Strategy</h4>
              <div class="split-visual">
                <div class="split-bar">
                  <div class="split-segment train" :style="{ width: stageDetail.data_split.train_ratio }">
                    <span>Train {{ stageDetail.data_split.train_ratio }}</span>
                  </div>
                  <div class="split-segment valid" :style="{ width: stageDetail.data_split.validation_ratio }">
                    <span>Valid {{ stageDetail.data_split.validation_ratio }}</span>
                  </div>
                  <div class="split-segment calib" :style="{ width: stageDetail.data_split.calibration_ratio }">
                    <span>Calib {{ stageDetail.data_split.calibration_ratio }}</span>
                  </div>
                </div>
                <div class="split-note">{{ stageDetail.data_split.note }}</div>
              </div>
            </div>
            <!-- Feature Selection -->
            <div class="detail-section" v-if="stageDetail.feature_selection">
              <h4>Feature Selection</h4>
              <div class="info-grid">
                <div class="info-item"><span class="info-label">Method</span><span class="info-val">{{ stageDetail.feature_selection.method }}</span></div>
                <div class="info-item"><span class="info-label">IV Threshold</span><span class="info-val mono">{{ stageDetail.feature_selection.threshold }}</span></div>
                <div class="info-item"><span class="info-label">Before</span><span class="info-val mono">{{ stageDetail.feature_selection.before }} features</span></div>
                <div class="info-item"><span class="info-label">After</span><span class="info-val mono highlight">{{ stageDetail.feature_selection.after }} features</span></div>
              </div>
              <div class="info-note" style="margin-top: 8px">{{ stageDetail.feature_selection.note }}</div>
            </div>
            <!-- Model Cards -->
            <div class="detail-section">
              <h4>Models</h4>
              <div class="model-detail-cards">
                <div v-for="(model, key) in stageDetail.models" :key="key" class="model-detail-card">
                  <div class="model-detail-header">
                    <span class="model-detail-name">{{ model.name }}</span>
                    <span class="model-detail-type mono">{{ model.type }}</span>
                  </div>
                  <div class="model-detail-body">
                    <div class="info-grid compact">
                      <div class="info-item"><span class="info-label">Features</span><span class="info-val mono">{{ model.features }}</span></div>
                    </div>
                    <!-- Hyperparams -->
                    <div class="hyperparams" v-if="model.hyperparams">
                      <div class="hyper-label">Hyperparameters</div>
                      <div class="hyper-grid">
                        <div v-for="(v, k) in model.hyperparams" :key="k" class="hyper-item">
                          <span class="hyper-key mono">{{ k }}</span>
                          <span class="hyper-val mono">{{ v }}</span>
                        </div>
                      </div>
                    </div>
                    <!-- Training Flow -->
                    <div class="train-flow" v-if="model.training_flow">
                      <div class="hyper-label">Training Flow</div>
                      <div class="flow-steps">
                        <div v-for="(fs, fi) in model.training_flow" :key="fi" class="flow-step">
                          <span class="flow-step-num">{{ fi + 1 }}</span>
                          <span class="flow-step-text">{{ fs }}</span>
                        </div>
                      </div>
                    </div>
                    <!-- Techniques -->
                    <div class="tech-tags" v-if="model.techniques">
                      <span v-for="t in model.techniques" :key="t" class="tech-tag">{{ t }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </template>

          <!-- Model Evaluation -->
          <template v-if="stageDetail.stage === 'model_evaluation'">
            <!-- Metrics Explanation -->
            <div class="detail-section" v-if="stageDetail.metrics_explanation">
              <h4>Metrics Explained</h4>
              <div class="metrics-explain-list">
                <div v-for="m in stageDetail.metrics_explanation" :key="m.name" class="metrics-explain-item">
                  <div class="metrics-explain-header">
                    <span class="metrics-explain-name mono">{{ m.name }}</span>
                    <span class="metrics-explain-full">{{ m.full }}</span>
                    <span class="metrics-explain-range mono">{{ m.range }}</span>
                  </div>
                  <div class="metrics-explain-meaning">{{ m.meaning }}</div>
                </div>
              </div>
            </div>
            <!-- Radar Chart -->
            <div class="detail-section">
              <h4>Metrics Comparison</h4>
              <v-chart v-if="evalRadarOption" :option="evalRadarOption" style="height: 280px" autoresize />
            </div>
            <!-- Results Table -->
            <div class="detail-section">
              <h4>Results</h4>
              <div class="eval-table">
                <div class="eval-header">
                  <span>Model</span>
                  <span>AUC</span>
                  <span>KS</span>
                  <span>Gini</span>
                  <span>F1</span>
                </div>
                <div v-for="m in stageDetail.models" :key="m.Model" class="eval-row">
                  <span class="eval-model">{{ m.Model }}</span>
                  <span class="eval-val highlight">{{ m.AUC?.toFixed(4) }}</span>
                  <span class="eval-val">{{ m.KS?.toFixed(4) }}</span>
                  <span class="eval-val highlight">{{ m.Gini?.toFixed(4) }}</span>
                  <span class="eval-val">{{ m['F1-Score']?.toFixed(4) }}</span>
                </div>
              </div>
            </div>
            <!-- Evaluation Process -->
            <div class="detail-section" v-if="stageDetail.evaluation_process">
              <h4>Evaluation Process</h4>
              <div class="flow-steps">
                <div v-for="(step, i) in stageDetail.evaluation_process" :key="i" class="flow-step">
                  <span class="flow-step-num">{{ i + 1 }}</span>
                  <span class="flow-step-text">{{ step }}</span>
                </div>
              </div>
            </div>
          </template>

          <!-- Policy Simulation -->
          <template v-if="stageDetail.stage === 'policy_simulation'">
            <div class="detail-section">
              <h4>Configuration</h4>
              <div class="info-grid">
                <div class="info-item"><span class="info-label">LGD</span><span class="info-val mono">{{ stageDetail.lgd }}</span></div>
                <div class="info-item"><span class="info-label">Cut-off Range</span><span class="info-val mono">{{ stageDetail.cutoff_range }}</span></div>
                <div class="info-item"><span class="info-label">Strategies</span><span class="info-val mono">{{ stageDetail.strategies_count }}</span></div>
              </div>
            </div>
            <div class="detail-section">
              <h4>Strategies</h4>
              <v-chart v-if="strategyChartOption" :option="strategyChartOption" style="height: 280px" autoresize />
              <div class="strategy-list" style="margin-top: 12px">
                <div v-for="s in stageDetail.strategies" :key="s.name" class="strategy-item">
                  <span class="strategy-name">{{ s.name }}</span>
                  <span class="strategy-cut mono">PD={{ s.cutoff }}</span>
                  <span class="strategy-rate">{{ s.approval_rate }}%</span>
                </div>
              </div>
            </div>
          </template>

          <!-- Monitoring -->
          <template v-if="stageDetail.stage === 'monitoring'">
            <div class="detail-section">
              <h4>Method</h4>
              <div class="info-grid">
                <div class="info-item"><span class="info-label">Metric</span><span class="info-val">{{ stageDetail.method }}</span></div>
                <div class="info-item"><span class="info-label">Stable</span><span class="info-val mono emerald">&lt;= {{ stageDetail.thresholds?.stable }}</span></div>
                <div class="info-item"><span class="info-label">Marginal</span><span class="info-val mono amber">{{ stageDetail.thresholds?.stable }} ~ {{ stageDetail.thresholds?.marginal }}</span></div>
                <div class="info-item"><span class="info-label">Unstable</span><span class="info-val mono rose">&gt; {{ stageDetail.thresholds?.marginal }}</span></div>
              </div>
            </div>
            <div class="detail-section">
              <h4>Results</h4>
              <div class="psi-summary">
                <div class="psi-item emerald"><span class="psi-num">{{ stageDetail.results?.stable }}</span><span class="psi-label">Stable</span></div>
                <div class="psi-item amber"><span class="psi-num">{{ stageDetail.results?.marginal }}</span><span class="psi-label">Marginal</span></div>
                <div class="psi-item rose"><span class="psi-num">{{ stageDetail.results?.unstable }}</span><span class="psi-label">Unstable</span></div>
              </div>
            </div>
            <div class="detail-section" v-if="stageDetail.top_unstable_features?.length">
              <h4>Top Unstable Features</h4>
              <v-chart v-if="psiDistOption" :option="psiDistOption" style="height: 200px" autoresize />
              <div class="bar-list" style="margin-top: 12px">
                <div v-for="f in stageDetail.top_unstable_features" :key="f.feature" class="bar-item">
                  <div class="bar-label">{{ f.feature }}</div>
                  <div class="bar-track"><div class="bar-fill rose" :style="{ width: Math.min(f.psi * 100, 100) + '%' }"></div></div>
                  <div class="bar-val">{{ f.psi }}</div>
                </div>
              </div>
            </div>
          </template>
        </div>
      </div>
    </transition>

    <!-- Charts Row: Target Distribution + Dataset Info + Model Trend -->
    <div class="charts-row charts-row-3 animate-in animate-in-delay-6" v-if="data && !loading">
      <div class="chart-card">
        <div class="chart-header">
          <span>Target Distribution</span>
          <span class="chart-badge">Train Set</span>
        </div>
        <v-chart :option="targetOption" style="height: 260px" autoresize />
      </div>

      <div class="chart-card">
        <div class="chart-header">
          <span>AUC Comparison</span>
          <span class="chart-badge">All Models</span>
        </div>
        <v-chart :option="aucCompareOption" style="height: 260px" autoresize />
      </div>

      <div class="chart-card">
        <div class="chart-header">
          <span>Model Performance Trend</span>
          <span class="chart-badge">AUC / KS / Gini</span>
        </div>
        <v-chart :option="modelTrendOption" style="height: 260px" autoresize />
      </div>
    </div>

    <!-- Dataset Info Summary -->
    <div class="dataset-info-bar animate-in animate-in-delay-6" v-if="data?.dataset_info && !loading">
      <div class="dinfo-item">
        <span class="dinfo-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        </span>
        <span class="dinfo-label">Train</span>
        <span class="dinfo-val">{{ data.dataset_info.train_rows?.toLocaleString() }} rows</span>
      </div>
      <div class="dinfo-sep"></div>
      <div class="dinfo-item">
        <span class="dinfo-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        </span>
        <span class="dinfo-label">Test</span>
        <span class="dinfo-val">{{ data.dataset_info.test_rows?.toLocaleString() }} rows</span>
      </div>
      <div class="dinfo-sep"></div>
      <div class="dinfo-item">
        <span class="dinfo-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        </span>
        <span class="dinfo-label">Features</span>
        <span class="dinfo-val">{{ data.n_features }}</span>
      </div>
      <div class="dinfo-sep"></div>
      <div class="dinfo-item">
        <span class="dinfo-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
        </span>
        <span class="dinfo-label">Numeric</span>
        <span class="dinfo-val">{{ data.dataset_info.numeric_features }}</span>
      </div>
      <div class="dinfo-sep"></div>
      <div class="dinfo-item">
        <span class="dinfo-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
        </span>
        <span class="dinfo-label">Train Cols</span>
        <span class="dinfo-val">{{ data.dataset_info.train_cols }}</span>
      </div>
    </div>

    <!-- Model Comparison Table -->
    <div class="table-section animate-in animate-in-delay-6" v-if="data?.models?.length && !loading">
      <div class="section-header">
        <div class="section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          <span>Model Comparison</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px">
          <span class="table-badge">{{ data.models.length }} Models</span>
          <button class="export-btn" @click="exportCSV">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
            CSV
          </button>
        </div>
      </div>
      <div class="table-wrap">
        <el-table :data="data.models" :header-cell-style="{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-muted)', fontWeight: 500, borderBottom: '1px solid var(--border-subtle)' }" :cell-style="{ borderBottom: '1px solid var(--border-subtle)' }">
          <el-table-column prop="Model" label="Model" min-width="160" fixed>
            <template #default="{ row }">
              <div class="model-name-cell" @click="goToModel(row.Model)" style="cursor: pointer">
                <span class="model-dot" :class="row.Model?.includes('XG') ? 'xgb' : row.Model?.includes('Light') ? 'lgb' : row.Model?.includes('Stack') ? 'stk' : 'lr'"></span>
                <span class="model-name">{{ row.Model || '-' }}</span>
                <span v-if="row.Model === bestModel" class="best-badge">Best</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column v-for="col in ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall', 'F1-Score']"
            :key="col" :label="col" min-width="100">
            <template #default="{ row }">
              <span class="metric-cell" :class="{ highlight: col === 'AUC' || col === 'Gini' }">
                {{ typeof row[col] === 'number' ? row[col].toFixed(4) : (row[col] || '-') }}
              </span>
            </template>
          </el-table-column>
          <el-table-column label="TP" min-width="80">
            <template #default="{ row }">
              <span class="cm-cell tp">{{ row.TP?.toLocaleString() || '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="TN" min-width="80">
            <template #default="{ row }">
              <span class="cm-cell tn">{{ row.TN?.toLocaleString() || '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="FP" min-width="80">
            <template #default="{ row }">
              <span class="cm-cell fp">{{ row.FP?.toLocaleString() || '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="FN" min-width="80">
            <template #default="{ row }">
              <span class="cm-cell fn">{{ row.FN?.toLocaleString() || '-' }}</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- Model Governance Registry -->
    <div class="registry-section animate-in animate-in-delay-7" v-if="registryModels.length && !registryLoading">
      <div class="section-header">
        <div class="section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
          <span>Model Governance</span>
        </div>
        <div class="registry-meta" v-if="registrySummary">
          <span class="registry-meta-item">{{ registrySummary.count }} Models</span>
          <span class="registry-meta-divider">|</span>
          <span class="registry-meta-item">Updated {{ formatTime(registrySummary.updatedAt) }}</span>
        </div>
      </div>

      <!-- Registry Summary Cards -->
      <div class="registry-summary" v-if="registrySummary?.latest">
        <div class="registry-summary-card latest">
          <div class="rsc-label">Latest Trained</div>
          <div class="rsc-value">{{ registrySummary.latest.model_name }}</div>
          <div class="rsc-sub">{{ formatTime(registrySummary.latest.trained_at) }}</div>
        </div>
        <div class="registry-summary-card">
          <div class="rsc-label">Best AUC</div>
          <div class="rsc-value gold">{{ Math.max(...registryModels.map(m => m.metrics?.AUC || 0)).toFixed(4) }}</div>
          <div class="rsc-sub">{{ registryModels.reduce((a, b) => (b.metrics?.AUC || 0) > (a.metrics?.AUC || 0) ? b : a).model_name }}</div>
        </div>
        <div class="registry-summary-card">
          <div class="rsc-label">Total Artifacts</div>
          <div class="rsc-value">{{ registryModels.reduce((s, m) => s + (m.artifact?.size_mb || 0), 0).toFixed(1) }} MB</div>
          <div class="rsc-sub">{{ registryModels.length }} model files</div>
        </div>
        <div class="registry-summary-card">
          <div class="rsc-label">Feature Version</div>
          <div class="rsc-value">{{ registrySummary.featureCount }}</div>
          <div class="rsc-sub">{{ registrySummary.latest?.features_hash?.slice(0, 8) || '-' }}</div>
        </div>
      </div>

      <!-- Registry Table -->
      <div class="table-wrap">
        <el-table :data="registryModels" :header-cell-style="{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-muted)', fontWeight: 500, borderBottom: '1px solid var(--border-subtle)' }" :cell-style="{ borderBottom: '1px solid var(--border-subtle)' }">
          <el-table-column label="Model" min-width="170" fixed>
            <template #default="{ row }">
              <div class="model-name-cell">
                <span class="model-dot" :class="row.model_type === 'xgboost' ? 'xgb' : row.model_type === 'lightgbm' ? 'lgb' : row.model_type === 'stacking' ? 'stk' : 'lr'"></span>
                <span class="model-name">{{ row.model_name }}</span>
                <span v-if="row.metrics?.AUC && row.metrics.AUC === Math.max(...registryModels.map(m => m.metrics?.AUC || 0))" class="best-badge">Best</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="AUC" min-width="90">
            <template #default="{ row }">
              <span class="metric-cell" :class="{ highlight: true }">{{ row.metrics?.AUC ? row.metrics.AUC.toFixed(4) : '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="KS" min-width="80">
            <template #default="{ row }">
              <span class="metric-cell">{{ row.metrics?.KS ? row.metrics.KS.toFixed(4) : '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Gini" min-width="80">
            <template #default="{ row }">
              <span class="metric-cell">{{ row.metrics?.Gini ? row.metrics.Gini.toFixed(4) : '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Features" min-width="80">
            <template #default="{ row }">
              <span class="mono-cell">{{ row.feature_count ?? '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Size" min-width="80">
            <template #default="{ row }">
              <span class="mono-cell">{{ row.artifact?.size_mb ? row.artifact.size_mb + ' MB' : '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Trained" min-width="130">
            <template #default="{ row }">
              <span class="mono-cell">{{ formatTime(row.trained_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Code Ver" min-width="90">
            <template #default="{ row }">
              <span class="mono-cell hash">{{ row.code_version || '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Fingerprint" min-width="110">
            <template #default="{ row }">
              <span class="mono-cell hash" :title="row.features_hash">{{ row.features_hash ? row.features_hash.slice(0, 12) + '...' : '-' }}</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-header { margin-bottom: 32px; }
.header-row { display: flex; align-items: flex-start; justify-content: space-between; }
.page-header h2 { margin: 0; font-family: var(--font-display); font-size: 28px; }
.page-desc { color: var(--text-muted); font-size: 12px; margin-top: 6px; font-family: var(--font-mono); letter-spacing: 0.5px; }

.refresh-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: 1px solid var(--border-subtle);
  background: rgba(255, 255, 255, 0.02);
  color: var(--text-secondary);
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 500;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.refresh-btn:hover:not(:disabled) {
  border-color: var(--accent-gold);
  color: var(--accent-gold);
  background: rgba(200, 170, 110, 0.06);
}

.refresh-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.refresh-icon { transition: transform 0.3s ease; }
.refresh-icon.spinning { animation: spin 1s linear infinite; }

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.export-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 5px 10px;
  border: 1px solid var(--border-subtle);
  background: rgba(255, 255, 255, 0.02);
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.25s;
}

.export-btn:hover {
  border-color: var(--accent-emerald);
  color: var(--accent-emerald);
  background: rgba(16, 185, 129, 0.08);
}

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 24px;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  backdrop-filter: blur(20px);
  position: relative;
  overflow: hidden;
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(200, 170, 110, 0.12), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}

.stat-card:hover {
  border-color: var(--border-accent);
  box-shadow: var(--shadow-glow-gold);
  transform: translateY(-1px);
}

.stat-card:hover::before { opacity: 1; }

.stat-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

.stat-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stat-icon.blue { background: rgba(74, 158, 255, 0.1); color: var(--accent-blue); }
.stat-icon.violet { background: rgba(167, 139, 250, 0.1); color: var(--accent-violet); }
.stat-icon.emerald { background: rgba(52, 211, 153, 0.1); color: var(--accent-emerald); }
.stat-icon.rose { background: rgba(251, 113, 133, 0.1); color: var(--accent-rose); }

.stat-tag {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.03);
  padding: 2px 8px;
  border-radius: 4px;
  letter-spacing: 0.3px;
}

.stat-tag.emerald { color: var(--accent-emerald); background: rgba(52, 211, 153, 0.08); }
.stat-tag.rose { color: var(--accent-rose); background: rgba(251, 113, 133, 0.08); }

.stat-number {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  font-family: var(--font-display);
  letter-spacing: -1px;
  line-height: 1;
  margin-bottom: 6px;
}

.stat-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 14px;
}

.stat-footer {
  padding-top: 12px;
  border-top: 1px solid var(--border-subtle);
}

.stat-foot-item {
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.stat-bar {
  height: 4px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 2px;
  overflow: hidden;
}

.stat-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.stat-bar-fill.emerald { background: var(--gradient-emerald); }
.stat-bar-fill.rose { background: var(--gradient-rose); }

/* Pipeline Section */
.pipeline-section, .table-section {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: 20px;
  backdrop-filter: blur(20px);
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  border-bottom: 1px solid var(--border-subtle);
}

.section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.pipeline-badge, .table-badge {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--accent-gold);
  background: rgba(200, 170, 110, 0.08);
  padding: 3px 10px;
  border-radius: 20px;
  font-weight: 500;
  letter-spacing: 0.3px;
}

/* Pipeline Horizontal Cards */
.pipeline-scroll {
  display: flex;
  gap: 8px;
  padding: 24px 20px;
  overflow-x: auto;
  position: relative;
  scroll-behavior: smooth;
}

.pipeline-scroll::-webkit-scrollbar {
  height: 6px;
}

.pipeline-scroll::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.02);
  border-radius: 3px;
}

.pipeline-scroll::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.08);
  border-radius: 3px;
}

.pipeline-scroll::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.15);
}

.pipe-card {
  flex: 0 0 auto;
  width: 240px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 20px;
  cursor: pointer;
  transition: all 0.25s ease;
  position: relative;
  backdrop-filter: blur(8px);
}

.pipe-card:hover {
  border-color: var(--border-accent);
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.pipe-card.active {
  border-color: rgba(59, 130, 246, 0.5);
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.1);
}

.pipe-step-num {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: rgba(200, 170, 110, 0.1);
  color: var(--accent-gold);
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-mono);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  border: 1px solid rgba(200, 170, 110, 0.2);
}

.pipe-card-head {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.pipe-card-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.pipe-card-icon.done {
  background: rgba(16, 185, 129, 0.12);
  color: var(--accent-emerald);
}

.pipe-card-icon.pending {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-muted);
}

.pipe-card-titles {
  flex: 1;
  min-width: 0;
}

.pipe-card-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  display: block;
  line-height: 1.3;
}

.pipe-card-en {
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  display: block;
  margin-top: 2px;
}

.pipe-card-badge {
  font-size: 9px;
  font-family: var(--font-mono);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  padding: 2px 6px;
  border-radius: 10px;
  flex-shrink: 0;
}

.pipe-card-badge.done {
  background: rgba(16, 185, 129, 0.12);
  color: var(--accent-emerald);
}

.pipe-card-badge.pending {
  background: rgba(255, 255, 255, 0.06);
  color: var(--text-muted);
}

.pipe-card-desc {
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.6;
  margin-bottom: 8px;
  min-height: 34px;
}

.pipe-card-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  padding: 4px 8px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 4px;
}

.meta-size {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.08);
  padding: 1px 6px;
  border-radius: 3px;
}

.meta-time {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text-muted);
}

.pipe-card-preview {
  border-top: 1px solid var(--border-subtle);
  padding-top: 14px;
}

/* Connector between cards */
.pipe-connector {
  display: flex;
  align-items: center;
  width: 32px;
  flex: 0 0 32px;
  align-self: center;
}

.connector-line {
  flex: 1;
  height: 2px;
  background: rgba(255, 255, 255, 0.08);
  position: relative;
  overflow: visible;
}

.connector-dot {
  position: absolute;
  top: 50%;
  left: 0%;
  transform: translate(0, -50%);
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent-gold);
  box-shadow: 0 0 6px rgba(200, 170, 110, 0.5);
  animation: flowDot 2s ease-in-out infinite;
}

@keyframes flowDot {
  0% { left: 0%; transform: translate(0, -50%); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { left: 100%; transform: translate(-100%, -50%); opacity: 0; }
}

.pipe-card.done + .pipe-connector .connector-line {
  background: rgba(16, 185, 129, 0.3);
}

.connector-arrow-wrap {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  color: var(--text-muted);
  opacity: 0.4;
  margin-left: -1px;
}

.pipe-card.done + .pipe-connector .connector-arrow-wrap {
  color: var(--accent-emerald);
  opacity: 0.6;
}

/* Preview: Bar Group (Data Loading) */
.preview-bars {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.preview-bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.preview-bar-label {
  font-size: 11px;
  color: var(--text-muted);
  width: 72px;
  flex-shrink: 0;
}

.preview-bar-track {
  flex: 1;
  height: 10px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 5px;
  overflow: hidden;
}

.preview-bar-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.6s ease;
}

.preview-bar-val {
  font-size: 12px;
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--text-primary);
  width: 30px;
  text-align: right;
  flex-shrink: 0;
}

.preview-extra {
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  margin-top: 8px;
  text-align: center;
}

/* Preview: Before/After (Data Processing) */
.preview-ba {
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: center;
}

.preview-ba-item {
  text-align: center;
}

.preview-ba-label {
  font-size: 10px;
  color: var(--text-muted);
  display: block;
}

.preview-ba-val {
  font-size: 12px;
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--text-primary);
  display: block;
  margin-top: 2px;
}

.preview-ba-item.after .preview-ba-val {
  color: var(--accent-emerald);
}

/* Preview: Pie Mini (Feature Engineering) */
.preview-pie-row {
  display: flex;
  gap: 12px;
  align-items: center;
}

.preview-pie-bars {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.preview-mini-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  height: 14px;
  position: relative;
}

.mini-bar-fill {
  height: 100%;
  border-radius: 3px;
  min-width: 4px;
  transition: width 0.6s ease;
}

.mini-bar-name {
  font-size: 9px;
  color: var(--text-muted);
  white-space: nowrap;
  flex-shrink: 0;
  min-width: 40px;
}

.mini-bar-val {
  font-size: 9px;
  font-family: var(--font-mono);
  color: var(--text-primary);
  font-weight: 600;
  flex-shrink: 0;
}

.preview-pie-total {
  text-align: center;
  flex-shrink: 0;
}

.pie-total-num {
  font-size: 22px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
  display: block;
  line-height: 1;
}

.pie-total-label {
  font-size: 10px;
  color: var(--text-muted);
  display: block;
  margin-top: 2px;
}

/* Preview: Bar Compare (Model Training) */
.preview-model-bars {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.preview-model-bar {
  display: flex;
  align-items: center;
  gap: 10px;
}

.model-bar-name {
  font-size: 11px;
  color: var(--text-muted);
  width: 65px;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.model-bar-track {
  flex: 1;
  height: 10px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 5px;
  overflow: hidden;
}

.model-bar-fill {
  height: 100%;
  border-radius: 5px;
  background: linear-gradient(90deg, #3b82f6, #06b6d4);
  transition: width 0.6s ease;
}

.model-bar-val {
  font-size: 12px;
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--accent-blue);
  width: 40px;
  text-align: right;
  flex-shrink: 0;
}

/* Preview: Metrics Row (Model Evaluation) */
.preview-metrics {
  display: flex;
  gap: 14px;
  justify-content: center;
}

.preview-metric-item {
  text-align: center;
}

.preview-metric-val {
  font-size: 18px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
  display: block;
  line-height: 1;
}

.preview-metric-key {
  font-size: 10px;
  color: var(--text-muted);
  display: block;
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

/* Preview: Strategy Count */
.preview-strategy {
  text-align: center;
}

.strategy-big-num {
  font-size: 28px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--accent-blue);
  display: block;
  line-height: 1;
}

.strategy-label {
  font-size: 11px;
  color: var(--text-muted);
  display: block;
  margin-top: 2px;
}

.strategy-range {
  font-size: 10px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  display: block;
  margin-top: 4px;
}

/* Preview: PSI Bars (Monitoring) */
.preview-psi {
  display: flex;
  justify-content: center;
}

.psi-bar-group {
  display: flex;
  gap: 12px;
}

.psi-bar-item {
  text-align: center;
  padding: 6px 10px;
  border-radius: 8px;
}

.psi-bar-item.emerald { background: rgba(16, 185, 129, 0.08); }
.psi-bar-item.amber { background: rgba(245, 158, 11, 0.08); }
.psi-bar-item.rose { background: rgba(244, 63, 94, 0.08); }

.psi-bar-num {
  font-size: 16px;
  font-weight: 700;
  font-family: var(--font-mono);
  display: block;
  line-height: 1;
}

.psi-bar-item.emerald .psi-bar-num { color: var(--accent-emerald); }
.psi-bar-item.amber .psi-bar-num { color: var(--accent-amber); }
.psi-bar-item.rose .psi-bar-num { color: var(--accent-rose); }

.psi-bar-label {
  font-size: 9px;
  color: var(--text-muted);
  display: block;
  margin-top: 2px;
}

/* Pipeline Detail Panel */
.detail-panel {
  background: var(--bg-secondary);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-md);
  margin-bottom: 20px;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(59, 130, 246, 0.1);
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-subtle);
  background: rgba(59, 130, 246, 0.05);
}

.panel-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.panel-close {
  width: 32px;
  height: 32px;
  border: none;
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-muted);
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.panel-close:hover { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); }

.panel-body {
  padding: 20px;
  max-height: 500px;
  overflow-y: auto;
}

.detail-section {
  margin-bottom: 20px;
}

.detail-section:last-child { margin-bottom: 0; }

.detail-section h4 {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent-blue);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(59, 130, 246, 0.1);
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.info-label { font-size: 11px; color: var(--text-muted); font-weight: 500; }
.info-val { font-size: 13px; color: var(--text-primary); font-weight: 500; }
.info-val.mono { font-family: var(--font-mono); }
.info-val.highlight { color: var(--accent-blue); font-weight: 600; }
.info-val.emerald { color: var(--accent-emerald); }
.info-val.amber { color: var(--accent-amber); }

.info-item.full-width { grid-column: 1 / -1; }
.info-note {
  font-size: 11px;
  color: var(--text-muted);
  font-style: italic;
  padding: 6px 10px;
  background: rgba(59, 130, 246, 0.06);
  border-radius: 6px;
  border-left: 2px solid var(--accent-blue);
}
.info-val.rose { color: var(--accent-rose); }

.aux-list { display: flex; flex-direction: column; gap: 6px; }
.aux-item { display: flex; gap: 12px; padding: 8px 12px; background: rgba(255, 255, 255, 0.02); border-radius: 6px; }
.aux-name { font-family: var(--font-mono); font-size: 12px; color: var(--accent-blue); font-weight: 500; min-width: 200px; }
.aux-desc { font-size: 12px; color: var(--text-secondary); }

.bar-list { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.bar-item { display: flex; align-items: center; gap: 10px; }
.bar-label { font-size: 11px; font-family: var(--font-mono); color: var(--text-secondary); min-width: 200px; }
.bar-track { flex: 1; height: 6px; background: rgba(255, 255, 255, 0.06); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--gradient-blue); border-radius: 3px; transition: width 0.5s; }
.bar-fill.rose { background: var(--gradient-rose); }
.bar-val { font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); min-width: 50px; text-align: right; }

.flow-visual {
  display: flex;
  align-items: center;
  gap: 0;
}

.flow-node {
  padding: 12px 20px;
  border-radius: 10px;
  border: 1px solid;
  text-align: center;
  min-width: 120px;
}

.flow-node.input {
  background: rgba(59, 130, 246, 0.06);
  border-color: rgba(59, 130, 246, 0.15);
}

.flow-node.output {
  background: rgba(16, 185, 129, 0.06);
  border-color: rgba(16, 185, 129, 0.15);
}

.flow-node-label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 600;
  margin-bottom: 4px;
}

.flow-node.input .flow-node-label { color: var(--accent-blue); }
.flow-node.output .flow-node-label { color: var(--accent-emerald); }

.flow-node-shape {
  font-size: 16px;
  color: var(--text-primary);
  font-weight: 700;
  line-height: 1.2;
}

.flow-node-sub {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 2px;
}

.flow-arrow-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 0 8px;
  flex-shrink: 0;
}

.flow-delta {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  justify-content: center;
}

.delta-badge {
  font-size: 9px;
  font-family: var(--font-mono);
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 3px;
  white-space: nowrap;
}

.delta-badge.cols { background: rgba(59, 130, 246, 0.1); color: var(--accent-blue); }
.delta-badge.cols.emerald { background: rgba(16, 185, 129, 0.1); color: var(--accent-emerald); }
.delta-badge.cols.rose { background: rgba(244, 63, 94, 0.1); color: var(--accent-rose); }

.flow-box { display: flex; align-items: center; gap: 12px; }
.flow-shape { padding: 8px 16px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.15); border-radius: 8px; font-size: 14px; color: var(--accent-blue); font-weight: 600; }

.step-list { display: flex; flex-direction: column; gap: 10px; }
.step-item { display: flex; gap: 12px; padding: 10px 12px; background: rgba(255, 255, 255, 0.02); border-radius: 8px; }
.step-num { width: 28px; height: 28px; border-radius: 8px; background: rgba(59, 130, 246, 0.1); color: var(--accent-blue); font-family: var(--font-mono); font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.step-name { font-size: 13px; font-weight: 600; color: var(--text-primary); margin-bottom: 2px; }
.step-desc { font-size: 12px; color: var(--text-secondary); }

.source-list { display: flex; flex-direction: column; gap: 10px; }
.source-item { padding: 12px; background: rgba(255, 255, 255, 0.02); border-radius: 8px; }
.source-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.source-name { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.source-count { font-family: var(--font-mono); font-size: 13px; font-weight: 700; color: var(--accent-emerald); background: rgba(16, 185, 129, 0.1); padding: 2px 8px; border-radius: 4px; }
.source-desc { font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
.source-file { font-size: 11px; color: var(--text-muted); }

.model-cards { display: flex; flex-direction: column; gap: 12px; }
.model-card { background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden; }
.model-card-header { padding: 12px 16px; border-bottom: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center; }
.model-card-name { font-weight: 600; color: var(--text-primary); font-size: 14px; }
.model-card-type { font-size: 11px; color: var(--text-muted); }
.model-card-body { padding: 12px 16px; }
.tech-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.tech-tag { font-size: 11px; padding: 3px 8px; background: rgba(139, 92, 246, 0.1); color: var(--accent-violet); border-radius: 4px; font-weight: 500; }

.metric-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.metric-tag { font-size: 12px; padding: 4px 12px; background: rgba(59, 130, 246, 0.08); color: var(--accent-blue); border-radius: 6px; font-family: var(--font-mono); font-weight: 500; }

.eval-table { display: flex; flex-direction: column; gap: 1px; background: var(--border-subtle); border-radius: 8px; overflow: hidden; }
.eval-header, .eval-row { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr; gap: 8px; padding: 10px 14px; background: var(--bg-secondary); }
.eval-header { background: rgba(255, 255, 255, 0.03); font-size: 11px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; }
.eval-model { font-weight: 600; color: var(--text-primary); font-size: 13px; }
.eval-val { font-family: var(--font-mono); font-size: 13px; color: var(--text-secondary); }
.eval-val.highlight { color: var(--accent-blue); font-weight: 600; }

.strategy-list { display: flex; flex-direction: column; gap: 6px; }
.strategy-item { display: flex; align-items: center; gap: 12px; padding: 8px 12px; background: rgba(255, 255, 255, 0.02); border-radius: 6px; }
.strategy-name { font-size: 13px; font-weight: 500; color: var(--text-primary); flex: 1; }
.strategy-cut { font-size: 12px; color: var(--accent-blue); }
.strategy-rate { font-family: var(--font-mono); font-size: 12px; color: var(--accent-emerald); font-weight: 600; }

.psi-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.psi-item { text-align: center; padding: 16px; border-radius: 8px; }
.psi-item.emerald { background: rgba(16, 185, 129, 0.08); }
.psi-item.amber { background: rgba(245, 158, 11, 0.08); }
.psi-item.rose { background: rgba(244, 63, 94, 0.08); }
.psi-num { display: block; font-size: 24px; font-weight: 700; font-family: var(--font-mono); }
.psi-item.emerald .psi-num { color: var(--accent-emerald); }
.psi-item.amber .psi-num { color: var(--accent-amber); }
.psi-item.rose .psi-num { color: var(--accent-rose); }
.psi-label { font-size: 11px; color: var(--text-muted); font-weight: 500; }

/* Expandable Lists */
.aux-expand-list, .step-expand-list, .source-expand-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.aux-expand-item, .step-expand-item, .source-expand-item {
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  overflow: hidden;
}

.aux-expand-header, .step-expand-header, .source-expand-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  cursor: pointer;
  transition: background 0.2s;
  gap: 12px;
}

.aux-expand-header:hover, .step-expand-header:hover, .source-expand-header:hover {
  background: rgba(255, 255, 255, 0.03);
}

.aux-expand-left, .step-expand-left, .source-expand-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
  min-width: 0;
}

.aux-expand-name, .source-expand-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  flex-shrink: 0;
}

.aux-expand-shape, .source-expand-count {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.08);
  padding: 2px 8px;
  border-radius: 4px;
  flex-shrink: 0;
}

.aux-expand-right, .source-expand-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.aux-expand-purpose, .source-expand-desc {
  font-size: 11px;
  color: var(--text-muted);
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.aux-chevron {
  color: var(--text-muted);
  transition: transform 0.2s;
  flex-shrink: 0;
}

.aux-chevron.open {
  transform: rotate(180deg);
}

.aux-expand-body, .step-expand-body, .source-expand-body {
  padding: 12px 14px;
  border-top: 1px solid var(--border-subtle);
  background: rgba(255, 255, 255, 0.01);
}

.aux-key-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.aux-key-label, .aux-col-label, .aux-sample-label, .hyper-label {
  font-size: 11px;
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  margin-bottom: 6px;
}

.aux-key-val {
  font-size: 12px;
  color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.08);
  padding: 2px 8px;
  border-radius: 4px;
}

.aux-columns {
  margin-bottom: 10px;
}

.aux-col-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
}

.aux-col-tag {
  font-size: 10px;
  padding: 2px 6px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 3px;
  color: var(--text-secondary);
}

.aux-sample-table-wrap, .sample-table-wrap {
  overflow-x: auto;
  margin-top: 6px;
}

.aux-sample-table, .sample-table {
  font-size: 11px;
  border-collapse: collapse;
  width: 100%;
}

.aux-sample-table th, .sample-table th {
  padding: 4px 8px;
  text-align: left;
  font-weight: 600;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
  font-size: 10px;
  white-space: nowrap;
}

.aux-sample-table td, .sample-table td {
  padding: 4px 8px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.02);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dtype-row td {
  color: var(--text-muted) !important;
  font-style: italic;
  font-size: 9px !important;
}

.val-null {
  color: var(--accent-rose) !important;
  opacity: 0.6;
  font-style: italic;
}

.sample-dtypes {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border-subtle);
}

/* Target column highlight */
.col-target {
  background: rgba(244, 63, 94, 0.15) !important;
  color: var(--accent-rose) !important;
}

.val-default {
  background: rgba(244, 63, 94, 0.12) !important;
  color: var(--accent-rose) !important;
  font-weight: 700 !important;
}

/* Missing overview layout */
.missing-overview {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

.missing-overview-left {
  flex: 1;
  min-width: 0;
}

.missing-overview-right {
  flex-shrink: 0;
  width: 160px;
}

/* Auxiliary table size badge */
.aux-expand-size {
  font-size: 10px;
  color: var(--accent-amber);
  background: rgba(245, 158, 11, 0.08);
  padding: 1px 6px;
  border-radius: 3px;
}

.dtype-chip {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
  font-size: 10px;
}

.dtype-col {
  color: var(--text-secondary);
  font-family: var(--font-mono);
}

.dtype-type {
  color: var(--text-muted);
  font-style: italic;
}

/* Step Example Grid */
/* Step slide transition */
.step-slide-enter-active {
  animation: stepSlideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.step-slide-leave-active {
  animation: stepSlideIn 0.2s ease reverse;
}
@keyframes stepSlideIn {
  from { opacity: 0; max-height: 0; transform: translateY(-8px); }
  to { opacity: 1; max-height: 600px; transform: translateY(0); }
}

/* Transform Viz (Step 1: Business Rule) */
.transform-viz {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
}

.transform-cell {
  flex: 1;
  text-align: center;
}

.transform-label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  font-weight: 600;
  letter-spacing: 0.3px;
  margin-bottom: 6px;
}

.transform-value {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  animation: transformPulse 0.6s ease 0.3s both;
}

.transform-value.before {
  background: rgba(244, 63, 94, 0.08);
  border: 1px solid rgba(244, 63, 94, 0.15);
  color: var(--accent-rose);
}

.transform-value.after {
  background: rgba(16, 185, 129, 0.08);
  border: 1px solid rgba(16, 185, 129, 0.15);
  color: var(--accent-emerald);
}

.strikethrough {
  text-decoration: line-through;
  text-decoration-color: rgba(244, 63, 94, 0.5);
}

.transform-arrow {
  flex-shrink: 0;
  color: var(--accent-blue);
  animation: arrowBounce 1s ease 0.4s infinite;
}

.transform-meta {
  text-align: center;
  margin-top: 8px;
}

.meta-affected {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.03);
  padding: 2px 8px;
  border-radius: 4px;
}

.new-flag {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  padding: 8px 12px;
  background: rgba(59, 130, 246, 0.06);
  border-left: 3px solid var(--accent-blue);
  border-radius: 0 6px 6px 0;
  animation: flagAppear 0.5s ease 0.6s both;
}

.flag-badge {
  font-size: 9px;
  font-weight: 700;
  color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.15);
  padding: 2px 6px;
  border-radius: 3px;
  text-transform: uppercase;
}

.flag-text {
  font-size: 11px;
  color: var(--text-secondary);
}

@keyframes transformPulse {
  0% { opacity: 0; transform: scale(0.9); }
  50% { transform: scale(1.03); }
  100% { opacity: 1; transform: scale(1); }
}

@keyframes arrowBounce {
  0%, 100% { transform: translateX(0); }
  50% { transform: translateX(3px); }
}

@keyframes flagAppear {
  from { opacity: 0; transform: translateX(-12px); }
  to { opacity: 1; transform: translateX(0); }
}

/* Missing Viz (Step 2) */
.missing-viz {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.missing-counter {
  display: flex;
  align-items: center;
  gap: 16px;
  justify-content: center;
}

.counter-block {
  text-align: center;
  padding: 12px 20px;
  border-radius: 10px;
}

.counter-block.before {
  background: rgba(244, 63, 94, 0.08);
  border: 1px solid rgba(244, 63, 94, 0.15);
  animation: transformPulse 0.5s ease 0.2s both;
}

.counter-block.after {
  background: rgba(16, 185, 129, 0.08);
  border: 1px solid rgba(16, 185, 129, 0.15);
  animation: transformPulse 0.5s ease 0.5s both;
}

.counter-num {
  font-size: 28px;
  font-weight: 700;
  color: var(--accent-rose);
  display: block;
  line-height: 1;
  animation: countUp 0.8s ease 0.3s both;
}

.counter-num.emerald { color: var(--accent-emerald); }

.counter-label {
  font-size: 10px;
  color: var(--text-muted);
  display: block;
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.counter-arrow {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--accent-emerald);
}

.arrow-line {
  width: 24px;
  height: 2px;
  background: var(--accent-emerald);
  animation: lineGrow 0.4s ease 0.3s both;
}

.missing-progress {
  padding: 10px 14px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
}

.progress-labels {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
}

.progress-label-before, .progress-label-after {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text-muted);
}

.progress-track {
  height: 8px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-emerald), #34d399);
  border-radius: 4px;
  animation: progressGrow 1s cubic-bezier(0.16, 1, 0.3, 1) 0.5s both;
  transform-origin: left;
}

.progress-subtitle {
  font-size: 10px;
  color: var(--accent-emerald);
  font-family: var(--font-mono);
  text-align: center;
  margin-top: 4px;
}

.missing-methods {
  display: flex;
  gap: 8px;
}

.method-chip {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
  font-size: 11px;
  color: var(--text-secondary);
}

.method-icon {
  width: 20px;
  height: 20px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 700;
  flex-shrink: 0;
}

.method-chip.numeric .method-icon {
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent-blue);
}

.method-chip.categorical .method-icon {
  background: rgba(139, 92, 246, 0.15);
  color: var(--accent-violet);
}

.missing-sample {
  padding: 8px 12px;
  background: rgba(59, 130, 246, 0.04);
  border-radius: 6px;
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.sample-arrow { color: var(--text-muted); font-weight: 600; }
.sample-col { color: var(--accent-blue); }
.sample-fill { color: var(--text-secondary); }

@keyframes countUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes lineGrow {
  from { width: 0; }
  to { width: 24px; }
}

@keyframes progressGrow {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

/* Encoding Viz (Step 3) */
.encoding-viz {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.encode-section {
  padding: 12px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
}

.encode-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.encode-badge {
  font-size: 9px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 3px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.encode-badge.onehot {
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent-blue);
}

.encode-badge.freq {
  background: rgba(139, 92, 246, 0.15);
  color: var(--accent-violet);
}

.encode-label {
  font-size: 11px;
  color: var(--text-muted);
}

.encode-transform {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.encode-cell {
  padding: 8px 12px;
  border-radius: 6px;
  animation: transformPulse 0.5s ease 0.2s both;
}

.encode-cell.before {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  flex: 1;
}

.encode-cell.after {
  background: rgba(59, 130, 246, 0.06);
  border: 1px solid rgba(59, 130, 246, 0.12);
  flex: 1;
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}

.encode-col {
  font-size: 11px;
  color: var(--text-muted);
  display: block;
  margin-bottom: 2px;
}

.encode-val {
  font-size: 12px;
  color: var(--text-primary);
}

.encode-bit {
  font-size: 11px;
  font-family: var(--font-mono);
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 3px;
}

.encode-bit.on {
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent-blue);
}

.encode-bit.off {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-muted);
}

.encode-arrow-cell {
  flex-shrink: 0;
  animation: arrowBounce 1s ease 0.4s infinite;
}

.encode-num {
  font-size: 16px;
  font-weight: 700;
  color: var(--accent-violet);
}

.encode-cols {
  font-size: 10px;
  color: var(--text-muted);
}

/* Outlier Viz (Step 4) */
.outlier-viz {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.outlier-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.outlier-badge {
  font-size: 9px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 3px;
  background: rgba(245, 158, 11, 0.15);
  color: var(--accent-amber);
  text-transform: uppercase;
}

.outlier-scope {
  font-size: 11px;
  color: var(--text-muted);
}

.clamp-visual {
  padding: 14px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
}

.clamp-axis {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.clamp-label {
  font-size: 9px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  white-space: nowrap;
  flex-shrink: 0;
  max-width: 80px;
  text-align: center;
}

.clamp-range {
  flex: 1;
  height: 24px;
  position: relative;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
  overflow: visible;
}

.clamp-bar {
  position: absolute;
  left: 20%;
  right: 25%;
  top: 0;
  bottom: 0;
  background: linear-gradient(90deg, rgba(59, 130, 246, 0.15), rgba(16, 185, 129, 0.15));
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.clamp-marker {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.clamp-marker.outlier {
  animation: markerPulse 1.5s ease infinite;
}

.clamp-marker.outlier::before {
  content: '';
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--accent-rose);
  box-shadow: 0 0 8px rgba(244, 63, 94, 0.4);
}

.clamp-marker.cutoff::before {
  content: '';
  width: 2px;
  height: 20px;
  background: var(--accent-amber);
  border-radius: 1px;
}

.clamp-marker-val {
  font-size: 9px;
  font-family: var(--font-mono);
  color: var(--accent-rose);
  white-space: nowrap;
  position: absolute;
  top: -16px;
}

.clamp-marker-label {
  font-size: 9px;
  font-family: var(--font-mono);
  color: var(--accent-amber);
  white-space: nowrap;
  position: absolute;
  top: -16px;
}

.clamp-note {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.clamp-col {
  font-size: 12px;
  color: var(--text-primary);
  font-weight: 600;
}

.clamp-from { font-size: 11px; }
.clamp-arrow-icon { color: var(--text-muted); }
.clamp-to { font-size: 11px; }

@keyframes markerPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Step example grid (kept for compatibility) */
.step-example-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.step-example-grid .full-width {
  grid-column: 1 / -1;
}

.example-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
}

.example-key {
  font-size: 10px;
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.example-val {
  font-size: 12px;
  color: var(--text-primary);
}

.example-val.rose { color: var(--accent-rose); }
.example-val.emerald { color: var(--accent-emerald); }
.example-val.highlight { color: var(--accent-blue); }

/* Formula List */
.formula-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.formula-item {
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
  border-left: 3px solid var(--accent-blue);
}

.formula-name {
  font-size: 13px;
  font-weight: 700;
  color: var(--accent-blue);
  margin-bottom: 4px;
}

.formula-expr {
  font-size: 12px;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.formula-eq {
  color: var(--text-muted);
  margin-right: 4px;
}

.formula-example {
  font-size: 11px;
  color: var(--text-secondary);
  margin-bottom: 2px;
}

.formula-label {
  color: var(--text-muted);
  font-weight: 600;
}

.formula-purpose {
  font-size: 11px;
  color: var(--text-muted);
  font-style: italic;
}

/* Aggregation List */
.agg-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.agg-source {
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.agg-item {
  padding: 8px 10px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
}

.agg-col {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.agg-funcs {
  display: flex;
  gap: 4px;
  margin-bottom: 4px;
}

.agg-func-tag {
  font-size: 10px;
  padding: 1px 6px;
  background: rgba(139, 92, 246, 0.1);
  color: var(--accent-violet);
  border-radius: 3px;
  font-family: var(--font-mono);
}

.agg-example {
  font-size: 11px;
  color: var(--text-muted);
}

/* Feature Tags */
.feat-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.feat-tag {
  font-size: 10px;
  padding: 3px 8px;
  background: rgba(59, 130, 246, 0.08);
  color: var(--accent-blue);
  border-radius: 4px;
}

/* Correlation heatmap note */
.corr-note {
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 10px;
  padding: 6px 10px;
  background: rgba(139, 92, 246, 0.06);
  border-left: 2px solid var(--accent-violet);
  border-radius: 4px;
}

/* Data Split Visual */
.split-visual {
  margin-top: 8px;
}

.split-bar {
  display: flex;
  height: 32px;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 8px;
}

.split-segment {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 600;
  color: #fff;
}

.split-segment.train { background: #3b82f6; }
.split-segment.valid { background: #8b5cf6; }
.split-segment.calib { background: #06b6d4; }

.split-note {
  font-size: 11px;
  color: var(--text-muted);
  text-align: center;
}

/* Model Detail Cards */
.model-detail-cards {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-detail-card {
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  overflow: hidden;
}

.model-detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-subtle);
  background: rgba(255, 255, 255, 0.02);
}

.model-detail-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.model-detail-type {
  font-size: 11px;
  color: var(--text-muted);
}

.model-detail-body {
  padding: 14px 16px;
}

.info-grid.compact {
  gap: 6px;
}

.hyperparams, .train-flow {
  margin-top: 12px;
}

.hyper-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
  margin-top: 6px;
}

.hyper-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 8px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 4px;
}

.hyper-key {
  font-size: 11px;
  color: var(--text-muted);
}

.hyper-val {
  font-size: 11px;
  color: var(--text-primary);
  font-weight: 600;
}

.flow-steps {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
}

.flow-step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
}

.flow-step-num {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  font-size: 11px;
  font-weight: 700;
  font-family: var(--font-mono);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.flow-step-text {
  font-size: 12px;
  color: var(--text-secondary);
}

/* Metrics Explanation */
.metrics-explain-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metrics-explain-item {
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
}

.metrics-explain-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 4px;
}

.metrics-explain-name {
  font-size: 14px;
  font-weight: 700;
  color: var(--accent-blue);
}

.metrics-explain-full {
  font-size: 12px;
  color: var(--text-primary);
  flex: 1;
}

.metrics-explain-range {
  font-size: 11px;
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.04);
  padding: 2px 8px;
  border-radius: 4px;
}

.metrics-explain-meaning {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

/* Panel transition */
.panel-enter-active { animation: fadeInUp 0.3s ease; }
.panel-leave-active { animation: fadeIn 0.2s ease reverse; }

/* Charts Row */
.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 20px;
}

.charts-row-3 {
  grid-template-columns: 1fr 1fr 1fr;
}

.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  backdrop-filter: blur(20px);
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--text-primary);
}

.chart-badge {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--accent-gold);
  background: rgba(200, 170, 110, 0.08);
  padding: 2px 8px;
  border-radius: 20px;
  font-weight: 500;
  letter-spacing: 0.3px;
}

.dataset-info-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  padding: 10px 20px;
  margin-bottom: 20px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  backdrop-filter: blur(20px);
}

.dinfo-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 16px;
}

.dinfo-icon {
  color: var(--text-muted);
  display: flex;
  align-items: center;
}

.dinfo-label {
  color: var(--text-muted);
  font-size: 11px;
  font-family: var(--font-sans);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.dinfo-val {
  color: var(--text-primary);
  font-size: 13px;
  font-family: var(--font-mono);
  font-weight: 600;
}

.dinfo-sep {
  width: 1px;
  height: 20px;
  background: var(--border-subtle);
}

.dim-footer {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border-subtle);
  border-top: 1px solid var(--border-subtle);
}

.dim-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 12px 8px;
  background: var(--bg-card);
}

.dim-label { font-size: 11px; color: var(--text-muted); font-weight: 500; }
.dim-val { font-size: 16px; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono); }

/* Table Section */
.table-wrap { overflow-x: auto; }

.model-name-cell { display: flex; align-items: center; gap: 8px; }
.model-dot { width: 6px; height: 6px; border-radius: 50%; }
.model-dot.xgb { background: var(--accent-blue); }
.model-dot.lr { background: var(--accent-violet); }
.model-dot.lgb { background: var(--accent-emerald); }
.model-dot.stk { background: var(--accent-amber); }

.model-name { font-weight: 600; color: var(--text-primary); font-size: 12px; }

.best-badge {
  font-size: 9px;
  font-weight: 700;
  color: var(--accent-gold);
  background: rgba(200, 170, 110, 0.15);
  padding: 1px 6px;
  border-radius: 3px;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  animation: bestBadgePulse 2s ease-in-out infinite;
}

@keyframes bestBadgePulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.metric-cell {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--text-secondary);
}

.metric-cell.highlight { color: var(--accent-blue); font-weight: 600; }

.cm-cell {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 4px;
  display: inline-block;
}

.cm-cell.tp { background: rgba(59, 130, 246, 0.12); color: var(--accent-blue); }
.cm-cell.tn { background: rgba(16, 185, 129, 0.12); color: var(--accent-emerald); }
.cm-cell.fp { background: rgba(244, 63, 94, 0.12); color: var(--accent-rose); }
.cm-cell.fn { background: rgba(245, 158, 11, 0.12); color: var(--accent-amber); }

:deep(.el-table) {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
}

:deep(.el-table__row:hover > td) {
  background: rgba(59, 130, 246, 0.05) !important;
}

/* Skeleton Loading */
@keyframes skeleton-pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.8; }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.skeleton-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 20px;
  display: flex;
  align-items: flex-start;
  gap: 14px;
  animation: skeleton-pulse 2s ease-in-out infinite;
}

.skeleton-icon,
.skeleton-node,
.skeleton-body,
.skeleton-header {
  background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 75%);
  background-size: 200% 100%;
  animation: shimmer 2s ease-in-out infinite;
}

.skeleton-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.04);
  flex-shrink: 0;
}

.skeleton-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton-line {
  height: 12px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.04);
}

.skeleton-line.short { width: 60%; }
.skeleton-line.medium { width: 80%; height: 18px; }
.skeleton-line.tiny { width: 50%; height: 10px; }

.skeleton-section {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  margin-bottom: 20px;
  animation: skeleton-pulse 1.5s ease-in-out infinite;
}

.skeleton-header {
  height: 16px;
  width: 120px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.06);
  margin-bottom: 16px;
}

.skeleton-pipeline {
  display: flex;
  justify-content: center;
  gap: 40px;
  padding: 20px 0;
}

.skeleton-stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.skeleton-node {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
}

.skeleton-charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}

.skeleton-chart {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  animation: skeleton-pulse 1.5s ease-in-out infinite;
}

.skeleton-chart .skeleton-header {
  padding: 14px 18px;
  margin: 0;
  border-bottom: 1px solid var(--border-subtle);
}

.skeleton-body {
  height: 220px;
  background: rgba(255, 255, 255, 0.03);
}

.skeleton-table {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton-row {
  display: flex;
  gap: 12px;
}

.skeleton-cell {
  height: 14px;
  flex: 1;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.06);
}

.skeleton-cell.wide {
  flex: 1.5;
}

/* ── Model Governance Registry ── */
.registry-section {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 24px;
  margin-top: 24px;
}

.registry-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.registry-meta-divider {
  color: var(--border-subtle);
}

.registry-summary {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 20px;
}

.registry-summary-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  transition: border-color 0.2s;
}

.registry-summary-card:hover {
  border-color: rgba(200, 170, 110, 0.2);
}

.registry-summary-card.latest {
  border-color: rgba(200, 170, 110, 0.15);
  background: rgba(200, 170, 110, 0.03);
}

.rsc-label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 6px;
  font-weight: 500;
}

.rsc-value {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.rsc-value.gold {
  color: var(--accent-gold);
}

.rsc-sub {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 4px;
  font-family: var(--font-mono);
}

.mono-cell {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
}

.mono-cell.hash {
  color: var(--text-muted);
  font-size: 11px;
  letter-spacing: 0.3px;
}

/* ── Executive Summary / Risk Briefing ── */
.briefing {
  display: grid;
  grid-template-columns: 220px 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 20px 24px;
}

.briefing-left {
  display: flex;
  flex-direction: column;
  gap: 16px;
  border-right: 1px solid var(--border-subtle);
  padding-right: 20px;
}

.risk-level-indicator {
  width: 8px; height: 8px; border-radius: 50%;
  animation: breathe 3s ease infinite;
}
.risk-level-indicator.healthy { background: var(--accent-emerald); box-shadow: 0 0 8px rgba(16,185,129,0.4); }
.risk-level-indicator.critical { background: var(--accent-rose); box-shadow: 0 0 8px rgba(244,63,94,0.4); }

.briefing-risk-badge { display: flex; align-items: center; gap: 10px; }
.risk-level-text { display: flex; flex-direction: column; }
.risk-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
.risk-status { font-size: 16px; font-weight: 700; font-family: var(--font-display); }
.risk-status.healthy { color: var(--accent-emerald); }
.risk-status.critical { color: var(--accent-rose); }

.briefing-default { display: flex; flex-direction: column; }
.briefing-default-rate { font-size: 32px; font-weight: 700; color: var(--accent-rose); font-family: var(--font-mono); line-height: 1; }
.briefing-default-label { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.briefing-default-sub { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); margin-top: 2px; }

.briefing-center {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding-right: 20px;
  border-right: 1px solid var(--border-subtle);
}

.briefing-model {}
.bm-header { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
.bm-icon { color: var(--accent-gold); }
.bm-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
.bm-name { font-size: 16px; font-weight: 600; color: var(--text-primary); font-family: var(--font-display); margin-bottom: 8px; }
.bm-metrics { display: flex; align-items: center; gap: 12px; }
.bm-metric { display: flex; flex-direction: column; }
.bm-metric-val { font-size: 18px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); }
.bm-metric-val.gold { color: var(--accent-gold); }
.bm-metric-val.rose { color: var(--accent-rose); }
.bm-metric-val.emerald { color: var(--accent-emerald); }
.bm-metric-key { font-size: 10px; color: var(--text-muted); text-transform: uppercase; }
.bm-sep { width: 1px; height: 24px; background: var(--border-subtle); }

.briefing-monitoring { display: flex; align-items: center; gap: 10px; }

.briefing-right { display: flex; flex-direction: column; }
.bs-header { margin-bottom: 10px; }

.strategy-cards { display: flex; gap: 10px; }
.strat-card {
  flex: 1;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  transition: border-color 0.2s;
}
.strat-card:hover { border-color: rgba(200,170,110,0.2); }
.strat-card.conservative { border-left: 3px solid var(--accent-emerald); }
.strat-card.balanced { border-left: 3px solid var(--accent-gold); }
.strat-card.growth { border-left: 3px solid var(--accent-blue); }

.strat-name { font-size: 12px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }
.strat-metrics { display: flex; flex-direction: column; gap: 3px; }
.strat-metrics > span { font-size: 10px; color: var(--text-muted); display: flex; justify-content: space-between; }
.strat-val { font-family: var(--font-mono); font-weight: 600; color: var(--text-secondary); }

@keyframes breathe {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
</style>
