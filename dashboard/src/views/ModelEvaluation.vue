<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart, RadarChart, HeatmapChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, RadarComponent, VisualMapComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import api from '../api'
import { chartAnim, tooltipStyle } from '../echartsTheme'

use([CanvasRenderer, LineChart, BarChart, RadarChart, HeatmapChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, RadarComponent, VisualMapComponent])

const modelType = ref('xgboost')
const metrics = ref(null)
const rocData = ref(null)
const ksData = ref(null)
const giniData = ref(null)
const pdDistData = ref(null)
const allModelsData = ref(null)
const loading = ref(true)

async function fetchData() {
  loading.value = true
  try {
    const [m, r, k, g, pd, overview] = await Promise.all([
      api.get(`/api/model/${modelType.value}/metrics`),
      api.get(`/api/model/${modelType.value}/roc`),
      api.get(`/api/model/${modelType.value}/ks`),
      api.get(`/api/model/${modelType.value}/gini`),
      api.get(`/api/model/${modelType.value}/pd_distribution`),
      api.get('/api/overview'),
    ])
    metrics.value = m.data
    rocData.value = r.data
    ksData.value = k.data
    giniData.value = g.data
    pdDistData.value = pd.data
    allModelsData.value = overview.data?.models || []
  } finally {
    loading.value = false
  }
}

onMounted(fetchData)
watch(modelType, fetchData)

const chartTheme = {
  ...chartAnim,
  textStyle: { fontFamily: 'Outfit, sans-serif', color: '#94a3b8' },
}

function isBest(model, col) {
  if (!allModelsData.value?.length) return false
  const vals = allModelsData.value.map(m => m[col]).filter(v => typeof v === 'number')
  if (!vals.length) return false
  const best = Math.max(...vals)
  return model[col] === best
}

const rocOption = ref({})
watch(rocData, (d) => {
  if (!d) return
  rocOption.value = {
    ...chartTheme,
    tooltip: { trigger: 'axis', ...tooltipStyle },
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'value', name: 'FPR', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: { type: 'value', name: 'TPR', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    series: [
      { type: 'line', data: d.fpr.map((x, i) => [x, d.tpr[i]]), smooth: true, lineStyle: { width: 2.5, color: '#3b82f6' }, areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(59,130,246,0.2)' }, { offset: 1, color: 'rgba(59,130,246,0)' }] } }, name: `AUC=${d.auc}`, showSymbol: false, itemStyle: { color: '#3b82f6' } },
      { type: 'line', data: [[0,0],[1,1]], lineStyle: { type: 'dashed', color: 'rgba(255,255,255,0.12)', width: 1 }, showSymbol: false, name: 'Random' },
    ],
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 12 } },
  }
})

const ksOption = ref({})
watch(ksData, (d) => {
  if (!d) return
  ksOption.value = {
    ...chartTheme,
    tooltip: { trigger: 'axis', ...tooltipStyle },
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'value', name: 'FPR', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: { type: 'value', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    series: [
      { type: 'line', data: d.fpr.map((x, i) => [x, d.tpr[i]]), name: 'TPR', showSymbol: false, lineStyle: { color: '#3b82f6', width: 2 }, itemStyle: { color: '#3b82f6' } },
      { type: 'line', data: d.fpr.map((x, i) => [x, 1 - d.fpr[i]]), name: '1-FPR', showSymbol: false, lineStyle: { color: '#94a3b8', width: 1.5, type: 'dashed' }, itemStyle: { color: '#94a3b8' } },
      { type: 'line', data: d.fpr.map((x, i) => [x, d.tpr[i] - d.fpr[i]]),
        name: `KS=${d.ks}`, lineStyle: { type: 'dashed', color: '#f59e0b', width: 2 }, showSymbol: false, itemStyle: { color: '#f59e0b' } },
    ],
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 12 } },
  }
})

const giniOption = ref({})
watch(giniData, (d) => {
  if (!d) return
  giniOption.value = {
    ...chartTheme,
    tooltip: { trigger: 'axis', ...tooltipStyle },
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'value', name: '累积客户占比', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: { type: 'value', name: '累积违约占比', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    series: [
      { type: 'line', data: d.x.map((x, i) => [x, d.y[i]]), smooth: true, showSymbol: false, name: `Gini=${d.gini}`, lineStyle: { width: 2.5, color: '#8b5cf6' }, areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(139,92,246,0.2)' }, { offset: 1, color: 'rgba(139,92,246,0)' }] } }, itemStyle: { color: '#8b5cf6' } },
      { type: 'line', data: [[0,0],[1,1]], lineStyle: { type: 'dashed', color: 'rgba(255,255,255,0.12)', width: 1 }, showSymbol: false, name: 'Random' },
    ],
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 12 } },
  }
})

const pdDistOption = ref({})
watch(pdDistData, (d) => {
  if (!d) return
  pdDistOption.value = {
    ...chartAnim,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      ...tooltipStyle,
      formatter: (params) => {
        const bin = params[0]?.axisValue
        let html = `<b>PD: ${bin}</b><br/>`
        params.forEach(p => { html += `${p.seriesName}: ${p.value.toLocaleString()}<br/>` })
        return html
      },
    },
    grid: { left: 60, right: 20, top: 30, bottom: 40 },
    xAxis: {
      type: 'category',
      data: d.bins,
      axisLabel: { color: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono', interval: 4 },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' },
    },
    series: [
      {
        name: 'Good',
        type: 'bar',
        stack: 'pd',
        data: d.good,
        itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#10b981' }, { offset: 1, color: '#059669' }] }, borderRadius: [0, 0, 0, 0] },
      },
      {
        name: 'Bad',
        type: 'bar',
        stack: 'pd',
        data: d.bad,
        itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#f43f5e' }, { offset: 1, color: '#e11d48' }] }, borderRadius: [3, 3, 0, 0] },
        markLine: {
          silent: true,
          symbol: 'none',
          data: [
            { xAxis: d.mean_pd, lineStyle: { color: '#3b82f6', type: 'dashed', width: 2 }, label: { formatter: `Mean: ${d.mean_pd}`, color: '#3b82f6', fontFamily: 'JetBrains Mono', fontSize: 11 } },
          ],
        },
      },
    ],
    legend: {
      bottom: 0,
      textStyle: { color: '#94a3b8', fontSize: 12 },
      data: ['Good', 'Bad'],
    },
  }
})

const compOption = ref({})
watch(allModelsData, (models) => {
  if (!models?.length) return
  const metrics = ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall']
  const colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b']
  compOption.value = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    grid: { left: 60, right: 20, top: 30, bottom: 50 },
    xAxis: { type: 'category', data: metrics, axisLabel: { color: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } } },
    yAxis: { type: 'value', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' } },
    series: models.map((m, i) => ({
      name: m.Model,
      type: 'bar',
      data: metrics.map(k => m[k] || 0),
      itemStyle: { color: colors[i % colors.length], borderRadius: [4, 4, 0, 0] },
      barMaxWidth: 24,
    })),
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 12 } },
  }
})

const radarOption = ref({})
watch(metrics, (m) => {
  if (!m) return
  const keys = ['auc', 'ks', 'gini', 'accuracy', 'precision', 'recall']
  const labels = ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall']
  const colorMap = { logistic: '#8b5cf6', xgboost: '#3b82f6', lightgbm: '#10b981', stacking: '#f59e0b' }
  const color = colorMap[modelType.value] || '#3b82f6'
  radarOption.value = {
    tooltip: { ...tooltipStyle },
    radar: {
      indicator: labels.map((l, i) => ({ name: l, max: 1 })),
      shape: 'polygon',
      axisName: { color: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono' },
      splitArea: { areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.04)'] } },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
    },
    series: [{
      type: 'radar',
      data: [{
        value: keys.map(k => m[k] || 0),
        name: m.model_type || modelType.value,
        lineStyle: { color, width: 2 },
        itemStyle: { color },
        areaStyle: { color, opacity: 0.15 },
      }],
    }],
  }
})

const multiRadarOption = ref({})
watch(allModelsData, (models) => {
  if (!models?.length) return
  const metricKeys = ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall']
  const colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b']
  multiRadarOption.value = {
    tooltip: { ...tooltipStyle },
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 12 } },
    radar: {
      indicator: metricKeys.map(k => ({ name: k, max: 1 })),
      shape: 'polygon',
      axisName: { color: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono' },
      splitArea: { areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.04)'] } },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
    },
    series: [{
      type: 'radar',
      data: models.map((m, i) => {
        const isBest = m.Model === models.reduce((b, x) => (x.AUC > (b?.AUC || 0)) ? x : b, null)?.Model
        return {
          name: m.Model,
          value: metricKeys.map(k => m[k] || 0),
          lineStyle: { color: colors[i % colors.length], width: isBest ? 3 : 1.5, type: isBest ? 'solid' : 'dashed' },
          itemStyle: { color: colors[i % colors.length] },
          areaStyle: { color: colors[i % colors.length], opacity: isBest ? 0.15 : 0.05 },
        }
      }),
    }],
  }
})

const cmHeatmapOption = computed(() => {
  if (!metrics.value?.confusion_matrix) return null
  const cm = metrics.value.confusion_matrix
  const data = [
    [0, 0, cm.tn], [1, 0, cm.fp],
    [0, 1, cm.fn], [1, 1, cm.tp],
  ]
  const maxVal = Math.max(cm.tn, cm.fp, cm.fn, cm.tp)
  return {
    tooltip: {
      formatter: (p) => {
        const labels = [['TN (实际正常, 预测正常)', 'FP (实际正常, 预测违约)'], ['FN (实际违约, 预测正常)', 'TP (实际违约, 预测违约)']]
        const pct = p.data[2] / (cm.tn + cm.fp + cm.fn + cm.tp) * 100
        return `<span style="font-family:Outfit">${labels[p.data[1]][p.data[0]]}</span><br/><span style="font-family:JetBrains Mono">${p.data[2].toLocaleString()} (${pct.toFixed(1)}%)</span>`
      },
      ...tooltipStyle,
    },
    grid: { left: 80, right: 40, top: 30, bottom: 40 },
    xAxis: { type: 'category', data: ['预测正常', '预测违约'], splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.04)'] } }, axisLabel: { color: '#94a3b8', fontSize: 12, fontFamily: 'Outfit' }, axisLine: { show: false }, axisTick: { show: false } },
    yAxis: { type: 'category', data: ['实际正常', '实际违约'], splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.04)'] } }, axisLabel: { color: '#94a3b8', fontSize: 12, fontFamily: 'Outfit' }, axisLine: { show: false }, axisTick: { show: false } },
    visualMap: { show: false, min: 0, max: maxVal, inRange: { color: ['#1e293b', '#3b82f6'] } },
    series: [{
      type: 'heatmap',
      data: data,
      label: {
        show: true,
        formatter: (p) => `${p.data[2].toLocaleString()}\n(${(p.data[2] / (cm.tn + cm.fp + cm.fn + cm.tp) * 100).toFixed(1)}%)`,
        color: '#f1f5f9',
        fontSize: 13,
        fontFamily: 'JetBrains Mono',
        lineHeight: 18,
      },
      itemStyle: {
        borderColor: 'var(--bg-primary)',
        borderWidth: 3,
        borderRadius: 6,
      },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } },
    }],
  }
})

const metricCards = [
  { key: 'AUC', color: 'blue' },
  { key: 'KS', color: 'cyan' },
  { key: 'Gini', color: 'violet' },
  { key: 'Accuracy', color: 'emerald' },
  { key: 'Precision', color: 'amber' },
  { key: 'Recall', color: 'rose' },
]
</script>

<template>
  <div v-loading="loading" element-loading-background="rgba(10, 14, 26, 0.8)">
    <div class="page-header animate-in">
      <div class="header-row">
        <div>
          <h2>模型评估</h2>
          <p class="page-desc">Model Performance Analysis</p>
        </div>
        <div class="header-actions">
          <button class="refresh-btn" @click="fetchData" :disabled="loading">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" :class="{ spinning: loading }"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
          </button>
          <div class="model-switch">
            <button :class="['switch-btn', { active: modelType === 'logistic' }]" @click="modelType = 'logistic'">Logistic</button>
            <button :class="['switch-btn', { active: modelType === 'xgboost' }]" @click="modelType = 'xgboost'">XGBoost</button>
            <button :class="['switch-btn', { active: modelType === 'lightgbm' }]" @click="modelType = 'lightgbm'">LightGBM</button>
            <button :class="['switch-btn best', { active: modelType === 'stacking' }]" @click="modelType = 'stacking'">Stacking <span class="best-tag">Best</span></button>
          </div>
        </div>
      </div>
    </div>

    <div class="metrics-row" v-if="metrics">
      <div v-for="(item, idx) in metricCards" :key="item.key" :class="['metric-card', item.color, 'animate-in']" :style="{ animationDelay: (idx * 0.05 + 0.1) + 's' }">
        <div class="metric-label">{{ item.key }}</div>
        <div class="metric-value">{{ metrics[item.key.toLowerCase()] || metrics[item.key] }}</div>
      </div>
    </div>

    <!-- Radar Charts -->
    <div class="charts-row" style="grid-template-columns: 1fr 1fr" v-if="metrics && allModelsData?.length > 1">
      <div class="chart-card animate-in animate-in-delay-3">
        <div class="chart-header">
          <span>Current Model Radar</span>
          <span class="chart-badge">{{ modelType }}</span>
        </div>
        <v-chart :option="radarOption" style="height: 280px" autoresize />
      </div>
      <div class="chart-card animate-in animate-in-delay-3">
        <div class="chart-header">
          <span>All Models Overlay</span>
          <span class="chart-badge">{{ allModelsData.length }} Models</span>
        </div>
        <v-chart :option="multiRadarOption" style="height: 280px" autoresize />
      </div>
    </div>

    <!-- Single Radar (when only 1 model) -->
    <div class="chart-full animate-in animate-in-delay-3" v-if="metrics && allModelsData?.length <= 1">
      <div class="chart-header">
        <span>Performance Radar</span>
        <span class="chart-badge">{{ modelType }}</span>
      </div>
      <v-chart :option="radarOption" style="height: 280px" autoresize />
    </div>

    <div class="charts-row animate-in animate-in-delay-4">
      <div class="chart-card">
        <div class="chart-header">ROC Curve</div>
        <v-chart :option="rocOption" style="height: 300px" autoresize />
      </div>
      <div class="chart-card">
        <div class="chart-header">KS Curve</div>
        <v-chart :option="ksOption" style="height: 300px" autoresize />
      </div>
      <div class="chart-card">
        <div class="chart-header">Gini Curve</div>
        <v-chart :option="giniOption" style="height: 300px" autoresize />
      </div>
    </div>

    <!-- PD Distribution -->
    <div class="chart-full animate-in animate-in-delay-5" v-if="pdDistData">
      <div class="chart-header">
        <span>PD Distribution (Good vs Bad)</span>
        <div class="pd-stats">
          <span class="pd-stat">Mean: <b>{{ pdDistData.mean_pd }}</b></span>
          <span class="pd-stat">Median: <b>{{ pdDistData.median_pd }}</b></span>
          <span class="pd-stat">Total: <b>{{ pdDistData.total?.toLocaleString() }}</b></span>
        </div>
      </div>
      <v-chart :option="pdDistOption" style="height: 260px" autoresize />
    </div>

    <!-- Model Comparison -->
    <div class="chart-full animate-in animate-in-delay-5" v-if="allModelsData?.length > 1">
      <div class="chart-header">
        <span>All Models Comparison</span>
        <span class="chart-badge">{{ allModelsData.length }} Models</span>
      </div>
      <v-chart :option="compOption" style="height: 280px" autoresize />
    </div>

    <!-- Model Comparison Table -->
    <div class="table-section animate-in animate-in-delay-5" v-if="allModelsData?.length > 1">
      <div class="section-header">
        <h3>Model Metrics Comparison</h3>
        <span class="chart-badge">{{ allModelsData.length }} Models</span>
      </div>
      <div class="table-wrap">
        <table class="comp-table">
          <thead>
            <tr>
              <th>Model</th>
              <th v-for="col in ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall', 'F1-Score']" :key="col">{{ col }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in allModelsData" :key="m.Model">
              <td class="model-name">
                <span class="model-dot" :class="m.Model?.includes('XG') ? 'xgb' : m.Model?.includes('Light') ? 'lgb' : m.Model?.includes('Stack') ? 'stk' : 'lr'"></span>
                {{ m.Model }}
              </td>
              <td v-for="col in ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall', 'F1-Score']" :key="col"
                :class="{ 'best-val': isBest(m, col) }">
                {{ typeof m[col] === 'number' ? m[col].toFixed(4) : '-' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="chart-full animate-in animate-in-delay-5" v-if="cmHeatmapOption">
      <div class="chart-header">
        <span>Confusion Matrix</span>
        <span class="chart-badge">Heatmap</span>
      </div>
      <v-chart :option="cmHeatmapOption" style="height: 280px" autoresize />
    </div>
  </div>
</template>

<style scoped>
.page-header { margin-bottom: 28px; }
.header-row { display: flex; align-items: flex-start; justify-content: space-between; }
.page-header h2 { margin: 0; font-family: var(--font-display); font-size: 28px; }
.page-desc { color: var(--text-muted); font-size: 12px; margin-top: 6px; font-family: var(--font-mono); letter-spacing: 0.5px; }

.model-switch {
  display: flex;
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 3px;
  gap: 2px;
}

.switch-btn {
  padding: 6px 16px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 500;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  letter-spacing: 0.3px;
}

.switch-btn.active {
  background: var(--accent-gold);
  color: #0a0b10;
}

.switch-btn:hover:not(.active) {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.04);
}

.switch-btn.best {
  position: relative;
}

.best-tag {
  font-size: 8px;
  font-weight: 700;
  color: var(--accent-gold);
  background: rgba(200, 170, 110, 0.2);
  padding: 1px 4px;
  border-radius: 2px;
  margin-left: 4px;
  vertical-align: middle;
  letter-spacing: 0.3px;
}

.switch-btn.best.active .best-tag {
  color: #0a0b10;
  background: rgba(0, 0, 0, 0.15);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.refresh-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: 1px solid var(--border-subtle);
  background: var(--bg-secondary);
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s ease;
}

.refresh-btn:hover:not(:disabled) {
  color: var(--accent-blue);
  border-color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.08);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.refresh-btn .spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.metrics-row {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(20px);
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
}

.metric-card.blue::before { background: var(--gradient-blue); }
.metric-card.cyan::before { background: linear-gradient(135deg, #06b6d4, #22d3ee); }
.metric-card.violet::before { background: var(--gradient-violet); }
.metric-card.emerald::before { background: var(--gradient-emerald); }
.metric-card.amber::before { background: var(--gradient-amber); }
.metric-card.rose::before { background: var(--gradient-rose); }

.metric-card:hover { border-color: var(--border-accent); transform: translateY(-1px); }

.metric-label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 8px;
  font-weight: 600;
}

.metric-value {
  font-size: 26px;
  font-weight: 700;
  color: var(--text-primary);
  font-family: var(--font-display);
  letter-spacing: -0.5px;
}

.charts-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 16px;
}

.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  backdrop-filter: blur(12px);
}

.chart-header {
  padding: 14px 18px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  border-bottom: 1px solid var(--border-subtle);
}

.chart-full {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: 20px;
}

.chart-full .chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.pd-stats {
  display: flex;
  gap: 16px;
}

.pd-stat {
  font-size: 12px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.pd-stat b {
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
.table-section {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: 20px;
}

.table-wrap { overflow-x: auto; }

.comp-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.comp-table th {
  padding: 10px 14px;
  text-align: left;
  font-size: 11px;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  background: rgba(255, 255, 255, 0.03);
  border-bottom: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
}

.comp-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
  color: var(--text-secondary);
}

.comp-table tr:hover td {
  background: rgba(59, 130, 246, 0.05);
}

.comp-table .model-name {
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 8px;
}

.comp-table .best-val {
  color: var(--accent-blue);
  font-weight: 700;
}

.model-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.model-dot.xgb { background: var(--accent-blue); }
.model-dot.lr { background: var(--accent-violet); }
.model-dot.lgb { background: var(--accent-emerald); }
.model-dot.stk { background: var(--accent-amber); }
</style>
