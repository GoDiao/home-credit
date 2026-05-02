<script setup>
import { ref, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, ScatterChart, HeatmapChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, VisualMapComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import api from '../api'
import { chartAnim, tooltipStyle } from '../echartsTheme'

use([CanvasRenderer, BarChart, ScatterChart, HeatmapChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, VisualMapComponent])

const modelType = ref('xgboost')
const importanceData = ref(null)
const ivData = ref(null)
const shapData = ref(null)
const corrData = ref(null)
const allImportanceData = ref(null)
const loading = ref(true)
const activeTab = ref('importance')
const shapSampleSize = ref(500)
const topN = ref(20)

async function fetchData() {
  loading.value = true
  try {
    const models = ['logistic', 'xgboost', 'lightgbm']
    const [imp, iv, shap, corr, ...allImp] = await Promise.all([
      api.get('/api/features/importance', { params: { model_type: modelType.value, top_n: topN.value } }),
      api.get('/api/features/iv', { params: { top_n: topN.value } }),
      api.get('/api/features/shap', { params: { model_type: modelType.value, sample_n: shapSampleSize.value } }),
      api.get('/api/features/correlation', { params: { top_n: 15 } }),
      ...models.map(m => api.get('/api/features/importance', { params: { model_type: m, top_n: 15 } }).catch(() => ({ data: null }))),
    ])
    importanceData.value = imp.data
    ivData.value = iv.data
    shapData.value = shap.data
    corrData.value = corr.data
    allImportanceData.value = {
      models: models,
      data: allImp.map(r => r.data).filter(Boolean),
    }
  } finally {
    loading.value = false
  }
}

onMounted(fetchData)
watch(modelType, fetchData)

function exportImportance() {
  if (!importanceData.value) return
  const header = 'Feature,Importance\n'
  const rows = importanceData.value.features.map((f, i) => `${f},${importanceData.value.importance[i]}`).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `feature_importance_${modelType.value}_top${topN.value}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function exportIV() {
  if (!ivData.value) return
  const header = 'Feature,IV,Level\n'
  const rows = ivData.value.features.map((f, i) => {
    const v = ivData.value.iv[i]
    const level = v >= 0.5 ? 'Strong' : v >= 0.1 ? 'Medium' : v >= 0.02 ? 'Weak' : 'Useless'
    return `${f},${v},${level}`
  }).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `iv_analysis_top${topN.value}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function exportSHAP() {
  if (!shapData.value?.data?.length) return
  const header = 'Feature,Feature Value,SHAP Value\n'
  const rows = shapData.value.data.map(r => `${r.feature},${r.feature_value},${r.shap_value}`).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `shap_values_${modelType.value}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function exportCorrelation() {
  if (!corrData.value) return
  const header = 'Feature1,Feature2,Correlation\n'
  const rows = corrData.value.matrix.map(m => `${corrData.value.features[m.x]},${corrData.value.features[m.y]},${m.value}`).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'correlation_matrix.csv'
  a.click()
  URL.revokeObjectURL(url)
}

async function refetchShap() {
  try {
    const res = await api.get('/api/features/shap', { params: { model_type: modelType.value, sample_n: shapSampleSize.value } })
    shapData.value = res.data
  } catch (e) {
    console.error(e)
  }
}

const compImportanceOption = ref({})
watch(allImportanceData, (d) => {
  if (!d?.data?.length) return
  // Collect top features across all models
  const allFeatures = new Set()
  d.data.forEach(imp => imp?.features?.slice(0, 10).forEach(f => allFeatures.add(f)))
  const features = [...allFeatures].slice(0, 12)
  const colors = ['#8b5cf6', '#3b82f6', '#10b981']
  const modelNames = ['Logistic', 'XGBoost', 'LightGBM']
  compImportanceOption.value = {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    grid: { left: 160, right: 30, top: 20, bottom: 40 },
    xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
    yAxis: { type: 'category', data: features.map(f => f.length > 18 ? f.slice(0, 16) + '..' : f).reverse(), axisLabel: { fontSize: 10, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: d.data.map((imp, i) => ({
      name: modelNames[i] || d.models[i],
      type: 'bar',
      data: features.map(f => {
        const idx = imp?.features?.indexOf(f)
        return idx >= 0 ? imp.importance[idx] : 0
      }).reverse(),
      itemStyle: { color: colors[i % colors.length], borderRadius: [0, 4, 4, 0] },
      barMaxWidth: 10,
    })),
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 11 } },
  }
})

const importanceOption = ref({})
watch(importanceData, (d) => {
  if (!d) return
  const features = [...d.features].reverse()
  const values = [...d.importance].reverse()
  importanceOption.value = {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    grid: { left: 160, right: 30, top: 10, bottom: 50 },
    dataZoom: [{ type: 'slider', yAxisIndex: 0, start: 0, end: 100, right: 10, width: 16, fillerColor: 'rgba(59,130,246,0.15)', borderColor: 'rgba(59,130,246,0.3)', handleStyle: { color: '#3b82f6' }, textStyle: { color: '#64748b', fontSize: 10 } }],
    xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: { type: 'category', data: features, axisLabel: { fontSize: 11, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'bar',
      data: values,
      itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#3b82f6' }, { offset: 1, color: '#06b6d4' }] }, borderRadius: [0, 6, 6, 0] },
      barMaxWidth: 18,
    }],
  }
})

const ivOption = ref({})
watch(ivData, (d) => {
  if (!d) return
  const features = [...d.features].reverse()
  const values = [...d.iv].reverse()
  const ivColors = {
    strong: { from: '#10b981', to: '#34d399' },
    medium: { from: '#3b82f6', to: '#60a5fa' },
    weak: { from: '#f59e0b', to: '#fbbf24' },
    none: { from: '#f43f5e', to: '#fb7185' },
  }
  ivOption.value = {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    grid: { left: 160, right: 30, top: 10, bottom: 50 },
    dataZoom: [{ type: 'slider', yAxisIndex: 0, start: 0, end: 60, right: 10, width: 16, fillerColor: 'rgba(59,130,246,0.15)', borderColor: 'rgba(59,130,246,0.3)', handleStyle: { color: '#3b82f6' }, textStyle: { color: '#64748b', fontSize: 10 } }],
    xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: { type: 'category', data: features, axisLabel: { fontSize: 11, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'bar',
      data: values.map(v => {
        const level = v >= 0.5 ? 'strong' : v >= 0.1 ? 'medium' : v >= 0.02 ? 'weak' : 'none'
        const c = ivColors[level]
        return { value: v, itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: c.from }, { offset: 1, color: c.to }] }, borderRadius: [0, 6, 6, 0] } }
      }),
      barMaxWidth: 18,
    }],
  }
})

const shapOption = ref({})
watch(shapData, (d) => {
  if (!d || !d.data?.length) return
  const topFeatures = [...new Set(d.data.map(r => r.feature))].slice(0, 10)
  const scatterData = []
  for (const feat of topFeatures) {
    const rows = d.data.filter(r => r.feature === feat)
    for (const r of rows) {
      scatterData.push({
        value: [Math.abs(r.shap_value), feat],
        symbolSize: Math.min(Math.abs(r.shap_value) * 20 + 4, 16),
        itemStyle: { color: r.feature_value >= 0 ? '#3b82f6' : '#f43f5e', opacity: 0.65 },
      })
    }
  }
  shapOption.value = {
    ...chartAnim,
    tooltip: {
      ...tooltipStyle,
      formatter: (p) => `<span style="font-family:Outfit">${p.value[1]}</span><br/><span style="font-family:JetBrains Mono">|SHAP| = ${p.value[0].toFixed(4)}</span>`,
    },
    grid: { left: 160, right: 30, top: 10, bottom: 20 },
    xAxis: { type: 'value', name: '|SHAP value|', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: { type: 'category', data: [...topFeatures].reverse(), axisLabel: { fontSize: 11, color: '#94a3b8', fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{
      type: 'scatter',
      data: scatterData,
    }],
  }
})

const corrOption = ref({})
watch(corrData, (d) => {
  if (!d) return
  const features = d.features
  const shortNames = features.map(f => f.length > 18 ? f.slice(0, 16) + '..' : f)
  corrOption.value = {
    ...chartAnim,
    tooltip: {
      ...tooltipStyle,
      formatter: (p) => `<span style="font-family:Outfit">${features[p.data[1]]} x ${features[p.data[0]]}</span><br/><span style="font-family:JetBrains Mono">r = ${p.data[2]}</span>`,
    },
    grid: { left: 160, right: 40, top: 10, bottom: 100 },
    xAxis: { type: 'category', data: shortNames, axisLabel: { color: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono', rotate: 45 }, axisLine: { show: false }, axisTick: { show: false }, splitArea: { show: false } },
    yAxis: { type: 'category', data: shortNames, axisLabel: { color: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }, axisLine: { show: false }, axisTick: { show: false }, splitArea: { show: false } },
    visualMap: {
      min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
      inRange: { color: ['#f43f5e', '#1e1b2b', '#3b82f6'] },
      textStyle: { color: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 11 },
    },
    series: [{
      type: 'heatmap',
      data: d.matrix.map(m => [m.x, m.y, m.value]),
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } },
    }],
  }
})

const tabs = [
  { key: 'importance', label: 'Feature Importance', icon: 'M12 20V10M18 20V4M6 20v-4' },
  { key: 'compare', label: 'Importance Compare', icon: 'M18 20V10M12 20V4M6 20v-6' },
  { key: 'iv', label: 'IV Analysis', icon: 'M3 3h18v18H3z' },
  { key: 'shap', label: 'SHAP Values', icon: 'M22 12h-4l-3 9L9 3l-3 9H2' },
  { key: 'correlation', label: 'Correlation', icon: 'M3 3h18v18H3zM3 12h18M12 3v18' },
]
</script>

<template>
  <div v-loading="loading" element-loading-background="rgba(10, 14, 26, 0.8)">
    <div class="page-header animate-in">
      <div class="header-row">
        <div>
          <h2>特征分析</h2>
          <p class="page-desc">Feature Importance & Selection</p>
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
        <div class="global-sample">
          <span class="sample-label">Top N:</span>
          <select v-model="topN" @change="fetchData" class="sample-select">
            <option :value="10">10</option>
            <option :value="20">20</option>
            <option :value="30">30</option>
            <option :value="50">50</option>
          </select>
        </div>
      </div>
    </div>

    <div class="tab-bar animate-in animate-in-delay-1">
      <button v-for="tab in tabs" :key="tab.key" :class="['tab-btn', { active: activeTab === tab.key }]" @click="activeTab = tab.key">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path :d="tab.icon"/></svg>
        {{ tab.label }}
      </button>
    </div>

    <div class="chart-panel animate-in animate-in-delay-2" v-if="activeTab === 'importance' && importanceData">
      <div class="panel-header">
        <span>Permutation Feature Importance</span>
        <div class="panel-actions">
          <button class="export-btn" @click="exportImportance">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            CSV
          </button>
          <span class="panel-badge">Top {{ topN }}</span>
        </div>
      </div>
      <v-chart :option="importanceOption" style="height: 520px" autoresize />
    </div>

    <div class="chart-panel animate-in animate-in-delay-2" v-if="activeTab === 'compare' && allImportanceData?.data?.length">
      <div class="panel-header">
        <span>Feature Importance Comparison</span>
        <span class="panel-badge">{{ allImportanceData.data.length }} Models</span>
      </div>
      <v-chart :option="compImportanceOption" style="height: 520px" autoresize />
    </div>

    <div class="chart-panel animate-in animate-in-delay-2" v-if="activeTab === 'iv' && ivData">
      <div class="panel-header">
        <span>Information Value (IV)</span>
        <div class="panel-actions">
          <button class="export-btn" @click="exportIV">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            CSV
          </button>
          <div class="iv-legend">
            <span class="iv-tag strong">>=0.5 Strong</span>
            <span class="iv-tag medium">>=0.1 Medium</span>
            <span class="iv-tag weak">>=0.02 Weak</span>
            <span class="iv-tag none"><0.02 Useless</span>
          </div>
        </div>
      </div>
      <v-chart :option="ivOption" style="height: 620px" autoresize />
    </div>

    <div class="chart-panel animate-in animate-in-delay-2" v-if="activeTab === 'shap' && shapData && !shapData.error">
      <div class="panel-header">
        <span>SHAP Summary</span>
        <div class="sample-control">
          <button class="export-btn" @click="exportSHAP">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            CSV
          </button>
          <span class="sample-label">Samples:</span>
          <select v-model="shapSampleSize" @change="refetchShap" class="sample-select">
            <option :value="200">200</option>
            <option :value="500">500</option>
            <option :value="1000">1000</option>
            <option :value="2000">2000</option>
          </select>
          <span class="panel-badge">Top 10 Features</span>
        </div>
      </div>
      <v-chart :option="shapOption" style="height: 520px" autoresize />
    </div>

    <div class="empty-state animate-in animate-in-delay-2" v-if="activeTab === 'shap' && shapData?.error">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <p>SHAP is not available</p>
    </div>

    <div class="chart-panel animate-in animate-in-delay-2" v-if="activeTab === 'correlation' && corrData">
      <div class="panel-header">
        <span>Feature Correlation Heatmap</span>
        <div class="panel-actions">
          <button class="export-btn" @click="exportCorrelation">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            CSV
          </button>
          <span class="panel-badge">Top 15 by Variance</span>
        </div>
      </div>
      <v-chart :option="corrOption" style="height: 560px" autoresize />
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
  font-size: 13px;
  font-weight: 500;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.switch-btn.active { background: var(--accent-gold); color: #0a0b10; }
.switch-btn:hover:not(.active) { color: var(--text-primary); background: rgba(255, 255, 255, 0.04); }

.switch-btn.best { position: relative; }
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
.switch-btn.best.active .best-tag { color: #0a0b10; background: rgba(0, 0, 0, 0.15); }

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

.refresh-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.refresh-btn .spinning { animation: spin 1s linear infinite; }

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.tab-bar {
  display: flex;
  gap: 4px;
  margin-bottom: 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 4px;
  width: fit-content;
}

.tab-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 500;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tab-btn.active { background: rgba(200, 170, 110, 0.08); color: var(--accent-gold); }
.tab-btn:hover:not(.active) { color: var(--text-primary); background: rgba(255, 255, 255, 0.03); }

.chart-panel {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  backdrop-filter: blur(12px);
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.panel-badge {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--accent-gold);
  background: rgba(200, 170, 110, 0.08);
  padding: 2px 8px;
  border-radius: 20px;
  font-weight: 500;
  letter-spacing: 0.3px;
}

.panel-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.export-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border: 1px solid var(--border-subtle);
  background: var(--bg-secondary);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 12px;
  font-family: var(--font-mono);
  cursor: pointer;
  transition: all 0.2s ease;
}

.export-btn:hover {
  color: var(--accent-emerald);
  border-color: var(--accent-emerald);
  background: rgba(16, 185, 129, 0.08);
}

.iv-legend { display: flex; gap: 8px; }

.iv-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 500;
  font-family: var(--font-mono);
}

.iv-tag.strong { background: rgba(16, 185, 129, 0.12); color: var(--accent-emerald); }
.iv-tag.medium { background: rgba(59, 130, 246, 0.12); color: var(--accent-blue); }
.iv-tag.weak { background: rgba(245, 158, 11, 0.12); color: var(--accent-amber); }
.iv-tag.none { background: rgba(244, 63, 94, 0.12); color: var(--accent-rose); }

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  gap: 16px;
}

.empty-state p {
  color: var(--text-muted);
  font-size: 14px;
}

.sample-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sample-label {
  font-size: 12px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.sample-select {
  padding: 4px 8px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 12px;
  cursor: pointer;
  outline: none;
}

.sample-select:focus {
  border-color: var(--accent-blue);
}

.global-sample {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
