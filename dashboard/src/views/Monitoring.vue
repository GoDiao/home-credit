<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart, PieChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import api from '../api'
import { chartAnim, tooltipStyle } from '../echartsTheme'

use([CanvasRenderer, BarChart, LineChart, PieChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent])

const psiData = ref(null)
const healthData = ref(null)
const vintageData = ref(null)
const rollRateData = ref(null)
const ewiData = ref(null)
const loading = ref(true)
const threshold = ref(0.25)
const autoRefresh = ref(false)
let refreshInterval = null

function toggleAutoRefresh() {
  autoRefresh.value = !autoRefresh.value
  if (autoRefresh.value) {
    refreshInterval = setInterval(fetchData, 60000)
  } else {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

onUnmounted(() => {
  if (refreshInterval) clearInterval(refreshInterval)
})

async function fetchData() {
  loading.value = true
  try {
    const [psiRes, healthRes, vintageRes, rollRes, ewiRes] = await Promise.all([
      api.get('/api/monitoring/psi', { params: { threshold: threshold.value } }),
      api.get('/api/monitoring/health', { params: { threshold: threshold.value } }).catch(() => ({ data: null })),
      api.get('/api/monitoring/vintage').catch(() => ({ data: null })),
      api.get('/api/monitoring/roll_rate').catch(() => ({ data: null })),
      api.get('/api/monitoring/ewi').catch(() => ({ data: null })),
    ])
    psiData.value = psiRes.data
    healthData.value = healthRes.data
    vintageData.value = vintageRes.data
    rollRateData.value = rollRes.data
    ewiData.value = ewiRes.data
  } finally {
    loading.value = false
  }
}

onMounted(fetchData)
watch(threshold, fetchData)

function exportPSI() {
  if (!psiData.value) return
  const header = 'Feature,PSI,Status\n'
  const rows = psiData.value.features.map((f, i) => {
    const psi = psiData.value.psi_values[i]
    const status = psi > threshold.value ? 'Unstable' : psi > 0.10 ? 'Marginal' : 'Stable'
    return `${f},${psi.toFixed(6)},${status}`
  }).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `psi_analysis_threshold${threshold.value.toFixed(2)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

const psiOption = ref({})
watch(psiData, (d) => {
  if (!d) return
  const sorted = d.features
    .map((f, i) => ({ feature: f, psi: d.psi_values[i] }))
    .sort((a, b) => b.psi - a.psi)
    .slice(0, 40)

  psiOption.value = {
    ...chartAnim,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      ...tooltipStyle,
      formatter: (params) => {
        const p = params[0]
        return `<span style="font-family:Outfit">${p.name}</span><br/><span style="font-family:JetBrains Mono">PSI = ${p.value.toFixed(4)}</span>`
      },
    },
    grid: { left: 160, right: 30, top: 10, bottom: 50 },
    dataZoom: [{ type: 'slider', yAxisIndex: 0, start: 0, end: 60, right: 10, width: 16, fillerColor: 'rgba(59,130,246,0.15)', borderColor: 'rgba(59,130,246,0.3)', handleStyle: { color: '#3b82f6' }, textStyle: { color: '#64748b', fontSize: 10 } }],
    xAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: {
      type: 'category',
      data: sorted.map(r => r.feature).reverse(),
      axisLabel: { fontSize: 11, color: '#94a3b8', fontFamily: 'JetBrains Mono' },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [{
      type: 'bar',
      data: sorted.map(r => {
        const isUnstable = r.psi > threshold.value
        const isMarginal = r.psi > 0.10 && r.psi <= threshold.value
        const colors = isUnstable
          ? { from: '#f43f5e', to: '#fb7185' }
          : isMarginal
          ? { from: '#f59e0b', to: '#fbbf24' }
          : { from: '#10b981', to: '#34d399' }
        return {
          value: r.psi,
          itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: colors.from }, { offset: 1, color: colors.to }] }, borderRadius: [0, 4, 4, 0] },
        }
      }).reverse(),
      barMaxWidth: 14,
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { type: 'dashed' },
        data: [
          { xAxis: 0.10, lineStyle: { color: 'rgba(245, 158, 11, 0.4)' }, label: { formatter: 'Marginal', position: 'end', color: '#f59e0b', fontSize: 11, fontFamily: 'JetBrains Mono' } },
          { xAxis: threshold.value, lineStyle: { color: 'rgba(244, 63, 94, 0.5)' }, label: { formatter: `Threshold`, position: 'end', color: '#f43f5e', fontSize: 11, fontFamily: 'JetBrains Mono' } },
        ],
      },
    }],
  }
})

// Vintage chart option
const vintageOption = computed(() => {
  if (!vintageData.value?.series?.length) return {}
  const d = vintageData.value
  const colors = ['#c8aa6e', '#60a5fa', '#34d399', '#f59e0b', '#a78bfa', '#f43f5e', '#06b6d4', '#fb923c', '#818cf8', '#e879f9', '#facc15', '#4ade80']
  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', ...tooltipStyle },
    legend: { top: 0, textStyle: { color: '#94a3b8', fontFamily: 'Outfit', fontSize: 10 }, type: 'scroll', pageTextStyle: { color: '#64748b' } },
    grid: { left: 50, right: 20, top: 36, bottom: 30 },
    xAxis: { type: 'category', data: d.months.map(m => m + 'm'), axisLabel: { color: '#64748b', fontFamily: 'Outfit' }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
    yAxis: { type: 'value', name: 'Default Rate %', nameTextStyle: { color: '#64748b', fontFamily: 'Outfit', fontSize: 10 }, axisLabel: { color: '#64748b', fontFamily: 'JetBrains Mono', fontSize: 10 }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
    series: d.series.map((s, i) => ({
      name: s.cohort, type: 'line', smooth: true, data: s.values, symbol: 'circle', symbolSize: 4,
      lineStyle: { width: 2, color: colors[i % colors.length] },
      itemStyle: { color: colors[i % colors.length] },
    })),
  }
})

// Roll Rate donut option
const rollRateOption = computed(() => {
  if (!rollRateData.value?.buckets?.length) return {}
  const d = rollRateData.value
  const colors = ['#34d399', '#60a5fa', '#f59e0b', '#fb923c', '#f43f5e', '#a78bfa']
  return {
    ...chartAnim,
    tooltip: { trigger: 'item', ...tooltipStyle, formatter: '{b}: {c} ({d}%)' },
    series: [{
      type: 'pie', radius: ['40%', '70%'], center: ['50%', '50%'],
      data: d.buckets.map((b, i) => ({ name: b, value: d.counts[i], itemStyle: { color: colors[i % colors.length] } })),
      label: { color: '#94a3b8', fontFamily: 'Outfit', fontSize: 11 },
      labelLine: { lineStyle: { color: 'rgba(255,255,255,0.15)' } },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } },
    }],
  }
})

const ewiItems = computed(() => {
  if (!ewiData.value) return []
  const d = ewiData.value
  const items = []
  if (d.default_rate != null) items.push({ label: 'Default Rate', value: (d.default_rate * 100).toFixed(2) + '%', cls: 'rose' })
  if (d.fpd_rate != null) items.push({ label: 'FPD Rate', value: (d.fpd_rate * 100).toFixed(2) + '%', cls: 'amber' })
  if (d.average_loan_amount != null) items.push({ label: 'Avg Loan Amount', value: '¥' + (d.average_loan_amount / 10000).toFixed(1) + '万', cls: 'blue' })
  if (d.average_income != null) items.push({ label: 'Avg Income', value: '¥' + (d.average_income / 10000).toFixed(1) + '万', cls: 'cyan' })
  if (d.debt_to_income_ratio != null) items.push({ label: 'Debt/Income', value: d.debt_to_income_ratio.toFixed(2) + 'x', cls: 'violet' })
  return items
})
</script>

<template>
  <div v-loading="loading" element-loading-background="rgba(10, 14, 26, 0.8)">
    <div class="page-header animate-in">
      <h2>稳定性监控</h2>
      <p class="page-desc">Population Stability Index (PSI)</p>
    </div>

    <div class="stats-row" v-if="psiData">
      <div class="stat-card animate-in animate-in-delay-1">
        <div class="stat-icon slate">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Total Features</div>
          <div class="stat-value">{{ psiData.total_features }}</div>
        </div>
      </div>

      <div class="stat-card animate-in animate-in-delay-2">
        <div class="stat-icon emerald">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="20 6 9 17 4 12"/></svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Stable (<=0.10)</div>
          <div class="stat-value emerald">{{ psiData.stable_count }}</div>
        </div>
      </div>

      <div class="stat-card animate-in animate-in-delay-3">
        <div class="stat-icon amber">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Marginal (0.10~T)</div>
          <div class="stat-value amber">{{ psiData.marginal_count }}</div>
        </div>
      </div>

      <div class="stat-card animate-in animate-in-delay-4">
        <div class="stat-icon rose">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
        </div>
        <div class="stat-content">
          <div class="stat-label">Unstable (>T)</div>
          <div class="stat-value rose">{{ psiData.unstable_count }}</div>
        </div>
      </div>
    </div>

    <!-- Health Status & Alerts -->
    <div class="health-section animate-in animate-in-delay-4" v-if="healthData">
      <div class="health-left">
        <div :class="['health-badge', healthData.color]">
          <div class="health-dot"></div>
          <span class="health-status">{{ healthData.status_label }}</span>
        </div>
        <div class="health-breakdown">
          <div class="health-bar">
            <div class="hb-fill stable" :style="{ width: healthData.stable_pct + '%' }"></div>
            <div class="hb-fill marginal" :style="{ width: healthData.marginal_pct + '%' }"></div>
            <div class="hb-fill unstable" :style="{ width: healthData.unstable_pct + '%' }"></div>
          </div>
          <div class="health-bar-legend">
            <span class="hbl-item"><span class="hbl-dot stable"></span>Stable {{ healthData.stable_pct }}%</span>
            <span class="hbl-item"><span class="hbl-dot marginal"></span>Marginal {{ healthData.marginal_pct }}%</span>
            <span class="hbl-item"><span class="hbl-dot unstable"></span>Unstable {{ healthData.unstable_pct }}%</span>
          </div>
        </div>
      </div>
      <div class="health-right">
        <div class="alert-list">
          <div v-for="(alert, i) in healthData.alerts" :key="i" :class="['alert-item', alert.level]">
            <div :class="['alert-icon', alert.level]">
              <svg v-if="alert.level === 'critical'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
              <svg v-else-if="alert.level === 'warning'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
              <svg v-else width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <div class="alert-content">
              <div class="alert-msg">{{ alert.message }}</div>
              <div class="alert-action">{{ alert.action }}</div>
            </div>
          </div>
        </div>
        <div class="top-drift" v-if="healthData.top_drift?.length">
          <div class="drift-title">Top Drift Features</div>
          <div class="drift-list">
            <div v-for="d in healthData.top_drift" :key="d.feature" class="drift-item">
              <span class="drift-name">{{ d.feature }}</span>
              <span :class="['drift-psi', d.psi > threshold ? 'unstable' : d.psi > 0.10 ? 'marginal' : 'stable']">{{ d.psi?.toFixed(4) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="chart-panel animate-in animate-in-delay-5">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          <span>PSI Distribution (Top 40)</span>
        </div>
        <div class="threshold-control">
          <button class="export-btn" @click="exportPSI">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            CSV
          </button>
          <button :class="['auto-btn', { active: autoRefresh }]" @click="toggleAutoRefresh">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" :class="{ spinning: autoRefresh }"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            Auto
          </button>
          <span class="threshold-label">Unstable Threshold</span>
          <el-slider
            v-model="threshold"
            :min="0.05"
            :max="1.0"
            :step="0.05"
            :style="{ width: '180px' }"
            :format-tooltip="(v) => v.toFixed(2)"
          />
          <span class="threshold-val">{{ threshold.toFixed(2) }}</span>
        </div>
      </div>
      <v-chart :option="psiOption" style="height: 620px" autoresize />
    </div>

    <!-- EWI Indicator Cards -->
    <div class="ewi-row animate-in" v-if="ewiData && Object.keys(ewiData).length">
      <div v-for="item in ewiItems" :key="item.label" :class="['ewi-card', item.cls]">
        <div class="ewi-label">{{ item.label }}</div>
        <div class="ewi-value">{{ item.value }}</div>
      </div>
    </div>

    <!-- Vintage + Roll Rate -->
    <div class="viz-grid animate-in" v-if="vintageData?.series?.length || rollRateData?.buckets?.length">
      <div class="chart-panel" v-if="vintageData?.series?.length">
        <div class="panel-header">
          <div class="panel-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            <span>Vintage Analysis — Cumulative Default Rate</span>
          </div>
        </div>
        <v-chart :option="vintageOption" style="height: 360px" autoresize />
      </div>
      <div class="chart-panel" v-if="rollRateData?.buckets?.length">
        <div class="panel-header">
          <div class="panel-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
            <span>DPD Bucket Distribution</span>
          </div>
        </div>
        <v-chart :option="rollRateOption" style="height: 360px" autoresize />
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-header { margin-bottom: 24px; }
.page-header h2 { margin: 0; font-family: var(--font-display); font-size: 28px; }
.page-desc { color: var(--text-muted); font-size: 12px; margin-top: 6px; font-family: var(--font-mono); letter-spacing: 0.5px; }

.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 18px;
  display: flex;
  align-items: center;
  gap: 14px;
  transition: all 0.3s ease;
}

.stat-card:hover { border-color: var(--border-accent); transform: translateY(-2px); }

.stat-icon {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.stat-icon.slate { background: rgba(148, 163, 184, 0.1); color: #94a3b8; }
.stat-icon.emerald { background: rgba(16, 185, 129, 0.12); color: var(--accent-emerald); }
.stat-icon.amber { background: rgba(245, 158, 11, 0.12); color: var(--accent-amber); }
.stat-icon.rose { background: rgba(244, 63, 94, 0.12); color: var(--accent-rose); }

.stat-content { min-width: 0; }
.stat-label { font-size: 12px; color: var(--text-muted); margin-bottom: 4px; font-weight: 500; }
.stat-value { font-size: 24px; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono); letter-spacing: -0.5px; }
.stat-value.emerald { color: var(--accent-emerald); }
.stat-value.amber { color: var(--accent-amber); }
.stat-value.rose { color: var(--accent-rose); }

.chart-panel {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  backdrop-filter: blur(12px);
}

/* ── Health Status ── */
.health-section {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 16px;
  margin-bottom: 20px;
}
.health-left, .health-right {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 20px;
}
.health-badge {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 16px;
}
.health-dot {
  width: 10px; height: 10px; border-radius: 50%;
  animation: breathe 3s ease infinite;
}
.health-badge.emerald .health-dot { background: var(--accent-emerald); box-shadow: 0 0 8px rgba(16,185,129,0.4); }
.health-badge.amber .health-dot { background: var(--accent-amber); box-shadow: 0 0 8px rgba(245,158,11,0.4); }
.health-badge.rose .health-dot { background: var(--accent-rose); box-shadow: 0 0 8px rgba(244,63,94,0.4); }
.health-status { font-size: 18px; font-weight: 700; font-family: var(--font-display); }
.health-badge.emerald .health-status { color: var(--accent-emerald); }
.health-badge.amber .health-status { color: var(--accent-amber); }
.health-badge.rose .health-status { color: var(--accent-rose); }
.health-breakdown {}
.health-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: rgba(255,255,255,0.04); }
.hb-fill { height: 100%; transition: width 0.6s ease; }
.hb-fill.stable { background: var(--accent-emerald); }
.hb-fill.marginal { background: var(--accent-amber); }
.hb-fill.unstable { background: var(--accent-rose); }
.health-bar-legend { display: flex; gap: 16px; margin-top: 8px; }
.hbl-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); }
.hbl-dot { width: 6px; height: 6px; border-radius: 50%; }
.hbl-dot.stable { background: var(--accent-emerald); }
.hbl-dot.marginal { background: var(--accent-amber); }
.hbl-dot.unstable { background: var(--accent-rose); }

.alert-list { display: flex; flex-direction: column; gap: 10px; margin-bottom: 16px; }
.alert-item { display: flex; gap: 10px; align-items: flex-start; }
.alert-icon { width: 24px; height: 24px; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.alert-icon.critical { background: rgba(244,63,94,0.12); color: var(--accent-rose); }
.alert-icon.warning { background: rgba(245,158,11,0.12); color: var(--accent-amber); }
.alert-icon.info { background: rgba(16,185,129,0.12); color: var(--accent-emerald); }
.alert-content {}
.alert-msg { font-size: 12px; color: var(--text-primary); font-weight: 500; }
.alert-action { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

.top-drift {}
.drift-title { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; font-weight: 500; }
.drift-list { display: flex; flex-wrap: wrap; gap: 6px; }
.drift-item { display: flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.03); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 4px 10px; }
.drift-name { font-size: 11px; color: var(--text-secondary); font-family: var(--font-mono); }
.drift-psi { font-size: 11px; font-family: var(--font-mono); font-weight: 600; }
.drift-psi.stable { color: var(--accent-emerald); }
.drift-psi.marginal { color: var(--accent-amber); }
.drift-psi.unstable { color: var(--accent-rose); }

@keyframes breathe {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border-subtle);
}

.panel-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.threshold-control {
  display: flex;
  align-items: center;
  gap: 12px;
}

.threshold-label { font-size: 12px; color: var(--text-muted); }
.threshold-val {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.1);
  padding: 3px 10px;
  border-radius: 6px;
  min-width: 42px;
  text-align: center;
}

:deep(.el-slider__runway) { background: rgba(255, 255, 255, 0.08) !important; }
:deep(.el-slider__bar) { background: var(--gradient-blue) !important; }
:deep(.el-slider__button) { border-color: var(--accent-blue) !important; width: 14px !important; height: 14px !important; }

.auto-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  border: 1px solid var(--border-subtle);
  background: var(--bg-secondary);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 12px;
  font-family: var(--font-mono);
  cursor: pointer;
  transition: all 0.2s ease;
}

.auto-btn:hover { border-color: var(--accent-blue); color: var(--text-primary); }
.auto-btn.active { background: rgba(59, 130, 246, 0.12); border-color: var(--accent-blue); color: var(--accent-blue); }
.auto-btn .spinning { animation: spin 2s linear infinite; }

.export-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
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

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* ── EWI Cards ── */
.ewi-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.ewi-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  text-align: center;
  transition: border-color 0.2s;
}
.ewi-card:hover { border-color: rgba(200,170,110,0.2); }
.ewi-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.ewi-value { font-size: 20px; font-weight: 700; font-family: var(--font-mono); }
.ewi-card.rose .ewi-value { color: var(--accent-rose); }
.ewi-card.amber .ewi-value { color: var(--accent-amber); }
.ewi-card.blue .ewi-value { color: var(--accent-blue); }
.ewi-card.cyan .ewi-value { color: #06b6d4; }
.ewi-card.violet .ewi-value { color: #a78bfa; }

/* ── Viz Grid ── */
.viz-grid {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 16px;
}
</style>
