<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import api from '../api'
import { chartAnim, tooltipStyle } from '../echartsTheme'

use([CanvasRenderer, BarChart, LineChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent])

const models = [
  { type: 'stacking', label: 'Stacking', cls: 'stk' },
  { type: 'xgboost', label: 'XGBoost', cls: 'xgb' },
  { type: 'lightgbm', label: 'LightGBM', cls: 'lgb' },
  { type: 'logistic', label: 'Logistic', cls: 'lr' },
]
const activeModel = ref('stacking')
const summary = ref(null)
const distribution = ref(null)
const liftData = ref(null)
const loading = ref(true)

async function fetchData() {
  loading.value = true
  try {
    const [sumRes, distRes, liftRes] = await Promise.all([
      api.get('/api/scorecard/summary', { params: { model_type: activeModel.value } }).catch(() => ({ data: null })),
      api.get('/api/scorecard/distribution', { params: { model_type: activeModel.value, n_bins: 20 } }).catch(() => ({ data: null })),
      api.get('/api/scorecard/lift', { params: { model_type: activeModel.value, n_bins: 10 } }).catch(() => ({ data: null })),
    ])
    summary.value = sumRes.data
    distribution.value = distRes.data
    liftData.value = liftRes.data
  } finally {
    loading.value = false
  }
}

onMounted(fetchData)
watch(activeModel, fetchData)

// Score Distribution Chart
const distOption = computed(() => {
  if (!distribution.value?.bins?.length) return {}
  const bins = distribution.value.bins
  const labels = bins.map(b => `${Math.round(b.score_min)}-${Math.round(b.score_max)}`)
  const goodCounts = bins.map(b => b.good)
  const badCounts = bins.map(b => b.bad)
  const badRates = bins.map(b => +(b.bad_rate * 100).toFixed(2))

  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    legend: { data: ['Good', 'Bad', 'Bad Rate %'], top: 4, textStyle: { color: '#94a3b8', fontFamily: 'Outfit', fontSize: 11 } },
    grid: { left: 60, right: 50, top: 46, bottom: 40 },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#64748b', fontFamily: 'Outfit', fontSize: 9, rotate: 35 }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
    yAxis: [
      { type: 'value', name: 'Count', nameTextStyle: { color: '#64748b', fontFamily: 'Outfit', fontSize: 10 }, axisLabel: { color: '#64748b', fontFamily: 'JetBrains Mono', fontSize: 10 }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
      { type: 'value', name: 'Bad Rate %', nameTextStyle: { color: '#64748b', fontFamily: 'Outfit', fontSize: 10 }, axisLabel: { color: '#64748b', fontFamily: 'JetBrains Mono', fontSize: 10 }, splitLine: { show: false } },
    ],
    series: [
      { name: 'Good', type: 'bar', stack: 'count', data: goodCounts, itemStyle: { color: 'rgba(16, 185, 129, 0.7)', borderRadius: [0, 0, 0, 0] } },
      { name: 'Bad', type: 'bar', stack: 'count', data: badCounts, itemStyle: { color: 'rgba(244, 63, 94, 0.7)', borderRadius: [2, 2, 0, 0] } },
      { name: 'Bad Rate %', type: 'line', yAxisIndex: 1, data: badRates, smooth: true, symbol: 'circle', symbolSize: 5, lineStyle: { color: '#fbbf24', width: 2 }, itemStyle: { color: '#fbbf24' } },
    ],
  }
})

// Lift / Gains Chart
const liftOption = computed(() => {
  if (!liftData.value?.bins?.length) return {}
  const bins = liftData.value.bins
  const deciles = bins.map(b => `D${b.decile}`)
  const liftVals = bins.map(b => +(b.lift).toFixed(2))
  const captureBad = bins.map(b => +(b.cumulative_bad_pct * 100).toFixed(1))
  const captureGood = bins.map(b => +(b.cumulative_good_pct * 100).toFixed(1))

  return {
    ...chartAnim,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, ...tooltipStyle },
    legend: { data: ['Lift', 'Bad Capture %', 'Good Capture %'], top: 4, textStyle: { color: '#94a3b8', fontFamily: 'Outfit', fontSize: 11 } },
    grid: { left: 50, right: 50, top: 46, bottom: 30 },
    xAxis: { type: 'category', data: deciles, axisLabel: { color: '#64748b', fontFamily: 'Outfit', fontSize: 11 }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
    yAxis: [
      { type: 'value', name: 'Lift', nameTextStyle: { color: '#64748b', fontFamily: 'Outfit', fontSize: 10 }, axisLabel: { color: '#64748b', fontFamily: 'JetBrains Mono', fontSize: 10 }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
      { type: 'value', name: 'Capture %', max: 100, nameTextStyle: { color: '#64748b', fontFamily: 'Outfit', fontSize: 10 }, axisLabel: { color: '#64748b', fontFamily: 'JetBrains Mono', fontSize: 10 }, splitLine: { show: false } },
    ],
    series: [
      { name: 'Lift', type: 'bar', data: liftVals, itemStyle: { color: 'rgba(200, 170, 110, 0.75)', borderRadius: [3, 3, 0, 0] } },
      { name: 'Bad Capture %', type: 'line', yAxisIndex: 1, data: captureBad, smooth: true, symbol: 'circle', symbolSize: 5, lineStyle: { color: '#f43f5e', width: 2 }, itemStyle: { color: '#f43f5e' } },
      { name: 'Good Capture %', type: 'line', yAxisIndex: 1, data: captureGood, smooth: true, symbol: 'circle', symbolSize: 5, lineStyle: { color: '#10b981', width: 2, type: 'dashed' }, itemStyle: { color: '#10b981' } },
    ],
  }
})

function exportBins() {
  if (!distribution.value?.bins) return
  const header = 'Score Min,Score Max,Count,Good,Bad,Bad Rate\n'
  const rows = distribution.value.bins.map(b =>
    `${Math.round(b.score_min)},${Math.round(b.score_max)},${b.count},${b.good},${b.bad},${(b.bad_rate * 100).toFixed(2)}%`
  ).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `scorecard_bins_${activeModel.value}.csv`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div>
    <div class="page-header animate-in">
      <div class="header-row">
        <div>
          <h2>Scorecard</h2>
          <p class="page-desc">PD-to-Score Mapping, Score Distribution & Lift Analysis</p>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
          <div class="model-switcher">
            <button v-for="m in models" :key="m.type"
              :class="['model-btn', m.cls, { active: activeModel === m.type }]"
              @click="activeModel = m.type">
              <span class="model-dot" :class="m.cls"></span>
              {{ m.label }}
            </button>
          </div>
          <button class="export-btn" @click="exportBins">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
            CSV
          </button>
        </div>
      </div>
    </div>

    <!-- Summary Stats -->
    <div class="stats-grid" v-if="summary && !loading">
      <div class="stat-card animate-in animate-in-delay-1">
        <div class="stat-top">
          <div class="stat-icon gold"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 20V10M18 20V4M6 20v-4"/></svg></div>
        </div>
        <div class="stat-number">{{ summary.score_mean?.toFixed(0) }}</div>
        <div class="stat-label">Mean Score</div>
        <div class="stat-footer"><span class="stat-sub">std {{ summary.score_std?.toFixed(0) }}</span></div>
      </div>
      <div class="stat-card animate-in animate-in-delay-2">
        <div class="stat-top">
          <div class="stat-icon blue"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg></div>
        </div>
        <div class="stat-number">{{ summary.score_median?.toFixed(0) }}</div>
        <div class="stat-label">Median Score</div>
        <div class="stat-footer"><span class="stat-sub">{{ summary.score_p25?.toFixed(0) }} - {{ summary.score_p75?.toFixed(0) }} (IQR)</span></div>
      </div>
      <div class="stat-card animate-in animate-in-delay-3">
        <div class="stat-top">
          <div class="stat-icon emerald"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></div>
        </div>
        <div class="stat-number">{{ summary.good_mean_score?.toFixed(0) }}</div>
        <div class="stat-label">Good Avg Score</div>
        <div class="stat-footer"><span class="stat-sub">{{ summary.total_good?.toLocaleString() }} good</span></div>
      </div>
      <div class="stat-card animate-in animate-in-delay-4">
        <div class="stat-top">
          <div class="stat-icon rose"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
        </div>
        <div class="stat-number">{{ summary.bad_mean_score?.toFixed(0) }}</div>
        <div class="stat-label">Bad Avg Score</div>
        <div class="stat-footer"><span class="stat-sub">{{ summary.total_bad?.toLocaleString() }} bad</span></div>
      </div>
    </div>

    <!-- Config Badge -->
    <div class="config-bar animate-in animate-in-delay-2" v-if="distribution?.config">
      <span class="config-item"><span class="config-label">Base Score</span><span class="config-val">{{ distribution.config.base_score }}</span></span>
      <span class="config-sep"></span>
      <span class="config-item"><span class="config-label">Base Odds</span><span class="config-val">{{ distribution.config.base_odds }}:1</span></span>
      <span class="config-sep"></span>
      <span class="config-item"><span class="config-label">PDO</span><span class="config-val">{{ distribution.config.pdo }}</span></span>
      <span class="config-sep"></span>
      <span class="config-item"><span class="config-label">Range</span><span class="config-val">{{ distribution.config.score_min }}-{{ distribution.config.score_max }}</span></span>
      <span class="config-sep"></span>
      <span class="config-item"><span class="config-label">KS</span><span class="config-val">{{ summary?.ks_stat?.toFixed(4) }}</span></span>
    </div>

    <!-- Charts Grid -->
    <div class="chart-grid animate-in animate-in-delay-3" v-if="!loading">
      <div class="panel">
        <div class="panel-header"><span class="panel-title">Score Distribution</span></div>
        <div class="panel-body"><v-chart :option="distOption" autoresize style="height:360px" /></div>
      </div>
      <div class="panel">
        <div class="panel-header"><span class="panel-title">Lift & Capture Rate</span></div>
        <div class="panel-body"><v-chart :option="liftOption" autoresize style="height:360px" /></div>
      </div>
    </div>

    <!-- Score Bin Table -->
    <div class="table-section animate-in animate-in-delay-4" v-if="distribution?.bins?.length && !loading">
      <div class="section-header">
        <div class="section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
          <span>Score Bins Detail</span>
        </div>
      </div>
      <div class="table-wrap">
        <el-table :data="distribution.bins" :header-cell-style="{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-muted)', fontWeight: 500, borderBottom: '1px solid var(--border-subtle)' }" :cell-style="{ borderBottom: '1px solid var(--border-subtle)' }">
          <el-table-column label="Score Range" min-width="140">
            <template #default="{ row }">
              <span class="mono-cell">{{ Math.round(row.score_min) }} - {{ Math.round(row.score_max) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="count" label="Count" min-width="90">
            <template #default="{ row }">
              <span class="mono-cell">{{ row.count?.toLocaleString() }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="good" label="Good" min-width="80">
            <template #default="{ row }">
              <span class="mono-cell" style="color:var(--accent-emerald)">{{ row.good?.toLocaleString() }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="bad" label="Bad" min-width="80">
            <template #default="{ row }">
              <span class="mono-cell" style="color:#f43f5e">{{ row.bad?.toLocaleString() }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Bad Rate" min-width="100">
            <template #default="{ row }">
              <div class="bad-rate-cell">
                <div class="bad-rate-bar"><div class="bad-rate-fill" :style="{ width: Math.min(row.bad_rate * 500, 100) + '%' }"></div></div>
                <span class="mono-cell">{{ (row.bad_rate * 100).toFixed(2) }}%</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="KS" min-width="80">
            <template #default="{ row }">
              <span class="mono-cell">{{ (row.ks * 100).toFixed(2) }}%</span>
            </template>
          </el-table-column>
          <el-table-column label="Cum Bad %" min-width="90">
            <template #default="{ row }">
              <span class="mono-cell">{{ (row.cumulative_bad_rate * 100).toFixed(1) }}%</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- Lift Table -->
    <div class="table-section animate-in animate-in-delay-5" v-if="liftData?.bins?.length && !loading">
      <div class="section-header">
        <div class="section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
          <span>Lift by Decile</span>
        </div>
        <span class="table-badge">Pop Bad Rate: {{ (liftData.population_bad_rate * 100).toFixed(2) }}%</span>
      </div>
      <div class="table-wrap">
        <el-table :data="liftData.bins" :header-cell-style="{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-muted)', fontWeight: 500, borderBottom: '1px solid var(--border-subtle)' }" :cell-style="{ borderBottom: '1px solid var(--border-subtle)' }">
          <el-table-column prop="decile" label="Decile" min-width="70">
            <template #default="{ row }">
              <span class="mono-cell">D{{ row.decile }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Score Range" min-width="140">
            <template #default="{ row }">
              <span class="mono-cell">{{ Math.round(row.score_min) }} - {{ Math.round(row.score_max) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="count" label="Count" min-width="80">
            <template #default="{ row }">
              <span class="mono-cell">{{ row.count?.toLocaleString() }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="bad" label="Bad" min-width="70">
            <template #default="{ row }">
              <span class="mono-cell" style="color:#f43f5e">{{ row.bad }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Bad Rate" min-width="90">
            <template #default="{ row }">
              <span class="mono-cell">{{ (row.bad_rate * 100).toFixed(2) }}%</span>
            </template>
          </el-table-column>
          <el-table-column label="Lift" min-width="80">
            <template #default="{ row }">
              <span class="mono-cell" :style="{ color: row.lift > 1.5 ? '#f43f5e' : row.lift < 0.5 ? 'var(--accent-emerald)' : 'var(--text-secondary)' }">{{ row.lift.toFixed(2) }}x</span>
            </template>
          </el-table-column>
          <el-table-column label="Bad Capture" min-width="100">
            <template #default="{ row }">
              <span class="mono-cell">{{ (row.cumulative_bad_pct * 100).toFixed(1) }}%</span>
            </template>
          </el-table-column>
          <el-table-column label="Good Capture" min-width="100">
            <template #default="{ row }">
              <span class="mono-cell" style="color:var(--accent-emerald)">{{ (row.cumulative_good_pct * 100).toFixed(1) }}%</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-header { margin-bottom: 24px; }
.header-row { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
.page-header h2 { margin: 0; font-family: var(--font-display); font-size: 28px; }
.page-desc { color: var(--text-muted); font-size: 13px; margin-top: 4px; }

.model-switcher { display: flex; gap: 6px; }
.model-btn {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 14px; border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle); background: transparent;
  color: var(--text-secondary); font-size: 12px; font-family: 'Outfit', sans-serif;
  cursor: pointer; transition: all 0.2s;
}
.model-btn:hover { border-color: rgba(200, 170, 110, 0.3); color: var(--text-primary); }
.model-btn.active { background: rgba(200, 170, 110, 0.08); border-color: var(--accent-gold); color: var(--accent-gold); }

.model-dot {
  width: 7px; height: 7px; border-radius: 50%;
}
.model-dot.stk { background: #d4a853; }
.model-dot.xgb { background: #60a5fa; }
.model-dot.lgb { background: #34d399; }
.model-dot.lr { background: #a78bfa; }

.export-btn {
  display: flex; align-items: center; gap: 5px;
  padding: 6px 12px; border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle); background: transparent;
  color: var(--text-muted); font-size: 11px; font-family: 'JetBrains Mono', monospace;
  cursor: pointer; transition: all 0.2s;
}
.export-btn:hover { border-color: rgba(16, 185, 129, 0.4); color: var(--accent-emerald); }

.stats-grid {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 14px; margin-bottom: 20px;
}
.stat-card {
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md); padding: 18px 20px;
  transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(200, 170, 110, 0.2); }
.stat-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.stat-icon {
  width: 32px; height: 32px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
}
.stat-icon.gold { background: rgba(200, 170, 110, 0.1); color: #c8aa6e; }
.stat-icon.blue { background: rgba(96, 165, 250, 0.1); color: #60a5fa; }
.stat-icon.emerald { background: rgba(16, 185, 129, 0.1); color: #10b981; }
.stat-icon.rose { background: rgba(244, 63, 94, 0.1); color: #f43f5e; }
.stat-number { font-size: 28px; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono); line-height: 1.1; }
.stat-label { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
.stat-footer { margin-top: 6px; }
.stat-sub { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); }

.config-bar {
  display: flex; align-items: center; gap: 16px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm); padding: 10px 18px;
  margin-bottom: 20px; flex-wrap: wrap;
}
.config-item { display: flex; align-items: center; gap: 6px; }
.config-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
.config-val { font-size: 13px; color: var(--accent-gold); font-family: var(--font-mono); font-weight: 600; }
.config-sep { width: 1px; height: 14px; background: var(--border-subtle); }

.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 24px; }
.panel { background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); overflow: hidden; }
.panel-header { padding: 14px 18px; border-bottom: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: space-between; }
.panel-title { font-size: 13px; font-weight: 500; color: var(--text-primary); }
.panel-body { padding: 8px; }

.table-section {
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md); padding: 20px; margin-bottom: 20px;
}
.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.section-title { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 500; color: var(--text-primary); }
.table-badge { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); background: rgba(255,255,255,0.03); padding: 3px 10px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); }

.mono-cell { font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary); }

.bad-rate-cell { display: flex; align-items: center; gap: 8px; }
.bad-rate-bar { width: 40px; height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
.bad-rate-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #f43f5e, #fb923c); }

.table-wrap { overflow-x: auto; }
</style>
