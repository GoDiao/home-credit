<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, MarkPointComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import api from '../api'
import { chartAnim, tooltipStyle } from '../echartsTheme'

use([CanvasRenderer, LineChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, MarkPointComponent])

const policyData = ref(null)
const interactiveResult = ref(null)
const recommendations = ref([])
const cutoff = ref(0.5)
const loading = ref(true)
const interactiveLoading = ref(false)

onMounted(async () => {
  try {
    const [sim, rec] = await Promise.all([
      api.get('/api/policy/simulation'),
      api.get('/api/policy/recommend'),
    ])
    policyData.value = sim.data
    recommendations.value = rec.data?.strategies || rec.data?.recommendations || []
    if (policyData.value?.pareto_front?.length) {
      cutoff.value = policyData.value.pareto_front[0]?.cutoff || 0.5
    }
    await fetchInteractive()
  } finally {
    loading.value = false
  }
})

async function fetchInteractive() {
  interactiveLoading.value = true
  try {
    const res = await api.get('/api/policy/interactive', { params: { cutoff: cutoff.value } })
    interactiveResult.value = res.data
  } finally {
    interactiveLoading.value = false
  }
}

let debounceTimer = null
function onCutoffChange(val) {
  cutoff.value = val
  clearTimeout(debounceTimer)
  debounceTimer = setTimeout(fetchInteractive, 300)
}

async function reload() {
  loading.value = true
  try {
    const [sim, rec] = await Promise.all([
      api.get('/api/policy/simulation'),
      api.get('/api/policy/recommend'),
    ])
    policyData.value = sim.data
    recommendations.value = rec.data?.strategies || rec.data?.recommendations || []
    await fetchInteractive()
  } finally {
    loading.value = false
  }
}

function exportStrategies() {
  if (!recommendations.value?.length) return
  const header = 'Strategy,Cut-off,Approval Rate,Avg PD,EL Rate,Bad Capture\n'
  const rows = recommendations.value.map(r =>
    `${strategyLabelMap[r.strategy] || r.strategy},${r.cutoff?.toFixed(4)},${(r.approval_rate * 100).toFixed(1)}%,${(r.avg_pd * 100).toFixed(2)}%,${(r.el_rate * 100).toFixed(2)}%,${(r.reject_bad_capture * 100).toFixed(1)}%`
  ).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'strategy_recommendations.csv'
  a.click()
  URL.revokeObjectURL(url)
}

const paretoOption = ref({})
watch(policyData, (d) => {
  if (!d?.pareto_front?.length) return
  const front = d.pareto_front
  paretoOption.value = {
    ...chartAnim,
    tooltip: {
      ...tooltipStyle,
      formatter: (p) => `<span style="font-family:Outfit">Cutoff: ${p.data[0].toFixed(3)}</span><br/><span style="font-family:JetBrains Mono">Approval: ${(p.data[1] * 100).toFixed(1)}%<br/>EL Rate: ${(p.data[2] * 100).toFixed(2)}%</span>`,
    },
    grid: { left: 60, right: 60, top: 20, bottom: 40 },
    xAxis: { type: 'value', name: 'Cut-off', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    yAxis: [
      { type: 'value', name: 'Approval Rate', min: 0, max: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
      { type: 'value', name: 'EL Rate', min: 0, splitLine: { show: false }, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    ],
    series: [
      {
        name: 'Approval Rate',
        type: 'scatter',
        yAxisIndex: 0,
        data: front.map(p => [p.cutoff, p.approval_rate]),
        symbolSize: 10,
        itemStyle: { color: '#10b981', shadowBlur: 6, shadowColor: 'rgba(16,185,129,0.3)' },
      },
      {
        name: 'EL Rate',
        type: 'scatter',
        yAxisIndex: 1,
        data: front.map(p => [p.cutoff, p.el_rate]),
        symbolSize: 10,
        itemStyle: { color: '#f43f5e', shadowBlur: 6, shadowColor: 'rgba(244,63,94,0.3)' },
      },
    ],
    legend: { bottom: 0, textStyle: { color: '#94a3b8', fontSize: 12 } },
  }
})

const strategyLabelMap = {
  max_el: 'Lowest EL',
  max_utility: 'Max Utility',
  max_efficiency: 'Max Efficiency',
  pareto_optimal: 'Pareto Optimal',
  elbow: 'Elbow Point',
  conservative: 'Conservative',
  moderate: 'Moderate',
  aggressive: 'Aggressive',
}

const strategyPresets = ['conservative', 'moderate', 'aggressive', 'max_utility', 'pareto_optimal']
const filteredRecommendations = computed(() => {
  return recommendations.value.filter(r => strategyPresets.includes(r.strategy))
})
</script>

<template>
  <div v-loading="loading" element-loading-background="rgba(10, 14, 26, 0.8)">
    <div class="page-header animate-in">
      <div class="header-row">
        <div>
          <h2>策略模拟</h2>
          <p class="page-desc">Credit Policy Simulation & Optimization</p>
        </div>
        <button class="refresh-btn" @click="reload" :disabled="loading">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" :class="{ spinning: loading }"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        </button>
      </div>
    </div>

    <div class="stats-row" v-if="interactiveResult">
      <div v-for="(item, idx) in [
        { label: 'Approval Rate', value: (interactiveResult.approval_rate * 100).toFixed(1) + '%', color: 'blue' },
        { label: 'Approved', value: interactiveResult.approved_count?.toLocaleString(), color: 'cyan' },
        { label: 'Avg PD', value: (interactiveResult.avg_pd * 100).toFixed(2) + '%', color: 'violet' },
        { label: 'EL Rate', value: (interactiveResult.el_rate * 100).toFixed(2) + '%', color: 'rose' },
        { label: 'Actual Default', value: (interactiveResult.actual_default_rate * 100).toFixed(2) + '%', color: 'amber' },
        { label: 'Bad Capture', value: (interactiveResult.reject_bad_capture * 100).toFixed(1) + '%', color: 'emerald' },
      ]" :key="item.label" :class="['stat-card', item.color, 'animate-in']" :style="{ animationDelay: (idx * 0.05 + 0.1) + 's' }">
        <div class="stat-label">{{ item.label }}</div>
        <div class="stat-value">{{ item.value }}</div>
      </div>
    </div>

    <!-- Strategy Recommendation Cards -->
    <div class="strat-rec animate-in animate-in-delay-2" v-if="recommendations.length">
      <div class="strat-rec-header">
        <span class="bm-label">Quick Strategy Presets</span>
      </div>
      <div class="strat-rec-cards">
        <div v-for="s in filteredRecommendations" :key="s.strategy" :class="['strat-rec-card', s.strategy]" @click="onCutoffChange(s.cutoff)">
          <div class="src-name">{{ strategyLabelMap[s.strategy] || s.strategy }}</div>
          <div class="src-cut-off">Cut-off {{ s.cutoff?.toFixed(3) }}</div>
          <div class="src-metrics">
            <div class="src-metric">
              <span class="src-metric-val">{{ (s.approval_rate * 100).toFixed(1) }}%</span>
              <span class="src-metric-key">Approval</span>
            </div>
            <div class="src-metric">
              <span class="src-metric-val">{{ (s.el_rate * 100).toFixed(2) }}%</span>
              <span class="src-metric-key">EL Rate</span>
            </div>
            <div class="src-metric">
              <span class="src-metric-val">{{ (s.reject_bad_capture * 100).toFixed(1) }}%</span>
              <span class="src-metric-key">Bad Capture</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="slider-panel animate-in animate-in-delay-3">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
          <span>Real-time Cut-off Analysis</span>
        </div>
        <div class="cutoff-display">
          <span class="cutoff-label">Threshold</span>
          <span class="cutoff-value">{{ cutoff.toFixed(3) }}</span>
        </div>
      </div>
      <div class="slider-body">
        <el-slider
          :model-value="cutoff"
          @input="onCutoffChange"
          :min="0.01"
          :max="0.99"
          :step="0.005"
          :format-tooltip="(v) => v.toFixed(3)"
          show-input
        />
        <div class="progress-row" v-if="interactiveResult">
          <div class="progress-item">
            <div class="progress-header">
              <span class="progress-label">Approval Rate</span>
              <span class="progress-val">{{ (interactiveResult.approval_rate * 100).toFixed(1) }}%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill blue" :style="{ width: (interactiveResult.approval_rate * 100) + '%' }"></div></div>
          </div>
          <div class="progress-item">
            <div class="progress-header">
              <span class="progress-label">Good Retention</span>
              <span class="progress-val">{{ (interactiveResult.good_keep_rate * 100).toFixed(1) }}%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill emerald" :style="{ width: (interactiveResult.good_keep_rate * 100) + '%' }"></div></div>
          </div>
          <div class="progress-item">
            <div class="progress-header">
              <span class="progress-label">Bad Capture</span>
              <span class="progress-val">{{ (interactiveResult.reject_bad_capture * 100).toFixed(1) }}%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill amber" :style="{ width: (interactiveResult.reject_bad_capture * 100) + '%' }"></div></div>
          </div>
        </div>
      </div>
    </div>

    <div class="table-panel animate-in animate-in-delay-4" v-if="recommendations.length">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          <span>Multi-Strategy Recommendations</span>
        </div>
        <button class="export-btn" @click="exportStrategies">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          CSV
        </button>
      </div>
      <div class="table-wrap">
        <el-table :data="recommendations" :header-cell-style="{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-muted)', fontWeight: 500, borderBottom: '1px solid var(--border-subtle)' }" :cell-style="{ borderBottom: '1px solid var(--border-subtle)' }">
          <el-table-column label="Strategy" min-width="140">
            <template #default="{ row }">
              <span class="strategy-name">{{ strategyLabelMap[row.strategy] || row.strategy }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="cutoff" label="Cut-off" :formatter="(_, __, val) => typeof val === 'number' ? val.toFixed(4) : val" />
          <el-table-column prop="approval_rate" label="Approval" :formatter="(_, __, val) => typeof val === 'number' ? (val * 100).toFixed(1) + '%' : val" />
          <el-table-column prop="avg_pd" label="Avg PD" :formatter="(_, __, val) => typeof val === 'number' ? (val * 100).toFixed(2) + '%' : val" />
          <el-table-column prop="el_rate" label="EL Rate" :formatter="(_, __, val) => typeof val === 'number' ? (val * 100).toFixed(2) + '%' : val" />
          <el-table-column prop="reject_bad_capture" label="Bad Capture" :formatter="(_, __, val) => typeof val === 'number' ? (val * 100).toFixed(1) + '%' : val" />
          <el-table-column label="Action" width="100" fixed="right">
            <template #default="{ row }">
              <button class="apply-btn" @click="onCutoffChange(row.cutoff)">Apply</button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <div class="chart-panel animate-in animate-in-delay-5" v-if="policyData?.pareto_front?.length">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          <span>Pareto Front</span>
        </div>
      </div>
      <v-chart :option="paretoOption" style="height: 380px" autoresize />
    </div>
  </div>
</template>

<style scoped>
.page-header { margin-bottom: 24px; }
.header-row { display: flex; align-items: flex-start; justify-content: space-between; }
.page-header h2 { margin: 0; font-family: var(--font-display); font-size: 28px; }
.page-desc { color: var(--text-muted); font-size: 12px; margin-top: 6px; font-family: var(--font-mono); letter-spacing: 0.5px; }

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

.stats-row {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
}

.stat-card.blue::before { background: var(--gradient-blue); }
.stat-card.cyan::before { background: linear-gradient(135deg, #06b6d4, #22d3ee); }
.stat-card.violet::before { background: var(--gradient-violet); }
.stat-card.rose::before { background: var(--gradient-rose); }
.stat-card.amber::before { background: var(--gradient-amber); }
.stat-card.emerald::before { background: var(--gradient-emerald); }

.stat-card:hover { border-color: var(--border-accent); transform: translateY(-2px); }

.stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 500; }
.stat-value { font-size: 18px; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono); letter-spacing: -0.3px; }

.slider-panel, .table-panel, .chart-panel {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: 16px;
  backdrop-filter: blur(12px);
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

.cutoff-display {
  display: flex;
  align-items: center;
  gap: 8px;
}

.cutoff-label { font-size: 12px; color: var(--text-muted); }
.cutoff-value {
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 700;
  color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.1);
  padding: 4px 12px;
  border-radius: 6px;
}

.slider-body { padding: 20px 24px; }

.progress-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-top: 20px;
}

.progress-item { }

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.progress-label { font-size: 12px; color: var(--text-muted); }
.progress-val { font-family: var(--font-mono); font-size: 13px; font-weight: 600; color: var(--text-primary); }

.progress-bar {
  height: 6px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s ease;
}

.progress-fill.blue { background: var(--gradient-blue); }
.progress-fill.emerald { background: var(--gradient-emerald); }
.progress-fill.amber { background: var(--gradient-amber); }

.strategy-name {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 13px;
}

.apply-btn {
  padding: 5px 14px;
  border: 1px solid var(--accent-blue);
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 500;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.apply-btn:hover {
  background: var(--accent-blue);
  color: #fff;
}

:deep(.el-table) {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
}

:deep(.el-table__row:hover > td) {
  background: rgba(59, 130, 246, 0.05) !important;
}

:deep(.el-input__wrapper) {
  background: var(--bg-secondary) !important;
  box-shadow: 0 0 0 1px var(--border-subtle) inset !important;
}

:deep(.el-input__inner) {
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
}

:deep(.el-slider__runway) { background: rgba(255, 255, 255, 0.08) !important; }
:deep(.el-slider__bar) { background: var(--gradient-blue) !important; }
:deep(.el-slider__button) { border-color: var(--accent-blue) !important; width: 16px !important; height: 16px !important; }

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

/* ── Strategy Recommendation Cards ── */
.strat-rec {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  margin-bottom: 20px;
}
.strat-rec-header { margin-bottom: 12px; }
.bm-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }

.strat-rec-cards { display: flex; gap: 12px; }
.strat-rec-card {
  flex: 1;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  cursor: pointer;
  transition: all 0.2s;
}
.strat-rec-card:hover {
  border-color: rgba(200,170,110,0.3);
  background: rgba(200,170,110,0.03);
  transform: translateY(-1px);
}
.strat-rec-card.conservative { border-left: 3px solid var(--accent-emerald); }
.strat-rec-card.moderate { border-left: 3px solid var(--accent-gold); }
.strat-rec-card.aggressive { border-left: 3px solid var(--accent-blue); }
.strat-rec-card.max_utility { border-left: 3px solid var(--accent-violet, #a78bfa); }
.strat-rec-card.pareto_optimal { border-left: 3px solid var(--accent-amber); }

.src-name { font-size: 13px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }
.src-cut-off { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); margin-bottom: 10px; }
.src-metrics { display: flex; gap: 16px; }
.src-metric { display: flex; flex-direction: column; }
.src-metric-val { font-size: 14px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); }
.src-metric-key { font-size: 9px; color: var(--text-muted); text-transform: uppercase; }
</style>
