import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'overview', component: () => import('../views/Overview.vue'), meta: { title: '概览' } },
  { path: '/model', name: 'model', component: () => import('../views/ModelEvaluation.vue'), meta: { title: '模型评估' } },
  { path: '/features', name: 'features', component: () => import('../views/FeatureAnalysis.vue'), meta: { title: '特征分析' } },
  { path: '/scorecard', name: 'scorecard', component: () => import('../views/Scorecard.vue'), meta: { title: '评分卡' } },
  { path: '/policy', name: 'policy', component: () => import('../views/PolicySimulation.vue'), meta: { title: '策略模拟' } },
  { path: '/monitoring', name: 'monitoring', component: () => import('../views/Monitoring.vue'), meta: { title: '稳定性监控' } },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
