<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const isCollapse = ref(false)

const menuItems = [
  { path: '/', icon: 'grid', title: 'Overview', desc: 'Dashboard' },
  { path: '/model', icon: 'trending-up', title: 'Models', desc: 'Evaluation' },
  { path: '/features', icon: 'layers', title: 'Features', desc: 'Analysis' },
  { path: '/scorecard', icon: 'score', title: 'Scorecard', desc: 'PD to Score' },
  { path: '/policy', icon: 'sliders', title: 'Policy', desc: 'Simulation' },
  { path: '/monitoring', icon: 'activity', title: 'Stability', desc: 'PSI Monitor' },
]

const activeMenu = computed(() => router.currentRoute.value.path)
</script>

<template>
  <div class="layout">
    <aside :class="['sidebar', { collapsed: isCollapse }]">
      <div class="sidebar-inner">
        <!-- Logo -->
        <div class="logo" @click="isCollapse = !isCollapse">
          <div class="logo-mark">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="9" fill="url(#logo-grad)" />
              <path d="M10 16L14 12L18 16L14 20Z" fill="white" opacity="0.95" />
              <path d="M14 12L18 16L22 12" stroke="white" stroke-width="1.5" stroke-linecap="round" opacity="0.5" />
              <path d="M14 20L18 16L22 20" stroke="white" stroke-width="1.5" stroke-linecap="round" opacity="0.5" />
              <defs>
                <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32">
                  <stop stop-color="#c8aa6e" />
                  <stop offset="1" stop-color="#8b7340" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <transition name="fade">
            <div v-if="!isCollapse" class="logo-text">
              <span class="logo-title">Home Credit</span>
              <span class="logo-sub">Risk Analytics</span>
            </div>
          </transition>
        </div>

        <!-- Navigation -->
        <nav class="nav-menu">
          <router-link
            v-for="item in menuItems"
            :key="item.path"
            :to="item.path"
            :class="['nav-item', { active: activeMenu === item.path }]"
          >
            <div class="nav-icon">
              <svg v-if="item.icon === 'grid'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/></svg>
              <svg v-if="item.icon === 'trending-up'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
              <svg v-if="item.icon === 'layers'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
              <svg v-if="item.icon === 'score'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
              <svg v-if="item.icon === 'sliders'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
              <svg v-if="item.icon === 'activity'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            </div>
            <transition name="fade">
              <div v-if="!isCollapse" class="nav-text">
                <span class="nav-title">{{ item.title }}</span>
                <span class="nav-desc">{{ item.desc }}</span>
              </div>
            </transition>
            <div v-if="activeMenu === item.path" class="active-bar"></div>
          </router-link>
        </nav>

        <!-- Footer -->
        <div class="sidebar-footer" v-if="!isCollapse">
          <div class="version-badge">
            <span class="version-dot"></span>
            <span>v2.0</span>
          </div>
        </div>
      </div>
    </aside>

    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="page" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ── Sidebar ── */
.sidebar {
  width: 220px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-subtle);
  transition: width 0.35s cubic-bezier(0.4, 0, 0.2, 1);
  flex-shrink: 0;
  position: relative;
  z-index: 10;
}

.sidebar::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 1px;
  background: linear-gradient(180deg, rgba(200, 170, 110, 0.1) 0%, transparent 50%, rgba(200, 170, 110, 0.05) 100%);
}

.sidebar.collapsed { width: 64px; }

.sidebar-inner {
  display: flex;
  flex-direction: column;
  height: 100%;
}

/* ── Logo ── */
.logo {
  height: 64px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 18px;
  cursor: pointer;
  border-bottom: 1px solid var(--border-subtle);
  transition: all 0.2s;
}

.sidebar.collapsed .logo { justify-content: center; padding: 0; }
.logo:hover { background: rgba(255, 255, 255, 0.02); }

.logo-mark { flex-shrink: 0; }

.logo-text {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.logo-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.3px;
  white-space: nowrap;
}

.logo-sub {
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  white-space: nowrap;
  font-weight: 500;
}

/* ── Navigation ── */
.nav-menu {
  flex: 1;
  padding: 12px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  text-decoration: none;
  color: var(--text-secondary);
  transition: all 0.2s;
  position: relative;
  overflow: hidden;
}

.sidebar.collapsed .nav-item { justify-content: center; padding: 10px; }

.nav-item:hover {
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-primary);
}

.nav-item.active {
  background: rgba(200, 170, 110, 0.06);
  color: var(--accent-gold);
}

.active-bar {
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 2px;
  height: 18px;
  background: var(--accent-gold);
  border-radius: 0 2px 2px 0;
}

.nav-icon {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
}

.nav-text {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
}

.nav-title {
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
}

.nav-desc {
  font-size: 10px;
  color: var(--text-muted);
  white-space: nowrap;
  margin-top: 1px;
  letter-spacing: 0.3px;
}

/* ── Footer ── */
.sidebar-footer {
  padding: 14px 18px;
  border-top: 1px solid var(--border-subtle);
}

.version-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.version-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent-emerald);
  animation: breathe 3s ease infinite;
}

/* ── Main Content ── */
.main-content {
  flex: 1;
  padding: 28px 32px;
  background: var(--bg-void);
  overflow-y: auto;
  overflow-x: hidden;
}

/* ── Transitions ── */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.2s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

.page-enter-active {
  animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.page-leave-active {
  animation: fadeIn 0.15s ease reverse both;
}
</style>
