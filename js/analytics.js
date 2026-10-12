/**
 * RL Lab — Learn to Play Tic-Tac-Toe
 * Module: analytics.js
 * Purpose: High-DPI responsive Canvas chart for real learning curves,
 *          metric tracking, and performance visualizations with Light/Dark theme support
 *          and glowing area gradients.
 */

class AnalyticsVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.dataPoints = []; // Array of { episode, winRate, drawRate, lossRate }
    this.hoverIndex = -1;

    if (this.canvas) {
      this.setupCanvasDpi();
      this.attachListeners();
      this.drawEmpty();
    }
  }

  isDark() {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  }

  /**
   * Adjust canvas resolution for high-DPI / Retina displays without compounding scale.
   */
  setupCanvasDpi() {
    if (!this.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : 700;
    const height = rect.height > 0 ? rect.height : 290;

    this.canvas.width = Math.floor(width * dpr);
    this.canvas.height = Math.floor(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.displayWidth = width;
    this.displayHeight = height;
  }

  /**
   * Resize listener for responsive redesign.
   */
  resize() {
    this.setupCanvasDpi();
    this.render();
  }

  /**
   * Theme change hook.
   */
  onThemeChanged() {
    this.render();
  }

  /**
   * Hover tooltip listeners.
   */
  attachListeners() {
    if (!this.canvas) return;

    window.addEventListener('resize', () => this.resize());
    window.addEventListener('themeChanged', () => this.onThemeChanged());

    const handlePointerMove = (e) => {
      if (this.dataPoints.length === 0) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const margin = { left: 45, right: 25, top: 25, bottom: 35 };
      const chartW = this.displayWidth - margin.left - margin.right;

      if (x < margin.left || x > this.displayWidth - margin.right) {
        if (this.hoverIndex !== -1) {
          this.hoverIndex = -1;
          this.render();
        }
        return;
      }

      const fraction = (x - margin.left) / chartW;
      const closestIdx = Math.round(fraction * (this.dataPoints.length - 1));
      const clampedIdx = Math.max(0, Math.min(this.dataPoints.length - 1, closestIdx));

      if (this.hoverIndex !== clampedIdx) {
        this.hoverIndex = clampedIdx;
        this.render();
      }
    };

    this.canvas.addEventListener('mousemove', handlePointerMove);
    this.canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length > 0) {
        handlePointerMove(e.touches[0]);
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoverIndex = -1;
      this.render();
    });
  }

  /**
   * Update data and re-render.
   * @param {Array} history - Array of { episode, winRate, drawRate, lossRate }
   */
  updateData(history) {
    this.dataPoints = [...history];
    this.render();
  }

  /**
   * Clear chart data.
   */
  clear() {
    this.dataPoints = [];
    this.hoverIndex = -1;
    this.drawEmpty();
  }

  /**
   * Draw empty chart state before any training runs.
   */
  drawEmpty() {
    if (!this.ctx) return;
    const w = this.displayWidth || 700;
    const h = this.displayHeight || 290;

    this.ctx.clearRect(0, 0, w, h);
    this.drawGridAndAxes(w, h, 0, 100);

    const dark = this.isDark();
    this.ctx.fillStyle = dark ? '#94a3b8' : '#64748b';
    this.ctx.font = '500 13px Inter, -apple-system, sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('No training data yet. Click "🚀 Train Agent" to observe the live learning curve.', w / 2, h / 2);
  }

  /**
   * Draw grid lines, axis labels, and percentage markers.
   */
  drawGridAndAxes(w, h, minX = 0, maxX = 100) {
    const margin = { left: 45, right: 25, top: 25, bottom: 35 };
    const chartW = w - margin.left - margin.right;
    const chartH = h - margin.top - margin.bottom;
    const dark = this.isDark();

    const gridColor = dark ? 'rgba(30, 41, 59, 0.9)' : '#e2e8f0';
    const axisColor = dark ? '#94a3b8' : '#64748b';
    const baselineColor = dark ? '#334155' : '#cbd5e1';

    this.ctx.strokeStyle = gridColor;
    this.ctx.lineWidth = 1;
    this.ctx.fillStyle = axisColor;
    this.ctx.font = '500 11px JetBrains Mono, monospace';
    this.ctx.textAlign = 'right';

    // Y-axis grid (0%, 25%, 50%, 75%, 100%)
    const yTicks = [0, 25, 50, 75, 100];
    for (const val of yTicks) {
      const y = margin.top + chartH - (val / 100) * chartH;
      this.ctx.beginPath();
      this.ctx.moveTo(margin.left, y);
      this.ctx.lineTo(w - margin.right, y);
      this.ctx.stroke();
      this.ctx.fillText(`${val}%`, margin.left - 8, y + 4);
    }

    // X-axis baseline
    this.ctx.beginPath();
    this.ctx.strokeStyle = baselineColor;
    this.ctx.moveTo(margin.left, margin.top + chartH);
    this.ctx.lineTo(w - margin.right, margin.top + chartH);
    this.ctx.stroke();

    // X-axis label
    this.ctx.font = '500 11px Inter, sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText(`Training Episodes (${minX.toLocaleString()} → ${maxX.toLocaleString()})`, margin.left + chartW / 2, h - 8);

    return { margin, chartW, chartH };
  }

  /**
   * Render complete learning curve with glowing gradients.
   */
  render() {
    if (!this.ctx) return;
    const w = this.displayWidth;
    const h = this.displayHeight;

    this.ctx.clearRect(0, 0, w, h);

    if (this.dataPoints.length === 0) {
      this.drawEmpty();
      return;
    }

    const maxEp = Math.max(10, ...this.dataPoints.map(d => d.episode));
    const minEp = Math.min(...this.dataPoints.map(d => d.episode));
    const { margin, chartW, chartH } = this.drawGridAndAxes(w, h, minEp, maxEp);

    const getX = (idx) => {
      if (this.dataPoints.length <= 1) return margin.left + chartW / 2;
      return margin.left + (idx / (this.dataPoints.length - 1)) * chartW;
    };

    const getY = (rate) => {
      const clamped = Math.max(0, Math.min(100, rate));
      return margin.top + chartH - (clamped / 100) * chartH;
    };

    const dark = this.isDark();
    const tealColor = dark ? '#14b8a6' : '#0d9488';
    const blueColor = dark ? '#3b82f6' : '#2563eb';
    const redColor = dark ? '#f87171' : '#ef4444';

    // 1. Draw glowing gradient area fill under Win Rate curve
    if (this.dataPoints.length > 1) {
      const winGrad = this.ctx.createLinearGradient(0, margin.top, 0, margin.top + chartH);
      winGrad.addColorStop(0, dark ? 'rgba(20, 184, 166, 0.28)' : 'rgba(13, 148, 136, 0.18)');
      winGrad.addColorStop(1, 'rgba(13, 148, 136, 0.0)');

      this.ctx.beginPath();
      this.ctx.moveTo(getX(0), getY(this.dataPoints[0].winRate));
      for (let i = 1; i < this.dataPoints.length; i++) {
        this.ctx.lineTo(getX(i), getY(this.dataPoints[i].winRate));
      }
      this.ctx.lineTo(getX(this.dataPoints.length - 1), margin.top + chartH);
      this.ctx.lineTo(getX(0), margin.top + chartH);
      this.ctx.closePath();
      this.ctx.fillStyle = winGrad;
      this.ctx.fill();
    }

    // 2. Draw lines
    const series = [
      { key: 'winRate', color: tealColor, label: 'Win Rate' },
      { key: 'drawRate', color: blueColor, label: 'Draw Rate' },
      { key: 'lossRate', color: redColor, label: 'Loss Rate' }
    ];

    series.forEach(({ key, color }) => {
      this.ctx.beginPath();
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = 2.5;
      this.ctx.lineJoin = 'round';

      this.dataPoints.forEach((pt, idx) => {
        const x = getX(idx);
        const y = getY(pt[key]);
        if (idx === 0) this.ctx.moveTo(x, y);
        else this.ctx.lineTo(x, y);
      });
      this.ctx.stroke();

      // Draw point markers
      this.dataPoints.forEach((pt, idx) => {
        const x = getX(idx);
        const y = getY(pt[key]);
        this.ctx.beginPath();
        this.ctx.fillStyle = color;
        this.ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        this.ctx.fill();
      });
    });

    // 3. Draw active hover tooltip
    if (this.hoverIndex >= 0 && this.hoverIndex < this.dataPoints.length) {
      const pt = this.dataPoints[this.hoverIndex];
      const x = getX(this.hoverIndex);

      // Vertical guide
      this.ctx.beginPath();
      this.ctx.strokeStyle = dark ? '#64748b' : '#94a3b8';
      this.ctx.lineWidth = 1;
      this.ctx.setLineDash([3, 3]);
      this.ctx.moveTo(x, margin.top);
      this.ctx.lineTo(x, margin.top + chartH);
      this.ctx.stroke();
      this.ctx.setLineDash([]);

      // Tooltip Card
      const tooltipW = 148;
      const tooltipH = 82;
      let tipX = x + 12;
      if (tipX + tooltipW > w - 10) tipX = x - tooltipW - 12;
      const tipY = margin.top + 10;

      this.ctx.fillStyle = dark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(15, 23, 42, 0.92)';
      this.ctx.strokeStyle = dark ? '#334155' : 'transparent';
      this.ctx.lineWidth = 1;
      this.ctx.beginPath();
      this.ctx.roundRect(tipX, tipY, tooltipW, tooltipH, 8);
      this.ctx.fill();
      if (dark) this.ctx.stroke();

      this.ctx.textAlign = 'left';
      this.ctx.fillStyle = '#ffffff';
      this.ctx.font = 'bold 11px Inter, sans-serif';
      this.ctx.fillText(`Episode: ${pt.episode.toLocaleString()}`, tipX + 10, tipY + 18);

      this.ctx.font = '500 11px JetBrains Mono, monospace';
      this.ctx.fillStyle = '#5eead4';
      this.ctx.fillText(`Win:  ${pt.winRate}%`, tipX + 10, tipY + 36);

      this.ctx.fillStyle = '#93c5fd';
      this.ctx.fillText(`Draw: ${pt.drawRate}%`, tipX + 10, tipY + 52);

      this.ctx.fillStyle = '#fca5a5';
      this.ctx.fillText(`Loss: ${pt.lossRate}%`, tipX + 10, tipY + 68);
    }
  }
}

// Export for module and window environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { AnalyticsVisualizer };
} else {
  window.AnalyticsVisualizer = AnalyticsVisualizer;
}
