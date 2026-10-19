/**
 * Liquid Glass JS Integration for Sentix AI
 * Provides background WebGL refraction, optical caustics, and specular highlights
 * without adding unwanted UI control elements.
 */

(function () {
  'use strict';

  // Optimal Glass preset configurations
  const GLASS_PRESETS = {
    visionOS: {
      edgeIntensity: 0.012,
      rimIntensity: 0.065,
      baseIntensity: 0.012,
      edgeDistance: 0.16,
      rimDistance: 0.85,
      baseDistance: 0.11,
      cornerBoost: 0.025,
      rippleEffect: 0.08,
      blurRadius: 6.0,
      tintOpacity: 0.22,
      warp: false
    }
  };

  // Initialize default glass controls
  window.glassControls = window.glassControls || Object.assign({}, GLASS_PRESETS.visionOS);

  // Hook into theme toggling to recapture liquid glass snapshot smoothly
  function hookThemeObserver() {
    const themeSwitch = document.getElementById('theme-switch');
    if (themeSwitch) {
      themeSwitch.addEventListener('change', () => {
        setTimeout(() => {
          triggerGlassRecapture();
        }, 320);
      });
    }
  }

  // Recapture page snapshot for WebGL glass refraction
  function triggerGlassRecapture() {
    if (typeof Container !== 'undefined' && typeof html2canvas !== 'undefined') {
      try {
        Container.pageSnapshot = null;
        Container.isCapturing = true;
        Container.waitingForSnapshot = (Container.instances || []).slice();

        html2canvas(document.body, {
          scale: 1,
          useCORS: true,
          allowTaint: true,
          backgroundColor: null,
          ignoreElements: function (element) {
            return (
              element.classList.contains('glass-container') ||
              element.classList.contains('glass-button') ||
              element.classList.contains('glass-button-text') ||
              element.id === 'webcam-feed' ||
              element.id === 'webcam-canvas' ||
              element.tagName === 'VIDEO'
            );
          }
        }).then((snapshot) => {
          Container.pageSnapshot = snapshot;
          Container.isCapturing = false;

          const waiting = Container.waitingForSnapshot.slice();
          Container.waitingForSnapshot = [];

          waiting.forEach((inst) => {
            if (inst && !inst.webglInitialized && typeof inst.initWebGL === 'function') {
              inst.initWebGL();
            } else if (inst && typeof inst.setupShader === 'function') {
              const img = new Image();
              img.src = snapshot.toDataURL();
              img.onload = () => {
                inst.setupShader(img);
                if (typeof inst.render === 'function') inst.render();
              };
            }
          });
        }).catch((err) => {
          console.warn('Glass snapshot capture skipped:', err);
          Container.isCapturing = false;
        });
      } catch (err) {
        console.warn('Error in triggerGlassRecapture:', err);
      }
    }
  }

  // Interactive subtle mouse glass refraction shimmer effect
  function setupInteractiveGlassShine() {
    const container = document.querySelector('.container');
    if (!container) return;

    let rafId = null;
    let targetX = 50;
    let targetY = 50;
    let currentX = 50;
    let currentY = 50;

    function onMouseMove(e) {
      const rect = container.getBoundingClientRect();
      targetX = ((e.clientX - rect.left) / rect.width) * 100;
      targetY = ((e.clientY - rect.top) / rect.height) * 100;

      if (!rafId) {
        rafId = requestAnimationFrame(updateGlint);
      }
    }

    function updateGlint() {
      currentX += (targetX - currentX) * 0.08;
      currentY += (targetY - currentY) * 0.08;

      container.style.setProperty('--glass-glint-x', `${currentX.toFixed(2)}%`);
      container.style.setProperty('--glass-glint-y', `${currentY.toFixed(2)}%`);

      if (Math.abs(targetX - currentX) > 0.1 || Math.abs(targetY - currentY) > 0.1) {
        rafId = requestAnimationFrame(updateGlint);
      } else {
        rafId = null;
      }
    }

    container.addEventListener('mousemove', onMouseMove, { passive: true });
    container.addEventListener('mouseleave', () => {
      targetX = 50;
      targetY = 20;
      if (!rafId) rafId = requestAnimationFrame(updateGlint);
    });
  }

  // Initialize everything when DOM is ready
  function init() {
    hookThemeObserver();
    setupInteractiveGlassShine();

    // Trigger initial snapshot after DOM layout settles
    setTimeout(() => {
      triggerGlassRecapture();
    }, 600);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose helper API globally
  window.SentixLiquidGlass = {
    recapture: triggerGlassRecapture
  };

})();
