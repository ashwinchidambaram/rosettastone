/* ============================================================
   RosettaStone UI — Interaction JavaScript
   ============================================================ */

(function () {
    "use strict";

    /* ---- Theme Toggle ---- */
    const THEME_KEY = "rosettastone-theme";

    function getPreferredTheme() {
        const stored = localStorage.getItem(THEME_KEY);
        if (stored) return stored;
        return window.matchMedia("(prefers-color-scheme: light)").matches
            ? "light"
            : "dark";
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem(THEME_KEY, theme);
        // Update toggle icon if present
        const btn = document.getElementById("theme-toggle");
        if (btn) {
            const icon = btn.querySelector("[data-lucide]");
            if (icon) {
                icon.setAttribute("data-lucide", theme === "dark" ? "sun" : "moon");
                if (typeof lucide !== "undefined") lucide.createIcons();
            }
        }
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        applyTheme(current === "dark" ? "light" : "dark");
    }

    /* ---- Slide-over Panel ---- */
    function openSlideout() {
        const backdrop = document.getElementById("slideout-backdrop");
        const panel = document.getElementById("slideout-panel");
        if (backdrop) backdrop.classList.add("open");
        if (panel) panel.classList.add("open");
        document.body.classList.add("scroll-locked");
    }

    function closeSlideout() {
        const backdrop = document.getElementById("slideout-backdrop");
        const panel = document.getElementById("slideout-panel");
        if (backdrop) backdrop.classList.remove("open");
        if (panel) panel.classList.remove("open");
        document.body.classList.remove("scroll-locked");
    }

    /* ---- Collapsible Sections ---- */
    function initCollapsibles() {
        document.querySelectorAll(".collapsible-trigger").forEach(function (trigger) {
            trigger.addEventListener("click", function () {
                var expanded = this.getAttribute("aria-expanded") === "true";
                this.setAttribute("aria-expanded", String(!expanded));
                var targetId = this.getAttribute("data-target");
                var content = document.getElementById(targetId);
                if (content) {
                    if (expanded) {
                        content.classList.remove("expanded");
                    } else {
                        content.classList.add("expanded");
                    }
                }
            });
        });
    }

    /* ---- Export Dropdown ---- */
    function initExportDropdown() {
        var toggle = document.getElementById("export-toggle");
        var dropdown = document.getElementById("export-dropdown");
        if (!toggle || !dropdown) return;

        toggle.addEventListener("click", function (e) {
            e.stopPropagation();
            dropdown.classList.toggle("hidden");
        });

        document.addEventListener("click", function (e) {
            if (!dropdown.contains(e.target) && e.target !== toggle) {
                dropdown.classList.add("hidden");
            }
        });

        // Keyboard nav
        dropdown.addEventListener("keydown", function (e) {
            var items = dropdown.querySelectorAll("a, button");
            var idx = Array.prototype.indexOf.call(items, document.activeElement);
            if (e.key === "ArrowDown") {
                e.preventDefault();
                if (idx < items.length - 1) items[idx + 1].focus();
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                if (idx > 0) items[idx - 1].focus();
            } else if (e.key === "Escape") {
                dropdown.classList.add("hidden");
                toggle.focus();
            }
        });
    }

    /* ---- HTMX Event Hooks ---- */
    function initHtmxHooks() {
        document.body.addEventListener("htmx:afterSwap", function (evt) {
            // Reinitialize Lucide icons after HTMX swaps in new content
            if (typeof lucide !== "undefined") {
                lucide.createIcons();
            }
            // If the diff panel was swapped in, open the slideout
            if (evt.detail.target && evt.detail.target.id === "diff-panel") {
                openSlideout();
            }
            // Re-init collapsibles in swapped content
            initCollapsibles();
        });

        document.body.addEventListener("htmx:beforeSwap", function (evt) {
            // Cleanup: destroy any Chart.js instances in the target before swap
            if (evt.detail.target) {
                var canvases = evt.detail.target.querySelectorAll("canvas");
                canvases.forEach(function (canvas) {
                    var chart = Chart.getChart(canvas);
                    if (chart) chart.destroy();
                });
            }
        });
    }

    /* ---- Escape key handler ---- */
    function initEscapeHandler() {
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") {
                closeSlideout();
                // Also close export dropdown
                var dropdown = document.getElementById("export-dropdown");
                if (dropdown) dropdown.classList.add("hidden");
            }
        });
    }

    /* ---- DOMContentLoaded ---- */
    document.addEventListener("DOMContentLoaded", function () {
        // Apply saved theme
        applyTheme(getPreferredTheme());

        // Initialize Lucide icons
        if (typeof lucide !== "undefined") {
            lucide.createIcons();
        }

        // Wire theme toggle
        var themeBtn = document.getElementById("theme-toggle");
        if (themeBtn) {
            themeBtn.addEventListener("click", toggleTheme);
        }

        // Wire slideout close
        var backdrop = document.getElementById("slideout-backdrop");
        if (backdrop) {
            backdrop.addEventListener("click", closeSlideout);
        }
        var closeBtn = document.getElementById("slideout-close");
        if (closeBtn) {
            closeBtn.addEventListener("click", closeSlideout);
        }

        // Init components
        initCollapsibles();
        initExportDropdown();
        initHtmxHooks();
        initEscapeHandler();
    });

    // Expose for HTMX event handlers
    window.rsOpenSlideout = openSlideout;
    window.rsCloseSlideout = closeSlideout;
})();
