/**
 * app.js — Shared interaction JS for RosettaStone Stitch UI
 *
 * - Theme toggle (data-theme + localStorage + prefers-color-scheme)
 * - Slide-over panel (open/close/animate/backdrop/Escape/scroll lock)
 * - Collapsible sections (aria-expanded + chevron)
 * - Export dropdown (click-open / click-outside-close)
 */

(function () {
    "use strict";

    /* ─── Theme Toggle ───────────────────────────────────────── */
    var THEME_KEY = "rosettastone-theme";

    function getStoredTheme() {
        return localStorage.getItem(THEME_KEY);
    }

    function getPreferredTheme() {
        var stored = getStoredTheme();
        if (stored) return stored;
        return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        if (theme === "dark") {
            document.documentElement.classList.add("dark");
        } else {
            document.documentElement.classList.remove("dark");
        }
    }

    // Apply on load
    applyTheme(getPreferredTheme());

    // Listen for OS changes
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
        if (!getStoredTheme()) {
            applyTheme(e.matches ? "dark" : "light");
        }
    });

    // Bind toggle buttons
    document.addEventListener("click", function (e) {
        var btn = e.target.closest("[data-action='toggle-theme']");
        if (!btn) return;
        var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        localStorage.setItem(THEME_KEY, next);
        applyTheme(next);
    });

    /* ─── Slide-over Panel ────────────────────────────────────── */
    function openSlideout() {
        var panel = document.getElementById("diff-panel");
        var backdrop = document.getElementById("diff-backdrop");
        if (!panel || !backdrop) return;
        backdrop.classList.remove("hidden");
        panel.classList.remove("hidden");
        // Force reflow before adding open class
        void panel.offsetWidth;
        panel.classList.remove("closed");
        panel.classList.add("open");
        document.body.style.overflow = "hidden";
    }

    function closeSlideout() {
        var panel = document.getElementById("diff-panel");
        var backdrop = document.getElementById("diff-backdrop");
        if (!panel || !backdrop) return;
        panel.classList.remove("open");
        panel.classList.add("closed");
        setTimeout(function () {
            backdrop.classList.add("hidden");
            panel.classList.add("hidden");
        }, 300);
        document.body.style.overflow = "";
    }

    // Close on backdrop click
    document.addEventListener("click", function (e) {
        if (e.target && e.target.id === "diff-backdrop") {
            closeSlideout();
        }
    });

    // Close button inside panel
    document.addEventListener("click", function (e) {
        var btn = e.target.closest("[data-action='close-slideout']");
        if (btn) closeSlideout();
    });

    // Escape key
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeSlideout();
    });

    // Listen for HTMX afterSwap on the diff panel to auto-open
    document.addEventListener("htmx:afterSwap", function (e) {
        if (e.detail.target && e.detail.target.id === "diff-content") {
            openSlideout();
        }
    });

    // Expose for inline onclick if needed
    window.openSlideout = openSlideout;
    window.closeSlideout = closeSlideout;

    /* ─── Mobile Navigation Drawer ────────────────────────────── */
    function openMobileNav() {
        var drawer = document.getElementById("mobile-nav-drawer");
        var backdrop = document.getElementById("mobile-nav-backdrop");
        var btn = document.getElementById("mobile-menu-btn");
        if (!drawer || !backdrop) return;
        drawer.classList.remove("hidden");
        backdrop.classList.remove("hidden");
        // Force reflow then animate in
        void drawer.offsetWidth;
        drawer.style.transform = "translateX(0)";
        if (btn) btn.setAttribute("aria-expanded", "true");
        document.body.style.overflow = "hidden";
    }

    function closeMobileNav() {
        var drawer = document.getElementById("mobile-nav-drawer");
        var backdrop = document.getElementById("mobile-nav-backdrop");
        var btn = document.getElementById("mobile-menu-btn");
        if (!drawer || !backdrop) return;
        drawer.style.transform = "translateX(100%)";
        if (btn) btn.setAttribute("aria-expanded", "false");
        setTimeout(function () {
            drawer.classList.add("hidden");
            backdrop.classList.add("hidden");
            // Return focus to hamburger menu button after drawer closes
            if (btn) btn.focus();
        }, 300);
        document.body.style.overflow = "";
    }

    // Toggle button
    document.addEventListener("click", function (e) {
        if (e.target.closest("[data-action='toggle-mobile-nav']")) {
            var drawer = document.getElementById("mobile-nav-drawer");
            if (drawer && drawer.classList.contains("hidden")) {
                openMobileNav();
            } else {
                closeMobileNav();
            }
        }
    });

    // Close button
    document.addEventListener("click", function (e) {
        if (e.target.closest("[data-action='close-mobile-nav']")) {
            closeMobileNav();
        }
    });

    // Close on backdrop click
    document.addEventListener("click", function (e) {
        if (e.target && e.target.id === "mobile-nav-backdrop") {
            closeMobileNav();
        }
    });

    // Close on link click
    document.addEventListener("click", function (e) {
        if (e.target.closest(".mobile-nav-link")) {
            closeMobileNav();
        }
    });

    // Escape key closes mobile nav too
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
            var drawer = document.getElementById("mobile-nav-drawer");
            if (drawer && !drawer.classList.contains("hidden")) {
                closeMobileNav();
            }
        }
    });

    /* ─── CSRF Token for HTMX ────────────────────────────────── */
    document.addEventListener("htmx:configRequest", function (e) {
        var meta = document.querySelector('meta[name="csrf-token"]');
        if (meta) {
            e.detail.headers["X-CSRF-Token"] = meta.getAttribute("content");
        }
    });

    /* ─── Collapsible Sections ────────────────────────────────── */
    document.addEventListener("click", function (e) {
        var trigger = e.target.closest("[data-action='toggle-collapse']");
        if (!trigger) return;
        var targetId = trigger.getAttribute("data-target");
        var content = document.getElementById(targetId);
        if (!content) return;

        var expanded = trigger.getAttribute("aria-expanded") === "true";
        trigger.setAttribute("aria-expanded", String(!expanded));
        content.classList.toggle("expanded");

        // Rotate chevron icon if present
        var chevron = trigger.querySelector(".collapse-chevron");
        if (chevron) {
            chevron.style.transform = expanded ? "rotate(0deg)" : "rotate(180deg)";
        }
    });

    /* ─── Export Dropdown ─────────────────────────────────────── */
    document.addEventListener("click", function (e) {
        var trigger = e.target.closest("[data-action='toggle-export']");
        if (trigger) {
            var dropdown = document.getElementById("export-dropdown");
            if (dropdown) dropdown.classList.toggle("hidden");
            return;
        }

        // Click outside — close
        var dropdown = document.getElementById("export-dropdown");
        if (dropdown && !dropdown.contains(e.target)) {
            dropdown.classList.add("hidden");
        }
    });

    /* ─── Settings Link ───────────────────────────────────────── */
    document.addEventListener("click", function (e) {
        if (e.target.closest("[data-action='settings-link']")) {
            window.location.href = "/ui/";
        }
    });

    /* ─── Account Menu Dropdown ───────────────────────────────── */
    document.addEventListener("click", function (e) {
        var trigger = e.target.closest("[data-action='toggle-account-menu']");
        if (trigger) {
            var dropdown = document.getElementById("account-dropdown");
            if (!dropdown) return;
            var isHidden = dropdown.classList.contains("hidden");
            dropdown.classList.toggle("hidden");
            trigger.setAttribute("aria-expanded", String(isHidden));
            return;
        }

        // Click outside — close
        var dropdown = document.getElementById("account-dropdown");
        var btn = document.querySelector("[data-action='toggle-account-menu']");
        if (dropdown && !dropdown.classList.contains("hidden") &&
            !dropdown.contains(e.target) && e.target !== btn && (!btn || !btn.contains(e.target))) {
            dropdown.classList.add("hidden");
            if (btn) btn.setAttribute("aria-expanded", "false");
        }
    });

    /* ─── Add Model Row ───────────────────────────────────────── */
    document.addEventListener("click", function (e) {
        if (!e.target.closest("[data-action='add-model-row']")) return;
        var rows = document.querySelectorAll(".model-input-row");
        if (!rows.length) return;
        var lastRow = rows[rows.length - 1];
        var clone = lastRow.cloneNode(true);
        // Clear input values in the clone
        clone.querySelectorAll("input").forEach(function (inp) { inp.value = ""; });
        lastRow.parentNode.insertBefore(clone, lastRow.nextSibling);
    });

    /* ─── Period Dropdown ─────────────────────────────────────── */
    document.addEventListener("click", function (e) {
        var trigger = e.target.closest("[data-action='toggle-period-dropdown']");
        if (trigger) {
            var dropdown = document.getElementById("period-dropdown");
            if (dropdown) dropdown.classList.toggle("hidden");
            return;
        }

        // Click outside — close
        var dropdown = document.getElementById("period-dropdown");
        if (dropdown && !dropdown.classList.contains("hidden") && !dropdown.contains(e.target)) {
            dropdown.classList.add("hidden");
        }
    });

    /* ─── Toast Notification ─────────────────────────────────── */
    window.showToast = function (message, duration) {
        duration = duration || 2500;
        var toast = document.createElement("div");
        toast.className = "fixed bottom-6 right-6 z-[300] px-5 py-3 bg-surface-container-highest border border-outline-variant/30 rounded-lg shadow-xl text-sm text-on-surface font-medium transition-all";
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(function () {
            toast.style.opacity = "0";
            setTimeout(function () { toast.remove(); }, 300);
        }, duration);
    };
})();
