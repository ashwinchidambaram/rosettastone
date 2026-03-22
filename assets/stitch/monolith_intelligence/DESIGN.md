# Design System Specification: The Celestial Monolith

## 1. Overview & Creative North Star
**Creative North Star: The Celestial Monolith**
This design system is built to feel like an advanced instrument discovered in deep space: silent, authoritative, and impossibly precise. We are moving away from the "busy" clutter of traditional developer tools. Instead, we embrace **Space-Forward Minimalism**—a philosophy where generous whitespace isn’t "empty," but represents the computational room required for high-level model intelligence.

The experience is defined by **chromatic tension**—the warmth of Terracotta and Sand set against the cold, vastness of Space Black. By utilizing intentional asymmetry and radical "Answer First" information architecture, we ensure the user is never hunting for data, but rather witnessing a revelation.

---

## 2. Colors & Surface Philosophy
We do not use color to decorate; we use it to signify state and depth. The palette is anchored in high-contrast legibility and tonal warmth.

### The "No-Line" Rule
Traditional UI relies on 1px borders to separate ideas. In this system, **borders are forbidden** for sectioning. We define boundaries through background color shifts. A `surface-container-low` section sitting on a `surface` background provides all the structural definition required without the visual "noise" of a line.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. We use Material-style tiers to define importance:
- **Background (`#131313`):** The deep space layer.
- **Surface Container Low (`#1c1b1b`):** The primary staging area for content.
- **Surface Container High (`#2a2a2a`):** Interactive or elevated content.
- **Surface Bright (`#393939`):** Elements that require immediate visual "pop."

### The Glass & Gradient Rule
To provide "soul" to the developer-focused experience:
- **The Top Bar (72px):** Must use a backdrop-blur (20px) with a semi-transparent `surface` color to allow content to bleed through as it scrolls, creating a sense of layered environment.
- **Primary CTAs:** Should utilize a subtle linear gradient from `primary` (#D4745E) to `primary-container` (#D67660) at a 135-degree angle. This prevents the "flat-toy" look of standard buttons.

---

## 3. Typography: Editorial Authority
We pair the utilitarian precision of **Inter** with the structural rigor of **JetBrains Mono**.

- **Display & Headlines (Inter):** Used for the "Answer First" principle. These should be bold, high-contrast, and carry the most visual weight.
- **Monospaced Data (JetBrains Mono):** Used for all model outputs, code snippets, and metadata. This signals to the developer that they are looking at "raw truth."

| Role | Font | Size | Intent |
| :--- | :--- | :--- | :--- |
| `display-lg` | Inter | 3.5rem | Heroic model conclusions. |
| `headline-md` | Inter | 1.75rem | Section headers. |
| `body-md` | Inter | 0.875rem | Standard UI text and descriptions. |
| `label-sm` | JetBrains | 0.6875rem | Technical metadata and timestamps. |

---

## 4. Elevation & Depth: Tonal Layering
We reject heavy, muddy drop shadows. Depth is achieved through light and layering.

- **The Layering Principle:** Place a `surface-container-lowest` card on a `surface-container-low` section. The subtle shift from `#0e0e0e` to `#1c1b1b` creates a sophisticated "lift."
- **Ambient Shadows:** For floating modals, use a large, 64px blur shadow with only 6% opacity. The shadow color should be tinted with `on-surface` (#E5E2E1) rather than pure black to simulate natural light refraction.
- **The Ghost Border:** If a boundary is strictly required for accessibility, use the `outline-variant` token at **15% opacity**. It should be felt, not seen.

---

## 5. Components

### The "Answer First" Card
Every layout must present the conclusion immediately.
- **The Answer (Top):** Set in `headline-sm` or `title-lg`.
- **The Evidence (Bottom):** A nested container using `surface-container-lowest` containing charts or JetBrains Mono logs.
- **Spacing:** Use `16` (5.5rem) padding between the Answer and the Evidence to provide "breathing room."

### Buttons & Interaction
- **Primary:** 8px radius (`md`). Solid `primary` fill or subtle gradient. No border.
- **Secondary:** Transparent fill with a `Ghost Border`. Text set in `label-md`.
- **Tertiary:** Text only, using `primary` color, strictly for low-priority actions.

### Chips & Status
- **Success/Error Chips:** Use a 10% opacity fill of the status color (`success` or `error`) with a solid `outline` of the same color at 30% opacity. This creates a "glow" effect suitable for a space-forward theme.

### Inputs
- **Text Fields:** Use `surface-container-highest` (#353535) for the field background. The active state is signaled by a 1px `primary` bottom-border only, rather than a full box stroke. This maintains the "Editorial" feel.

---

## 6. Do’s and Don’ts

### Do
- **Use extreme whitespace.** If you think there is enough space, add 20% more.
- **Align to the grid, but break it intentionally.** Let a hero image or a code block bleed 40px off the standard container margin to create visual tension.
- **Use Lucide icons at 2px stroke.** Keep them consistent at `24px` size within a `40px` touch target.

### Don’t
- **Don't use dividers.** Use the `spacing-8` (2.75rem) or background color shifts to separate content.
- **Don't use pure black (#000000).** It kills the "Space Forward" depth. Use `surface-container-lowest` (#0e0e0e).
- **Don't use "standard" blue for info.** Use the specified `Info 400` (#7A9FB5) to maintain the muted, premium palette.

---

## 7. Spacing Scale
Follow the 8px base unit strictly for alignment, but use the upper end of the scale for layout.

- **Component Internal Padding:** `2` (0.7rem) or `3` (1rem).
- **Section Gaps:** `8` (2.75rem) to `12` (4rem).
- **Hero Margins:** `16` (5.5rem) or `20` (7rem).

This system is designed to be a quiet partner to the user's intelligence—providing the "Answer" with clarity and the "Evidence" with sophisticated depth.