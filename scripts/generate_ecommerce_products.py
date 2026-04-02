#!/usr/bin/env python3
"""
Generate ecommerce product description dataset for RosettaStone benchmarking.

Produces 300 synthetic product specs across 6 categories, then collects
structured marketing descriptions from GPT-4o and Haiku.

Usage:
    uv run python scripts/generate_ecommerce_products.py
"""

import json
import os
import re
import sys
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path("/Users/ashwinchidambaram/dev/projects/rosettastone/.env")
OUTPUT_DIR = ROOT / "examples" / "datasets" / "ecommerce_products"

RANDOM_SEED = 20260401
TOTAL_PRODUCTS = 300
CATEGORY_COUNTS = {
    "electronics": 60,
    "kitchen": 60,
    "outdoor_fitness": 50,
    "fashion": 50,
    "complex_products": 40,
    "sparse_input": 40,
}

SUPERLATIVES_PATTERN = re.compile(
    r"\b(best|amazing|revolutionary|unparalleled|incredible|unmatched|superior|"
    r"ultimate|extraordinary|unbeatable|world-class|industry-leading)\b",
    re.IGNORECASE,
)

TUNING_SAMPLE_SIZE = 30
TUNING_PASS_RATE = 0.90

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert product copywriter for an ecommerce platform. You write "
    "concise, factual product descriptions.\n\n"
    "RULES -- follow every one exactly:\n"
    "1. The description MUST be 80-150 words total. This is a hard requirement. "
    "Even if the product has few specs, write a full 80-150 word description by "
    "elaborating on the stated features with helpful context. Never go below 80 words.\n"
    "2. Start with one opening sentence that names the product and its category.\n"
    "3. Follow with exactly 3-5 bullet points (use \"- \" to start each bullet). "
    "Each bullet should be a complete thought of at least 8 words.\n"
    "4. NEVER use these words: best, amazing, revolutionary, unparalleled, "
    "incredible, unmatched, superior, ultimate, extraordinary, unbeatable, "
    "world-class, industry-leading.\n"
    "5. End with exactly one sentence starting with \"Perfect for \" that "
    "identifies the target customer.\n"
    "6. ONLY mention features, specs, and attributes that are explicitly listed "
    "in the product specification below. Do NOT invent or assume any feature "
    "that is not stated. You may describe the given features in more detail, "
    "but do not add features that were not provided.\n"
    "7. Keep the tone professional and informative.\n\n"
    "STRUCTURE:\n"
    "<opening sentence>\n\n"
    "- <bullet 1>\n"
    "- <bullet 2>\n"
    "- <bullet 3>\n"
    "[optionally bullet 4 and 5]\n\n"
    "Perfect for <target customer description>."
)


def build_user_prompt(spec: dict) -> str:
    """Format a product spec into the user message."""
    lines = [f"Product: {spec['name']}", f"Category: {spec['category']}"]
    if spec.get("subcategory"):
        lines.append(f"Subcategory: {spec['subcategory']}")
    lines.append(f"Features/Specs: {spec['features']}")
    lines.append(f"Target customer: {spec['target']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_description(text: str, input_features: str) -> dict:
    """Validate a generated description against quality requirements."""
    words = len(text.split())
    bullet_lines = re.findall(r"(?:^|\n)\s*[-*]\s+.+", text)
    bullet_count = len(bullet_lines)
    ends_with_perfect = bool(
        re.search(r"Perfect for .+\.?\s*$", text, re.IGNORECASE)
    )
    no_superlatives = not bool(SUPERLATIVES_PATTERN.search(text))

    valid = (
        80 <= words <= 150
        and 3 <= bullet_count <= 5
        and ends_with_perfect
        and no_superlatives
    )
    return {
        "valid": valid,
        "word_count": words,
        "bullet_count": bullet_count,
        "ends_with_perfect": ends_with_perfect,
        "no_superlatives": no_superlatives,
    }


# ---------------------------------------------------------------------------
# Product spec generators
# ---------------------------------------------------------------------------


def generate_electronics(fake: Faker, rng: random.Random, count: int) -> list[dict]:
    """Generate electronics product specs with measurable technical attributes."""
    products = []

    audio_products = [
        {
            "name": "ProSound ANC Wireless Earbuds",
            "subcategory": "audio",
            "features": "Driver size: 10mm dynamic | Frequency: 20Hz-20kHz | Battery: 8h (32h with case) | Connectivity: Bluetooth 5.3 | ANC: Active noise cancellation (-30dB) | Charging: USB-C, 10min=1h | Water resistance: IPX5",
            "target": "daily commuters and remote workers",
        },
        {
            "name": "StudioFlex Over-Ear Headphones",
            "subcategory": "audio",
            "features": "Driver size: 40mm planar magnetic | Frequency: 10Hz-40kHz | Impedance: 32 ohms | Cable: Detachable 3.5mm + 6.35mm adapter | Ear cups: Memory foam, 90-degree swivel | Weight: 265g | Connectivity: Wired only",
            "target": "audio engineers and music producers who need flat reference sound",
        },
        {
            "name": "BassDrop Portable Speaker",
            "subcategory": "audio",
            "features": "Output: 20W (2x10W stereo) | Battery: 12h at 50% volume | Bluetooth: 5.2 with multipoint | Water resistance: IP67 | Size: 180mm x 75mm | Weight: 540g | Aux input: 3.5mm | Speakerphone: Built-in mic",
            "target": "outdoor gatherings and poolside listening",
        },
        {
            "name": "ClearVoice USB Condenser Microphone",
            "subcategory": "audio",
            "features": "Capsule: 16mm condenser | Polar pattern: Cardioid | Frequency: 20Hz-20kHz | Sample rate: 48kHz/16-bit | Connectivity: USB-C plug-and-play | Monitoring: 3.5mm zero-latency headphone out | Gain: Hardware dial | Mount: Integrated desk stand + 5/8-inch thread",
            "target": "podcasters, streamers, and video call professionals",
        },
        {
            "name": "SoundBar Slim Home Theater System",
            "subcategory": "audio",
            "features": "Channels: 2.1 with wireless subwoofer | Total output: 150W | Connectivity: HDMI eARC, optical, Bluetooth 5.0 | Dimensions: 880mm x 58mm x 85mm | Subwoofer: 6.5-inch driver, 60W | Decoding: Dolby Digital, DTS | Wall-mountable: Yes",
            "target": "apartment dwellers wanting theater-quality sound without a full surround system",
        },
        {
            "name": "MelodyBuds Kids Wireless Headphones",
            "subcategory": "audio",
            "features": "Volume limit: 85dB maximum | Driver: 30mm | Battery: 24h | Connectivity: Bluetooth 5.1 | Weight: 150g | Ear cushions: Hypoallergenic foam | Foldable design | Built-in mic for calls",
            "target": "parents seeking hearing-safe headphones for children ages 4-12",
        },
        {
            "name": "VinylStream Turntable with Bluetooth",
            "subcategory": "audio",
            "features": "Speeds: 33-1/3 and 45 RPM | Cartridge: Audio-Technica AT3600L pre-installed | Output: Bluetooth 5.0 + RCA line out | Platter: Die-cast aluminum | Motor: Belt-drive DC | Built-in preamp: Switchable phono/line | Dust cover: Hinged acrylic",
            "target": "vinyl enthusiasts who want to stream records to wireless speakers",
        },
        {
            "name": "QuietSpace Noise-Masking Sleep Earbuds",
            "subcategory": "audio",
            "features": "Driver: 6mm micro | Battery: 10h continuous | Bluetooth: 5.3 LE | Profile: 5mm protrusion for side-sleeping | Weight: 3.2g per bud | Sounds: 20 built-in soundscapes | Alarm: Vibration + gentle tone | Charging: Wireless case",
            "target": "light sleepers and travelers in noisy environments",
        },
        {
            "name": "BoomBox Retro Bluetooth Speaker",
            "subcategory": "audio",
            "features": "Output: 40W (2x15W + 10W woofer) | Battery: 8h | Bluetooth: 5.0 | FM Radio: Built-in | USB playback: MP3/WAV | Handle: Integrated carry | Weight: 2.4kg | LED: Front-panel ambient lights",
            "target": "outdoor party hosts who enjoy retro aesthetics",
        },
        {
            "name": "AudioLink Wireless DAC/Amp",
            "subcategory": "audio",
            "features": "DAC chip: ESS ES9038Q2M | Output power: 250mW at 32 ohms | Bluetooth: 5.2 with LDAC, aptX Adaptive | Wired input: USB-C | Battery: 12h | Impedance support: 8-300 ohms | THD+N: <0.0005% | Weight: 75g",
            "target": "audiophiles who want wireless convenience with wired-quality sound from any headphone",
        },
        {
            "name": "PodStudio XR Recording Interface",
            "subcategory": "audio",
            "features": "Inputs: 2x XLR/TRS combo | Phantom power: 48V | Preamp gain: 60dB | Sample rate: up to 192kHz/24-bit | Connectivity: USB-C | Latency: <1ms direct monitoring | Headphone outs: 2 independent | Software: Includes lite DAW license",
            "target": "home studio musicians and podcast duos recording multi-track audio",
        },
        {
            "name": "EarGuard Pro Hearing Protection Earbuds",
            "subcategory": "audio",
            "features": "Noise reduction: NRR 22dB passive + electronic limiting at 82dB | Frequency: 20Hz-16kHz | Battery: 30h | Connectivity: Bluetooth 5.3 | Ambient pass-through: Adjustable | IP rating: IP54 | Ear tips: 5 sizes included",
            "target": "concert-goers, construction workers, and motorsport enthusiasts",
        },
    ]

    peripherals = [
        {
            "name": "TypeForce TKL Mechanical Keyboard",
            "subcategory": "computing_peripherals",
            "features": "Switch type: Cherry MX Red linear | Layout: 75% TKL | Backlighting: Per-key RGB | Connectivity: USB-C + Bluetooth 5.0 dual-mode | Battery: 4000mAh (wireless mode ~200h) | Build: Aluminum top case, PBT keycaps | Hot-swappable: Yes",
            "target": "programmers and gamers who value typing feel and desk space",
        },
        {
            "name": "GlideTrack Wireless Ergonomic Mouse",
            "subcategory": "computing_peripherals",
            "features": "Sensor: PAW3395 optical | DPI: 100-26000 adjustable | Buttons: 6 programmable | Connectivity: 2.4GHz dongle + Bluetooth 5.1 | Battery: 70h (2.4GHz), 90h (BT) | Weight: 68g | Grip: Right-hand ergonomic, 57-degree vertical tilt | Charging: USB-C",
            "target": "office workers experiencing wrist strain who need precision and comfort",
        },
        {
            "name": "DeskHub Pro USB-C Docking Station",
            "subcategory": "computing_peripherals",
            "features": "Ports: 2x HDMI 2.1, 1x DisplayPort 1.4, 3x USB-A 3.2, 2x USB-C (1 downstream, 1 upstream PD), 1x Gigabit Ethernet, SD/microSD | Power delivery: 100W pass-through | Display support: Dual 4K@60Hz or single 8K@30Hz | Compatible: Windows, macOS, ChromeOS",
            "target": "professionals managing multi-monitor setups from a single laptop cable",
        },
        {
            "name": "SharpView 27-inch 4K Monitor",
            "subcategory": "computing_peripherals",
            "features": "Panel: IPS | Resolution: 3840x2160 | Refresh: 60Hz | Color: 99% sRGB, Delta E<2 | Brightness: 350 nits | Ports: HDMI 2.0, DisplayPort 1.4, USB-C (65W PD) | Stand: Height/tilt/swivel/pivot adjustable | VESA: 100x100mm",
            "target": "graphic designers and photographers who need color-accurate displays",
        },
        {
            "name": "SwiftCharge 140W GaN USB-C Charger",
            "subcategory": "computing_peripherals",
            "features": "Total output: 140W | Ports: 2x USB-C (100W+30W), 1x USB-A (18W) | Technology: GaN III | Input: 100-240V universal | Size: 68mm x 68mm x 32mm | Weight: 198g | Safety: UL certified, overcurrent/overvoltage/overtemp protection | Folding prongs",
            "target": "frequent travelers charging a laptop, phone, and tablet from one compact adapter",
        },
        {
            "name": "SecureVault 2TB Encrypted Portable SSD",
            "subcategory": "computing_peripherals",
            "features": "Capacity: 2TB | Speed: 2000MB/s read, 1800MB/s write (USB 3.2 Gen 2x2) | Encryption: AES-256 hardware | Interface: USB-C with USB-A adapter included | Durability: IP55, 3-meter drop-tested | Size: 110mm x 55mm x 10mm | Weight: 58g",
            "target": "creative professionals and security-conscious users who transport large files",
        },
        {
            "name": "NetMesh AX6000 Wi-Fi 6E Router",
            "subcategory": "computing_peripherals",
            "features": "Speed: AX6000 tri-band (2.4GHz: 1148Mbps, 5GHz: 2402Mbps, 6GHz: 2402Mbps) | Coverage: up to 3000 sq ft | Ports: 1x 2.5G WAN, 4x Gigabit LAN | Processor: Quad-core 1.8GHz | RAM: 1GB | Mesh: Supports up to 6 nodes | Security: WPA3, built-in VPN server",
            "target": "households with 30+ connected devices needing fast, reliable coverage",
        },
        {
            "name": "PixelPad Pro Drawing Tablet",
            "subcategory": "computing_peripherals",
            "features": "Active area: 10x6 inches | Pressure levels: 8192 | Tilt recognition: +/-60 degrees | Resolution: 5080 LPI | Report rate: 220 PPS | Connectivity: USB-C + Bluetooth 5.0 | Express keys: 8 customizable | Stylus: Battery-free electromagnetic",
            "target": "digital illustrators and photo retouchers who want natural pen-on-paper feel",
        },
        {
            "name": "StreamDeck Mini Control Pad",
            "subcategory": "computing_peripherals",
            "features": "Keys: 6 customizable LCD buttons | Display: 72x72 pixel per key | Connectivity: USB 2.0 | Software: Windows 10+, macOS 12+ | Profiles: Unlimited switchable | Actions: Multi-action macros, folder nesting | Dimensions: 84mm x 60mm x 30mm | Weight: 108g",
            "target": "content creators and streamers who need quick access to scenes, media, and shortcuts",
        },
        {
            "name": "LaptopRise Adjustable Aluminum Stand",
            "subcategory": "computing_peripherals",
            "features": "Material: 6061 aluminum alloy | Height range: 5-10 inches adjustable | Weight capacity: 20 lbs | Compatibility: 10-17 inch laptops | Ventilation: Open-air design for airflow | Base: Non-slip silicone pads | Cable management: Rear channel | Foldable: Yes, for travel",
            "target": "remote workers who want to raise their laptop to eye level and reduce neck strain",
        },
        {
            "name": "ClearCam 4K Webcam",
            "subcategory": "computing_peripherals",
            "features": "Resolution: 4K @ 30fps, 1080p @ 60fps | Sensor: Sony STARVIS IMX335 | FOV: 85-degree, adjustable to 65-degree | Autofocus: PDAF, 10cm minimum | Microphone: Dual omnidirectional | Mount: Monitor clip + tripod thread | Privacy: Physical lens cover | Connectivity: USB-C",
            "target": "remote professionals and content creators who need broadcast-quality video calls",
        },
        {
            "name": "KeySafe Hardware Security Key",
            "subcategory": "computing_peripherals",
            "features": "Protocols: FIDO2/WebAuthn, U2F, OTP, PIV | Connectivity: USB-C + NFC | Cryptography: ECC P-256 | Certifications: FIDO Alliance certified | Durability: IP68, crush-resistant | OS: Windows, macOS, Linux, iOS, Android | Setup: No battery, no driver required",
            "target": "security-conscious professionals protecting online accounts with hardware-based 2FA",
        },
    ]

    home_electronics = [
        {
            "name": "BreezeComfort Smart Tower Fan",
            "subcategory": "home_electronics",
            "features": "Airflow modes: 12 speed settings | Oscillation: 90-degree wide | Height: 42 inches | Noise level: 28dB (lowest) to 52dB (max) | Timer: 1-12 hours | Connectivity: Wi-Fi + app control | Voice: Works with Alexa and Google Home | Remote: Included IR remote",
            "target": "apartment residents who need quiet, scheduled cooling without AC",
        },
        {
            "name": "LuxGlow Smart LED Bulb (4-pack)",
            "subcategory": "home_electronics",
            "features": "Wattage: 9W (60W equivalent) | Lumens: 800 | Color: 16 million colors + tunable white (2700K-6500K) | Connectivity: Wi-Fi, no hub needed | Dimmable: 1-100% | Lifespan: 25000 hours | Fitting: E26/A19 | Voice: Alexa, Google, Siri Shortcuts",
            "target": "smart home enthusiasts who want room-by-room lighting customization",
        },
        {
            "name": "PureAir HEPA Air Purifier",
            "subcategory": "home_electronics",
            "features": "Filter: True HEPA H13 + activated carbon | CADR: 240 CFM | Coverage: up to 450 sq ft | Fan speeds: 4 (Sleep, Low, Medium, High) | Noise: 24dB on Sleep | Sensors: PM2.5 real-time display | Filter life: 6-8 months | Weight: 5.2kg",
            "target": "allergy sufferers and pet owners in urban environments",
        },
        {
            "name": "AquaPure Reverse Osmosis Water Filter",
            "subcategory": "home_electronics",
            "features": "Filtration: 4-stage reverse osmosis | TDS reduction: up to 99% | Flow rate: 0.5 gallons per minute | Tank capacity: 3 gallons | Installation: Under-sink, DIY-friendly | Filter replacement: Every 6 months (sediment), 24 months (RO membrane) | Certifications: NSF 42, 53, 58",
            "target": "health-conscious families wanting purified drinking water at home",
        },
        {
            "name": "WarmFloor Radiant Heating Mat",
            "subcategory": "home_electronics",
            "features": "Coverage: 30 sq ft per mat | Wattage: 360W (12W per sq ft) | Voltage: 120V | Thickness: 1/8 inch | Thermostat: Programmable 7-day, Wi-Fi enabled | Material: Twin-conductor shielded cable | Flooring: Compatible with tile, stone, laminate | Warranty: 25-year limited",
            "target": "homeowners renovating bathrooms or kitchens in cold climates",
        },
        {
            "name": "ZenMist Ultrasonic Humidifier",
            "subcategory": "home_electronics",
            "features": "Tank capacity: 4.5L | Runtime: up to 36 hours | Output: Cool mist, adjustable | Coverage: 400 sq ft | Noise: <30dB | Auto shutoff: When empty | Display: LED humidity readout | Essential oil tray: Yes",
            "target": "dry-climate residents and parents of infants needing consistent room humidity",
        },
        {
            "name": "GuardView Wireless Doorbell Camera",
            "subcategory": "home_electronics",
            "features": "Resolution: 2K (2560x1440) | FOV: 160-degree wide-angle | Night vision: IR, 25-foot range | Power: Rechargeable battery (6 months) or hardwired | Storage: Local microSD (up to 256GB) + optional cloud | Two-way audio: Full-duplex | Motion detection: Adjustable zones | Wi-Fi: 2.4GHz + 5GHz",
            "target": "homeowners wanting package theft prevention and visitor monitoring",
        },
        {
            "name": "SolarEdge Outdoor Motion Light",
            "subcategory": "home_electronics",
            "features": "Power: Solar panel with 2200mAh Li-ion battery | Lumens: 1000 | LED count: 72 | Motion sensor: PIR, 120-degree, 26-foot range | Modes: Steady-on / dim-to-bright / sensor-only | Weather: IP65 | Charge time: 6h full sun | Mount: Screw or adhesive",
            "target": "homeowners illuminating driveways, gardens, and walkways without wiring",
        },
        {
            "name": "PowerStation 500Wh Portable Battery",
            "subcategory": "home_electronics",
            "features": "Capacity: 500Wh (135000mAh) | Output: 2x AC (500W pure sine), 2x USB-C PD (100W), 2x USB-A, 1x 12V DC | Charging: Wall (2.5h), solar panel (MPPT, 200W max), car 12V | Weight: 6.4kg | Display: LCD showing watts in/out | UPS mode: 20ms switchover",
            "target": "campers, outdoor event planners, and homeowners preparing for power outages",
        },
        {
            "name": "AirSense Smart Thermostat",
            "subcategory": "home_electronics",
            "features": "Display: 3.5-inch color touchscreen | Sensors: Temperature, humidity, occupancy | Connectivity: Wi-Fi + Thread/Matter | Voice: Alexa, Google, HomeKit | Learning: 7-day adaptive scheduling | Compatibility: Works with most 24V HVAC systems | Power: C-wire or included adapter | Geofencing: Auto home/away",
            "target": "homeowners looking to automate climate control and reduce energy bills",
        },
        {
            "name": "LaundryFresh Portable Clothes Steamer",
            "subcategory": "home_electronics",
            "features": "Power: 1500W | Heat-up: 25 seconds | Tank: 260ml (15 minutes continuous) | Steam output: 28g/min | Cord: 9-foot power cord | Weight: 0.9kg | Attachments: Fabric brush, crease clip | Auto shutoff: 8-minute safety timer",
            "target": "business travelers and professionals who need wrinkle-free clothes without an ironing board",
        },
        {
            "name": "RoboVac MapStar Laser Navigation Robot Vacuum",
            "subcategory": "home_electronics",
            "features": "Suction: 4000Pa | Navigation: LiDAR mapping | Battery: 5200mAh (180 min runtime) | Dustbin: 600ml | Mopping: Vibrating mop pad attachment | Noise: 55dB (quiet mode) | App: Zone cleaning, no-go areas, scheduling | Height: 95mm (fits under most furniture)",
            "target": "busy households with mixed flooring (carpet and hard floors) and pets",
        },
    ]

    wearables = [
        {
            "name": "FitPulse GPS Running Watch",
            "subcategory": "wearables",
            "features": "GPS: Dual-band (L1+L5) | Heart rate: Optical wrist-based | Battery: 14 days (smartwatch), 40h (GPS mode) | Display: 1.4-inch AMOLED, always-on | Water: 5ATM | Sensors: Barometric altimeter, compass, SpO2 | Navigation: Breadcrumb trail | Weight: 47g",
            "target": "distance runners and triathletes who train with pace and heart-rate zones",
        },
        {
            "name": "SleepRing Health Tracker",
            "subcategory": "wearables",
            "features": "Sensors: PPG heart rate, skin temperature, 3-axis accelerometer | Battery: 7 days | Weight: 4g (size 8) | Material: Titanium with inner epoxy coating | Water: 100m | Data: Sleep stages, HRV, readiness score | Connectivity: Bluetooth 5.2 | Sizes: 6-13",
            "target": "health-conscious individuals focused on sleep quality and recovery metrics",
        },
        {
            "name": "TrailSight AR Hiking Glasses",
            "subcategory": "wearables",
            "features": "Display: Micro-LED heads-up, 20-degree FOV | Battery: 8h (display on) | Navigation: Waypoint arrows overlay | Camera: 12MP photo, 1080p video | Audio: Open-ear directional speakers | Weight: 45g | Lens: Polarized, interchangeable | App: Route planning and sync",
            "target": "tech-forward hikers and trail runners who want navigation without checking a phone",
        },
        {
            "name": "HearClear OTC Hearing Enhancer",
            "subcategory": "wearables",
            "features": "Amplification: Up to 35dB gain, 4 frequency bands | Battery: 16h rechargeable | Fit: 3 silicone dome sizes | Noise reduction: Directional microphone + DSP | Bluetooth: 5.3 for phone calls and audio streaming | Weight: 5.2g per ear | Charging: Pocket case with 3 additional charges | FDA: Class II OTC cleared",
            "target": "adults with mild-to-moderate hearing loss who want an affordable, discreet solution",
        },
        {
            "name": "PostureAlign Wearable Trainer",
            "subcategory": "wearables",
            "features": "Sensor: 9-axis IMU | Vibration alert: Gentle buzz when slouching | Battery: 10 days | Attachment: Magnetic clip for collar or skin-safe adhesive | Weight: 12g | App: Posture score, daily/weekly trends | Connectivity: Bluetooth 5.0 | Calibration: 15-second auto-calibrate",
            "target": "desk workers and students who want to build better posture habits",
        },
        {
            "name": "UVSense Solar Exposure Tracker",
            "subcategory": "wearables",
            "features": "Sensor: UV-A and UV-B photodiode | Battery: 30 days (solar-assisted) | Display: E-ink UV index readout | Alert: Vibration when recommended exposure limit reached | Water: IPX7 | Weight: 8g | Wearable: Clip-on or wristband | App: Cumulative daily UV log with skin-type personalization",
            "target": "outdoor workers and sun-sensitive individuals monitoring UV exposure",
        },
        {
            "name": "BreathCoach Smart Respiration Band",
            "subcategory": "wearables",
            "features": "Sensor: Strain gauge respiratory monitoring | Metrics: Breathing rate, depth, pattern regularity | Battery: 5 days | Material: Stretchy moisture-wicking fabric | Connectivity: Bluetooth 5.1 | Guided sessions: 20 pre-loaded breathing exercises | Haptic: Inhale/exhale vibration guides | Sizes: S/M/L",
            "target": "meditation practitioners and anxiety sufferers looking for data-driven breathing training",
        },
        {
            "name": "MotionCapture Sport Sensor Pod",
            "subcategory": "wearables",
            "features": "Sensors: 9-axis IMU + barometer | Attachment: Universal clip for shoe, bat, racket, club | Battery: 20h active | Weight: 9g | Connectivity: Bluetooth 5.2 + ANT+ | Metrics: Swing speed, cadence, stride length (sport-dependent) | Water: IPX6 | App: 3D motion replay",
            "target": "amateur athletes and coaches analyzing swing mechanics or running form",
        },
        {
            "name": "NightGlow Safety Vest with LED",
            "subcategory": "wearables",
            "features": "LEDs: 360-degree visibility, 3 modes (steady, flash, pulse) | Brightness: Visible at 500m | Battery: USB-C rechargeable, 8h runtime | Material: Breathable mesh, reflective 3M strips | Sizes: Adjustable S-XL | Weight: 180g | Water resistance: IPX4",
            "target": "runners, cyclists, and dog walkers active during dawn, dusk, or nighttime",
        },
        {
            "name": "TempSmart Fever Monitoring Patch",
            "subcategory": "wearables",
            "features": "Sensor: Medical-grade thermistor | Accuracy: +/-0.1 degrees C | Monitoring: Continuous, real-time to app | Adhesive: Hypoallergenic, latex-free, 48h wear | Battery: Single-use CR1620, replaceable | Alert: Configurable fever threshold push notification | Weight: 6g | Connectivity: Bluetooth 5.0 LE",
            "target": "parents monitoring young children's fevers overnight",
        },
        {
            "name": "StepFlex Smart Insole Pair",
            "subcategory": "wearables",
            "features": "Sensors: 16 pressure zones per insole | Metrics: Gait symmetry, strike pattern, weight distribution | Battery: 30 days standby, 10h active tracking | Connectivity: Bluetooth 5.0 | Sizes: Trimmable, US 5-14 | Thickness: 3mm | Material: Antibacterial EVA foam | App: Real-time heatmap and stride analysis",
            "target": "physiotherapy patients and runners optimizing gait to prevent injuries",
        },
        {
            "name": "HydroRemind Smart Water Bottle Lid",
            "subcategory": "wearables",
            "features": "Fits: Standard wide-mouth bottles (63mm) | Sensor: Capacitive water level | LED ring: Glows when hydration reminder due | Battery: 2 months (rechargeable) | Connectivity: Bluetooth 5.1 | App: Daily intake goal tracking | Dishwasher safe: Top rack only | Weight: 52g",
            "target": "office workers and athletes who forget to drink enough water throughout the day",
        },
    ]

    smart_home = [
        {
            "name": "LockSmart Pro Fingerprint Deadbolt",
            "subcategory": "smart_home",
            "features": "Unlock methods: Fingerprint (capacity: 100), PIN, app, key backup | Fingerprint speed: 0.3 seconds | Connectivity: Wi-Fi + Bluetooth | Battery: 4x AA, 12-month life | Auto-lock: Configurable timer | Compatibility: Standard US deadbolt prep | ANSI Grade: 2 | Weather: IP53",
            "target": "homeowners upgrading to keyless entry with multiple family members and guests",
        },
        {
            "name": "GrowSmart Indoor Herb Garden",
            "subcategory": "smart_home",
            "features": "Pods: 6 plant slots | Light: Full-spectrum 20W LED, 16h auto cycle | Water: 1.5L self-watering reservoir, low-water alert | Height: Adjustable arm (up to 18 inches) | Power: AC adapter | App: Growth tracking, nutrient reminders | Includes: 6 herb seed pods (basil, cilantro, parsley, mint, dill, chive)",
            "target": "urban apartment dwellers who want fresh herbs year-round without outdoor space",
        },
        {
            "name": "PetWatch Automated Feeder",
            "subcategory": "smart_home",
            "features": "Capacity: 4L dry food (about 16 cups) | Portions: 1-10 portions per meal, up to 6 meals/day | Connectivity: Wi-Fi app control | Camera: 1080p with night vision and two-way audio | Power: AC adapter with 3x D-cell battery backup | Bowl: Stainless steel, dishwasher safe | Lock lid: Prevents pet tampering",
            "target": "pet owners with irregular schedules who want to monitor and feed remotely",
        },
        {
            "name": "LeakGuard Smart Water Sensor (3-pack)",
            "subcategory": "smart_home",
            "features": "Detection: Water and freeze (below 40 degrees F) | Alert: 100dB siren + push notification | Connectivity: Zigbee 3.0 (requires compatible hub) | Battery: CR2 (2-year life) | Placement: Floor-level with extension cable probe | Size: 70mm x 70mm x 28mm | IP rating: IPX7 on sensor contacts",
            "target": "homeowners protecting basements, laundry rooms, and under-sink areas from water damage",
        },
        {
            "name": "SceneSwitch Smart Dimmer",
            "subcategory": "smart_home",
            "features": "Load: 600W incandescent, 150W LED | Connectivity: Wi-Fi, no hub required | Voice: Alexa, Google, HomeKit | Dimming: 1-100% with adjustable low-end trim | Installation: Requires neutral wire | Scenes: Sunrise/sunset scheduling | Faceplate: Included in white, almond, and black | Wattage metering: Real-time in app",
            "target": "homeowners automating room lighting and creating mood scenes",
        },
        {
            "name": "AirQuality Monitor Hub",
            "subcategory": "smart_home",
            "features": "Sensors: PM2.5, CO2, VOC, temperature, humidity | Display: 3.2-inch color e-ink (always visible) | Connectivity: Wi-Fi + Thread | Refresh: Readings every 2 minutes | History: 30-day on-device log | Alerts: Configurable thresholds via app | Power: USB-C (no battery) | Calibration: Auto-baseline correction",
            "target": "families, allergy sufferers, and home office workers monitoring indoor air health",
        },
        {
            "name": "GarageSmart Wi-Fi Controller",
            "subcategory": "smart_home",
            "features": "Compatibility: Most chain, belt, and screw-drive openers made after 1993 | Sensor: Tilt sensor for open/close detection | Connectivity: Wi-Fi 2.4GHz | Features: Open/close via app, scheduling, auto-close timer | Alerts: Push notification on state change | Voice: Alexa, Google | Installation: 15-minute DIY | Power: Existing opener outlet",
            "target": "homeowners who forget to close the garage and want remote status monitoring",
        },
        {
            "name": "IrrigationIQ Smart Sprinkler Controller",
            "subcategory": "smart_home",
            "features": "Zones: 8 independently programmable | Connectivity: Wi-Fi | Weather: Automatic rain/freeze skip using local forecast | Scheduling: Flexible calendar or interval | Flow sensor: Optional (sold separately) for leak detection | Valve: 24V AC standard | App: Seasonal adjust, soak cycles, run history | Voice: Alexa, Google",
            "target": "homeowners looking to reduce water waste while keeping lawns and gardens healthy",
        },
        {
            "name": "SmokeSense Smart Detector",
            "subcategory": "smart_home",
            "features": "Detection: Photoelectric smoke + electrochemical CO + heat | Battery: Sealed 10-year lithium | Connectivity: Wi-Fi | Alerts: 85dB siren, voice announcement, push notification | Self-test: Monthly automatic | Interconnect: App-linked (up to 12 units) | Certifications: UL 217, UL 2034 | Silence: App or button",
            "target": "homeowners upgrading to connected fire and CO safety with whole-home alert linking",
        },
        {
            "name": "MirrorPro Smart Bathroom Mirror",
            "subcategory": "smart_home",
            "features": "Display: 21.5-inch embedded LCD behind anti-fog glass | Resolution: 1080p | Lighting: Adjustable edge-lit LED (3000K-6000K), CRI 90+ | Features: Time, weather, calendar widget | Connectivity: Wi-Fi | Audio: Bluetooth speaker built into frame | Defogger: Built-in heated pad | Dimensions: 24x32 inches | Power: Hardwired 120V",
            "target": "homeowners remodeling bathrooms who want a functional smart mirror",
        },
        {
            "name": "BirdWatch AI Feeder Camera",
            "subcategory": "smart_home",
            "features": "Camera: 1080p with 4x digital zoom | AI: Identifies 6000+ bird species | Storage: Cloud (free 3 days) + local microSD | Power: Solar panel + 5200mAh backup battery | Seed capacity: 1.5L | Connectivity: Wi-Fi 2.4GHz | Notifications: Species-identified photo alerts | Weather: IP65",
            "target": "backyard birdwatchers who want automatic species logging and photo collection",
        },
        {
            "name": "ScentPod Smart Diffuser",
            "subcategory": "smart_home",
            "features": "Tank: 300ml | Runtime: 8h continuous, 16h intermittent | Coverage: 400 sq ft | Mist: Ultrasonic cool mist | Connectivity: Wi-Fi | Scheduling: App and voice (Alexa, Google) | LED: 7 ambient color options | Auto shutoff: When tank empty | Noise: <25dB",
            "target": "aromatherapy users who want scheduled whole-room scenting with smart home integration",
        },
    ]

    all_electronics = audio_products + peripherals + home_electronics + wearables + smart_home
    rng.shuffle(all_electronics)
    for p in all_electronics[:count]:
        p["category"] = "electronics"
        products.append(p)
    return products


def generate_kitchen(fake: Faker, rng: random.Random, count: int) -> list[dict]:
    """Generate kitchen product specs with benefit-led, lifestyle language."""
    products = [
        {
            "name": "Heritage Enameled Cast Iron Dutch Oven",
            "subcategory": "cookware",
            "features": "Capacity: 5.5-quart | Material: Enameled cast iron | Weight: 11.4 lbs | Oven-safe: up to 500 degrees F | Lid: Self-basting with condensation ridges | Colors: 6 options | Dishwasher: Not recommended | Made in: France",
            "target": "home cooks who braise, stew, and batch-cook on weekends",
        },
        {
            "name": "MorningBrew 12-Cup Programmable Drip Coffee Maker",
            "subcategory": "appliances",
            "features": "Capacity: 12 cups (60 oz) | Brew strength: Regular and bold | Carafe: Double-wall thermal stainless steel (4h heat retention) | Programmable: 24h advance timer | Water filter: Charcoal | Brew time: 8 minutes full pot | Auto shutoff: 2 hours | Showerhead: 9-hole for even saturation",
            "target": "coffee-drinking households that want a hot pot waiting every morning",
        },
        {
            "name": "ChefEdge 8-Inch Japanese Gyuto Knife",
            "subcategory": "cutlery",
            "features": "Blade: VG-10 stainless steel core, 67-layer Damascus cladding | Hardness: HRC 60 | Length: 8 inches | Weight: 7.2 oz | Handle: Pakkawood, ambidextrous | Edge angle: 15 degrees per side | Balance point: At the bolster",
            "target": "home cooks and aspiring chefs who value sharp, precise knife work",
        },
        {
            "name": "PrepStation Bamboo Cutting Board Set",
            "subcategory": "prep",
            "features": "Set includes: 3 boards (18x12, 14x10, 10x7 inches) | Material: Moso bamboo, FSC certified | Juice groove: On large and medium boards | Thickness: 0.75 inches | Care: Hand wash, oil periodically | Non-slip: Silicone corner grips",
            "target": "home cooks wanting a durable, eco-friendly prep surface in multiple sizes",
        },
        {
            "name": "FreshSeal Vacuum Food Sealer",
            "subcategory": "appliances",
            "features": "Sealing width: 12 inches | Modes: Dry, moist, gentle (delicate items) | Vacuum strength: -0.8 bar | Bag cutter: Built-in with roll storage | Compatible bags: Standard channel bags | Pulse mode: Manual vacuum control | Seal-only mode: For chips and zipper bags | Power: 120W",
            "target": "meal preppers and sous-vide cooks who buy in bulk and batch-freeze",
        },
        {
            "name": "AeroWhisk Immersion Blender Set",
            "subcategory": "appliances",
            "features": "Motor: 500W | Speeds: Variable trigger + turbo | Shaft: 8-inch stainless steel, detachable | Attachments: Blending arm, whisk, chopper (2-cup) | Blade: 4-point hardened stainless | Cord: 5 feet | Weight: 1.5 lbs (blending arm attached) | BPA-free: All plastic components",
            "target": "home cooks who blend soups in the pot, whip cream, and make quick salsas",
        },
        {
            "name": "CrispAir Digital Air Fryer Oven",
            "subcategory": "appliances",
            "features": "Capacity: 6-quart | Temperature: 180-400 degrees F | Presets: 8 one-touch (fries, chicken, fish, steak, shrimp, cake, pizza, dehydrate) | Timer: Up to 60 minutes | Basket: Non-stick, dishwasher safe | Power: 1700W | Display: Digital LED touchscreen | Shake reminder: Audio alert at half-time",
            "target": "health-conscious families who want crispy results with minimal oil",
        },
        {
            "name": "BrewPerfect Gooseneck Electric Kettle",
            "subcategory": "appliances",
            "features": "Capacity: 0.9L | Temperature control: 140-212 degrees F in 1-degree increments | Hold temp: 60-minute keep-warm | Flow rate: Precision gooseneck spout | Material: 304 stainless steel body | Display: LCD with real-time temperature | Base: 360-degree swivel | Boil time: Under 5 minutes",
            "target": "pour-over coffee and loose-leaf tea enthusiasts who need precise water temperature",
        },
        {
            "name": "FermentPro Glass Jar Set with Airlocks",
            "subcategory": "prep",
            "features": "Set: 4 wide-mouth 32oz mason jars | Lids: BPA-free silicone with built-in airlock | Weights: 4 glass fermentation weights | Material: Lead-free borosilicate glass | Measurement: Embossed oz and ml markings | Includes: Recipe booklet with 12 ferments | Dishwasher: Jars and weights safe",
            "target": "home fermenters making sauerkraut, kimchi, pickles, and kombucha",
        },
        {
            "name": "GrindMaster Burr Coffee Grinder",
            "subcategory": "appliances",
            "features": "Burrs: 40mm conical stainless steel | Grind settings: 18 click positions (espresso to French press) | Hopper: 250g capacity | Timer: Adjustable 5-30 seconds | Retention: <0.5g | Speed: 450 RPM (low heat) | Noise: 70dB | Motor: DC with anti-static feature",
            "target": "coffee enthusiasts grinding fresh beans daily for espresso or filter methods",
        },
        {
            "name": "SizzlePro Cast Iron Skillet 12-Inch",
            "subcategory": "cookware",
            "features": "Diameter: 12 inches | Material: Pre-seasoned cast iron (flaxseed oil) | Weight: 8 lbs | Oven-safe: up to 650 degrees F | Compatible: Gas, electric, induction, campfire, grill | Handle: Assist handle on opposite side | Pour spouts: Dual | Made in: USA",
            "target": "cooks who sear steaks, bake cornbread, and want heirloom-quality cookware",
        },
        {
            "name": "SliceMaster Adjustable Mandoline Slicer",
            "subcategory": "prep",
            "features": "Thickness: 0.5mm to 8mm adjustable dial | Blades: Straight, waffle, julienne (1.5mm and 3mm) | Material: Stainless steel blade, BPA-free body | Safety: Integrated hand guard with prongs | Feet: Non-slip rubber | Size: 14 x 5 inches | Dishwasher: Hand wash recommended for blades",
            "target": "home cooks who want uniform vegetable slices for gratins, salads, and stir-fries",
        },
        {
            "name": "NutriBlend Personal Blender",
            "subcategory": "appliances",
            "features": "Motor: 900W | Blade: 6-point stainless steel | Cups: 2 included (24oz and 12oz) with flip-top lids | Speed: One-touch pulse | BPA-free: All cups and lids | Dimensions: 6.5 x 6.5 x 15 inches | Cord: Integrated storage | Cleanup: Cups dishwasher safe",
            "target": "on-the-go professionals blending smoothies, protein shakes, and baby food",
        },
        {
            "name": "PastaRoller Manual Pasta Machine",
            "subcategory": "prep",
            "features": "Roller width: 6 inches | Thickness settings: 9 positions (0.3mm to 3mm) | Cutters: Fettuccine (6.5mm) and spaghetti (2mm) included | Material: Chrome-plated steel rollers | Clamp: Adjustable table clamp (up to 2-inch edge) | Handle: Detachable stainless steel | Weight: 5.6 lbs",
            "target": "home cooks who enjoy making fresh pasta from scratch on weekends",
        },
        {
            "name": "SpiceVault 24-Jar Revolving Rack",
            "subcategory": "storage",
            "features": "Jars: 24 glass jars, 4 oz each | Labels: 48 pre-printed + 24 blank | Rack: Stainless steel countertop carousel | Lids: Dual-flap (shaker and pour) | Dimensions: 10-inch diameter x 13.5 inches tall | Rotation: 360-degree smooth bearing | Jars: Dishwasher safe",
            "target": "organized cooks who want all spices visible and within arm's reach",
        },
        {
            "name": "SteamPro Bamboo Steamer Set",
            "subcategory": "cookware",
            "features": "Tiers: 2 steaming trays + 1 lid | Diameter: 10 inches | Material: Natural bamboo, stainless steel bands | Capacity: Fits standard wok or 10+ inch pot | Liner: 50 perforated parchment liners included | Care: Hand wash, air dry | Food safe: No chemical coatings or glues",
            "target": "home cooks making dumplings, vegetables, and dim sum",
        },
        {
            "name": "QuickBrine Stainless Steel Marinating Container",
            "subcategory": "prep",
            "features": "Capacity: 5-quart | Material: 18/10 stainless steel | Lid: Airtight silicone-sealed | Rack: Internal elevated rack for even brining | Dimensions: 12 x 9 x 4 inches | Stackable: Flat lid design | Fridge-safe: Yes | Dishwasher: All components safe",
            "target": "grill enthusiasts and holiday cooks who brine poultry and marinate large cuts",
        },
        {
            "name": "SmartScale Pro Kitchen Scale",
            "subcategory": "tools",
            "features": "Capacity: 11 lbs / 5 kg | Precision: 1g / 0.05oz increments | Display: Backlit LCD | Modes: g, oz, lb:oz, ml (water/milk) | Platform: Stainless steel, 7-inch diameter | Tare: One-touch | Power: 2x AAA (included), auto-off after 2 minutes | Connectivity: None (no app)",
            "target": "bakers and meal-prep enthusiasts who need precise ingredient measurements",
        },
        {
            "name": "ZestPro Citrus Juicer Press",
            "subcategory": "tools",
            "features": "Mechanism: Lever-action press | Material: Die-cast zinc alloy with enamel coating | Strainer: Built-in seed/pulp filter | Capacity: Handles lemons, limes, small oranges | Drip tray: Removable | Height: 14 inches | Colors: 5 options | Cleanup: Hand wash",
            "target": "cocktail enthusiasts and home cooks who juice citrus daily",
        },
        {
            "name": "HeatKeep Insulated Casserole Carrier",
            "subcategory": "storage",
            "features": "Fits: 9x13-inch baking dishes | Insulation: Thermal lining, maintains temperature for 2 hours | Closure: Full-zip with reinforced handles | Material: 600D polyester exterior, food-safe PEVA lining | Pockets: 1 exterior for utensils | Flat bottom: Stable transport | Cleanup: Wipe-clean interior",
            "target": "potluck-goers, holiday hosts, and families transporting dishes to gatherings",
        },
        {
            "name": "ColdBrew Tower Slow-Drip Coffee Maker",
            "subcategory": "appliances",
            "features": "Capacity: 6-8 cups | Brew time: 3-5 hours (adjustable drip rate) | Material: Borosilicate glass tower, stainless steel filter | Drip rate: Adjustable valve (1 drop per second to steady stream) | Height: 24 inches | Coffee bed: Holds 100g ground coffee | No electricity required | Includes: Cleaning brush",
            "target": "cold brew enthusiasts who enjoy the ritual of slow-drip extraction",
        },
        {
            "name": "WokMaster Carbon Steel Round Bottom Wok",
            "subcategory": "cookware",
            "features": "Diameter: 14 inches | Material: 1.8mm carbon steel, pre-seasoned | Handle: Wooden main handle + steel helper handle | Weight: 4.2 lbs | Heat source: Gas stove recommended | Oven-safe: up to 500 degrees F | Includes: Wok ring for flat-top stoves | Patina: Develops natural nonstick with use",
            "target": "home cooks who stir-fry at high heat and value quick, even heating",
        },
        {
            "name": "BreadBox Proofing Basket Set",
            "subcategory": "baking",
            "features": "Set: 2 bannetons (9-inch round, 10-inch oval) | Material: Natural rattan cane | Includes: 2 linen liners, 1 dough scraper, 1 bread lame with 5 blades | Dough capacity: Up to 2 lbs each | Care: Brush clean, air dry (no water) | Flour dusting: Rattan pattern imprints on loaf",
            "target": "sourdough bakers who want artisan-shaped boules and batards",
        },
        {
            "name": "ThermoSip Double-Wall Espresso Cups (Set of 4)",
            "subcategory": "drinkware",
            "features": "Capacity: 2.7 oz (80ml) each | Material: Borosilicate glass, hand-blown | Insulation: Double-wall vacuum (keeps hot 3x longer than ceramic) | Set: 4 cups | Dimensions: 2.4 x 2.6 inches | Microwave: Yes | Dishwasher: Top rack safe | Design: Suspended-liquid visual effect",
            "target": "espresso drinkers who want to see the crema without burning their fingers",
        },
        {
            "name": "IceForge Clear Ice Ball Maker",
            "subcategory": "drinkware",
            "features": "Produces: 4 crystal-clear 2.3-inch ice spheres | Method: Directional freezing (top-down) | Mold material: Insulated silicone + polycarbonate frame | Freeze time: 18-22 hours | Water capacity: Standard freezer shelf | No electricity: Passive insulation method | Dimensions: 7 x 7 x 5 inches",
            "target": "whiskey and cocktail lovers who want perfectly clear, slow-melting ice",
        },
        {
            "name": "FlameGuard Silicone Oven Mitt Pair",
            "subcategory": "tools",
            "features": "Heat resistance: up to 500 degrees F | Length: 14.7 inches (forearm coverage) | Material: BPA-free silicone exterior, soft cotton lining | Grip: Textured non-slip palm and fingers | Waterproof: Outer silicone layer | Fit: One-size, flexible | Care: Machine washable | Pair: Left and right included",
            "target": "home bakers and grillers who handle hot pans, sheets, and grill grates",
        },
        {
            "name": "HerbScissor 5-Blade Kitchen Shears",
            "subcategory": "tools",
            "features": "Blades: 5 parallel stainless steel blades | Blade length: 3.25 inches | Handle: Soft-grip ergonomic | Cleaning comb: Included for between-blade herbs | Use: Herbs, green onions, nori, shredded paper | Dishwasher: Yes | Safety: Blade guard included",
            "target": "home cooks who mince fresh herbs quickly and evenly",
        },
        {
            "name": "PortaGrill Stovetop Smokeless Grill Pan",
            "subcategory": "cookware",
            "features": "Surface: 13 x 10 inch nonstick ridged grill plate | Material: Die-cast aluminum | Drip tray: Water-filled channel reduces smoke by 80% | Compatible: Gas, electric, ceramic stovetops | Handles: Stay-cool stainless steel | Weight: 4.5 lbs | Oven-safe: up to 450 degrees F | Cleanup: Dishwasher safe",
            "target": "apartment cooks who want indoor grilling without setting off the smoke alarm",
        },
        {
            "name": "FreshKeep Herb Saver Pod Set",
            "subcategory": "storage",
            "features": "Set: 3 pods (small, medium, large) | Design: Inner basket suspends herbs in water | Material: BPA-free Tritan plastic | Herb life: Extends freshness up to 3 weeks | Fridge door: Tall pod fits standard door shelf | Capacity: Holds a full supermarket bunch each | Cleaning: Top-rack dishwasher safe",
            "target": "cooks who buy fresh herbs weekly and hate throwing away wilted bunches",
        },
        {
            "name": "RisePro Sourdough Starter Jar",
            "subcategory": "baking",
            "features": "Capacity: 24 oz | Material: Wide-mouth borosilicate glass | Lid: Wooden with silicone-sealed vent hole | Markings: Raised silicone date band + 4 rubber band markers for rise tracking | Thermometer strip: Adhesive liquid crystal on side | Includes: Feeding ratio guide card | Dishwasher: Jar only",
            "target": "beginner and experienced sourdough bakers tracking their starter's activity",
        },
        {
            "name": "OilMist Refillable Cooking Spray Bottle",
            "subcategory": "tools",
            "features": "Capacity: 200ml | Spray: Fine mist pump, no propellant | Material: Glass bottle with stainless steel sprayer | Spray volume: ~1ml per pump | Suitable for: Any cooking oil, vinegar, citrus juice | Cleaning: Fully disassembleable, hand wash | Dimensions: 2.5 x 8.5 inches",
            "target": "health-conscious cooks who want to control oil portions without aerosol cans",
        },
        {
            "name": "CrustPerfect Pizza Steel",
            "subcategory": "baking",
            "features": "Material: 3/8-inch A36 steel plate | Dimensions: 16 x 14 inches | Weight: 23 lbs | Oven use: Placed on middle rack, preheat 45 min at max temp | Heat conductivity: 18x more than ceramic stone | Broiler-safe: Yes | Maintenance: Light oil coating after wash | Also works for: Bread, flatbreads, frozen pizza",
            "target": "pizza enthusiasts chasing charred, blistered Neapolitan-style crust at home",
        },
        {
            "name": "MixBowl Nesting Set with Lids",
            "subcategory": "prep",
            "features": "Set: 5 bowls (1, 1.5, 2, 3, 5 quart) | Material: 18/8 stainless steel | Lids: Airtight BPA-free silicone (5 included) | Base: Silicone non-slip ring | Measurement markings: Interior ml and cup lines | Nesting: All 5 fit inside the largest | Dishwasher: Bowls safe, hand wash lids",
            "target": "organized home cooks who prep, mix, store, and serve in the same bowls",
        },
        {
            "name": "SipCraft Cocktail Smoker Kit",
            "subcategory": "drinkware",
            "features": "Smoker: Stainless steel chimney top fits standard rocks glasses | Wood chips: 4 flavors included (cherry, oak, hickory, apple, 20 uses each) | Torch: Butane culinary torch with safety lock (butane not included) | Mesh filter: Removable for cleaning | Includes: Recipe card set for 10 smoked cocktails",
            "target": "cocktail hobbyists and home bartenders who want smoky Old Fashioneds and Manhattans",
        },
        {
            "name": "TempCheck Instant-Read Food Thermometer",
            "subcategory": "tools",
            "features": "Speed: 2-3 second reading | Range: -58 to 572 degrees F | Accuracy: +/-0.9 degrees F | Probe: 4.5-inch foldable | Display: Backlit, auto-rotating | Hold: Press to lock reading | Calibration: One-button ice-water reset | Battery: CR2032 (3000h life) | Water resistance: IP67",
            "target": "grillers, bakers, and home cooks who need accurate internal temperatures every time",
        },
        {
            "name": "StrainEasy Expandable Over-Sink Colander",
            "subcategory": "prep",
            "features": "Collapsed width: 14 inches | Extended width: 19.5 inches (fits sinks 14-19 inches) | Material: BPA-free silicone with stainless steel frame | Capacity: 6-quart | Hole pattern: Fine mesh (keeps orzo and quinoa) | Handles: Non-slip rubber grip | Foldable: Collapses to 2-inch height for storage | Dishwasher: Safe",
            "target": "small-kitchen cooks who need a colander that stores flat and spans the sink",
        },
        {
            "name": "ButterBell French Butter Crock",
            "subcategory": "storage",
            "features": "Capacity: 1 standard stick of butter | Material: Stoneware, lead-free glaze | Seal: Water-seal keeps butter fresh for 30 days unrefrigerated | Dimensions: 4 x 4 x 4 inches | Colors: 8 glazed options | Care: Dishwasher safe | Temperature: Best below 76 degrees F room temp",
            "target": "toast lovers and bakers who want spreadable butter without a microwave",
        },
        {
            "name": "GrillMat Reusable BBQ Sheet Set",
            "subcategory": "tools",
            "features": "Set: 5 mats (15.75 x 13 inches each) | Material: PTFE-coated fiberglass | Heat resistance: Up to 500 degrees F | Thickness: 0.2mm | Reusable: Up to 1000 uses per mat | Use: Grill, oven, or smoker | Prevents: Small food from falling through grates | Cleanup: Dishwasher safe or wipe with soapy cloth",
            "target": "grillers cooking fish, vegetables, and delicate items that stick or fall through grates",
        },
        {
            "name": "DoughMaster Silicone Baking Mat Set",
            "subcategory": "baking",
            "features": "Set: 2 half-sheet (16.5 x 11.5 inches) + 1 round (9-inch) | Material: Food-grade silicone with fiberglass mesh | Markings: Cookie spacing guide, pie crust circle, dough rolling measurements | Temperature: -40 to 480 degrees F | Nonstick: Eliminates need for parchment or cooking spray | Thickness: 0.75mm | Dishwasher: Safe",
            "target": "bakers who want consistent results and less waste from disposable parchment",
        },
        {
            "name": "PourOver Ceramic Dripper with Stand",
            "subcategory": "appliances",
            "features": "Dripper: Single-serve, holds #2 cone filters | Material: Handmade ceramic (food-safe glaze) | Ribbing: Interior spiral channels for optimal flow | Stand: Walnut wood with stainless steel arms | Fits: Cups and carafes 3-5 inches wide | Capacity: Brews 12-16 oz per pour | Includes: 40 unbleached paper filters | Weight: 1.1 lbs total",
            "target": "manual coffee brewers who appreciate craft aesthetics on the countertop",
        },
        {
            "name": "ChopBlock End-Grain Butcher Block",
            "subcategory": "prep",
            "features": "Dimensions: 18 x 12 x 2 inches | Material: End-grain acacia wood | Feet: 4 stainless steel, non-slip | Juice groove: Full perimeter | Weight: 12 lbs | Finish: Food-safe mineral oil | Reversible: Grooved side + flat side | Care: Hand wash, re-oil monthly",
            "target": "serious home cooks who want a professional-grade cutting surface that protects knives",
        },
        {
            "name": "LunchStack Bento Box System",
            "subcategory": "storage",
            "features": "Tiers: 3 stackable compartments (total 50 oz) | Material: 18/8 stainless steel, no plastic liners | Closure: Silicone band holds tiers together | Leak-resistant: Each tier has silicone-rimmed lid | Dimensions: 6 x 4.5 x 5 inches stacked | Weight: 14 oz empty | Dishwasher: Safe | Microwave: Not suitable (metal)",
            "target": "eco-conscious commuters packing plastic-free, portion-controlled lunches",
        },
        {
            "name": "RackStar Adjustable Pot Lid Organizer",
            "subcategory": "storage",
            "features": "Slots: 7 adjustable dividers | Material: Stainless steel wire | Width: Extends 10-22 inches | Height: 7 inches | Use: Pot lids, cutting boards, baking sheets, plates | Mounting: Freestanding (countertop or cabinet) | Feet: Non-scratch rubber pads | Assembly: No tools needed",
            "target": "small-kitchen organizers tired of avalanching lids and baking sheets",
        },
        {
            "name": "MillStone Granite Mortar and Pestle",
            "subcategory": "tools",
            "features": "Material: Solid Thai granite | Capacity: 2-cup (16 oz) mortar bowl | Weight: 8 lbs total | Interior: Unpolished for grinding texture | Exterior: Polished | Pestle length: 6 inches | Base: Flat, stable | Use: Spice grinding, curry pastes, pesto, guacamole",
            "target": "cooks who grind whole spices, make curry pastes, and prefer hand-ground texture",
        },
        {
            "name": "SousChef Precision Immersion Circulator",
            "subcategory": "appliances",
            "features": "Power: 1100W | Temperature range: 77-210 degrees F | Accuracy: +/-0.1 degrees F | Flow rate: 2.4 gallons/min | Connectivity: Wi-Fi + Bluetooth | Clamp: Adjustable, fits pots 5-10 inches deep | Display: 2.4-inch TFT color | Timer: Up to 99 hours | Noise: <40dB",
            "target": "home cooks who want restaurant-quality steak, fish, and egg doneness every time",
        },
        {
            "name": "FilterFlask Cold Brew Pitcher",
            "subcategory": "drinkware",
            "features": "Capacity: 2-quart | Material: Borosilicate glass pitcher | Filter: Stainless steel mesh insert (removable) | Lid: Airtight BPA-free with pour spout | Brew time: 12-24 hours recommended | Dishwasher: All parts safe | Dimensions: 4.5 x 4.5 x 10 inches | Also brews: Iced tea, fruit-infused water",
            "target": "iced coffee drinkers who make large batches for the week",
        },
        {
            "name": "CrispKeep Bread Storage Box",
            "subcategory": "storage",
            "features": "Material: Powder-coated steel with bamboo lid/cutting board | Dimensions: 15.7 x 9.4 x 6.3 inches | Ventilation: Rear air holes for crust preservation | Lid: Doubles as a cutting surface | Capacity: Fits 2 standard loaves | Finish: Matte white or matte black | Cleaning: Wipe with damp cloth",
            "target": "bread bakers and artisan loaf buyers who want to keep crust crisp for days",
        },
        {
            "name": "StackPour Nesting Measuring Cup Set",
            "subcategory": "tools",
            "features": "Set: 7 cups (1/8, 1/4, 1/3, 1/2, 2/3, 3/4, 1 cup) | Material: Heavy-gauge 18/8 stainless steel | Handle: Engraved measurements (won't fade) | Nesting: All cups nest flat | Leveler: Magnetic straight-edge clips to 1-cup handle | Pour spout: Integrated on each cup | Dishwasher: Safe",
            "target": "detail-oriented bakers who need accurate dry-ingredient scooping without guesswork",
        },
        {
            "name": "WhiskPro Stainless Steel Balloon Whisk",
            "subcategory": "tools",
            "features": "Wires: 12 stainless steel | Length: 11 inches | Handle: Silicone-wrapped, non-slip | Weight: 3.2 oz | Head shape: Balloon for maximum aeration | Dishwasher: Safe | Heat-resistant: Handle up to 450 degrees F",
            "target": "home bakers who whisk eggs, cream, and batters by hand",
        },
        {
            "name": "BrewTimer Digital Kitchen Timer",
            "subcategory": "tools",
            "features": "Timers: 2 independent countdowns | Display: Large backlit LCD | Alerts: Loud beep (70dB) + flashing light | Magnet: Strong rear magnet for fridge | Stand: Retractable kickstand + hook hole | Battery: AAA (included) | Memory: Recalls last timer setting",
            "target": "busy cooks juggling multiple dishes and brew times simultaneously",
        },
        {
            "name": "NoriRoller Bamboo Sushi Mat Set",
            "subcategory": "prep",
            "features": "Set: 2 bamboo rolling mats + 1 rice paddle + 1 spreader | Mat size: 9.5 x 9.5 inches | Material: Natural bamboo slats with cotton string | Includes: 5 pairs disposable chopsticks | Care: Hand wash, air dry | Food safe: No coatings",
            "target": "home sushi makers who want traditional tools for maki and temaki rolls",
        },
        {
            "name": "CrispRack Stainless Steel Cooling Rack Set",
            "subcategory": "baking",
            "features": "Set: 2 racks (17x12 inches, fits half-sheet pan) | Material: 18/8 stainless steel | Grid: Tight crosswire, 0.5-inch spacing | Legs: Raised 0.75 inches for airflow | Oven-safe: up to 575 degrees F | Use: Cooling, glazing, roasting, broiling | Dishwasher: Safe | Weight: 1.1 lbs each",
            "target": "bakers and roasters who need even airflow for cooling cookies and crisping meats",
        },
        {
            "name": "TasteRing Silicone Egg Ring Set",
            "subcategory": "tools",
            "features": "Set: 4 rings | Diameter: 3.5 inches | Material: Food-grade silicone | Heat resistance: up to 450 degrees F | Handle: Stainless steel fold-out | Nonstick: Naturally releases | Use: Eggs, pancakes, English muffins | Dishwasher: Safe",
            "target": "breakfast cooks who want uniform round eggs and pancakes",
        },
        {
            "name": "StockPot Heavy-Gauge 12-Quart",
            "subcategory": "cookware",
            "features": "Capacity: 12-quart | Material: 18/10 stainless steel, aluminum-encapsulated base | Lid: Tempered glass with steam vent | Handles: Riveted stainless steel, stay-cool | Compatible: All stovetops including induction | Oven-safe: up to 500 degrees F | Dishwasher: Safe | Interior markings: Quart lines",
            "target": "batch cookers making stocks, soups, pasta, and canning large quantities",
        },
        {
            "name": "CrepeMaker Electric Griddle",
            "subcategory": "appliances",
            "features": "Surface: 12-inch nonstick cooking plate | Temperature: Adjustable dial (200-400 degrees F) | Heat-up: 2 minutes | Power: 1000W | Includes: Wooden spreader + spatula | Cord: 36-inch with wrap storage | Cleaning: Nonstick surface wipes clean | Weight: 3.4 lbs",
            "target": "crepe and flatbread enthusiasts who want thin, even results at home",
        },
        {
            "name": "JarPop Canning Funnel and Lifter Set",
            "subcategory": "tools",
            "features": "Set: Wide-mouth funnel + jar lifter + magnetic lid wand + bubble popper/measurer | Funnel: Fits regular and wide-mouth mason jars | Material: Stainless steel funnel, silicone-coated lifter tongs | Dishwasher: All pieces safe | Storage: Nests together",
            "target": "home canners and preservers who jar jams, pickles, and sauces",
        },
        {
            "name": "TeaCaddy Airtight Tin Storage Set",
            "subcategory": "storage",
            "features": "Set: 4 tins | Capacity: 4 oz each | Material: Tin-plated steel with food-safe lining | Seal: Double-lid (inner + outer) for airtight storage | Shape: Cylindrical, 3 x 3.5 inches | Labels: 8 chalkboard labels + chalk marker | Use: Loose leaf tea, spices, herbs",
            "target": "tea drinkers and spice collectors who want light-sealed, organized countertop storage",
        },
        {
            "name": "ProTong 12-Inch Locking Kitchen Tongs",
            "subcategory": "tools",
            "features": "Length: 12 inches | Material: Stainless steel with silicone tips | Heat resistance: Tips safe to 480 degrees F | Lock: Pull-ring locking mechanism for storage | Grip: Scalloped silicone edges | Weight: 5.6 oz | Dishwasher: Safe | Use: Grilling, serving, plating, tossing salads",
            "target": "grillers and home cooks who need all-purpose tongs that lock closed for drawer storage",
        },
        {
            "name": "MapleCraft Wooden Spoon Set",
            "subcategory": "tools",
            "features": "Set: 5 spoons (cooking spoon, slotted spoon, spatula, corner spoon, tasting spoon) | Material: North American hard maple | Finish: Food-safe mineral oil | Length: 12 inches each | Handle: Rounded, comfortable grip | Cookware safe: Will not scratch nonstick | Care: Hand wash, oil periodically",
            "target": "home cooks who prefer wooden utensils and want a complete matching set",
        },
        {
            "name": "RotaryPeel Vegetable Peeler Set",
            "subcategory": "tools",
            "features": "Set: 3 peelers (straight, serrated, julienne) | Blade: Ceramic, stays sharp 10x longer than steel | Handle: Soft-touch ergonomic grip | Weight: 1.2 oz each | Dishwasher: Safe | Colors: Color-coded by blade type | Right and left-hand friendly: Swivel blade",
            "target": "home cooks who peel vegetables daily and want blades that stay sharp without sharpening",
        },
    ]
    rng.shuffle(products)
    for p in products[:count]:
        p["category"] = "kitchen"
    return products[:count]


def generate_outdoor_fitness(fake: Faker, rng: random.Random, count: int) -> list[dict]:
    """Generate outdoor/fitness product specs with performance-focused language."""
    products = [
        {
            "name": "TrailBlaze 12L Hydration Pack",
            "subcategory": "packs",
            "features": "Capacity: 12L main + 2L water reservoir | Weight: 320g (empty) | Fit: Unisex, XS-XL adjustable sternum and waist straps | Pockets: 8 external + front phone sleeve | Material: 210D ripstop nylon | Hydration: Compatible with 2L reservoir (included) | Back panel: Ventilated mesh",
            "target": "trail runners and day hikers covering 10-25 mile distances",
        },
        {
            "name": "PowerGrip Adjustable Dumbbell (5-52.5 lbs)",
            "subcategory": "strength",
            "features": "Weight range: 5 to 52.5 lbs in 2.5 lb increments (first 25 lbs) then 5 lb increments | Mechanism: Quick-turn dial selector | Material: Steel plates with molded urethane coating | Handle: Contoured, non-slip | Footprint: 16.9 x 8.3 inches per dumbbell | Replaces: 15 sets of dumbbells | Tray: Included storage cradle",
            "target": "home gym builders who need a full dumbbell rack in minimal floor space",
        },
        {
            "name": "SummitLite 2-Person Backpacking Tent",
            "subcategory": "camping",
            "features": "Weight: 3 lbs 6 oz (packed) | Setup: Freestanding, 2-pole hubbed | Floor area: 28 sq ft | Peak height: 42 inches | Vestibules: 2 (8 sq ft each) | Rainfly: 20D sil-nylon, 1500mm HH | Floor: 30D nylon, 3000mm HH | Packed size: 6 x 18 inches | Seasons: 3-season",
            "target": "backpackers and thru-hikers who count every ounce on multi-day trips",
        },
        {
            "name": "FlexBand Pro Resistance Band Set",
            "subcategory": "strength",
            "features": "Bands: 5 latex loops (10, 20, 30, 40, 50 lbs) | Length: 12 inches (loop), 48 inches (stretched) | Material: Natural latex rubber | Includes: Door anchor, 2 handles, 2 ankle straps, carry bag | Stackable: Combine for up to 150 lbs | Color-coded: Resistance by color | Latex-free option: TPE version available",
            "target": "travelers, home gym users, and physiotherapy patients needing portable progressive resistance",
        },
        {
            "name": "RideSteady Indoor Cycling Trainer",
            "subcategory": "cycling",
            "features": "Resistance: Magnetic, 16 levels via handlebar lever | Noise: <65dB at 20mph | Compatibility: 26-29 inch wheels and 700c | Clamp: Quick-release rear axle (130/135mm) | Flywheel: 6 lbs | Power accuracy: +/-5% (with optional sensor) | Folding: Legs fold for storage | Weight capacity: 250 lbs",
            "target": "cyclists maintaining fitness indoors during bad weather without waking the household",
        },
        {
            "name": "GripForce Climbing Hangboard",
            "subcategory": "climbing",
            "features": "Material: Polyurethane resin | Holds: 22 positions (edges: 6mm-35mm, slopers, jugs, pockets: 2/3/4 finger) | Mounting: 4-bolt (hardware included) | Dimensions: 22 x 6.5 x 2.2 inches | Weight: 5.5 lbs | Texture: Skin-friendly fine grain | Training guide: Included 8-week program PDF",
            "target": "rock climbers training finger strength and grip endurance at home",
        },
        {
            "name": "AquaStroke Folding Rowing Machine",
            "subcategory": "cardio",
            "features": "Resistance: Water flywheel (adjustable fill level, 2-6 intensity) | Rail: Anodized aluminum, 38-inch stroke length | Seat: Contoured with ball-bearing rollers | Display: LCD (time, distance, strokes, calories, 500m split) | Folding: Upright storage, 26 x 22 inch footprint | Capacity: 300 lbs | Tank: 4.2 gallon polycarbonate | Noise: Water swoosh only",
            "target": "home gym users who want full-body, low-impact cardio with a natural rowing feel",
        },
        {
            "name": "HeadLamp X2 Rechargeable Trail Light",
            "subcategory": "accessories",
            "features": "Lumens: 600 (max) / 30 (low) / red night mode | Beam: Spot + flood hybrid | Battery: 2600mAh USB-C rechargeable | Runtime: 3h (high), 50h (low), 80h (red) | Weight: 75g including band | Water: IPX6 | Band: Reflective, adjustable, silicone-grip | Lock mode: Prevents accidental activation",
            "target": "trail runners, campers, and early-morning hikers who need hands-free lighting",
        },
        {
            "name": "CoreWheel Ab Roller with Knee Pad",
            "subcategory": "strength",
            "features": "Wheel: 3.5-inch diameter, dual-wheel for stability | Material: Steel axle, TPR rubber treads | Handles: Foam grip, ergonomic angle | Knee pad: 15mm thick, 13 x 7 inches | Weight capacity: 440 lbs | Assembled weight: 1.3 lbs | Width: 7 inches (prevents tipping) | Surface: Works on hard floor, carpet, or gym mat",
            "target": "core training enthusiasts building abdominal and upper-body strength at home",
        },
        {
            "name": "StridePro Carbon Trekking Poles (Pair)",
            "subcategory": "hiking",
            "features": "Material: 100% carbon fiber shafts | Weight: 7.2 oz each | Length: Adjustable 100-135cm (lever lock) | Grip: Cork with extended EVA foam | Strap: Padded, quick-release | Tips: Carbide + rubber, interchangeable | Basket: Standard + snow basket included | Sections: 3-section collapsible (15 inches packed)",
            "target": "long-distance hikers and trekkers who want ultralight support on technical terrain",
        },
        {
            "name": "BalanceBoard Pro Wobble Trainer",
            "subcategory": "strength",
            "features": "Deck: 16-inch diameter, birch plywood with grip tape | Base: TPE rubber hemisphere, 15-degree max tilt | Weight capacity: 350 lbs | Height: 3.5 inches | Surface: Anti-slip textured | Use: Balance training, ankle rehab, core activation, desk standing | Weight: 3.2 lbs",
            "target": "athletes rehabbing ankle injuries and desk workers incorporating movement into their day",
        },
        {
            "name": "CampChef Ultralight Backpacking Stove",
            "subcategory": "camping",
            "features": "Fuel: Isobutane/propane canisters (screw-on) | Boil time: 3 minutes (1L water) | Output: 10000 BTU | Weight: 2.6 oz (stove head only) | Packed size: 2.1 x 1.5 inches | Pot support: 4 arms, fits cookware up to 8 inches | Piezo igniter: Built-in | Wind: Partial windscreen design",
            "target": "ultralight backpackers and bike-tourers who need fast hot water with minimal pack weight",
        },
        {
            "name": "SprintFlex Agility Ladder Set",
            "subcategory": "training",
            "features": "Length: 20 feet (12 rungs) | Rung spacing: Adjustable 15-20 inches | Material: Heavy-duty nylon straps, plastic rungs | Includes: 12 disc cones + carry bag | Stakes: 4 ground stakes for outdoor use | Width: 17 inches | Weight: 1.2 lbs | Flat design: Lies flat on any surface",
            "target": "athletes, coaches, and fitness enthusiasts improving footwork, speed, and coordination",
        },
        {
            "name": "ThermoFlask 32oz Insulated Water Bottle",
            "subcategory": "hydration",
            "features": "Capacity: 32 oz | Insulation: Double-wall vacuum stainless steel | Cold retention: 24 hours | Hot retention: 12 hours | Lid: Leak-proof flip-top with carry loop | Material: 18/8 food-grade stainless steel | BPA-free: All components | Mouth: Wide-mouth (fits ice cubes) | Weight: 14.4 oz empty",
            "target": "gym-goers, hikers, and office workers who want cold water available all day",
        },
        {
            "name": "SuspendFit Bodyweight Training Straps",
            "subcategory": "strength",
            "features": "Strap length: Adjustable 36-78 inches | Material: 1.5-inch nylon webbing, 1500 lb rated | Handles: Padded neoprene with rubber grip | Anchor: Door anchor + ceiling/beam carabiner mount | Weight capacity: 400 lbs | Exercises: 100+ bodyweight movements | Carry bag: Drawstring included | Setup: Under 60 seconds",
            "target": "travelers and home exercisers who want a portable full-body gym",
        },
        {
            "name": "QuickDry Microfiber Camp Towel",
            "subcategory": "camping",
            "features": "Size: 60 x 30 inches (bath) | Material: Suede microfiber | Weight: 7.4 oz | Dry time: 3x faster than cotton | Absorbency: Holds 5x its weight in water | Packed size: 5 x 3 inches (with snap loop) | Includes: Carrying pouch | Antimicrobial: Silver-ion treated",
            "target": "backpackers, gym-goers, and swimmers who need a lightweight, fast-drying towel",
        },
        {
            "name": "PeakView Compact Binoculars",
            "subcategory": "optics",
            "features": "Magnification: 10x42 | Prism: BAK-4 roof | Lens coating: Fully multi-coated | FOV: 341 ft at 1000 yards | Eye relief: 16mm | Weight: 22 oz | Close focus: 6.5 feet | Water/fog: IPX7 nitrogen-purged | Includes: Padded case, neck strap, lens caps",
            "target": "birdwatchers, hunters, and outdoor enthusiasts who want clear optics in a packable size",
        },
        {
            "name": "FlexGrip Yoga Mat 6mm",
            "subcategory": "yoga",
            "features": "Thickness: 6mm | Length: 72 inches | Width: 26 inches | Material: TPE (thermoplastic elastomer), latex-free | Weight: 2.8 lbs | Surface: Dual-texture (grip top, cushion bottom) | Closed-cell: Sweat and moisture resistant | Includes: Carry strap | Alignment marks: Laser-etched center line",
            "target": "yoga practitioners who need joint cushioning on hard floors without sacrificing grip",
        },
        {
            "name": "MountainFilter Gravity Water Purifier",
            "subcategory": "camping",
            "features": "Method: Hollow fiber membrane (0.1 micron) | Flow rate: 1.5L per minute (gravity) | Capacity: 4L dirty bag + 4L clean bag | Removes: 99.99% bacteria, 99.99% protozoa | Weight: 11.5 oz (system) | Cartridge life: 4000 liters | Backflush: Tool-free, 10-second process | Packed: Rolls to 9 x 3 inches",
            "target": "group campers and backcountry hikers filtering water for 2-6 people at camp",
        },
        {
            "name": "PaceTracker Jump Rope with Counter",
            "subcategory": "cardio",
            "features": "Display: Built-in LCD (jumps, calories, timer) | Rope: PVC-coated steel cable, 3mm | Length: Adjustable up to 9.8 feet | Handles: Weighted (each 170g) with ball bearing swivel | Battery: 2x AAA (included) | Weight: 12.7 oz total | Memory: Stores cumulative totals | Replacement cable: Included",
            "target": "boxers, CrossFit athletes, and anyone who wants a measurable cardio warm-up",
        },
        {
            "name": "SunShield UPF 50+ Hiking Hat",
            "subcategory": "accessories",
            "features": "Protection: UPF 50+ certified | Brim: 3.5-inch all-around | Material: Nylon ripstop with mesh crown panels | Chin strap: Adjustable toggle | Fit: Drawcord adjustment, one-size (fits 21.5-24 inches) | Weight: 2.8 oz | Packable: Folds flat without crease | Quick-dry: Moisture-wicking sweatband",
            "target": "hikers, anglers, and gardeners who need all-day sun protection that packs flat",
        },
        {
            "name": "RecoverRoll Vibrating Foam Roller",
            "subcategory": "recovery",
            "features": "Vibration: 5 intensity levels (1200-3700 RPM) | Length: 18 inches | Diameter: 5.5 inches | Surface: High-density EPP foam, textured | Battery: 4h runtime, USB-C rechargeable | Weight: 2.2 lbs | Noise: <45dB | Auto shutoff: 10 minutes | Includes: Carry bag",
            "target": "athletes and runners who need deep-tissue myofascial release after training",
        },
        {
            "name": "AnchorStrap Tree-Friendly Hammock Straps (Pair)",
            "subcategory": "camping",
            "features": "Length: 10 feet each | Width: 1 inch (tree-safe) | Loops: 20 adjustment points per strap | Material: Nautical-grade polyester | Capacity: 400 lbs combined | Weight: 7 oz per strap | Stretch: <3% | Setup: Looped end + carabiner (2 included), under 60 seconds | No knots needed",
            "target": "hammock campers who want fast, adjustable setup without damaging bark",
        },
        {
            "name": "FrostGuard Insulated Trail Gloves",
            "subcategory": "accessories",
            "features": "Insulation: 150g synthetic | Shell: Softshell with DWR treatment | Lining: Brushed fleece | Touchscreen: Index finger and thumb conductive tips | Grip: Silicone palm print | Cuff: Elastic with pull-on tab | Sizes: XS-XXL | Temperature range: 20-45 degrees F",
            "target": "hikers, runners, and dog walkers active in cold weather who need dexterity and phone access",
        },
        {
            "name": "TideRunner Neoprene Water Shoes",
            "subcategory": "water",
            "features": "Upper: 3mm neoprene + breathable mesh | Sole: Rubber with multi-directional tread | Drainage: 4 mesh ports per shoe | Closure: Adjustable bungee toggle | Weight: 7.6 oz per shoe (men's 10) | Protection: Reinforced toe cap | Sizes: Men's 6-14, Women's 5-12 | Quick-dry: Under 2 hours",
            "target": "kayakers, stand-up paddlers, and beach hikers crossing rocky shorelines",
        },
        {
            "name": "PedalLock Folding Bike Lock",
            "subcategory": "cycling",
            "features": "Lock type: Folding bar (6 riveted links) | Length: 33 inches (extended) | Material: Hardened steel bars, 5mm thick | Lock cylinder: Disc detainer, pick-resistant | Weight: 1.7 lbs | Folded size: 3.5 x 2 x 1.5 inches | Mount: Frame bracket included | Keys: 3 included (1 with LED light)",
            "target": "urban cyclists who need a compact, high-security lock for daily commuting",
        },
        {
            "name": "WindBreaker Packable Running Jacket",
            "subcategory": "running",
            "features": "Material: 20D ripstop nylon with DWR | Weight: 3.6 oz (men's M) | Packable: Stuffs into chest pocket (4 x 3 inches) | Wind resistance: Tested to 45 mph | Water resistance: Light rain for 30 min | Reflective: 360-degree reflective accents | Ventilation: Laser-cut back panel | Fit: Athletic, drop-tail hem",
            "target": "runners and cyclists who need a windproof layer that disappears into a pocket",
        },
        {
            "name": "GripTape Athletic Kinesiology Tape Roll",
            "subcategory": "recovery",
            "features": "Length: 16.4 feet (5m) per roll | Width: 2 inches | Material: 97% cotton, 3% spandex | Adhesive: Acrylic, latex-free, hypoallergenic | Stretch: 140-180% longitudinal elasticity | Wear time: Up to 5 days | Water resistant: Stays on through sweat and showers | Pre-cut: 10-inch strips with finger-tear backing",
            "target": "athletes and physical therapy patients supporting joints and muscles during activity",
        },
        {
            "name": "SummitStick Collapsible Trekking Stool",
            "subcategory": "camping",
            "features": "Weight capacity: 250 lbs | Seat height: 20 inches | Weight: 2 lbs | Material: 7075 aluminum frame, 600D polyester seat | Packed size: 13.5 x 3.5 inches | Setup: Pull open, auto-locks in 5 seconds | Legs: 3-point with anti-sink feet | Seat: Breathable mesh center",
            "target": "hikers, festival-goers, and photographers who need a lightweight seat for rest stops",
        },
        {
            "name": "NightTrail Reflective Running Vest",
            "subcategory": "running",
            "features": "Visibility: 360-degree 3M Scotchlite reflective | Fit: Adjustable side straps, one-size (fits over jackets) | Weight: 3.2 oz | Material: Breathable mesh back | Pockets: 2 front zip (phone + keys) | LED: Optional clip-on LED loop (not included) | Closure: Front zipper | Wash: Machine washable",
            "target": "early-morning and evening runners who train on roads and need high visibility",
        },
        {
            "name": "QuickFold Camp Chair",
            "subcategory": "camping",
            "features": "Weight capacity: 300 lbs | Seat height: 18 inches | Weight: 4.2 lbs | Material: Steel frame, 600D polyester seat | Packed size: 34 x 6 x 6 inches | Cup holder: Mesh, built-in on armrest | Carry bag: Included | Setup: Unfolds in 3 seconds",
            "target": "car campers and tailgaters who want a sturdy, comfortable seat with no assembly",
        },
        {
            "name": "SplashGuard Dry Bag 20L",
            "subcategory": "water",
            "features": "Capacity: 20L | Material: 500D PVC tarpaulin | Closure: Roll-top with buckle | Waterproof: IPX8 (submersible) | Strap: Removable shoulder strap | Weight: 12 oz | Welded seams | Colors: 4 high-visibility options | Dimensions: 22 x 10 inches (unrolled)",
            "target": "kayakers, rafters, and beach-goers protecting electronics and dry clothes",
        },
        {
            "name": "PeakGrip Approach Shoes",
            "subcategory": "climbing",
            "features": "Upper: Suede leather with synthetic mesh | Sole: Stealth C4 rubber, 4mm lugs | Midsole: Compression-molded EVA | Lacing: Asymmetric, climbing-zone wrap | Weight: 13.5 oz per shoe (men's 9) | Toe: Rubber toe cap | Sizes: Men's 6-14, Women's 5-11 | Use: Scrambling, light climbing, approach trails",
            "target": "climbers and scramblers who hike to the crag and need grip on rock and trail",
        },
        {
            "name": "EndureFuel Energy Gel Flask",
            "subcategory": "hydration",
            "features": "Capacity: 150ml (5 gel servings) | Material: BPA-free soft TPU | Valve: One-hand squeeze, self-sealing | Weight: 28g empty | Attachment: Bungee loop for waistbelt or vest | Opening: Wide-mouth for refilling | Dishwasher: Top rack safe | Markings: 50ml increment lines",
            "target": "marathon runners and ultra-distance athletes who refill with bulk gel",
        },
        {
            "name": "CampLight Rechargeable Lantern",
            "subcategory": "camping",
            "features": "Lumens: 300 (high), 30 (low), red mode | Battery: 4400mAh USB-C rechargeable | Runtime: 8h (high), 100h (low) | Weight: 7.6 oz | Water: IPX4 | Hang loop: Built-in carabiner clip | Collapsible: Packs to 1.5 inches tall | Power bank: Can charge a phone",
            "target": "campers and backpackers who need a compact lantern that doubles as a power bank",
        },
        {
            "name": "PowerStep Weighted Jump Rope (1 lb)",
            "subcategory": "cardio",
            "features": "Weight: 1 lb (handles + rope) | Rope: PVC-coated steel, 9.5 feet adjustable | Handles: Weighted foam grip (each 0.25 lbs) | Bearing: Double ball bearing swivel | Diameter: 6mm rope | Replacement rope: Included | Travel bag: Drawstring pouch",
            "target": "boxers and HIIT athletes who want added upper-body engagement during rope work",
        },
        {
            "name": "TractionSpike Trail Running Shoes",
            "subcategory": "running",
            "features": "Upper: Engineered mesh with TPU overlays | Sole: Vibram Megagrip, 5mm lugs | Midsole: Dual-density EVA, 25mm stack | Drop: 4mm | Weight: 10.2 oz (men's 9) | Rock plate: TPU forefoot protection | Drainage: Mesh ports for water crossings | Sizes: Men's 7-14, Women's 5-12",
            "target": "trail runners tackling muddy, rocky terrain who need aggressive grip and foot protection",
        },
        {
            "name": "FlexTube Pilates Resistance Ring",
            "subcategory": "strength",
            "features": "Diameter: 14 inches | Material: Fiberglass core with foam-padded handles | Resistance: Medium (about 10 lbs) | Handles: Dual-sided padded grip | Weight: 12 oz | Use: Inner/outer thigh, chest, arms, core | Travel-friendly: Fits in a gym bag | Latex-free",
            "target": "Pilates practitioners and rehab patients working inner thigh and core stability",
        },
        {
            "name": "SummitSack Compression Stuff Sack Set",
            "subcategory": "camping",
            "features": "Set: 3 sacks (5L, 10L, 20L) | Material: 30D sil-nylon ripstop | Closure: Drawstring + 4 compression straps | Weight: 1.2 oz (5L), 1.8 oz (10L), 2.6 oz (20L) | Water resistance: DWR treated (not submersible) | Use: Sleeping bags, clothing, insulation | Colors: Color-coded by size",
            "target": "backpackers who need to minimize pack volume on multi-day trips",
        },
        {
            "name": "ImpactShield Mouthguard with Case",
            "subcategory": "training",
            "features": "Material: Medical-grade EVA | Fit: Boil-and-bite custom mold | Thickness: 4mm at impact zones | Breathing: Front channel for unrestricted airflow | Size: Adult (ages 12+) | Case: Ventilated hard case with clip | BPA-free: Yes | Flavor: Mint-infused (optional)",
            "target": "contact sport athletes in boxing, MMA, basketball, and rugby",
        },
        {
            "name": "TrailFix Multi-Tool Bike Repair Kit",
            "subcategory": "cycling",
            "features": "Tools: 16 functions (hex keys 2-8mm, Torx T25, screwdrivers, chain tool, spoke wrench, tire lever) | Material: Chrome vanadium steel | Weight: 6.4 oz | Size: 3.5 x 1.5 x 0.8 inches | Case: Neoprene pouch with Velcro strap mount | Compatibility: Most road and mountain bikes",
            "target": "cyclists who need a compact trailside repair kit for rides far from a bike shop",
        },
        {
            "name": "StretchZone Resistance Loop Band Set",
            "subcategory": "strength",
            "features": "Bands: 4 fabric loops (light, medium, heavy, extra heavy) | Material: Woven polyester-cotton-latex blend | Width: 3 inches (won't roll up) | Loop size: 13 x 3 inches | Carry bag: Mesh drawstring | Wash: Machine washable | Use: Glute activation, hip abduction, warm-ups",
            "target": "glute-focused trainers and physical therapy patients doing banded hip work",
        },
        {
            "name": "BivyStar Emergency Shelter Bivy",
            "subcategory": "camping",
            "features": "Material: Aluminized polyethylene (reflects 90% body heat) | Weight: 3.8 oz | Packed: 5 x 3 x 1 inches | Capacity: 1 person | Opening: Full-length zipper for entry | Reusable: Yes (unlike emergency blankets) | Visibility: Bright orange exterior | Use: Emergency survival, unexpected weather",
            "target": "hikers and trail runners carrying an ultralight emergency backup shelter",
        },
        {
            "name": "AquaFloat Inflatable Stand-Up Paddleboard",
            "subcategory": "water",
            "features": "Length: 10 feet 6 inches | Width: 32 inches | Thickness: 6 inches | Weight capacity: 275 lbs | Material: Military-grade PVC, triple-layer drop-stitch | Weight: 19 lbs (board) | Inflates: 10-15 minutes with included hand pump | PSI: 15 | Includes: Adjustable paddle, leash, fin, backpack carry bag",
            "target": "recreational paddlers and lake-goers who want a portable board that fits in a car trunk",
        },
        {
            "name": "GravityStrap Inversion Table",
            "subcategory": "recovery",
            "features": "Height range: 4 ft 10 in to 6 ft 6 in | Weight capacity: 300 lbs | Inversion range: 20-180 degrees (adjustable tether) | Ankle lock: Foam rollers with ratchet clamp | Frame: Heavy-gauge steel | Backrest: Memory foam with lumbar bridge | Foldable: 20 x 28 inch footprint when folded | Weight: 49 lbs",
            "target": "back pain sufferers and athletes seeking decompression and spinal traction at home",
        },
        {
            "name": "SpeedCone Training Set (50-pack)",
            "subcategory": "training",
            "features": "Quantity: 50 cones | Height: 2 inches | Diameter: 7.5 inches | Material: Flexible LDPE plastic (won't crack) | Colors: 5 colors x 10 each | Carry strap: Included | Weight: 3 lbs total | Use: Agility drills, boundary markers, parking, sports practice",
            "target": "coaches, PE teachers, and athletes setting up agility courses and field boundaries",
        },
        {
            "name": "EnduroGel Cycling Shorts (Men's)",
            "subcategory": "cycling",
            "features": "Pad: 3D multi-density gel chamois | Material: 80% nylon, 20% spandex | Compression: Graduated | Inseam: 9 inches | Waistband: Wide silicone-dotted elastic | Leg grippers: Silicone-print hem | UPF: 50+ | Flatlock seams: Chafe-free | Sizes: S-XXL",
            "target": "road and gravel cyclists riding 30+ miles who need long-ride comfort",
        },
        {
            "name": "TrailNav Handheld GPS Unit",
            "subcategory": "hiking",
            "features": "GPS: Multi-GNSS (GPS + GLONASS + Galileo) | Display: 3-inch sunlight-readable color | Battery: 16h GPS mode (2x AA) | Memory: 8GB internal + microSD | Maps: Preloaded topo (continental) | Navigation: Waypoints, routes, tracks, breadcrumb | Water: IPX7 | Weight: 5 oz | Connectivity: Bluetooth for phone notifications",
            "target": "backcountry hikers and hunters who want reliable navigation without relying on a phone",
        },
        {
            "name": "FoamFit High-Density Yoga Block Set",
            "subcategory": "yoga",
            "features": "Set: 2 blocks | Dimensions: 9 x 6 x 4 inches each | Material: High-density EVA foam | Weight: 7 oz each | Surface: Non-slip beveled edges | Firmness: Supportive, does not compress under body weight | Colors: 4 solid options | Latex-free and moisture-resistant",
            "target": "yoga practitioners who need stable support for balance poses and flexibility work",
        },
        {
            "name": "PowerPull Portable Chin-Up Bar",
            "subcategory": "strength",
            "features": "Mounting: Doorframe pressure mount (no screws) | Fits: Doorways 24-36 inches wide | Grip positions: 3 (wide, neutral, close) | Capacity: 300 lbs | Material: Steel bar with foam grips | Bar diameter: 1.25 inches | Weight: 5.8 lbs | Extra: Can be used as floor push-up bar (angled grips)",
            "target": "home bodyweight trainers who want pull-ups, chin-ups, and hanging exercises without permanent installation",
        },
    ]
    rng.shuffle(products)
    for p in products[:count]:
        p["category"] = "outdoor_fitness"
    return products[:count]


def generate_fashion(fake: Faker, rng: random.Random, count: int) -> list[dict]:
    """Generate fashion product specs with aesthetic/sensory language."""
    products = [
        {
            "name": "CloudKnit Merino Wool Crew",
            "subcategory": "tops",
            "features": "Material: 100% 17.5-micron merino wool | Weight: 200gsm | Fit: Relaxed | Sizes: XS-3XL | Colors: 12 seasonal colorways | Care: Machine wash cold, lay flat to dry | Origin: Responsibly sourced New Zealand wool | Odor resistance: Natural antimicrobial",
            "target": "travelers who prioritize packability and natural odor resistance",
        },
        {
            "name": "StretchDenim High-Rise Slim Jeans",
            "subcategory": "bottoms",
            "features": "Material: 92% cotton, 6% polyester, 2% elastane | Weight: 11 oz denim | Rise: 10.5 inches | Inseam: 28, 30, 32 inches | Closure: Zip fly with branded button | Pockets: 5-pocket classic | Wash: Medium indigo, enzyme-washed | Sizes: 24-36",
            "target": "everyday wearers who want a polished look with all-day stretch comfort",
        },
        {
            "name": "VelvetStep Suede Chelsea Boots",
            "subcategory": "footwear",
            "features": "Upper: Genuine calf suede | Sole: Stacked leather heel (1.5 inches) with rubber outsole | Construction: Blake stitched | Elastic: Gore panels on both sides | Pull tab: Rear canvas loop | Lining: Breathable leather | Sizes: Men's 7-13, half sizes available | Care: Includes suede brush",
            "target": "style-conscious professionals who transition from office to evening wear",
        },
        {
            "name": "ArcticDown Puffer Jacket",
            "subcategory": "outerwear",
            "features": "Fill: 700-fill-power RDS-certified goose down | Shell: 20D ripstop nylon, DWR coated | Weight: 12.8 oz (men's M) | Warmth: Comfort-rated to 20 degrees F | Packable: Stuffs into internal chest pocket | Pockets: 2 hand + 1 internal zip | Collar: Stand-up with chin guard | Sizes: XS-XXL",
            "target": "commuters and travelers who need serious warmth that packs into a daypack",
        },
        {
            "name": "SilkTouch Satin Wrap Dress",
            "subcategory": "dresses",
            "features": "Material: 95% recycled polyester satin, 5% spandex | Weight: Lightweight, fluid drape | Closure: Self-tie wrap with internal snap | Length: Midi (hits below knee) | Sleeves: Three-quarter | Sizes: 0-16 | Colors: 8 solid tones | Care: Machine wash cold, line dry",
            "target": "professional women who need a desk-to-dinner wardrobe staple",
        },
        {
            "name": "TrailWeave Waxed Canvas Backpack",
            "subcategory": "bags",
            "features": "Material: 18 oz waxed canvas body, full-grain leather straps | Capacity: 22L | Laptop: Padded sleeve fits up to 15 inches | Closure: Drawstring + leather buckle flap | Pockets: 2 side, 1 front zip, 1 internal organizer | Dimensions: 17 x 12 x 6 inches | Weight: 2.8 lbs | Hardware: Antique brass",
            "target": "urban commuters who want a heritage-style bag that handles rain and daily use",
        },
        {
            "name": "LineFrame Titanium Aviator Sunglasses",
            "subcategory": "accessories",
            "features": "Frame: Beta-titanium, 18g total | Lenses: Polarized CR-39, UV400 protection | Lens width: 58mm | Bridge: 15mm | Temple: 140mm | Hinge: 5-barrel spring | Includes: Hard case, microfiber cloth | Colors: Gold/green, silver/gray, matte black/brown",
            "target": "drivers and outdoor enthusiasts who want glare reduction in a lightweight, durable frame",
        },
        {
            "name": "CashmereMist V-Neck Sweater",
            "subcategory": "tops",
            "features": "Material: 100% Grade-A Mongolian cashmere (2-ply) | Gauge: 12-gauge knit | Weight: 8.5 oz (men's M) | Fit: Regular | Sizes: XS-XXL | Neckline: V-neck with ribbed trim | Cuffs: Ribbed | Colors: 10 muted tones | Care: Hand wash or dry clean",
            "target": "professionals layering under blazers who value soft texture against skin",
        },
        {
            "name": "UrbanStride Leather Minimalist Sneakers",
            "subcategory": "footwear",
            "features": "Upper: Full-grain Italian calf leather | Sole: Margom rubber, 25mm | Lining: Leather | Insole: Memory foam with leather cover | Construction: Cemented with stitch reinforcement | Weight: 13 oz per shoe (men's 9) | Sizes: Men's 7-13, Women's 5-11 | Colors: White, black, navy",
            "target": "minimalist dressers who want a clean sneaker that works with chinos or jeans",
        },
        {
            "name": "HarborWash Linen Button-Down Shirt",
            "subcategory": "tops",
            "features": "Material: 100% European flax linen | Weight: 150gsm | Weave: Plain, garment-washed for softness | Collar: Button-down, unfused | Fit: Regular with back pleat | Buttons: Mother-of-pearl | Sizes: S-XXL | Colors: 6 muted coastal tones | Hem: Curved, wearable untucked",
            "target": "warm-climate dressers who want breathable texture with a relaxed weekend feel",
        },
        {
            "name": "FlexWool Tailored Trousers",
            "subcategory": "bottoms",
            "features": "Material: 98% merino wool, 2% elastane | Weight: 240gsm tropical weight | Fit: Slim-tapered | Rise: Mid (10 inches) | Closure: Hook and bar with zip | Waistband: Half-lined, split-back for give | Crease: Permanent center crease | Sizes: 28-40 (inseam: 30, 32, 34) | Dry clean recommended",
            "target": "office professionals who want wrinkle-resistant trousers that stretch through a full workday",
        },
        {
            "name": "DriftKnit Recycled Sneaker",
            "subcategory": "footwear",
            "features": "Upper: 3D-knit from recycled plastic bottles (12 bottles per pair) | Sole: SugarFoam (sugarcane-based EVA) | Insole: Castor bean oil memory foam | Weight: 7.5 oz per shoe (men's 9) | Washable: Machine wash cold | Sizes: Men's 7-14, Women's 5-12 | Colors: 8 heathered tones | Carbon footprint: 5.5 kg CO2 per pair",
            "target": "eco-conscious consumers who want comfortable everyday footwear with transparent sustainability metrics",
        },
        {
            "name": "VintageGrain Leather Belt",
            "subcategory": "accessories",
            "features": "Material: Full-grain vegetable-tanned cowhide | Width: 1.25 inches | Thickness: 4mm | Buckle: Solid brass, removable (fits standard 1.25-inch buckles) | Holes: 5 (with additional punch tool included) | Edge: Burnished by hand | Sizes: 30-44 (measure at pant waist) | Patina: Develops character with wear",
            "target": "men and women who appreciate leather goods that age into a personal patina",
        },
        {
            "name": "StormShell 3-Layer Rain Jacket",
            "subcategory": "outerwear",
            "features": "Membrane: 20K/20K waterproof/breathable | Shell: 40D recycled nylon face | Seams: Fully taped | Hood: Adjustable, helmet-compatible, stows in collar | Pockets: 2 hand zip, 1 chest zip, all with storm flaps | Ventilation: Pit zips | Packed size: 5 x 7 inches | Weight: 11 oz (men's M)",
            "target": "hikers and cyclists who need reliable rain protection that breathes during high-output activity",
        },
        {
            "name": "OceanThread Striped Breton Tee",
            "subcategory": "tops",
            "features": "Material: 100% heavyweight combed cotton (220gsm) | Knit: Interlock (smooth both sides) | Neckline: Boat neck | Fit: Relaxed, dropped shoulder | Stripes: 14mm navy on off-white | Sizes: XS-XL | Shrinkage: Pre-shrunk (<1%) | Origin: Made in Portugal",
            "target": "classic dressers who want a timeless striped tee that holds its shape wash after wash",
        },
        {
            "name": "NomadRoll Packable Duffel Bag",
            "subcategory": "bags",
            "features": "Capacity: 40L | Material: 100D nylon with DWR coating | Weight: 14 oz | Packed size: Folds into 8 x 6 inch internal pocket | Carry: Padded shoulder strap + hand grips | Opening: U-shaped full zip | Pockets: 1 internal mesh, 1 external zip | Lockable: YKK zippers with loops for TSA lock",
            "target": "travelers who need an extra bag for souvenirs that takes zero space when packed",
        },
        {
            "name": "SoftStep Merino Ankle Socks (6-Pack)",
            "subcategory": "basics",
            "features": "Material: 68% merino wool, 28% nylon, 4% lycra | Weight: Lightweight cushion | Height: Ankle (just above shoe line) | Seamless toe: Linked closure, no bump | Arch support: Light compression band | Sizes: S (W 4-6.5), M (W 7-10 / M 6-8.5), L (M 9-12.5) | Care: Machine wash warm | Odor resistant: Natural merino property",
            "target": "daily wearers who want socks that stay odor-free and cushioned from morning to night",
        },
        {
            "name": "CanvasWear Relaxed Chino Short",
            "subcategory": "bottoms",
            "features": "Material: 98% organic cotton twill, 2% elastane | Weight: 7 oz | Inseam: 7 inches | Fit: Relaxed through thigh, 9.5 inch rise | Closure: Button fly | Pockets: Slant front, welt back with button | Waist: Interior drawcord option | Sizes: 28-40 | Colors: 6 garment-dyed earth tones",
            "target": "casual weekend dressers who want a clean short with enough stretch to move freely",
        },
        {
            "name": "CloudWalk Cushioned Running Shoe",
            "subcategory": "footwear",
            "features": "Midsole: Nitrogen-infused EVA foam, 34mm stack height | Upper: Engineered mesh, single-layer | Drop: 8mm | Weight: 9.1 oz (men's 9) | Outsole: Carbon rubber at heel, blown rubber at forefoot | Lacing: Traditional, reflective laces | Tongue: Gusseted, no-slip | Sizes: Men's 7-15, Women's 5-13",
            "target": "daily trainers logging 30-50 miles per week who want plush cushioning that lasts",
        },
        {
            "name": "WrapScarf Oversized Wool Blanket Scarf",
            "subcategory": "accessories",
            "features": "Material: 80% lambswool, 20% cashmere | Dimensions: 80 x 24 inches | Weight: 10 oz | Pattern: Windowpane check | Fringe: 3-inch hand-knotted | Colors: 4 seasonal plaids | Origin: Woven in Scotland | Care: Dry clean or hand wash cold",
            "target": "cold-weather commuters who want a versatile piece that works as scarf, shawl, or lap blanket",
        },
        {
            "name": "HeritageCap Waxed Cotton Five-Panel",
            "subcategory": "accessories",
            "features": "Material: British Millerain waxed cotton | Lining: Cotton flannel | Brim: Pre-curved, 2.75 inches | Closure: Leather strap with brass buckle | Eyelets: 2 per side (ventilation) | One size: Fits 56-62 cm | Water resistance: Naturally water-repellent | Re-waxing: Annual (wax bar sold separately)",
            "target": "outdoor-minded individuals who want a weather-ready cap with traditional craft details",
        },
        {
            "name": "FlexFrame Crossbody Phone Bag",
            "subcategory": "bags",
            "features": "Material: Pebbled leather with microfiber lining | Fits: Phones up to 6.7 inches | Compartments: Front phone slot + rear zip for cards/cash | Strap: Adjustable crossbody, 22-52 inches | Closure: Magnetic flap | Dimensions: 7.5 x 4 x 1.5 inches | Weight: 5.2 oz | Colors: 6 neutrals",
            "target": "minimalists who carry only a phone, cards, and keys and want to leave the big bag behind",
        },
        {
            "name": "SunWeave Wide-Brim Straw Hat",
            "subcategory": "accessories",
            "features": "Material: Natural toyo straw | Brim: 4-inch wide, shapeable wire edge | Crown: 4 inches, pinched | Sweatband: Grosgrain ribbon | Chin strap: Removable, adjustable | Sizes: S, M, L (56-60 cm) | Packable: Rolls without cracking | UPF: 50+",
            "target": "beach-goers and garden hosts who need sun coverage that looks polished",
        },
        {
            "name": "TwillMaster Stretch Blazer",
            "subcategory": "outerwear",
            "features": "Material: 72% polyester, 25% rayon, 3% spandex | Weight: 240gsm stretch twill | Construction: Half-canvas, natural shoulder | Buttons: 2-button closure, horn-effect | Pockets: 3 external flap, 2 internal | Lining: Half-lined for breathability | Sizes: 36S-48L | Care: Machine washable",
            "target": "professionals who want a polished blazer that moves like a jacket and survives a home wash",
        },
        {
            "name": "LoopStitch Chunky Cardigan",
            "subcategory": "tops",
            "features": "Material: 60% cotton, 40% acrylic | Gauge: 3-gauge chunky knit | Fit: Oversized, drop-shoulder | Closure: 5 tortoiseshell buttons | Pockets: 2 patch pockets | Length: Hip-length (26 inches at size M) | Sizes: XS-XXL | Care: Machine wash gentle | Colors: 5 heathered neutrals",
            "target": "cozy-at-home dressers who layer a chunky knit over everything from tees to turtlenecks",
        },
        {
            "name": "NightShift Silk Sleep Set",
            "subcategory": "loungewear",
            "features": "Material: 100% 22-momme mulberry silk, OEKO-TEX certified | Set: Button-up long-sleeve top + elastic-waist straight-leg pants | Piping: Contrast satin piping on collar and cuffs | Sizes: XS-XL | Colors: Midnight navy, dusty rose, ivory | Care: Hand wash cold or silk cycle | Button: Mother-of-pearl",
            "target": "sleepers who run hot and want a temperature-regulating, skin-friendly sleep set",
        },
        {
            "name": "FieldCanvas Work Jacket",
            "subcategory": "outerwear",
            "features": "Material: 12 oz washed duck canvas (100% cotton) | Lining: Quilted flannel in body, brushed twill in sleeves | Pockets: 4 front (2 chest snap, 2 hand) + 2 internal | Closure: Full-zip with snap storm flap | Cuffs: Adjustable snap | Sizes: S-3XL | Colors: Tan, olive, black | Hem: Drawcord adjustable",
            "target": "tradespeople and weekend builders who need rugged layering for cool-weather outdoor work",
        },
        {
            "name": "GlideStitch Seamless Sports Bra",
            "subcategory": "activewear",
            "features": "Material: 75% nylon, 25% elastane | Support: Medium (A-D cup) | Construction: Seamless knit, no tag | Band: Wide elastic, stays put without digging | Straps: Racerback with keyhole detail | Moisture: Quick-dry, wicking | Sizes: XS-XL | Colors: 8 options | Care: Machine wash cold",
            "target": "active women who want chafe-free support for yoga, cycling, and gym sessions",
        },
        {
            "name": "TimberFrame Wooden Watch",
            "subcategory": "accessories",
            "features": "Case: 42mm, natural walnut wood | Movement: Japanese Miyota quartz | Dial: Minimalist, applied hour markers | Band: Walnut wood link bracelet with fold-over clasp | Water resistance: Splash-proof (not for swimming) | Weight: 48g | Case thickness: 11mm | Includes: Engraving-ready caseback",
            "target": "eco-minded gift shoppers looking for a unique, natural-material timepiece",
        },
        {
            "name": "AlpineKnit Fair Isle Beanie",
            "subcategory": "accessories",
            "features": "Material: 100% lambswool | Knit: Fair Isle pattern in 4 colors | Lining: Fleece ear band | Fit: Cuffed, one-size (stretches 21-24 inch head) | Weight: 2.4 oz | Origin: Knitted in the Scottish Borders | Care: Hand wash cold, reshape flat | Pom: Detachable faux-fur",
            "target": "winter dressers who want a traditional knit beanie with warmth and heritage character",
        },
        {
            "name": "BalancePack Everyday Tote",
            "subcategory": "bags",
            "features": "Material: Water-resistant 500D nylon with leather handles | Capacity: 18L | Laptop: Interior padded sleeve for 14 inches | Closure: Top zip + magnetic snap | Pockets: Exterior zip, interior zip, 2 slip pockets | Dimensions: 15 x 13 x 6 inches | Strap: Removable crossbody (47 inch) | Weight: 1.3 lbs",
            "target": "working parents who carry a laptop, water bottle, and kid supplies in one polished bag",
        },
        {
            "name": "WoolBlend Herringbone Flat Cap",
            "subcategory": "accessories",
            "features": "Material: 70% wool, 30% polyester herringbone tweed | Lining: Satin | Brim: Sewn-down, 2.5 inches | Fit: Structured, low-profile | Sizes: S (55cm), M (57cm), L (59cm), XL (61cm) | Sweatband: Moisture-wicking | Colors: Charcoal, brown, navy | Origin: Tailored in Ireland",
            "target": "classic dressers who appreciate traditional British and Irish headwear with modern sizing",
        },
        {
            "name": "PureCotton Everyday Crew Tee (3-Pack)",
            "subcategory": "basics",
            "features": "Material: 100% Supima cotton, ring-spun | Weight: 180gsm | Fit: Standard crew neck | Seams: Side-seamed for reduced twist | Collar: Ribbed, reinforced with taping | Pre-shrunk: Yes (<3%) | Sizes: XS-3XL | Pack: 3 shirts (white, heather gray, black) | Care: Machine wash warm",
            "target": "wardrobe essentialists who want a reliable, well-fitting crew tee for everyday rotation",
        },
        {
            "name": "RainDash Waterproof Ankle Boot",
            "subcategory": "footwear",
            "features": "Upper: Waterproof rubberized leather | Sole: Lugged rubber, 1.75-inch heel | Lining: Recycled polyester fleece | Construction: Vulcanized | Closure: Front zip with gusset | Weight: 15 oz per boot (women's 8) | Sizes: Women's 5-12 | Colors: Black, forest green, burgundy | Waterproof: Seam-sealed",
            "target": "city dwellers who walk through rain and puddles and still want a boot that looks intentional",
        },
        {
            "name": "ArchSupport Leather Penny Loafer",
            "subcategory": "footwear",
            "features": "Upper: Hand-stained calf leather | Sole: Leather with rubber heel and toe taps | Construction: Goodyear welted | Insole: Cork footbed with leather cover | Fit: E width (accommodates wider feet) | Sizes: Men's 7-14 | Break-in: Minimal, supple leather | Resoleable: Yes | Colors: Cognac, dark brown, black",
            "target": "professionals who want a dressy loafer that can be resoled and worn for years",
        },
        {
            "name": "TechFleece Zip Hoodie",
            "subcategory": "activewear",
            "features": "Material: 66% cotton, 34% polyester tech fleece | Weight: 320gsm | Construction: Bonded two-layer with spacer mesh core | Fit: Slim athletic | Hood: 3-panel, no drawstring (clean look) | Pockets: 2 zip hand + 1 internal media | Sizes: XS-XXL | Cuffs: Ribbed, thumbhole option | Colors: 6",
            "target": "gym-to-street wearers who want structured warmth that looks sharp outside the weight room",
        },
        {
            "name": "MidnightSatin Tie Collection (3-Pack)",
            "subcategory": "accessories",
            "features": "Material: 100% silk twill | Width: 3 inches (modern slim) | Length: 58 inches | Construction: 7-fold, hand-rolled edges | Keeper loop: Self-fabric | Pack: Navy solid, burgundy micro-dot, charcoal diagonal stripe | Care: Dry clean only | Handmade: Hand-finished in Como, Italy",
            "target": "professionals building a versatile tie rotation that covers suits from navy to charcoal",
        },
        {
            "name": "ComfortLane Slide Sandal",
            "subcategory": "footwear",
            "features": "Upper: Molded EVA with soft-touch lining | Footbed: Contoured arch support, 10mm cushion | Sole: Textured rubber outsole | Weight: 5.3 oz per sandal (men's 10) | Sizes: Men's 7-14, Women's 5-12 | Water-friendly: Quick-dry, pool/beach safe | Colors: 4 matte tones | Vegan: 100% animal-free materials",
            "target": "post-workout and casual wearers who want supportive slides for recovery and errands",
        },
        {
            "name": "EdgewoodFrame Clear Lens Glasses",
            "subcategory": "accessories",
            "features": "Frame: Acetate, handmade | Lens: Clear demo with blue-light filtering (non-prescription) | Shape: Round, 48mm lens width | Bridge: Keyhole, 20mm | Temple: 145mm with spring hinge | Weight: 24g | Includes: Hardshell case and cleaning cloth | Prescription-ready: Lenses can be swapped by an optician",
            "target": "remote workers and students wanting blue-light protection with a vintage-inspired frame",
        },
        {
            "name": "FieldReady Canvas Belt Bag",
            "subcategory": "bags",
            "features": "Material: Organic cotton canvas with leather trim | Capacity: 2L | Compartments: Main zip + front snap pocket | Strap: Adjustable webbing, 28-48 inches, wearable as crossbody or waist | Dimensions: 10 x 6 x 3 inches | Hardware: Matte black zinc alloy | Weight: 7 oz | Water resistance: Light splash only",
            "target": "hands-free travelers and festival-goers carrying phone, wallet, and sunscreen",
        },
        {
            "name": "SilkEdge Pocket Square Set (3-Pack)",
            "subcategory": "accessories",
            "features": "Material: 100% silk twill | Size: 13 x 13 inches each | Edge: Hand-rolled | Pack: White, burgundy, navy | Weight: 0.5 oz each | Care: Dry clean or hand wash cold | Origin: Printed and finished in Italy",
            "target": "professionals adding a polished finishing touch to blazers and suit jackets",
        },
        {
            "name": "CorduroyClassic Wide-Leg Trousers",
            "subcategory": "bottoms",
            "features": "Material: 98% cotton, 2% elastane corduroy (8-wale) | Fit: Wide-leg, high-rise | Rise: 11.5 inches | Inseam: 30, 32, 34 inches | Closure: Zip fly with button | Pockets: 4-pocket | Colors: Chestnut, forest green, navy, cream | Sizes: 26-38",
            "target": "trend-forward dressers who want retro texture with modern proportions",
        },
        {
            "name": "WoolWrap Shawl Cardigan",
            "subcategory": "tops",
            "features": "Material: 70% wool, 30% nylon | Weight: 400gsm | Fit: Oversized, open front (no buttons) | Collar: Draped shawl | Length: Knee-length (38 inches at size M) | Pockets: 2 side seam | Sizes: XS/S, M/L, XL/XXL | Care: Dry clean recommended | Colors: Oatmeal, charcoal, camel",
            "target": "layering enthusiasts who want a cozy wrap for reading, travel, or working from home",
        },
        {
            "name": "RetroRunner Suede Sneaker",
            "subcategory": "footwear",
            "features": "Upper: Pig suede with nylon mesh panels | Sole: Gum rubber, retro cup sole | Midsole: EVA, 20mm cushion | Lacing: Flat cotton laces | Weight: 11 oz per shoe (men's 9) | Sizes: Men's 7-13, Women's 5-11 | Colors: Gray/navy, tan/green, black/white | Inspired by: 1970s running silhouettes",
            "target": "casual sneaker wearers who appreciate vintage sport aesthetics with modern comfort",
        },
        {
            "name": "PrismWear Color-Block Windbreaker",
            "subcategory": "outerwear",
            "features": "Material: 100% recycled nylon ripstop | Weight: 6 oz (men's M) | Fit: Regular, hip-length | Hood: Stowable in collar | Pockets: 2 side zip + 1 chest zip | Lining: Mesh body lining | Packable: Stuffs into chest pocket | Colors: 4 color-block combos | Water resistance: DWR treated",
            "target": "streetwear-influenced dressers who want a lightweight layer with bold color blocking",
        },
        {
            "name": "HelixStitch Cable-Knit Beanie",
            "subcategory": "accessories",
            "features": "Material: 50% merino wool, 50% acrylic | Knit: Cable pattern | Fit: Cuffed, slouch option | One size: Stretches to fit 21-24 inch heads | Lining: Fleece-lined for extra warmth | Weight: 3.2 oz | Colors: 6 muted tones | Care: Machine wash cold, lay flat",
            "target": "cold-weather commuters who want a warm, stylish beanie that fits under a hood",
        },
        {
            "name": "GrainLeather Dopp Kit",
            "subcategory": "bags",
            "features": "Material: Full-grain vegetable-tanned leather | Lining: Water-resistant nylon | Closure: YKK brass zipper | Dimensions: 10 x 5 x 5 inches | Base: Flat bottom, stands upright | Interior: 2 mesh pockets + elastic strap organizer | Weight: 9 oz | Patina: Darkens and softens over time",
            "target": "travelers who want a durable toiletry bag that improves with age",
        },
        {
            "name": "BreezeLinen Wide-Leg Palazzo Pants",
            "subcategory": "bottoms",
            "features": "Material: 100% French flax linen | Fit: Wide-leg, high-rise | Waist: Elastic back panel with flat front | Inseam: 30 inches | Pockets: 2 side seam | Colors: White, sand, olive, black | Sizes: XS-XL | Care: Machine wash cold, tumble dry low | Wrinkle: Intentional linen texture",
            "target": "warm-climate dressers who want an effortless, breezy pant for weekends and vacations",
        },
        {
            "name": "OxfordWeave Button-Down Shirt",
            "subcategory": "tops",
            "features": "Material: 100% long-staple cotton Oxford cloth (140gsm) | Collar: Button-down, unlined | Fit: Slim with back dart | Cuffs: Adjustable 2-button barrel | Buttons: Natural corozo nut | Hem: Split gusset, wearable tucked or untucked | Sizes: S-XXL | Colors: White, blue, pink, chambray stripe | Care: Machine wash warm",
            "target": "professionals and smart-casual dressers who want a versatile shirt for office and weekend",
        },
        {
            "name": "WildStitch Embroidered Jean Jacket",
            "subcategory": "outerwear",
            "features": "Material: 100% cotton denim, 13 oz | Wash: Medium vintage wash | Embroidery: Floral motif across yoke and cuffs (chain stitch) | Closure: Metal button front | Pockets: 2 chest flap + 2 hand | Lining: Unlined | Fit: Classic trucker, slightly cropped | Sizes: XS-XL | Care: Machine wash cold, inside out",
            "target": "statement-piece dressers who want a conversation-starting layer with artisan embroidery",
        },
    ]
    rng.shuffle(products)
    for p in products[:count]:
        p["category"] = "fashion"
    return products[:count]


def generate_complex_products(fake: Faker, rng: random.Random, count: int) -> list[dict]:
    """Generate complex multi-function product specs with many features."""
    products = [
        {
            "name": "AirBlend Smart Air Purifier with Humidifier",
            "subcategory": "home",
            "features": "Purification: True HEPA H13 filter | Humidification: Evaporative, 500ml/h output | CADR: 280 CFM | Coverage: 550 sq ft | Air quality sensor: PM2.5, VOC, humidity | Display: Color-coded ring + LCD | Scheduling: 24-hour timer via app | Connectivity: Wi-Fi, Alexa/Google | Tank: 2.5L, top-fill | Noise: 26dB (sleep mode) | Filter life indicator: Automatic",
            "target": "allergy sufferers and parents in dry climates who want combined purification and humidification",
        },
        {
            "name": "ChefStation 10-in-1 Multicooker",
            "subcategory": "kitchen",
            "features": "Functions: Pressure cook, slow cook, steam, saute, rice, yogurt, sous vide, air fry lid, warm, ferment | Capacity: 6-quart stainless steel inner pot | Pressure: 10.2-11.6 PSI | Display: Color touchscreen with guided recipes | Presets: 15 one-touch | Safety: 10 verified mechanisms (lid lock, pressure regulator, anti-blockage, etc.) | Accessories: Steam rack, condensation collector, measuring cup | Power: 1000W | Delay start: Up to 24 hours",
            "target": "busy families who want one appliance to replace a pressure cooker, slow cooker, and air fryer",
        },
        {
            "name": "VisionDesk Pro Motorized Standing Desk",
            "subcategory": "furniture",
            "features": "Height range: 25.2-50.8 inches (electric dual motor) | Desktop: 60 x 30 inches, bamboo or laminate options | Lift capacity: 350 lbs | Speed: 1.5 inches/second | Presets: 4 programmable memory positions | Anti-collision: Sensor stops motor on obstruction | Cable management: Integrated tray + grommet holes | Stability: T-leg with crossbar | Noise: <50dB while adjusting | Assembly: 2-person, ~45 minutes | Warranty: 15-year frame, 5-year motor",
            "target": "remote professionals who alternate between sitting and standing throughout the workday",
        },
        {
            "name": "TravelEase Carry-On Spinner with Tech Pocket",
            "subcategory": "travel",
            "features": "Dimensions: 22 x 14 x 9 inches (fits most US carry-on) | Material: Polycarbonate hardshell | Wheels: 4 dual-spinner (360-degree) | Weight: 7.2 lbs | Capacity: 39L | Laptop pocket: Quick-access front, fits 16 inches | Compression: Interior strap system | Lock: TSA-approved combination | Interior: Full zip divider + mesh pockets | Handle: Telescoping aluminum, 3-position | Warranty: Lifetime",
            "target": "frequent business flyers who need organized, one-bag carry-on travel with quick laptop access",
        },
        {
            "name": "FitHub Home Gym Cable Machine",
            "subcategory": "fitness",
            "features": "Resistance: 5-200 lbs, 5-lb increments (digital magnetic) | Pulleys: Dual adjustable (floor to 7 feet) | Footprint: 20 x 48 inches | Display: 10.1-inch HD touchscreen with exercise library | Connectivity: Wi-Fi + Bluetooth | Programs: 500+ guided workouts | Accessories: Bar, handles, ankle straps, bench (optional) | Weight: 130 lbs | Power: AC 120V | Max user height: 6 ft 6 in",
            "target": "home gym enthusiasts who want cable-machine versatility without a commercial-gym footprint",
        },
        {
            "name": "SoundStage Wireless Home Theater System",
            "subcategory": "audio",
            "features": "Configuration: 5.1.2 (soundbar + 2 surrounds + subwoofer + 2 upfiring Atmos) | Total output: 720W | Connectivity: HDMI eARC, optical, Bluetooth 5.2, Wi-Fi, AirPlay 2 | Decoding: Dolby Atmos, DTS:X | Room calibration: Mic-based auto EQ | Subwoofer: 8-inch wireless, 200W | App: EQ adjustment, speaker placement guide | Dimensions: Soundbar 41 x 2.5 x 4 inches | Night mode: Dynamic range compression | Streaming: Built-in Chromecast",
            "target": "movie enthusiasts who want an immersive Dolby Atmos experience without running wires to every speaker",
        },
        {
            "name": "PrintForge 3D Printer with Auto-Leveling",
            "subcategory": "maker",
            "features": "Build volume: 220 x 220 x 250mm | Resolution: 50-300 micron layer height | Nozzle: 0.4mm (swappable) | Max temp: 260 degrees C hotend, 100 degrees C heated bed | Leveling: 25-point automatic mesh | Filament: PLA, PETG, TPU, ABS | Connectivity: USB, microSD, Wi-Fi | Camera: Built-in 1080p for remote monitoring | Resume: Power-loss recovery | Display: 4.3-inch color touchscreen | Frame: Dual Z-axis with lead screws | Speed: Up to 180mm/s",
            "target": "hobbyist makers and educators who want reliable prints with minimal calibration effort",
        },
        {
            "name": "GardenBot Solar-Powered Robotic Mower",
            "subcategory": "outdoor",
            "features": "Coverage: Up to 0.25 acres | Navigation: RTK GPS (no boundary wire) | Cutting width: 9 inches | Cutting height: 1-3.5 inches (adjustable) | Blade: 3 pivoting razor | Battery: 5.2Ah Li-ion + solar top panel | Runtime: 80 minutes (solar extends by ~20%) | Charging: Auto-return to dock | Rain sensor: Pauses and docks | Anti-theft: PIN + GPS tracking + alarm | App: Zone mapping, scheduling, mow history | Noise: 58dB | Slope: Handles up to 35% grade",
            "target": "homeowners who want a maintained lawn without manual mowing or boundary wire installation",
        },
        {
            "name": "StudyPad E-Ink Tablet with Stylus",
            "subcategory": "electronics",
            "features": "Display: 10.3-inch E-Ink Carta 1200, 300 PPI | Stylus: Wacom EMR, 4096 pressure levels, no battery | Storage: 64GB | RAM: 4GB | OS: Custom Android 11 (no app store, focused) | Formats: PDF, EPUB, DOCX | Handwriting: OCR + handwriting search | Templates: 40 (notebooks, planners, sheet music, storyboards) | Sync: Dropbox, Google Drive, OneDrive | Battery: 3000mAh (3 weeks typical) | Weight: 390g | Frontlight: Adjustable warm/cool",
            "target": "students, researchers, and professionals who annotate documents and take handwritten notes without screen fatigue",
        },
        {
            "name": "BrewStation All-in-One Espresso Machine",
            "subcategory": "kitchen",
            "features": "Grinder: Built-in conical burr, 15 settings | Boiler: Thermoblock with PID control | Brew pressure: 15 bar (adjustable 9-bar profiling) | Steam: Automatic milk frother with temperature control | Water tank: 67 oz removable | Bean hopper: 8 oz with seal | Presets: Espresso, lungo, americano, cappuccino, latte, flat white | Display: 3.5-inch color touchscreen | Self-clean: Automatic descale and rinse cycle | Drip tray: Removable with full indicator | Dimensions: 11 x 17 x 14 inches | Power: 1450W",
            "target": "home baristas who want cafe-quality espresso drinks with built-in grinding and automatic milk frothing",
        },
        {
            "name": "SafeHome Wireless Security System Kit",
            "subcategory": "home",
            "features": "Hub: Wi-Fi + cellular backup + 8h battery | Kit includes: 5 door/window sensors, 2 motion detectors, 1 keypad, 1 siren (105dB) | Cameras: Add-on compatible (sold separately) | Monitoring: Professional optional ($10/mo) or self-monitor free | Sensors: Up to 50 per hub | Arming modes: Home, away, night | App: Real-time alerts, arm/disarm, event log | Voice: Alexa, Google, HomeKit | Installation: Peel-and-stick, 30-minute setup | Encryption: AES-128 | Battery life: 3 years (sensors)",
            "target": "renters and homeowners who want a professional-grade security system with no contracts or drilling",
        },
        {
            "name": "FoldGo Electric Commuter Scooter",
            "subcategory": "transport",
            "features": "Motor: 350W brushless hub | Speed: Up to 19 mph (3 speed modes) | Range: 25 miles per charge | Battery: 36V 10.4Ah Li-ion | Braking: Regenerative electronic + rear disc | Tires: 10-inch pneumatic | Weight: 28.6 lbs | Folded size: 44 x 17 x 19 inches | Lights: Front LED + rear brake light | Display: LED dashboard (speed, battery, mode) | Suspension: Front fork | Max rider weight: 220 lbs | IP rating: IP54",
            "target": "urban commuters covering 3-8 miles to transit or work who need a fold-and-carry last-mile vehicle",
        },
        {
            "name": "CleanBot Self-Emptying Robot Vacuum and Mop",
            "subcategory": "home",
            "features": "Suction: 6000Pa | Navigation: LiDAR + 3D structured light obstacle avoidance | Mopping: Sonic vibration pad (3000 RPM) with auto-lift on carpet | Self-empty: 2.5L dust station, 60-day capacity | Water tank: 200ml with electronic flow control | Runtime: 180 minutes | Noise: 55dB (quiet mode) | App: Multi-floor mapping, room-specific settings, no-mop zones | Object avoidance: Detects cables, shoes, pet waste | Edge cleaning: Side brush + edge-hugging mode | Dustbin: 400ml | Mop wash: Optional hot-water dock (sold separately)",
            "target": "busy pet-owning households who want daily automated vacuuming and mopping with minimal maintenance",
        },
        {
            "name": "WandCast Portable Smart Projector",
            "subcategory": "entertainment",
            "features": "Resolution: Native 1080p (4K input supported) | Brightness: 500 ANSI lumens | Lamp: LED (30000-hour life) | Throw: 1.2:1 (100-inch image at 8.5 feet) | Keystone: Auto vertical + manual horizontal | Focus: Autofocus | Speakers: 2x 5W Harman Kardon | Connectivity: HDMI, USB-A, Wi-Fi 6, Bluetooth 5.0 | OS: Built-in Android TV with Netflix, YouTube | Battery: Built-in 10000mAh (2.5h playback) | Weight: 3.3 lbs | Dimensions: 7 x 7 x 5 inches",
            "target": "backyard movie hosts and travelers who want a big-screen experience without mounting hardware",
        },
        {
            "name": "CraftDesk Modular Workshop Bench",
            "subcategory": "workshop",
            "features": "Surface: 60 x 30 inches, 1.5-inch solid beech hardwood | Height: Adjustable 28-36 inches (crank handle) | Weight capacity: 1000 lbs | Vise: Built-in 7-inch quick-release front vise | Dog holes: 20mm grid (24 holes) | Storage: 2 full-width drawers (ball-bearing slides) + lower shelf | Leveling: 4 adjustable feet | Power: 4-outlet strip mounted to rear rail | Pegboard: Optional rear panel (included) | Casters: Optional locking (included) | Assembly: 60 minutes, hardware included",
            "target": "woodworkers and makers who need a sturdy, adaptable workbench for assembly, carving, and project storage",
        },
        {
            "name": "NutriTrack Smart Kitchen Scale with App",
            "subcategory": "kitchen",
            "features": "Capacity: 11 lbs / 5 kg | Precision: 1g increments | Connectivity: Bluetooth 5.0 | App: Nutritional database (500K+ foods), recipe scaling, meal logging | Modes: Weight (g/oz/lb/ml), nutritional (calories, protein, carbs, fat, fiber) | Platform: Tempered glass, 7.5 inches | Display: Backlit LCD on device + app dashboard | Tare: Physical button + app control | Battery: USB-C rechargeable (90-day standby) | API: Export to Apple Health, Google Fit, MyFitnessPal | Waterproof: IPX5",
            "target": "macro-tracking athletes and dieters who want automatic nutritional breakdown as they cook",
        },
        {
            "name": "TravelSafe Carry-On Garment Duffel",
            "subcategory": "travel",
            "features": "Modes: Unfolds to 44-inch garment bag, folds to 22 x 12 x 10 duffel | Material: Water-resistant 1680D ballistic nylon | Garment: Holds 2 suits + 3 shirts on built-in hanger | Shoe pocket: Ventilated, fits size 13 | Laptop: Padded 15-inch sleeve | Pockets: 8 total (toiletry, wet/dry, document, 2 mesh, 2 exterior) | Strap: Removable padded shoulder + luggage sleeve | Hardware: YKK zippers, gunmetal | Weight: 4.1 lbs | Warranty: Lifetime",
            "target": "business travelers who need suits wrinkle-free and refuse to check a bag",
        },
        {
            "name": "AquaGarden Self-Watering Indoor Planter System",
            "subcategory": "home",
            "features": "Pods: 12 plant slots in 3 tiers | Lighting: Full-spectrum LED, 30W, auto 16h/8h cycle | Water reservoir: 4L with float indicator | Pump: Silent recirculating (hydroponic) | Nutrients: 3-month starter supply included | App: Growth tracking, water/feed reminders | Height: Adjustable 18-30 inches | Base: Bamboo frame | Power: AC adapter | Pod variety: Herbs, leafy greens, flowers, peppers | Germination: Sponge-based pods (20 included)",
            "target": "apartment dwellers and kitchen gardeners growing herbs and salad greens year-round indoors",
        },
        {
            "name": "SnapFrame Digital Art Display",
            "subcategory": "home",
            "features": "Display: 21.5-inch IPS, anti-glare matte | Resolution: 1920x1080 | Color: 99% sRGB, hardware calibrated | Art library: 10000+ curated works (subscription) + upload your own | Sensors: Ambient light (adjusts brightness/warmth) + motion (sleep/wake) | Frame: Interchangeable magnetic bezels (walnut, oak, black, white) | Connectivity: Wi-Fi | App: Schedule art rotations, upload from phone | Mount: Wall or easel stand (both included) | Power: AC, 18W average | Dimensions: 22 x 14 x 1.5 inches",
            "target": "art lovers and interior decorators who want rotating gallery-quality art on their wall",
        },
        {
            "name": "PetCare Automatic Litter System",
            "subcategory": "pet",
            "features": "Mechanism: Rotating sifting with rake backup | Cycle: Auto-cleans 15 minutes after cat exits | Sensor: Weight-based cat detection (3-25 lbs) | Waste drawer: Carbon-filtered, lined, 7-day capacity (1 cat) | Litter type: Works with clumping clay and crystal | Globe: Removable for deep cleaning | Safety: Anti-pinch, cat-present detection pauses cycle | Odor: Carbon filter + optional lavender pod | Dimensions: 27 x 25 x 27 inches | Power: AC with 9V battery backup | App: Usage tracking, health insights (frequency/weight), order filters",
            "target": "cat owners who want hands-off litter maintenance with health monitoring for up to 3 cats",
        },
        {
            "name": "MediaWall Modular Shelving System",
            "subcategory": "furniture",
            "features": "Configuration: Customizable grid (sold as 4-cube base + expansion kits) | Material: Powder-coated steel frame + engineered oak veneer shelves | Cube size: 14 x 14 x 14 inches internal | Weight capacity: 30 lbs per shelf | Expansion: Stackable up to 4 rows x 6 columns | Back panel: Optional fabric, pegboard, or open | Accessories: Drawer inserts, door fronts, cable management clips (all optional) | Finish: Matte black frame with natural or walnut shelf | Wall anchor: Anti-tip hardware included | Tools: Allen key included, no drill required for base assembly",
            "target": "apartment dwellers and home office workers who need flexible storage that adapts when they reorganize or move",
        },
        {
            "name": "OutdoorChef Portable Pizza Oven",
            "subcategory": "outdoor",
            "features": "Fuel: Propane (10K-15K BTU) or wood pellet (dual-fuel) | Max temp: 950 degrees F in 15 minutes | Stone: 13-inch cordierite pizza stone | Cook time: 60-90 seconds per pizza | Body: Insulated stainless steel, double-walled | Thermometer: Built-in dome + stone probe | Chimney: Adjustable damper for airflow | Folding legs: Freestanding at table height | Weight: 26 lbs | Accessories: Peel, cutter, carry cover included | Pizza size: Up to 12-inch | Ignition: Push-button (propane mode)",
            "target": "backyard entertainers and pizza enthusiasts who want restaurant-style Neapolitan pizza at home",
        },
        {
            "name": "SleepCloud Adjustable Base with Massage",
            "subcategory": "furniture",
            "features": "Positions: Head 0-60 degrees, foot 0-45 degrees, zero-gravity preset, anti-snore preset | Massage: 3 zones (head, back, foot), 3 intensities, wave/pulse/constant modes | Motor: Dual DC (whisper-quiet, <40dB) | Connectivity: Bluetooth app + wireless remote + voice (Alexa) | USB: 2 ports (1A + 2.1A) on each side | Under-bed light: LED strip with motion sensor | Capacity: 750 lbs | Fits: Standard bed frames | Timer: 10/20/30 minute auto-off for massage | Wall-hugger: Slides forward to maintain nightstand distance | Sizes: Twin XL, Full, Queen, Split King | Power: Emergency battery lowers head in outage",
            "target": "couples and individuals with back pain or acid reflux who want customizable sleep positions and relaxation features",
        },
        {
            "name": "DroneView Foldable Camera Drone",
            "subcategory": "electronics",
            "features": "Camera: 4K/60fps, 1/1.3-inch CMOS sensor | Gimbal: 3-axis stabilized | Flight time: 46 minutes | Range: 12 km (FCC) | Obstacle avoidance: Omnidirectional (6 sensors) | Weight: 249g (under FAA registration limit) | Folded: 5.3 x 3.6 x 2.5 inches | Storage: MicroSD up to 256GB | Tracking: ActiveTrack 5.0 (follow subject) | Modes: Hyperlapse, panorama, quickshots, waypoint | Wind resistance: Level 5 (24 mph) | Transmission: O4 video (1080p live feed) | Return to home: Auto on low battery or signal loss",
            "target": "travel photographers and content creators who want cinematic aerial footage in a pocket-sized package",
        },
        {
            "name": "CycleStation Indoor Bike with Screen",
            "subcategory": "fitness",
            "features": "Screen: 22-inch HD touchscreen, swivel | Resistance: Magnetic, 100 micro-levels | Drive: Belt (silent) | Flywheel: Equivalent 35 lbs | Metrics: Cadence, power (watts), heart rate (chest strap compatible), calories | Seat: 4-way adjustable | Handlebars: Multi-grip, fore/aft adjustable | Pedals: Dual-sided (SPD clip + toe cage) | Speakers: 2x 10W front-facing | Camera: Built-in for virtual group rides | Connectivity: Wi-Fi, Bluetooth, ANT+ | Dimensions: 48 x 24 x 54 inches | Weight: 140 lbs | Max rider: 300 lbs",
            "target": "spin class fans who want live and on-demand instructor-led rides at home with real-time metrics",
        },
        {
            "name": "SmartMirror Fitness Display",
            "subcategory": "fitness",
            "features": "Display: 43-inch HD LCD behind mirrored glass | Camera: 12MP with form-correction AI | Speakers: 2x 10W stereo | Microphone: Built-in for voice control | Content: 10000+ classes (strength, HIIT, yoga, boxing, stretching) | Heart rate: Bluetooth monitor included | Space: Mirror when off, display when on | Dimensions: 52 x 22 x 1.4 inches | Wall mount: Included (can lean or mount) | Weight: 70 lbs | Connectivity: Wi-Fi | Subscription: Required for content ($40/mo)",
            "target": "home exercisers who want guided workouts in a small footprint that blends into living room decor",
        },
        {
            "name": "PowerHub Whole-Home Battery System",
            "subcategory": "energy",
            "features": "Capacity: 13.5 kWh per unit (stackable to 40.5 kWh) | Power output: 7 kW continuous, 10 kW peak | Round-trip efficiency: 90% | Inverter: Integrated hybrid (grid-tied + off-grid) | Solar input: 10 kW max, MPPT | Transfer time: <20ms (seamless backup) | Monitoring: App with real-time production/consumption/storage | Grid services: Time-of-use optimization, demand response | Warranty: 10-year, 70% capacity guarantee | Dimensions: 45 x 30 x 6 inches (wall-mount) | Weight: 265 lbs | Connectivity: Wi-Fi + Ethernet | Operating temp: -4 to 122 degrees F",
            "target": "solar-equipped homeowners who want energy independence and backup power during grid outages",
        },
        {
            "name": "VanBuild Modular Camper Conversion Kit",
            "subcategory": "outdoor",
            "features": "Fits: Standard cargo vans (Ford Transit, RAM ProMaster, Sprinter) | Modules: Bed platform (full-size), kitchenette (sink + single-burner + 30L fridge), overhead cabinets, fold-down table | Electrical: 200Ah LiFePO4 battery, 200W solar panel, 2000W inverter, shore power input | Water: 20-gallon fresh + 10-gallon gray | Material: Baltic birch plywood, powder-coated steel brackets | Insulation: Thinsulate 3M (included for walls and ceiling) | Installation: Bolt-in, no welding (reversible) | Assembly time: 2-day weekend build | Ventilation: MaxxAir fan mount pre-cut | Weight: 380 lbs (all modules)",
            "target": "van-life beginners and weekend adventurers who want a professional camper build without permanent vehicle modification",
        },
        {
            "name": "ProStudio Podcast Recording Kit",
            "subcategory": "audio",
            "features": "Microphones: 2x dynamic cardioid (XLR) | Interface: 2-channel USB-C audio interface (24-bit/96kHz) | Boom arms: 2x desk-clamp adjustable | Pop filters: 2x dual-layer nylon | Headphones: 2x closed-back studio monitor | Software: 1-year DAW license + 6 months hosting | Cables: 2x 10-foot XLR, 1x USB-C | Carrying case: Padded, fits all components | Phantom power: 48V on both channels | Total weight: 14 lbs (kit) | Latency: <1ms direct monitoring",
            "target": "new podcasters launching an interview-format show who want a complete two-person setup out of the box",
        },
        {
            "name": "ClimateSuit Smart HVAC Zoning System",
            "subcategory": "home",
            "features": "Zones: Up to 8 independently controlled | Dampers: Motorized with position feedback | Thermostat: Smart, per-zone (Wi-Fi, occupancy sensor) | Hub: Central controller, hardwired to HVAC | Sensors: Temperature + humidity per zone | Scheduling: Per-room, per-day | Learning: Adaptive based on occupancy patterns | Compatibility: Forced-air systems (heat/AC) | App: Zone-by-zone control, energy reports | Voice: Alexa, Google, HomeKit | Installation: Professional recommended (2-4 hours) | Energy savings: Up to 30% (manufacturer estimate)",
            "target": "homeowners with hot and cold spots who want room-by-room temperature control from a single HVAC system",
        },
        {
            "name": "KidsCode STEM Robot Building Kit",
            "subcategory": "education",
            "features": "Pieces: 280+ snap-together parts | Configurations: 12 buildable robots | Controller: Bluetooth app (iOS/Android) | Sensors: IR distance, light, gyroscope | Motors: 2 DC + 1 servo | Programming: Blockly (visual) + Python (advanced) | Battery: Rechargeable Li-Po (2h runtime) | Age: 8-14 | Curriculum: 30 lesson plans included | Storage: Parts organizer box | Materials: ABS plastic, RoHS certified | Connectivity: Bluetooth 5.0",
            "target": "parents and educators introducing children to robotics, coding, and engineering through hands-on building",
        },
        {
            "name": "AllTrail GPS Cycling Computer",
            "subcategory": "cycling",
            "features": "GPS: Multi-band (GPS + GLONASS + Galileo + BeiDou) | Display: 2.6-inch color touchscreen, sunlight-readable | Battery: 32h GPS mode | Navigation: Turn-by-turn with rerouting | Maps: Preloaded topo + road | Connectivity: Bluetooth, ANT+, Wi-Fi | Sensors: Compatible with power, HR, cadence, radar | Climbing: ClimbPro gradient profile | Safety: Live tracking, incident detection | Training: Structured workouts, FTP test | Mount: Out-front + stem included | Water: IPX7 | Weight: 75g | Data fields: Customizable, 10 per screen",
            "target": "road and gravel cyclists who want comprehensive navigation, training metrics, and safety features in one head unit",
        },
        {
            "name": "HomeBarista Water Treatment System",
            "subcategory": "kitchen",
            "features": "Filtration: 3-stage (sediment + carbon block + ion exchange) | Flow rate: 0.75 gal/min | Capacity: 1000 gallons per filter set | Installation: Under-sink + dedicated faucet | TDS target: Adjustable mineral content dial (50-150 ppm) | Bypass: Built-in for unfiltered water | Certifications: NSF 42, 53 | Minerals: Adds back calcium and magnesium after filtration | Pressure gauge: Filter life indicator | Compatibility: Fits standard US plumbing | Dimensions: 14 x 6 x 16 inches | Included: Faucet, installation kit, 2 filter sets",
            "target": "espresso and specialty coffee enthusiasts who want precisely mineralized water for optimal extraction",
        },
        {
            "name": "StageLight DMX LED Par Kit",
            "subcategory": "entertainment",
            "features": "Kit: 4 LED par cans + 1 DMX controller + cables + stand | LEDs: 18x 10W RGBW per can | Modes: DMX-512 (7-channel), sound-active, auto, master/slave | Beam angle: 25 degrees | Dimming: 0-100% smooth | Strobe: 0-20Hz | Controller: 192-channel, 12 fixtures x 16 channels | Stand: T-bar, adjustable 5-9 feet, 40 lb capacity | Power: Daisy-chain up to 8 cans | Cooling: Silent convection (no fan) | Weight: 4.4 lbs per can",
            "target": "mobile DJs, event planners, and small venue owners setting up stage lighting without a dedicated tech crew",
        },
        {
            "name": "AeroGarden Smart Indoor Growing System",
            "subcategory": "home",
            "features": "Pods: 24 plant positions in rotating carousel | Lighting: 60W full-spectrum LED, auto 16h/8h | Water: 5-gallon reservoir with level sensor | Nutrients: Automated dosing pump | pH: Built-in sensor with alert | Height: Adjustable light panel (6-24 inches from pods) | App: Camera timelapse, growth tracking, harvest reminders | Seeds: Compatible with proprietary pods or DIY sponge inserts | Power: 120V AC | Connectivity: Wi-Fi | Dimensions: 30 x 18 x 36 inches | Weight: 22 lbs",
            "target": "serious indoor gardeners and urban farmers growing lettuce, herbs, tomatoes, and peppers year-round",
        },
        {
            "name": "TotalClean Ultrasonic Jewelry Cleaner",
            "subcategory": "home",
            "features": "Tank: 20 oz (600ml) stainless steel | Frequency: 42kHz ultrasonic | Power: 35W | Timer: 5 preset cycles (90s, 180s, 300s, 480s, 600s) | Touch panel: LED display with mode selection | Basket: Removable with watch holder | Degas function: Removes dissolved air for deeper cleaning | Auto shutoff: After cycle completion | Dimensions: 7.5 x 6 x 5 inches | Weight: 2.2 lbs | Items: Jewelry, eyeglasses, coins, dentures, razor heads, small parts | Safe for: Gold, silver, platinum, titanium, stainless steel, glass",
            "target": "jewelry owners and eyeglass wearers who want professional-grade cleaning at home without chemicals",
        },
        {
            "name": "CargoMax Roof Box Carrier",
            "subcategory": "travel",
            "features": "Capacity: 16 cubic feet (450 liters) | Dimensions: 75 x 32 x 16 inches | Weight capacity: 165 lbs | Material: ABS double-shell with UV protection | Opening: Dual-side, 180-degree | Lock: Integrated key lock (4 keys) + anti-pinch safety | Mount: Universal clamp fits square, round, and aero crossbars (adjustable 24-36 inches apart) | Aerodynamic: Low-profile design, tested to 90 mph | Struts: Gas-assist lid lifters | Lining: Carpeted interior base | Colors: Matte black, gloss white | Weight: 42 lbs",
            "target": "road-trip families and ski-season travelers needing extra cargo space without a trailer",
        },
        {
            "name": "SmartScale Body Composition Analyzer",
            "subcategory": "health",
            "features": "Metrics: Weight, BMI, body fat %, muscle mass, bone mass, water %, visceral fat, metabolic age, protein % | Sensors: 8 BIA electrodes (4 foot + 4 hand via retractable handles) | Accuracy: Weight +/-0.1 lb | Capacity: 400 lbs | Platform: Tempered glass, 12 x 12 inches | Connectivity: Wi-Fi + Bluetooth | App: Trend graphs, goal tracking, family profiles (up to 16) | Display: LED segmented | Export: Apple Health, Google Fit, Fitbit | Power: 4x AAA | Users: Auto-recognition",
            "target": "fitness enthusiasts and health-trackers who want detailed body composition beyond just weight",
        },
        {
            "name": "WorkFlow Kanban Whiteboard System",
            "subcategory": "office",
            "features": "Board: 48 x 36 inches, magnetic glass surface | Columns: 5 preprinted (Backlog, To Do, In Progress, Review, Done) with removable header magnets | Cards: 100 magnetic task cards (4 colors, 3 x 2 inches) + 50 blank | Markers: 4 glass-board markers (low-odor, no ghost) | Eraser: Magnetic microfiber | Mounting: Wall-mount hardware for drywall and concrete | Tray: Full-width aluminum marker/card tray | Accessories: 20 dot magnets for priority flagging | Surface: Anti-glare, never stains",
            "target": "agile teams, project managers, and home-office workers who want a physical kanban board for visual workflow management",
        },
        {
            "name": "MixMaster DJ Controller with Audio Interface",
            "subcategory": "audio",
            "features": "Channels: 2-deck with 4-channel mixer | Jog wheels: 6-inch touch-sensitive, mechanical | Faders: 3 line + 1 crossfader (replaceable) | Effects: 16 onboard (filter, echo, flanger, reverb, etc.) | Pads: 16 velocity-sensitive (hot cues, loops, samples) | Audio interface: 24-bit/48kHz, 2 stereo outputs (main + headphone cue) | Connectivity: USB-C bus-powered (no external PSU) | Software: Full DJ software license included | Inputs: Mic with EQ and gain | Display: Per-channel LED meters | Weight: 5.3 lbs | Dimensions: 21 x 12 x 2.5 inches | Build: Metal faceplate with rubberized knobs",
            "target": "beginner and mobile DJs who want a professional-feeling controller that runs off laptop power with no extra gear",
        },
    ]
    rng.shuffle(products)
    for p in products[:count]:
        p["category"] = "complex_products"
    return products[:count]


def generate_sparse_input(fake: Faker, rng: random.Random, count: int) -> list[dict]:
    """Generate sparse product specs with minimal features only."""
    products = [
        {"name": "Bamboo Cutting Board", "subcategory": "kitchen", "features": "bamboo material, juice groove", "target": "home cooks"},
        {"name": "Resistance Band Set", "subcategory": "outdoor_fitness", "features": "5 resistance levels", "target": "fitness enthusiasts"},
        {"name": "Canvas Tote Bag", "subcategory": "fashion", "features": "cotton canvas", "target": "everyday use"},
        {"name": "Stainless Steel Water Bottle", "subcategory": "outdoor_fitness", "features": "stainless steel, 24 oz", "target": "gym-goers"},
        {"name": "Ceramic Mug Set", "subcategory": "kitchen", "features": "ceramic, set of 4", "target": "coffee drinkers"},
        {"name": "Desk Organizer", "subcategory": "home", "features": "wooden, 3 compartments", "target": "office workers"},
        {"name": "Cotton Throw Blanket", "subcategory": "home", "features": "100% cotton, machine washable", "target": "couch loungers"},
        {"name": "Silicone Spatula Set", "subcategory": "kitchen", "features": "heat-resistant silicone, 3 sizes", "target": "home bakers"},
        {"name": "Yoga Block", "subcategory": "outdoor_fitness", "features": "high-density foam", "target": "yoga practitioners"},
        {"name": "Leather Keychain", "subcategory": "fashion", "features": "genuine leather", "target": "gift shoppers"},
        {"name": "Glass Food Storage Containers", "subcategory": "kitchen", "features": "glass with snap-lock lids, set of 5", "target": "meal preppers"},
        {"name": "Wool Beanie", "subcategory": "fashion", "features": "wool blend, one size", "target": "cold-weather wearers"},
        {"name": "Plant Mister Bottle", "subcategory": "home", "features": "glass, fine mist spray", "target": "plant owners"},
        {"name": "Foam Roller", "subcategory": "outdoor_fitness", "features": "18-inch, medium density", "target": "runners"},
        {"name": "Linen Napkin Set", "subcategory": "kitchen", "features": "linen, set of 6", "target": "dinner hosts"},
        {"name": "Nylon Drawstring Backpack", "subcategory": "fashion", "features": "nylon, drawstring closure", "target": "students"},
        {"name": "Cork Coaster Set", "subcategory": "home", "features": "natural cork, set of 8", "target": "homeowners"},
        {"name": "Jump Rope", "subcategory": "outdoor_fitness", "features": "adjustable length", "target": "cardio exercisers"},
        {"name": "Stoneware Baking Dish", "subcategory": "kitchen", "features": "stoneware, 9x13 inch", "target": "home cooks"},
        {"name": "Woven Belt", "subcategory": "fashion", "features": "elastic woven, fits 28-40", "target": "casual dressers"},
        {"name": "Beeswax Candle", "subcategory": "home", "features": "100% beeswax, unscented", "target": "natural home enthusiasts"},
        {"name": "Microfiber Cleaning Cloths", "subcategory": "home", "features": "microfiber, pack of 12", "target": "household cleaners"},
        {"name": "Wooden Salad Servers", "subcategory": "kitchen", "features": "acacia wood, 12 inches", "target": "salad lovers"},
        {"name": "Ankle Weights", "subcategory": "outdoor_fitness", "features": "adjustable, 2 lbs each", "target": "walkers and runners"},
        {"name": "Cotton Bandana", "subcategory": "fashion", "features": "100% cotton, 22x22 inches", "target": "outdoor adventurers"},
        {"name": "Magnetic Phone Mount", "subcategory": "electronics", "features": "magnetic, dashboard mount", "target": "drivers"},
        {"name": "Ceramic Planter Pot", "subcategory": "home", "features": "ceramic with drainage hole, 6 inches", "target": "indoor gardeners"},
        {"name": "Rubber Kitchen Mat", "subcategory": "kitchen", "features": "anti-fatigue, rubber, 20x36 inches", "target": "home cooks who stand at the counter"},
        {"name": "Pocket Notebook", "subcategory": "stationery", "features": "3.5x5.5 inches, lined pages", "target": "note-takers"},
        {"name": "Terrycloth Headband", "subcategory": "outdoor_fitness", "features": "terrycloth, moisture-absorbing", "target": "athletes"},
        {"name": "Stainless Steel Mixing Bowls", "subcategory": "kitchen", "features": "stainless steel, set of 3", "target": "home cooks"},
        {"name": "Canvas Pencil Case", "subcategory": "stationery", "features": "canvas, zipper closure", "target": "students and artists"},
        {"name": "Pine Shoe Rack", "subcategory": "home", "features": "pine wood, 3 tiers", "target": "entryway organizers"},
        {"name": "Mesh Laundry Bag", "subcategory": "home", "features": "mesh, drawstring, large", "target": "laundry doers"},
        {"name": "Silicone Ice Cube Tray", "subcategory": "kitchen", "features": "silicone, 15 large cubes", "target": "cold drink enthusiasts"},
        {"name": "Paracord Bracelet", "subcategory": "outdoor_fitness", "features": "550 paracord, buckle closure", "target": "campers"},
        {"name": "Felt Coaster Set", "subcategory": "home", "features": "wool felt, round, set of 6", "target": "home decor shoppers"},
        {"name": "Cotton Apron", "subcategory": "kitchen", "features": "cotton, adjustable neck strap", "target": "home cooks and bakers"},
        {"name": "Steel Carabiner Clip", "subcategory": "outdoor_fitness", "features": "steel, locking gate", "target": "climbers and hikers"},
        {"name": "Striped Crew Socks", "subcategory": "fashion", "features": "cotton blend, striped pattern", "target": "everyday wearers"},
        {"name": "Wooden Picture Frame", "subcategory": "home", "features": "solid wood, 5x7 inch", "target": "home decorators"},
        {"name": "Stainless Steel Straw Set", "subcategory": "kitchen", "features": "stainless steel, 4 straws with brush", "target": "eco-conscious drinkers"},
        {"name": "Fleece Neck Gaiter", "subcategory": "outdoor_fitness", "features": "fleece, one size", "target": "cold-weather outdoor enthusiasts"},
        {"name": "Jute Rug", "subcategory": "home", "features": "natural jute, 4x6 feet", "target": "home decorators"},
    ]
    rng.shuffle(products)
    for p in products[:count]:
        p["category"] = "sparse_input"
    return products[:count]


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------

def call_model(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
) -> tuple[str, float]:
    """Call a model via litellm and return (response_text, cost)."""
    from litellm import completion

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
            cost = response._hidden_params.get("response_cost", 0.0) or 0.0
            return text, cost
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = max(wait, 5)
            print(f"  Retry {attempt + 1}/{max_retries} for {model} (wait {wait}s): {e.__class__.__name__}")
            time.sleep(wait)
    return "", 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_all_products(seed: int = RANDOM_SEED) -> list[dict]:
    """Generate all 300 product specs."""
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)

    all_products = []
    all_products.extend(generate_electronics(fake, rng, CATEGORY_COUNTS["electronics"]))
    all_products.extend(generate_kitchen(fake, rng, CATEGORY_COUNTS["kitchen"]))
    all_products.extend(generate_outdoor_fitness(fake, rng, CATEGORY_COUNTS["outdoor_fitness"]))
    all_products.extend(generate_fashion(fake, rng, CATEGORY_COUNTS["fashion"]))
    all_products.extend(generate_complex_products(fake, rng, CATEGORY_COUNTS["complex_products"]))
    all_products.extend(generate_sparse_input(fake, rng, CATEGORY_COUNTS["sparse_input"]))

    print(f"Generated {len(all_products)} product specs:")
    from collections import Counter
    cats = Counter(p.get("category", "unknown") for p in all_products)
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")

    return all_products


def run_tuning(products: list[dict], api_key: str) -> float:
    """Test prompt on a sample of 30 products, return tuning cost."""
    rng = random.Random(RANDOM_SEED + 1)
    by_cat = {}
    for p in products:
        cat = p.get("category", "unknown")
        by_cat.setdefault(cat, []).append(p)

    sample = []
    for cat, prods in sorted(by_cat.items()):
        k = min(5, len(prods))
        sample.extend(rng.sample(prods, k))

    print(f"\n--- TUNING PHASE: Testing {len(sample)} products with GPT-4o ---")

    total_cost = 0.0
    results = {"valid": 0, "invalid": 0, "details": []}

    for i, spec in enumerate(sample):
        user_prompt = build_user_prompt(spec)
        text, cost = call_model(
            model="openrouter/openai/gpt-4o",
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        total_cost += cost
        v = validate_description(text, spec["features"])
        status = "PASS" if v["valid"] else "FAIL"
        if v["valid"]:
            results["valid"] += 1
        else:
            results["invalid"] += 1
        results["details"].append({**v, "name": spec["name"], "category": spec.get("category")})
        print(f"  [{i+1}/{len(sample)}] {status} | {spec['name']} | words={v['word_count']} bullets={v['bullet_count']} perfect={v['ends_with_perfect']} sups={v['no_superlatives']}")

        if not v["valid"]:
            print(f"    RESPONSE: {text[:200]}...")

    pass_rate = results["valid"] / len(sample) if sample else 0
    print(f"\nTuning results: {results['valid']}/{len(sample)} pass ({pass_rate:.0%})")
    print(f"Tuning cost: ${total_cost:.4f}")

    if pass_rate < TUNING_PASS_RATE:
        print(f"WARNING: Pass rate {pass_rate:.0%} < {TUNING_PASS_RATE:.0%} target")
    else:
        print(f"PASS: Pass rate meets {TUNING_PASS_RATE:.0%} target")

    return total_cost


def run_production(
    products: list[dict],
    model: str,
    api_key: str,
    output_path: Path,
    source_model_label: str,
) -> tuple[float, dict]:
    """Run full production generation and save JSONL."""
    print(f"\n--- PRODUCTION: {source_model_label} ({len(products)} products) ---")

    total_cost = 0.0
    valid_count = 0
    invalid_count = 0
    fail_details = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rate limit: Anthropic allows 50 req/min, OpenRouter is higher.
    # Use 1.5s delay for Anthropic models, 0.3s for others.
    is_anthropic = "anthropic" in model.lower()
    request_delay = 1.5 if is_anthropic else 0.3

    with open(output_path, "w") as f:
        for i, spec in enumerate(products):
            user_prompt = build_user_prompt(spec)
            text, cost = call_model(
                model=model,
                api_key=api_key,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            total_cost += cost

            v = validate_description(text, spec["features"])
            if v["valid"]:
                valid_count += 1
            else:
                invalid_count += 1
                fail_details.append({"name": spec["name"], "category": spec.get("category"), **v})

            record = {
                "prompt": user_prompt,
                "response": text,
                "source_model": source_model_label,
                "metadata": {
                    "dataset": "ecommerce_products",
                    "variant": spec.get("category", "unknown"),
                    "product_name": spec["name"],
                    "subcategory": spec.get("subcategory", ""),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 25 == 0 or (i + 1) == len(products):
                pct = valid_count / (i + 1) * 100
                print(f"  [{i+1}/{len(products)}] valid={valid_count} invalid={invalid_count} ({pct:.0f}% pass) cost=${total_cost:.4f}")

            # Rate limiting
            time.sleep(request_delay)

    print(f"\nProduction complete: {valid_count}/{len(products)} valid ({valid_count/len(products)*100:.1f}%)")
    print(f"Production cost: ${total_cost:.4f}")
    print(f"Output: {output_path}")

    stats = {
        "valid": valid_count,
        "invalid": invalid_count,
        "total": len(products),
        "pass_rate": valid_count / len(products) if products else 0,
        "cost": total_cost,
        "failures": fail_details[:10],
    }
    return total_cost, stats


def spot_check(output_path: Path, count_per_category: int = 5):
    """Spot-check records from each category."""
    print(f"\n--- SPOT CHECK: {output_path.name} ---")

    by_variant = {}
    with open(output_path) as f:
        for line in f:
            rec = json.loads(line)
            variant = rec["metadata"]["variant"]
            by_variant.setdefault(variant, []).append(rec)

    for variant, records in sorted(by_variant.items()):
        rng = random.Random(RANDOM_SEED + 42)
        sample = rng.sample(records, min(count_per_category, len(records)))
        print(f"\n  [{variant}] ({len(records)} total, checking {len(sample)}):")
        for rec in sample:
            v = validate_description(rec["response"], "")
            status = "PASS" if v["valid"] else "FAIL"
            name = rec["metadata"].get("product_name", "?")
            print(f"    {status} {name}: words={v['word_count']} bullets={v['bullet_count']} perfect={v['ends_with_perfect']} sups={v['no_superlatives']}")
            if not v["valid"]:
                print(f"      First 150 chars: {rec['response'][:150]}...")


def main():
    load_dotenv(ENV_PATH)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openrouter_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate product specs
    products = generate_all_products()

    # Step 2: Tuning phase
    tuning_cost = run_tuning(products, openrouter_key)

    # Step 3: Production -- GPT-4o
    gpt4o_path = OUTPUT_DIR / "ecommerce_products_gpt4o.jsonl"
    gpt4o_cost, gpt4o_stats = run_production(
        products=products,
        model="openrouter/openai/gpt-4o",
        api_key=openrouter_key,
        output_path=gpt4o_path,
        source_model_label="openai/gpt-4o",
    )

    # Step 3b: Production -- Haiku
    haiku_path = OUTPUT_DIR / "ecommerce_products_haiku.jsonl"
    haiku_cost, haiku_stats = run_production(
        products=products,
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=anthropic_key,
        output_path=haiku_path,
        source_model_label="anthropic/claude-haiku-4-5-20251001",
    )

    # Step 4: Spot checks
    spot_check(gpt4o_path)
    spot_check(haiku_path)

    # Step 5: Cost summary
    cost_summary = [
        {
            "model": "openai/gpt-4o",
            "tuning_cost_usd": round(tuning_cost, 4),
            "production_cost_usd": round(gpt4o_cost, 4),
            "total_cost_usd": round(tuning_cost + gpt4o_cost, 4),
            "pairs_generated": 300,
            "pass_rate": round(gpt4o_stats["pass_rate"], 4),
        },
        {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "tuning_cost_usd": 0.0,
            "production_cost_usd": round(haiku_cost, 4),
            "total_cost_usd": round(haiku_cost, 4),
            "pairs_generated": 300,
            "pass_rate": round(haiku_stats["pass_rate"], 4),
        },
    ]

    cost_path = OUTPUT_DIR / "cost_summary.json"
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"\nCost summary saved to {cost_path}")

    haiku_fail_rate = haiku_stats["invalid"] / haiku_stats["total"] if haiku_stats["total"] else 0
    if haiku_fail_rate > 0.10:
        print(f"\nWARNING: Haiku format failure rate is {haiku_fail_rate:.1%} (>10%)")
        print("This will be documented in SOURCES.md")

    total_cost = tuning_cost + gpt4o_cost + haiku_cost
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"GPT-4o:  {gpt4o_stats['valid']}/{gpt4o_stats['total']} valid ({gpt4o_stats['pass_rate']:.1%}) | ${tuning_cost + gpt4o_cost:.4f}")
    print(f"Haiku:   {haiku_stats['valid']}/{haiku_stats['total']} valid ({haiku_stats['pass_rate']:.1%}) | ${haiku_cost:.4f}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"{'='*60}")

    return {
        "tuning_cost": tuning_cost,
        "gpt4o_cost": gpt4o_cost,
        "haiku_cost": haiku_cost,
        "total_cost": total_cost,
        "gpt4o_stats": gpt4o_stats,
        "haiku_stats": haiku_stats,
        "haiku_fail_rate_high": haiku_fail_rate > 0.10,
    }


if __name__ == "__main__":
    result = main()
