# Spec: Weather Display Feature

**Jira Epic**: [SCRUM-10](https://shuvojit0194.atlassian.net/browse/SCRUM-10)
**PRD**: Weather Display Feature v1.1
**Status**: Ready for Development
**Priority**: Medium
**Weather API**: Open-Meteo — NO API key required, no signup, no credit card

---

## Overview

Add real-time weather data to the Agent Assistant home screen. The feature surfaces current conditions, icons, location detection, city search, 5-day forecast, unit preferences, detailed metrics, auto-refresh, weather-aware chat suggestions, and severe weather alerts.

---

## Architecture

**New backend endpoint:**
```
GET /weather?lat={lat}&lon={lon}&units={fahrenheit|celsius}
```

Calls Open-Meteo API — no API key needed:
```
https://api.open-meteo.com/v1/forecast
  ?latitude={lat}
  &longitude={lon}
  &current=temperature_2m,weathercode,windspeed_10m,relativehumidity_2m,apparent_temperature,is_day
  &daily=temperature_2m_max,temperature_2m_min,weathercode
  &hourly=temperature_2m,weathercode
  &temperature_unit={fahrenheit|celsius}
  &windspeed_unit=mph
  &timezone=auto
```

**City search endpoint:**
```
GET /weather/search?q={city}
```
Calls Open-Meteo Geocoding API — also no key needed:
```
https://geocoding-api.open-meteo.com/v1/search?name={city}&count=5&language=en&format=json
```

**Severe weather alerts (SCRUM-20):**
```
GET /weather/alerts?lat={lat}&lon={lon}
```
Calls NWS API — free, no key, US only:
```
https://api.weather.gov/alerts/active?point={lat},{lon}
```

**Frontend additions in `static/index.html`:**
- Weather widget component on home screen
- Geolocation JS using `navigator.geolocation`
- City search autocomplete
- C/F toggle with localStorage persistence
- Auto-refresh polling every 30 minutes
- Severe weather alert banner above chat

**No environment variables needed** — Open-Meteo requires no API key.

---

## WMO Weather Code Reference

Open-Meteo uses WMO weather codes. Use this mapping for icons and condition labels:

| Code | Condition | Icon |
|------|-----------|------|
| 0 | Clear sky | ☀️ |
| 1, 2, 3 | Partly cloudy | ⛅ |
| 45, 48 | Foggy | 🌫️ |
| 51, 53, 55 | Drizzle | 🌦️ |
| 61, 63, 65 | Rain | 🌧️ |
| 71, 73, 75 | Snow | 🌨️ |
| 80, 81, 82 | Rain showers | 🌧️ |
| 85, 86 | Snow showers | 🌨️ |
| 95 | Thunderstorm | ⛈️ |
| 96, 99 | Thunderstorm with hail | ⛈️ |

Night variant: when `is_day === 0` and code is 0 → show 🌙 instead of ☀️

---

## Implementation Order

| Sprint | Jira | Story | Priority | Effort |
|--------|------|-------|----------|--------|
| 1 | [SCRUM-11](https://shuvojit0194.atlassian.net/browse/SCRUM-11) | View Current Weather on Chat Home Screen | High | M |
| 2 | [SCRUM-12](https://shuvojit0194.atlassian.net/browse/SCRUM-12) | Display Weather Icon and Condition Label | High | S |
| 3 | [SCRUM-13](https://shuvojit0194.atlassian.net/browse/SCRUM-13) | Auto-Detect User Location for Weather | High | M |
| 4 | [SCRUM-14](https://shuvojit0194.atlassian.net/browse/SCRUM-14) | Manually Search and Set Weather Location | High | S |
| 5 | [SCRUM-15](https://shuvojit0194.atlassian.net/browse/SCRUM-15) | Display Weather in Celsius or Fahrenheit | Medium | S |
| 6 | [SCRUM-16](https://shuvojit0194.atlassian.net/browse/SCRUM-16) | Display 5-Day Weather Forecast | Medium | M |
| 7 | [SCRUM-17](https://shuvojit0194.atlassian.net/browse/SCRUM-17) | Show Humidity, Wind Speed, and Feels-Like Temperature | Low | S |
| 8 | [SCRUM-18](https://shuvojit0194.atlassian.net/browse/SCRUM-18) | Weather Data Auto-Refresh | Low | S |
| 9 | [SCRUM-19](https://shuvojit0194.atlassian.net/browse/SCRUM-19) | Show Weather-Aware Chat Suggestions | Medium | L |
| 10 | [SCRUM-20](https://shuvojit0194.atlassian.net/browse/SCRUM-20) | Severe Weather Alert Banner | Medium | L |

---

## Story Specs

---

### SCRUM-11 — View Current Weather on Chat Home Screen

**Acceptance Criteria:**
- [ ] AC1: WHEN the user opens the app THE SYSTEM SHALL display a weather widget showing current temperature, condition, and location
- [ ] AC2: WHEN weather data is loading THE SYSTEM SHALL display a loading skeleton
- [ ] AC3: WHEN weather data fails to load THE SYSTEM SHALL display a friendly error message with a retry option

**Files to Modify:** `main.py`, `static/index.html`

**Technical Notes:**
- Open-Meteo URL: `https://api.open-meteo.com/v1/forecast` — no API key needed
- Required current fields: `temperature_2m,weathercode,windspeed_10m,relativehumidity_2m,apparent_temperature,is_day`
- Cache response for 30 minutes on backend, key: `weather_{lat}_{lon}_{units}`
- Sample Python call:
```python
import httpx
async def get_weather(lat, lon, units="fahrenheit"):
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,weathercode,windspeed_10m,relativehumidity_2m,apparent_temperature,is_day",
        "temperature_unit": units, "windspeed_unit": "mph", "timezone": "auto"
    }
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        return r.json()
```

**Branch:** `feat/SCRUM-11-current-weather-widget`
**Commit:** `feat(SCRUM-11): add current weather widget to home screen`

---

### SCRUM-12 — Display Weather Icon and Condition Label

**Acceptance Criteria:**
- [ ] AC1: WHEN weather data is loaded THE SYSTEM SHALL display a weather icon matching the WMO weather code
- [ ] AC2: WHEN weather data is loaded THE SYSTEM SHALL display a human-readable condition label
- [ ] AC3: WHEN displaying at night THE SYSTEM SHALL show night-specific icons

**Files to Modify:** `static/index.html`

**Technical Notes:**
- Use WMO code + `is_day` field from Open-Meteo response
- Implement as a JS lookup object — no external icon CDN needed, use emoji
- Night icon: `is_day === 0 && weathercode === 0` → 🌙

```javascript
const WMO_LABELS = {
  0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
  45: "Foggy", 48: "Icy Fog", 51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
  61: "Light Rain", 63: "Rain", 65: "Heavy Rain", 71: "Light Snow", 73: "Snow",
  75: "Heavy Snow", 80: "Rain Showers", 81: "Heavy Showers", 82: "Violent Showers",
  85: "Snow Showers", 86: "Heavy Snow Showers", 95: "Thunderstorm",
  96: "Thunderstorm with Hail", 99: "Heavy Thunderstorm"
};
```

**Branch:** `feat/SCRUM-12-weather-icons`
**Commit:** `feat(SCRUM-12): add weather icons and condition labels`

---

### SCRUM-13 — Auto-Detect User Location for Weather

**Acceptance Criteria:**
- [ ] AC1: WHEN the user opens the app for the first time THE SYSTEM SHALL request browser geolocation permission
- [ ] AC2: WHEN the user grants permission THE SYSTEM SHALL use coordinates to fetch local weather
- [ ] AC3: WHEN the user denies permission THE SYSTEM SHALL prompt them to manually enter a city name
- [ ] AC4: WHEN location is detected THE SYSTEM SHALL display the detected city name

**Files to Modify:** `static/index.html`, `main.py`

**Technical Notes:**
- Reverse geocoding for city name (free, no key):
  `https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json`
  → use `address.city` or `address.town`
- Store in localStorage: `weather_location = {name, latitude, longitude}`
- Default fallback: St. Louis, MO (lat: 38.6270, lon: -90.1994)
- Only prompt geolocation once — check localStorage first on every load

**Branch:** `feat/SCRUM-13-auto-detect-location`
**Commit:** `feat(SCRUM-13): add auto-detect geolocation for weather`

---

### SCRUM-14 — Manually Search and Set Weather Location

**Acceptance Criteria:**
- [ ] AC1: WHEN the user clicks the location name THE SYSTEM SHALL open a city search input
- [ ] AC2: WHEN the user types at least 2 characters THE SYSTEM SHALL show autocomplete city suggestions
- [ ] AC3: WHEN the user selects a city THE SYSTEM SHALL update the weather widget
- [ ] AC4: WHEN the user sets a city THE SYSTEM SHALL save the preference for next visit

**Files to Modify:** `main.py`, `static/index.html`

**Technical Notes:**
- Open-Meteo Geocoding API (no key):
  `https://geocoding-api.open-meteo.com/v1/search?name={city}&count=5&language=en&format=json`
- Response fields: `name`, `country`, `admin1` (state), `latitude`, `longitude`
- Display as: `{name}, {admin1}, {country}`
- Debounce input by 300ms
- Save to localStorage: `weather_location = {name, latitude, longitude}`

**Branch:** `feat/SCRUM-14-manual-city-search`
**Commit:** `feat(SCRUM-14): add manual city search for weather location`

---

### SCRUM-15 — Display Weather in Celsius or Fahrenheit

**Acceptance Criteria:**
- [ ] AC1: WHEN weather is displayed THE SYSTEM SHALL show a C / F toggle button
- [ ] AC2: WHEN the user clicks toggle THE SYSTEM SHALL re-display all temperatures in the new unit
- [ ] AC3: WHEN user selects a unit THE SYSTEM SHALL persist that preference

**Files to Modify:** `static/index.html`

**Technical Notes:**
- Store in localStorage: `weather_units` = `fahrenheit` or `celsius`
- On toggle: re-call `/weather` with new `units` param — do NOT convert client-side
- Open-Meteo handles unit conversion natively via `temperature_unit` param

**Branch:** `feat/SCRUM-15-celsius-fahrenheit-toggle`
**Commit:** `feat(SCRUM-15): add Celsius/Fahrenheit toggle for weather`

---

### SCRUM-16 — Display 5-Day Weather Forecast

**Acceptance Criteria:**
- [ ] AC1: WHEN weather is loaded THE SYSTEM SHALL display a 5-day forecast below current conditions
- [ ] AC2: WHEN displaying forecast THE SYSTEM SHALL show day name, icon, high, and low temp per day
- [ ] AC3: WHEN the user clicks a forecast day THE SYSTEM SHALL expand to show hourly breakdown

**Files to Modify:** `main.py`, `static/index.html`

**Technical Notes:**
- Add to Open-Meteo call: `&daily=temperature_2m_max,temperature_2m_min,weathercode&hourly=temperature_2m,weathercode`
- Daily array: `daily.time[0..4]` = next 5 days
- Hourly: 168 values (7 days × 24hrs) — filter by matching date prefix for day breakdown
- Collapse other expanded days when a new day is clicked

**Branch:** `feat/SCRUM-16-5-day-forecast`
**Commit:** `feat(SCRUM-16): add 5-day weather forecast`

---

### SCRUM-17 — Show Humidity, Wind Speed, and Feels-Like Temperature

**Acceptance Criteria:**
- [ ] AC1: WHEN weather is loaded THE SYSTEM SHALL display humidity, wind speed, and feels-like in a secondary row
- [ ] AC2: WHEN hovering over any metric THE SYSTEM SHALL show a tooltip
- [ ] AC3: WHEN wind speed exceeds 30 mph THE SYSTEM SHALL highlight the value in amber

**Files to Modify:** `static/index.html`

**Technical Notes:**
- All from Open-Meteo `current` object: `relativehumidity_2m`, `windspeed_10m`, `apparent_temperature`
- Wind already in mph (set via `windspeed_unit=mph`)
- Amber threshold: `windspeed_10m > 30`

**Branch:** `feat/SCRUM-17-detailed-weather-metrics`
**Commit:** `feat(SCRUM-17): add humidity, wind speed, feels-like metrics`

---

### SCRUM-18 — Weather Data Auto-Refresh

**Acceptance Criteria:**
- [ ] AC1: WHEN the app is open THE SYSTEM SHALL refresh weather every 30 minutes
- [ ] AC2: WHEN refreshing THE SYSTEM SHALL update the widget without a page reload
- [ ] AC3: WHEN refresh fails THE SYSTEM SHALL retain last data and show a last updated timestamp
- [ ] AC4: WHEN user returns after 30+ minutes THE SYSTEM SHALL trigger an immediate refresh

**Files to Modify:** `static/index.html`

**Technical Notes:**
- `setInterval(fetchWeather, 1_800_000)`
- `document.addEventListener('visibilitychange')` — if hidden > 30 min → refresh on return
- Track last fetch: `window._weatherLastFetch = Date.now()`
- Show "Last updated: X min ago" in small gray text

**Branch:** `feat/SCRUM-18-weather-auto-refresh`
**Commit:** `feat(SCRUM-18): add 30-minute weather auto-refresh`

---

### SCRUM-19 — Show Weather-Aware Chat Suggestions

**Acceptance Criteria:**
- [ ] AC1: WHEN conditions are extreme THE SYSTEM SHALL show at least one weather-contextual prompt suggestion
- [ ] AC2: WHEN the user clicks a suggestion THE SYSTEM SHALL pre-fill the chat input
- [ ] AC3: WHEN conditions are normal THE SYSTEM SHALL show standard suggestions

**Files to Modify:** `static/index.html`

**Technical Notes:**
Map WMO codes to suggestions:
```javascript
const WEATHER_SUGGESTIONS = {
  rain: { codes: [51,53,55,61,63,65,80,81,82], prompt: "What should I pack for a rainy day commute?" },
  snow: { codes: [71,73,75,85,86], prompt: "What are good indoor activities for a snowy day?" },
  storm: { codes: [95,96,99], prompt: "Is it safe to drive in thunderstorm conditions?" },
  fog:  { codes: [45,48], prompt: "Tips for driving safely in foggy conditions?" },
  // heat: apparent_temperature > 95 → "How can I stay cool in extreme heat?"
};
```

**Branch:** `feat/SCRUM-19-weather-aware-suggestions`
**Commit:** `feat(SCRUM-19): add weather-aware chat prompt suggestions`

---

### SCRUM-20 — Severe Weather Alert Banner

**Acceptance Criteria:**
- [ ] AC1: WHEN an alert is active THE SYSTEM SHALL display a prominent banner above the chat interface
- [ ] AC2: WHEN displaying an alert THE SYSTEM SHALL show type, severity level, and brief description
- [ ] AC3: WHEN the user clicks the banner THE SYSTEM SHALL expand to full alert details
- [ ] AC4: WHEN the user dismisses THE SYSTEM SHALL hide until a new alert is issued
- [ ] AC5: WHEN no alerts exist THE SYSTEM SHALL not display the banner

**Files to Modify:** `main.py`, `static/index.html`

**Technical Notes:**
- NWS API (free, no key, US only): `https://api.weather.gov/alerts/active?point={lat},{lon}`
- Response: `features[].properties` → `event`, `severity`, `description`, `id`
- Severity colors: Extreme=red, Severe=orange, Moderate=amber, Minor=yellow
- Store dismissed IDs in localStorage: `dismissed_alerts` (array)
- For non-US locations: skip alert fetch, show no banner silently

**Branch:** `feat/SCRUM-20-severe-weather-alerts`
**Commit:** `feat(SCRUM-20): add severe weather alert banner`

---

## Global Definition of Done

- [ ] `/health` returns `agent: true` and `rag_chain: true`
- [ ] Existing `/chat` works without regression
- [ ] Weather widget loads asynchronously — chat never blocked
- [ ] No API keys anywhere — Open-Meteo and NWS are both keyless
- [ ] All API calls routed through backend (never called directly from frontend JS)
- [ ] Branch: `feat/SCRUM-{id}-{short-name}`
- [ ] Commit: `feat(SCRUM-{id}): description`
