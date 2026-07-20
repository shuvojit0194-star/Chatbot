# Spec: Weather Display Feature

**Jira Epic**: [SCRUM-10](https://shuvojit0194.atlassian.net/browse/SCRUM-10)
**PRD**: Weather Display Feature v1.0
**Status**: Ready for Development
**Priority**: Medium
**API**: OpenWeatherMap (recommended) — store key as `WEATHER_API_KEY` in Render env vars

---

## Overview

Add real-time weather data to the Agent Assistant home screen. The feature surfaces current conditions, icons, location detection, city search, 5-day forecast, unit preferences, detailed metrics, auto-refresh, weather-aware chat suggestions, and severe weather alerts.

---

## Architecture

**New backend endpoint:**
```
GET /weather?lat={lat}&lon={lon}&units={metric|imperial}
```
- Calls OpenWeatherMap API
- Caches response for 30 minutes to reduce API calls
- Returns: temperature, condition, icon code, humidity, wind speed, feels-like, 5-day forecast, alerts

**Frontend additions in `static/index.html`:**
- Weather widget component on home screen
- Geolocation JS using `navigator.geolocation`
- City search autocomplete
- C/F toggle with localStorage persistence
- Auto-refresh polling every 30 minutes
- Severe weather alert banner above chat

**Environment variable to add in Render:**
```
WEATHER_API_KEY=your_openweathermap_api_key
```

---

## Implementation Order

Stories must be implemented in this order due to dependencies:

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

**As a** user of the Agent Assistant chatbot
**I want to** see the current weather conditions displayed on the home screen
**So that** I can get a quick weather snapshot without leaving the app

**Acceptance Criteria:**
- [ ] AC1: WHEN the user opens the app THE SYSTEM SHALL display a weather widget on the home screen showing current temperature, condition, and location
- [ ] AC2: WHEN weather data is loading THE SYSTEM SHALL display a loading skeleton so the user knows data is being fetched
- [ ] AC3: WHEN weather data fails to load THE SYSTEM SHALL display a friendly error message with a retry option

**Files to Modify:**
- `main.py` — add `/weather` endpoint with OpenWeatherMap API call and 30-min caching
- `static/index.html` — add weather widget HTML/CSS/JS, fetch from `/weather`

**Technical Notes:**
- Widget must load asynchronously — chat interface must not block on weather data
- Use skeleton loader (pulsing gray bars) during fetch
- Error state must include a "Retry" button that re-calls `/weather`
- Cache key: `weather_{lat}_{lon}_{units}`

**Definition of Done:**
- [ ] All ACs pass at http://localhost:8080
- [ ] /health still returns agent: true
- [ ] Weather widget does not block chat from loading
- [ ] Branch: feat/SCRUM-11-current-weather-widget
- [ ] Commit: feat(SCRUM-11): add current weather widget to home screen

---

### SCRUM-12 — Display Weather Icon and Condition Label

**As a** user glancing at the weather widget
**I want to** see a clear weather icon and condition label
**So that** I can understand the weather at a glance

**Acceptance Criteria:**
- [ ] AC1: WHEN weather data is loaded THE SYSTEM SHALL display a weather condition icon (SVG or image) that matches the current condition
- [ ] AC2: WHEN weather data is loaded THE SYSTEM SHALL display a human-readable condition label (e.g. Partly Cloudy, Heavy Rain)
- [ ] AC3: WHEN displaying at night THE SYSTEM SHALL show night-specific icons (e.g. moon instead of sun)

**Files to Modify:**
- `static/index.html` — add icon rendering logic using OpenWeatherMap icon codes

**Technical Notes:**
- OpenWeatherMap returns icon codes (e.g. `01d` = clear sky day, `01n` = clear sky night)
- Use OpenWeatherMap icon CDN: `https://openweathermap.org/img/wn/{icon}@2x.png`
- Map icon codes to human-readable labels in a JS lookup table

**Definition of Done:**
- [ ] All ACs pass
- [ ] Night icons render correctly when system time is after sunset
- [ ] Branch: feat/SCRUM-12-weather-icons
- [ ] Commit: feat(SCRUM-12): add weather icons and condition labels

---

### SCRUM-13 — Auto-Detect User Location for Weather

**As a** user who has not manually set a location
**I want to** have the app automatically detect my current location
**So that** I do not have to manually configure my city every time

**Acceptance Criteria:**
- [ ] AC1: WHEN the user opens the app for the first time THE SYSTEM SHALL request browser geolocation permission
- [ ] AC2: WHEN the user grants permission THE SYSTEM SHALL use their coordinates to fetch and display local weather
- [ ] AC3: WHEN the user denies permission THE SYSTEM SHALL prompt them to manually enter a city name instead
- [ ] AC4: WHEN location is detected THE SYSTEM SHALL display the detected city name alongside the weather data

**Files to Modify:**
- `static/index.html` — add `navigator.geolocation.getCurrentPosition()` logic

**Technical Notes:**
- Only request geolocation once — store result in localStorage
- If permission denied, show city search input immediately (do not show blank widget)
- Use reverse geocoding from OpenWeatherMap to get city name from coordinates
- Default fallback city if all else fails: New York (lat: 40.7128, lon: -74.0060)

**Definition of Done:**
- [ ] All ACs pass
- [ ] Geolocation only requested once per browser session
- [ ] Denial gracefully shows city search
- [ ] Branch: feat/SCRUM-13-auto-detect-location
- [ ] Commit: feat(SCRUM-13): add auto-detect geolocation for weather

---

### SCRUM-14 — Manually Search and Set Weather Location

**As a** user who wants to check weather for a different city
**I want to** search for and set a specific city
**So that** I can check weather for any city, not just my current location

**Acceptance Criteria:**
- [ ] AC1: WHEN the user clicks the location name in the weather widget THE SYSTEM SHALL open a city search input field
- [ ] AC2: WHEN the user types at least 2 characters THE SYSTEM SHALL show autocomplete city suggestions
- [ ] AC3: WHEN the user selects a city THE SYSTEM SHALL update the weather widget to show weather for that city
- [ ] AC4: WHEN the user sets a new city THE SYSTEM SHALL save the preference so it persists on next visit

**Files to Modify:**
- `main.py` — add `/weather/search?q={city}` endpoint using OpenWeatherMap Geocoding API
- `static/index.html` — add city search input, autocomplete dropdown, localStorage save

**Technical Notes:**
- Use OpenWeatherMap Geocoding API: `http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5`
- Debounce search input by 300ms to avoid hammering the API
- Save selected city as `{name, lat, lon}` in localStorage under key `weather_location`

**Definition of Done:**
- [ ] All ACs pass
- [ ] Debounce implemented on search input
- [ ] City preference persists after page reload
- [ ] Branch: feat/SCRUM-14-manual-city-search
- [ ] Commit: feat(SCRUM-14): add manual city search for weather location

---

### SCRUM-15 — Display Weather in Celsius or Fahrenheit

**As a** user who prefers a specific temperature unit
**I want to** toggle between Celsius and Fahrenheit
**So that** I can read temperatures in the unit I am most comfortable with

**Acceptance Criteria:**
- [ ] AC1: WHEN weather is displayed THE SYSTEM SHALL show a C / F toggle button on the weather widget
- [ ] AC2: WHEN the user clicks the toggle THE SYSTEM SHALL immediately convert and re-display all temperatures in the selected unit
- [ ] AC3: WHEN the user selects a unit preference THE SYSTEM SHALL persist that preference across sessions

**Files to Modify:**
- `static/index.html` — add C/F toggle button, conversion logic, localStorage persistence

**Technical Notes:**
- Store preference in localStorage under key `weather_units` (`metric` or `imperial`)
- Convert all temperature fields: current temp, feels-like, forecast high/low
- Toggle must update all displayed temperatures without re-fetching from API
- Pass `units` param to `/weather` endpoint on next fetch

**Definition of Done:**
- [ ] All ACs pass
- [ ] All temperature values convert correctly (°C to °F: multiply by 9/5 then add 32)
- [ ] Preference persists after reload
- [ ] Branch: feat/SCRUM-15-celsius-fahrenheit-toggle
- [ ] Commit: feat(SCRUM-15): add Celsius/Fahrenheit toggle for weather

---

### SCRUM-16 — Display 5-Day Weather Forecast

**As a** user planning activities across the week
**I want to** view a 5-day weather forecast below the current weather widget
**So that** I can plan my week without switching to a separate weather app

**Acceptance Criteria:**
- [ ] AC1: WHEN weather data is loaded THE SYSTEM SHALL display a 5-day forecast row below the current conditions
- [ ] AC2: WHEN displaying the forecast THE SYSTEM SHALL show day name, weather icon, high temperature, and low temperature for each day
- [ ] AC3: WHEN the user clicks a forecast day THE SYSTEM SHALL expand to show hourly breakdown for that day

**Files to Modify:**
- `main.py` — extend `/weather` endpoint to include 5-day forecast data from OpenWeatherMap
- `static/index.html` — add forecast row and expandable hourly breakdown UI

**Technical Notes:**
- OpenWeatherMap 5-day forecast endpoint: `/forecast` (returns 3-hour intervals)
- Group 3-hour intervals by day to get daily high/low
- Hourly breakdown: show time, icon, and temperature for each 3-hour slot
- Collapse other expanded days when a new one is clicked

**Definition of Done:**
- [ ] All ACs pass
- [ ] Forecast data grouped correctly by calendar day
- [ ] Hourly breakdown expands/collapses correctly
- [ ] Branch: feat/SCRUM-16-5-day-forecast
- [ ] Commit: feat(SCRUM-16): add 5-day weather forecast

---

### SCRUM-17 — Show Humidity, Wind Speed, and Feels-Like Temperature

**As a** user who wants detailed weather context
**I want to** view humidity, wind speed, and feels-like temperature
**So that** I can make better decisions about what to wear or whether to go outside

**Acceptance Criteria:**
- [ ] AC1: WHEN weather data is loaded THE SYSTEM SHALL display humidity percentage, wind speed, and feels-like temperature in a secondary row below the main weather display
- [ ] AC2: WHEN the user hovers over any metric THE SYSTEM SHALL show a tooltip explaining what it means
- [ ] AC3: WHEN wind speed exceeds 30 mph THE SYSTEM SHALL highlight the wind speed value in amber

**Files to Modify:**
- `static/index.html` — add secondary metrics row, tooltips, amber highlight logic

**Technical Notes:**
- All three values are already returned by the `/weather` endpoint (from OpenWeatherMap current weather)
- Tooltip text: Humidity = "Percentage of moisture in the air", Wind = "Current wind speed", Feels Like = "What the temperature feels like accounting for wind and humidity"
- Amber threshold: 30 mph = 48.28 km/h = 13.41 m/s (convert based on units)

**Definition of Done:**
- [ ] All ACs pass
- [ ] Amber highlight triggers at correct wind speed threshold in both metric and imperial
- [ ] Branch: feat/SCRUM-17-detailed-weather-metrics
- [ ] Commit: feat(SCRUM-17): add humidity, wind speed, feels-like metrics

---

### SCRUM-18 — Weather Data Auto-Refresh

**As a** user who leaves the app open for extended periods
**I want to** have weather data automatically refresh every 30 minutes
**So that** I do not have to manually reload the page

**Acceptance Criteria:**
- [ ] AC1: WHEN the app is open THE SYSTEM SHALL automatically refresh weather data every 30 minutes
- [ ] AC2: WHEN a refresh occurs THE SYSTEM SHALL update the weather widget without a full page reload
- [ ] AC3: WHEN a refresh fails THE SYSTEM SHALL retain the last successful data and display a subtle last updated timestamp
- [ ] AC4: WHEN the user returns to the app after being away for more than 30 minutes THE SYSTEM SHALL trigger an immediate refresh

**Files to Modify:**
- `static/index.html` — add `setInterval` polling, Page Visibility API handler, last updated timestamp

**Technical Notes:**
- Use `setInterval` with 30 minutes (1,800,000ms)
- Use `document.addEventListener('visibilitychange')` to detect tab return
- Store last fetch timestamp in memory; compare on visibility change
- Show "Last updated: X min ago" below the widget in small gray text
- On failed refresh, keep previous data visible — do not show error state

**Definition of Done:**
- [ ] All ACs pass
- [ ] Tab return after 30+ min triggers immediate refresh
- [ ] Failed refresh retains stale data with timestamp visible
- [ ] Branch: feat/SCRUM-18-weather-auto-refresh
- [ ] Commit: feat(SCRUM-18): add 30-minute weather auto-refresh

---

### SCRUM-19 — Show Weather-Aware Chat Suggestions

**As a** user interacting with the Agent Assistant
**I want to** see contextual chat prompt suggestions relevant to current weather
**So that** I am inspired to ask weather-related questions

**Acceptance Criteria:**
- [ ] AC1: WHEN weather data is loaded and conditions are extreme (rain, snow, heat) THE SYSTEM SHALL show at least one weather-contextual prompt suggestion on the home screen
- [ ] AC2: WHEN the user clicks a weather suggestion THE SYSTEM SHALL pre-fill the chat input with that prompt
- [ ] AC3: WHEN conditions are normal THE SYSTEM SHALL show standard prompt suggestions instead

**Files to Modify:**
- `static/index.html` — add weather-aware suggestion logic based on condition codes
- `main.py` — optionally inject weather context into agent system prompt

**Technical Notes:**
- Map OpenWeatherMap condition codes to suggestion templates:
  - Rain (500-531): "What should I pack for a rainy day commute?"
  - Snow (600-622): "What are good indoor activities for a snowy day?"
  - Extreme heat (temp > 95°F / 35°C): "How can I stay cool in extreme heat?"
  - Thunderstorm (200-232): "Is it safe to drive in thunderstorm conditions?"
  - Normal (clear/clouds): show default suggestions
- Pre-fill is frontend only — click sets `chatInput.value` and focuses the input

**Definition of Done:**
- [ ] All ACs pass
- [ ] At least 4 weather condition categories mapped to suggestions
- [ ] Click pre-fills chat input and focuses it
- [ ] Branch: feat/SCRUM-19-weather-aware-suggestions
- [ ] Commit: feat(SCRUM-19): add weather-aware chat prompt suggestions

---

### SCRUM-20 — Severe Weather Alert Banner

**As a** user in an area with active weather alerts
**I want to** be notified of severe weather alerts relevant to my location
**So that** I am aware of dangerous conditions that may affect my safety

**Acceptance Criteria:**
- [ ] AC1: WHEN a government weather alert is active for the user's location THE SYSTEM SHALL display a prominent alert banner above the chat interface
- [ ] AC2: WHEN displaying an alert THE SYSTEM SHALL show the alert type, severity level, and a brief description
- [ ] AC3: WHEN the user clicks the alert banner THE SYSTEM SHALL expand to show the full alert details
- [ ] AC4: WHEN the user dismisses the alert THE SYSTEM SHALL hide the banner until a new alert is issued
- [ ] AC5: WHEN no active alerts exist THE SYSTEM SHALL not display the alert banner

**Files to Modify:**
- `main.py` — extend `/weather` endpoint to include alerts from OpenWeatherMap One Call API
- `static/index.html` — add alert banner component above chat, expand/dismiss logic

**Technical Notes:**
- OpenWeatherMap One Call API 3.0 includes `alerts` array (requires paid plan or free trial)
- Alternative: NWS API (free, US only) at `https://api.weather.gov/alerts/active?point={lat},{lon}`
- Store dismissed alert IDs in localStorage to avoid re-showing same alert
- Banner color by severity: Extreme = red, Severe = orange, Moderate = amber, Minor = yellow
- Banner must be dismissible with an X button

**Definition of Done:**
- [ ] All ACs pass
- [ ] Dismissed alerts do not reappear until a new alert event ID is received
- [ ] Banner does not render when alerts array is empty
- [ ] Branch: feat/SCRUM-20-severe-weather-alerts
- [ ] Commit: feat(SCRUM-20): add severe weather alert banner

---

## Global Definition of Done (all stories)

- [ ] `/health` returns `agent: true` and `rag_chain: true`
- [ ] Existing `/chat` functionality works without regression
- [ ] Weather widget loads asynchronously — chat is never blocked
- [ ] `WEATHER_API_KEY` is never exposed in frontend JS
- [ ] All API calls routed through backend `/weather` endpoint
- [ ] Branch naming: `feat/SCRUM-{id}-{short-name}`
- [ ] Commit format: `feat(SCRUM-{id}): description`
