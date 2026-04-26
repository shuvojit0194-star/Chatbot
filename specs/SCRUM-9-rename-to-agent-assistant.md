# Spec: Rename "Chatbot" to "Agent Assistant" across the UI

**Jira**: [SCRUM-9](https://shuvojit0194.atlassian.net/browse/SCRUM-9)
**Epic**: [SCRUM-7 — UI Branding & Aesthetic Improvements](https://shuvojit0194.atlassian.net/browse/SCRUM-7)
**Status**: Ready for Development
**Priority**: Low

---

## Summary
Rename all visible instances of "Chatbot" to "Agent Assistant" in the frontend UI for a more descriptive and professional brand name.

## Background & Motivation
Following the previous rename from "MCP Chatbot" to "Chatbot" (SCRUM-8), the product name is being updated again to "Agent Assistant" to better reflect the agentic nature of the tool powered by MCP. This is a text-only change with no functional impact.

## Acceptance Criteria

- [ ] AC1: WHEN the user loads the app THE SYSTEM SHALL display "Agent Assistant" as the browser tab title (not "Chatbot")
- [ ] AC2: WHEN the user is on the home screen THE SYSTEM SHALL display "Agent Assistant" as the hero/header title
- [ ] AC3: WHEN the user focuses on the message input THE SYSTEM SHALL show placeholder text "Message Agent Assistant..." (not "Message Chatbot...")
- [ ] AC4: WHEN the user views the sidebar THE SYSTEM SHALL display "Agent Assistant" as the sidebar header
- [ ] AC5: WHEN any other visible UI text references "Chatbot" as a display name THE SYSTEM SHALL display "Agent Assistant" instead

## Out of Scope
- Changes to backend code (main.py)
- Changes to API responses or metadata
- Changes to the subtitle/description text ("A helpful AI assistant powered by the Model Context Protocol...")
- Changes to scraped_urls.json, Dockerfile, or requirements.txt
- Any CSS, layout, or color changes

## Files to Modify
- `static/index.html` — all text changes confined to this file only

## Technical Notes
Search for all occurrences of "Chatbot" used as a visible display name in index.html and replace with "Agent Assistant". Be careful NOT to replace "Chatbot" in:
- HTML class names or IDs (e.g. class="chatbot-container")
- JavaScript variable names or function names
- API endpoint references
- Comments

Specific strings to find and replace:
- `<title>Chatbot</title>` → `<title>Agent Assistant</title>`
- Any `<h1>` or heading containing "Chatbot" → "Agent Assistant"
- Placeholder text "Message Chatbot..." → "Message Agent Assistant..."
- Sidebar label "Chatbot" used as app name → "Agent Assistant"

## Definition of Done
- [ ] All 5 ACs pass when tested in browser at http://localhost:8080
- [ ] No CSS, layout, or functionality changes introduced
- [ ] /health still returns agent: true
- [ ] No regressions on existing /chat functionality
- [ ] Branch pushed: feat/SCRUM-9-rename-to-agent-assistant
- [ ] Commit: feat(SCRUM-9): rename Chatbot to Agent Assistant in UI
