# Spec: Rename "MCP Chatbot" to "Chatbot" across the UI

**Jira**: [SCRUM-8](https://shuvojit0194.atlassian.net/browse/SCRUM-8)
**Epic**: [SCRUM-7 — UI Branding & Aesthetic Improvements](https://shuvojit0194.atlassian.net/browse/SCRUM-7)
**Status**: Ready for Development
**Priority**: Low

---

## Summary
Rename all visible instances of "MCP Chatbot" to "Chatbot" in the frontend UI for a cleaner, simpler brand name.

## Background & Motivation
The current UI displays "MCP Chatbot" in multiple places. The product should simply be called "Chatbot" for a cleaner aesthetic. This is a text-only change with no functional impact.

## Acceptance Criteria

- [ ] AC1: WHEN the user loads the app THE SYSTEM SHALL display "Chatbot" as the browser tab title (not "MCP Chatbot")
- [ ] AC2: WHEN the user is on the home screen THE SYSTEM SHALL display "Chatbot" as the hero/header title
- [ ] AC3: WHEN the user focuses on the message input THE SYSTEM SHALL show placeholder text "Message Chatbot..." (not "Message MCP Chatbot...")
- [ ] AC4: WHEN the user views the sidebar THE SYSTEM SHALL display "Chatbot" as the sidebar header (not "MCP")
- [ ] AC5: WHEN any other visible UI text references "MCP Chatbot" THE SYSTEM SHALL display "Chatbot" instead

## Out of Scope
- Changes to backend code (main.py)
- Changes to API responses or metadata
- Changes to the subtitle/description text ("A helpful AI assistant powered by the Model Context Protocol...")
- Changes to scraped_urls.json or any other config files
- Any CSS or layout changes

## Files to Modify
- `static/index.html` — all text changes confined to this file only

## Technical Notes
Search for all occurrences of "MCP Chatbot" and "MCP" used as a display name in index.html and replace with "Chatbot". Use find-and-replace carefully — do NOT replace "MCP" in comments, class names, or non-visible text that refers to the protocol itself.

Specific strings to find and replace:
- `<title>MCP Chatbot</title>` → `<title>Chatbot</title>`
- Any `<h1>` or heading containing "MCP Chatbot" → "Chatbot"
- Placeholder text "Message MCP Chatbot..." → "Message Chatbot..."
- Sidebar label "MCP" used as app name → "Chatbot"

## Definition of Done
- [ ] All 5 ACs pass when tested in browser at http://localhost:8080
- [ ] No other UI text or functionality is affected
- [ ] /health still returns agent: true
- [ ] Branch pushed: feat/SCRUM-8-rename-to-chatbot
- [ ] Commit: feat(SCRUM-8): rename MCP Chatbot to Chatbot in UI
