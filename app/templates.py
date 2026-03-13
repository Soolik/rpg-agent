from __future__ import annotations

from textwrap import dedent


NPC_TEMPLATE = dedent("""
# NPC

## Identity
- Name:
- Role in campaign:
- Status:
- Faction:
- Related threads:
- Related locations:

## Short description

## Appearance and vibe

## Personality

## Motivations

## Secrets

## Relationships

## What they know

## GM usage notes
""").strip()


LOCATION_TEMPLATE = dedent("""
# Location

## Identity
- Name:
- Type:
- Region:
- Status:
- Related factions:
- Related threads:

## Short description

## Appearance and atmosphere

## Narrative purpose

## Key features

## Threats and tensions

## Secrets

## Related NPCs

## GM usage notes
""").strip()


FACTION_TEMPLATE = dedent("""
# Faction

## Identity
- Name:
- Type:
- Status:
- Area of influence:
- Related locations:
- Related threads:

## Short description

## Public goals

## Hidden goals

## Methods and culture

## Structure

## Relationships

## Secrets

## Narrative purpose
""").strip()


THREAD_TEMPLATE = dedent("""
# Thread

## Identity
- ID:
- Title:
- Status:
- Stakes:
- Last updated:
- Related NPCs:
- Related locations:
- Related factions:

## What this thread is about

## Why it matters

## Current state

## Possible developments

## Clues and leads

## Secrets

## Next GM steps
""").strip()


SECRET_TEMPLATE = dedent("""
# Secret

## Identity
- Name:
- Category:
- Status:
- Related threads:
- Related NPCs:
- Related locations:

## Truth

## Who knows

## How it can be discovered

## Consequences of reveal
""").strip()
