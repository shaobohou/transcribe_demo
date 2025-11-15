# Documentation Sitemap

This is the central index for all transcribe-demo documentation. Use this to navigate to the documentation you need.

## Quick Navigation

**I want to...**
- 🚀 **Get started** → [README.md](README.md)
- 💻 **Contribute code** → [CLAUDE.md](CLAUDE.md)
- 🏗️ **Understand the architecture** → [DESIGN.md](DESIGN.md)
- 🔧 **Find refactoring opportunities** → [TODO.md](TODO.md)
- 📊 **Work with session logs** → [SESSIONS.md](SESSIONS.md)
- 🤖 **Work with AI assistants** → [CLAUDE.md](CLAUDE.md) or [GEMINI.md](GEMINI.md)

---

## Documentation Organization

### User Documentation

**[README.md](README.md)** - *Start here if you're a user*
- Installation and setup instructions
- Feature overview and usage examples
- Command-line interface reference
- Configuration options for both backends
- Session logging overview

**[SESSIONS.md](SESSIONS.md)** - *Session logging and replay guide*
- Complete session log format specification
- Directory structure and file formats
- JSON schema for `session.json`
- Session replay utility (`transcribe-session`)
- How to list, inspect, and retranscribe sessions
- Python API and command-line examples
- Audio file format details

---

### Developer Documentation

**[CLAUDE.md](CLAUDE.md)** - *Development workflow and implementation rules*
- Common commands and development workflow
- Branch strategy and PR process
- Key files and critical implementation rules
- Testing strategy and gotchas
- Pre-commit requirements
- **Use this for**: Day-to-day development, testing, and contributing

**[DESIGN.md](DESIGN.md)** - *Architecture and design rationale*
- Design philosophy and goals
- Architecture overview with diagrams
- Backend design decisions
- VAD-based chunking strategy
- Stitching and punctuation cleanup logic
- Session logging and replay design
- **Use this for**: Understanding why the system works the way it does

**[TODO.md](TODO.md)** - *Refactoring opportunities and technical debt*
- Implementation-level refactoring opportunities
- Code quality improvements
- Prioritized action plans
- When and how to refactor
- **Use this for**: Finding areas to improve code quality

---

### Refactoring History

**[REFACTORING_HISTORY.md](REFACTORING_HISTORY.md)** - *Chronicle of major refactoring efforts*
- Backend protocol refactoring (2025-11-14): Protocol-based architecture, configuration dataclasses
- CLI refactoring (2025-11): Protocol adoption, diff module extraction, code reduction
- Migration guides and lessons learned
- Code metrics and impact analysis
- **Use this for**: Understanding what was refactored and why (historical reference)

---

### AI Assistant Instructions

**[AGENTS.md](AGENTS.md)** - *Instructions for AI coding agents*
- References CLAUDE.md for project details

**[GEMINI.md](GEMINI.md)** - *Instructions for Gemini AI*
- References CLAUDE.md for project details

---

## Document Boundaries

Each document has a specific purpose to avoid overlap:

| Document | Answers | Audience | Update When |
|----------|---------|----------|-------------|
| **README.md** | "How do I use this?" | End users | Adding features, changing CLI |
| **CLAUDE.md** | "How do I develop this?" | Contributors | Changing workflow, adding tests |
| **DESIGN.md** | "Why is it designed this way?" | Developers, maintainers | Major architecture changes |
| **TODO.md** | "What should we improve?" | Contributors | Finding refactoring opportunities |
| **SESSIONS.md** | "How do sessions work?" | Users, automation scripts | Changing log format or replay utility |
| **REFACTORING_HISTORY.md** | "What was refactored and why?" | Developers | Historical reference (rarely updated) |

---

## Documentation Workflow

### When Creating New Documentation

1. Add the new document to this sitemap
2. Update the appropriate section (User, Developer, or AI Assistant)
3. Add a quick navigation link if appropriate
4. Do NOT add cross-references to other documents (use this sitemap instead)

### When Reading Documentation

1. Start here (SITEMAP.md) to find the right document
2. Navigate to the specific document you need
3. Return here if you need to find related information

### When Updating Documentation

1. Update the relevant document
2. If the update is significant, update this sitemap
3. Do NOT update cross-references in other documents

---

## Common Scenarios

### 🎯 I'm a new user
1. Start with [README.md](README.md) for installation and basic usage
2. Check [SESSIONS.md](SESSIONS.md) to understand session logging and replay

### 🎯 I want to contribute code
1. Read [CLAUDE.md](CLAUDE.md) for development workflow
2. Check [DESIGN.md](DESIGN.md) to understand the architecture
3. Look at [TODO.md](TODO.md) for areas that need improvement

### 🎯 I'm debugging an issue
1. Check [DESIGN.md](DESIGN.md) for how the system is supposed to work
2. Review [SESSIONS.md](SESSIONS.md) to analyze session data
3. See [CLAUDE.md](CLAUDE.md) for testing and debugging guidelines

### 🎯 I'm planning a refactoring
1. Review [TODO.md](TODO.md) for existing refactoring opportunities
2. Check [DESIGN.md](DESIGN.md) to understand current architecture
3. Follow [CLAUDE.md](CLAUDE.md) testing guidelines before and after

### 🎯 I'm using an AI assistant
1. The AI should read [CLAUDE.md](CLAUDE.md) for project-specific rules
2. It may reference [DESIGN.md](DESIGN.md) for architecture understanding
3. It can suggest improvements from [TODO.md](TODO.md)

---

## Keeping Documentation Updated

**Golden Rule**: When in doubt about where information belongs, use this sitemap to find the right document.

**Avoid**: Cross-referencing between documents (except through this sitemap)

**Instead**: Update this sitemap to add new documents or reorganize navigation

---

*Last Updated: 2025-11-15*
