# Contributing to GradSleuth

Thank you for your interest in contributing to GradSleuth! This document provides guidelines for contributing to the project.

## Table of Contents

- [Branch Strategy](#branch-strategy)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Branch Strategy

GradSleuth uses a simplified Git branching model:

```
main (protected, production-ready code)
├── develop (integration branch for ongoing work)
└── feature/* (feature development branches)
```

### Branch Purposes

- **main**: Contains production-ready code. All code here should be stable and tested.
- **develop**: Integration branch where features come together. This is the base for feature branches.
- **feature/\***: Individual feature development branches.

## Branch Naming Conventions

Use the following naming patterns for branches:

### Feature Branches
```
feature/[description]
feature/[service]-[description]  # When working with specific services
```

Examples:
- `feature/faculty-search`
- `feature/pubmed-integration`
- `feature/ml-biobert-embeddings`

### Bug Fix Branches
```
bugfix/[issue-number]-[description]
bugfix/[description]  # If no issue exists
```

Examples:
- `bugfix/42-search-timeout`
- `bugfix/faculty-scraper-encoding`

### Hotfix Branches
```
hotfix/[severity]-[description]
```

Examples:
- `hotfix/critical-api-error`
- `hotfix/security-xss-vulnerability`

### Documentation Branches
```
docs/[description]
```

Examples:
- `docs/api-documentation`
- `docs/setup-instructions`

## Development Workflow

### Starting New Work

1. **Always start from `develop`**:
   ```bash
   git checkout develop
   git pull origin develop
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Develop and commit regularly**:
   ```bash
   git add .
   git commit -m "descriptive commit message"
   ```

4. **Keep your branch updated**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout feature/your-feature-name
   git merge develop
   ```

5. **Push your branch**:
   ```bash
   git push -u origin feature/your-feature-name
   ```

### Merging to Develop

**Current Stage**: As a single-developer project, you can merge directly to `develop`:

```bash
git checkout develop
git merge feature/your-feature-name
git push origin develop
```

**Future**: When the team grows, use Pull Requests (see below).

### Merging to Main

**Always** merge to `main` through `develop`:

```bash
# Ensure develop is up to date and tested
git checkout develop
git pull origin develop

# Merge to main
git checkout main
git merge develop
git push origin main
```

## Code Standards

### JavaScript

- Use ES6+ syntax where appropriate
- Use semicolons
- Use 2-space indentation
- Use meaningful variable and function names
- Add comments for complex logic

### HTML/CSS

- Use semantic HTML5 elements
- Follow BEM naming convention for CSS classes (when applicable)
- Keep styles organized and modular

### General

- **Security First**: Never commit sensitive data (API keys, credentials, etc.)
- **No console.logs in production**: Remove debug statements before merging to main
- **Test your changes**: Ensure code works as expected before committing
- **Keep commits focused**: One logical change per commit

## Commit Messages

Write clear, descriptive commit messages following this format:

```
[Type] Brief description (50 chars or less)

More detailed explanation if needed (wrap at 72 chars).
- Bullet points for multiple changes
- Explain the "why" not just the "what"
```

### Commit Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring (no feature change)
- **test**: Adding or updating tests
- **chore**: Maintenance tasks, dependency updates

### Examples

```
feat: Add PubMed API integration for faculty research

Implemented search functionality using nickleby.js to query
PubMed for faculty publications based on research topics.

- Added search interface
- Integrated with NCBI E-utilities
- Added state-based affiliation filtering
```

```
fix: Resolve search timeout for long queries

Increased API timeout to 30s and added retry logic
for network failures.
```

## Pull Request Process

**Note**: PRs are **optional** for now as this is a single-developer project. Use them when you want formal review or for significant changes.

### When to Use PRs

- Major architectural changes
- Security-sensitive changes
- When you want to document a feature thoroughly
- When collaborating with others

### PR Guidelines

1. **Create from feature branch to develop**
2. **Write a clear title and description**:
   - What does this PR do?
   - Why is it needed?
   - What testing was done?

3. **Link related issues**: Reference issue numbers with `#issue-number`

4. **Keep PRs focused**: One feature or fix per PR

5. **Update documentation**: Include relevant doc updates in the PR

### PR Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- List of specific changes
- Another change

## Testing
How was this tested?

## Screenshots (if applicable)
Visual changes should include before/after screenshots.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data committed
- [ ] Tests pass (when CI is added)
```

## Future Enhancements

As the project grows, we may add:

- Automated testing requirements
- CI/CD pipeline integration
- Code coverage requirements
- Automated code quality checks
- CODEOWNERS for team-based review

## Questions?

For questions about contributing, please:
- Open an issue for discussion
- Review existing documentation in `/docs`
- Check the [ARCHITECTURE.md](ARCHITECTURE.md) for system design

---

**Remember**: These guidelines are here to help maintain code quality and project organization. They'll evolve as the project grows!
