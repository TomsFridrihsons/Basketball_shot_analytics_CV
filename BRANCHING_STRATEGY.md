# Git Branching Strategy

This repository uses a multi-environment branching strategy to separate development, testing, and production code.

## Branch Structure

```
main (Production)
  ↑
testing (Testing/Staging)
  ↑
development (Development)
  ↑
colab-training (Colab Training)
```

## Branch Descriptions

### 🌟 `main` - Production Environment
- **Purpose**: Final, production-ready code
- **Status**: Stable, tested, and working
- **Who can push**: Only after testing approval
- **What goes here**: 
  - Fully tested and verified code
  - Final model versions
  - Production-ready features
  - Stable configurations

### 🧪 `testing` - Testing/Staging Environment
- **Purpose**: Pre-production testing environment
- **Status**: Code ready for production testing
- **Who can push**: After development completion
- **What goes here**:
  - Code that passed development testing
  - Features ready for production validation
  - Models tested in development
  - Final checks before production

### 💻 `development` - Development Environment
- **Purpose**: Active development workspace
- **Status**: Can be unstable, experimental
- **Who can push**: Developers actively working
- **What goes here**:
  - New features in development
  - Experimental code
  - Breaking changes
  - Work-in-progress features
  - Can mess things up - that's OK!

### 🚀 `colab-training` - Colab Training Environment
- **Purpose**: Model training in Google Colab
- **Status**: Training experiments
- **Who can push**: After training completion
- **What goes here**:
  - Trained models from Colab
  - Training scripts and configurations
  - Training results and metrics

## Workflow

### Standard Development Flow

```
1. development → 2. testing → 3. main
```

#### Step 1: Development
```bash
# Start from development branch
git checkout development
git pull origin development

# Create feature branch (optional but recommended)
git checkout -b feature/new-feature

# Make changes, commit
git add .
git commit -m "Add new feature"

# Push to development
git push origin feature/new-feature
# Or merge to development if working directly
```

#### Step 2: Testing
```bash
# Merge development → testing
git checkout testing
git pull origin testing
git merge development
git push origin testing

# Test the code thoroughly
# Fix any issues found
# If fixes needed, go back to development
```

#### Step 3: Production
```bash
# Merge testing → main (only after testing passes)
git checkout main
git pull origin main
git merge testing
git push origin main

# Tag the release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Colab Training Flow

```
colab-training → (optional) → development → testing → main
```

```bash
# Train model in Colab
# Commit trained model to colab-training branch

# When ready, merge to development
git checkout development
git merge colab-training

# Then follow standard flow: dev → testing → main
```

## Branch Protection Rules (GitHub Settings)

### Recommended Settings:

#### `main` Branch
- ✅ Require pull request reviews
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Do not allow force pushes
- ✅ Do not allow deletions

#### `testing` Branch
- ✅ Require pull request reviews (optional)
- ✅ Require status checks to pass
- ⚠️ Allow force pushes (for quick fixes)

#### `development` Branch
- ⚠️ Allow force pushes
- ⚠️ Allow deletions
- No restrictions (free development)

## Quick Reference Commands

### Creating Feature Branches
```bash
# From development
git checkout development
git pull
git checkout -b feature/feature-name
# Work on feature
git push origin feature/feature-name
```

### Merging Between Environments
```bash
# Development → Testing
git checkout testing
git merge development
git push origin testing

# Testing → Production
git checkout main
git merge testing
git push origin main
```

### Hotfixes (Emergency Production Fixes)
```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/critical-fix
# Fix the issue
git commit -m "Fix critical issue"
git checkout main
git merge hotfix/critical-fix
git push origin main

# Also merge back to development and testing
git checkout development
git merge hotfix/critical-fix
git checkout testing
git merge hotfix/critical-fix
```

## Best Practices

### ✅ DO:
- Always pull before starting work
- Create feature branches for major changes
- Test thoroughly in development before moving to testing
- Write clear commit messages
- Keep `main` branch stable and deployable
- Merge sequentially: dev → testing → main

### ❌ DON'T:
- Push directly to `main` without testing
- Skip the testing environment
- Force push to `main` or `testing`
- Commit broken code to `main`
- Mix development and production code

## Environment-Specific Configurations

### Development
- Debug mode enabled
- Verbose logging
- Experimental features allowed
- Can break things

### Testing
- Production-like configuration
- Full test suite runs
- Performance testing
- Security checks

### Production
- Optimized settings
- Error logging only
- Stable features only
- Production-ready models

## Current Branch Status

- ✅ `main` - Production (stable)
- ✅ `testing` - Testing environment (ready)
- ✅ `development` - Development environment (ready)
- ✅ `colab-training` - Colab training branch (ready)

## Getting Started

1. **For new development**: Checkout `development` branch
2. **For testing**: Checkout `testing` branch
3. **For production**: Use `main` branch
4. **For training**: Use `colab-training` branch

---

**Remember**: `development` is where you can break things. `testing` validates. `main` is production-ready!
