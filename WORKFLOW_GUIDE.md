# Workflow Guide - Quick Reference

## 🎯 Quick Start

### I want to develop new features
```bash
git checkout development
# Make your changes
git add .
git commit -m "Add new feature"
git push origin development
```

### I want to test my code
```bash
# Merge development → testing
git checkout testing
git merge development
git push origin testing
# Test thoroughly, then merge to main if all good
```

### I want to deploy to production
```bash
# Only after testing passes!
git checkout main
git merge testing
git push origin main
```

## 📋 Common Workflows

### Feature Development
```bash
# 1. Start from development
git checkout development
git pull

# 2. Create feature branch (optional)
git checkout -b feature/my-feature

# 3. Develop
# ... make changes ...
git add .
git commit -m "Implement feature X"

# 4. Push feature branch
git push origin feature/my-feature

# 5. Merge to development (or work directly on development)
git checkout development
git merge feature/my-feature
git push origin development
```

### Testing Flow
```bash
# 1. Merge development → testing
git checkout testing
git pull
git merge development
git push origin testing

# 2. Test the code
# ... run tests, check functionality ...

# 3. If issues found, fix in development
git checkout development
# ... fix issues ...
git push origin development

# 4. Re-merge to testing
git checkout testing
git merge development
git push origin testing

# 5. When tests pass, merge to main
git checkout main
git merge testing
git push origin main
```

### Colab Training → Production
```bash
# 1. Train model in Colab (on colab-training branch)
# 2. Merge to development
git checkout development
git merge colab-training
git push origin development

# 3. Follow standard flow: dev → testing → main
```

### Hotfix (Emergency Production Fix)
```bash
# 1. Create hotfix from main
git checkout main
git checkout -b hotfix/critical-bug

# 2. Fix the issue
# ... make fix ...
git add .
git commit -m "Fix critical bug"
git push origin hotfix/critical-bug

# 3. Merge to main
git checkout main
git merge hotfix/critical-bug
git push origin main

# 4. Merge back to development and testing
git checkout development
git merge hotfix/critical-bug
git push origin development

git checkout testing
git merge hotfix/critical-bug
git push origin testing
```

## 🔄 Branch Relationships

```
main (Production)
  ↑ merge when tested
testing (Staging)
  ↑ merge when ready
development (Dev)
  ↑ merge when complete
feature branches / colab-training
```

## ⚠️ Important Rules

1. **Never push directly to `main`** - Always go through testing first
2. **`development` can break** - That's its purpose
3. **`testing` should be stable** - Fix issues before merging to main
4. **`main` is production** - Only stable, tested code

## 📊 Branch Status

| Branch | Purpose | Stability | Can Break? |
|--------|---------|-----------|------------|
| `main` | Production | ✅ Stable | ❌ No |
| `testing` | Pre-production | ✅ Stable | ⚠️ Shouldn't |
| `development` | Active dev | ⚠️ Unstable | ✅ Yes |
| `colab-training` | Model training | ⚠️ Experimental | ✅ Yes |

## 🚀 Getting Help

- See `BRANCHING_STRATEGY.md` for detailed strategy
- See branch-specific README files for environment details
- Check GitHub branch protection settings

---

**Current branch**: Check with `git branch`
