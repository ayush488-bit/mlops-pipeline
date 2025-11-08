# 🌳 Git Workflow Guide

Professional branching strategy for MLOps pipeline development and deployment.

---

## 📋 Branch Structure

```
main (production)
  ↑
staging (pre-production)
  ↑
dev (development)
  ↑
feature/* (feature branches)
```

---

## 🌿 Branch Descriptions

### `main` - Production Branch
- **Purpose**: Production-ready code
- **Protection**: Protected, requires PR approval
- **Deployment**: Auto-deploys to production
- **Stability**: Highest - only tested, approved code
- **Merges from**: `staging` only

### `staging` - Pre-Production Branch
- **Purpose**: Final testing before production
- **Protection**: Protected, requires PR approval
- **Deployment**: Auto-deploys to staging environment
- **Stability**: High - thoroughly tested code
- **Merges from**: `dev` only

### `dev` - Development Branch
- **Purpose**: Integration of features
- **Protection**: Protected, requires PR review
- **Deployment**: Auto-deploys to dev environment
- **Stability**: Medium - tested features
- **Merges from**: `feature/*` branches

### `feature/*` - Feature Branches
- **Purpose**: Individual feature development
- **Naming**: `feature/feature-name` or `feature/issue-123`
- **Protection**: None
- **Stability**: Low - work in progress
- **Merges to**: `dev` only

---

## 🚀 Workflow Steps

### 1. Create Feature Branch

```bash
# Start from dev
git checkout dev
git pull origin dev

# Create feature branch
git checkout -b feature/add-new-model
```

### 2. Develop Feature

```bash
# Make changes
git add .
git commit -m "feat: add new model architecture"

# Push to remote
git push origin feature/add-new-model
```

### 3. Merge to Dev

```bash
# Create Pull Request: feature/add-new-model → dev
# After review and approval, merge

# Or via command line:
git checkout dev
git pull origin dev
git merge feature/add-new-model
git push origin dev

# Delete feature branch
git branch -d feature/add-new-model
git push origin --delete feature/add-new-model
```

### 4. Promote to Staging

```bash
# Create Pull Request: dev → staging
# After testing in dev, merge to staging

git checkout staging
git pull origin staging
git merge dev
git push origin staging
```

### 5. Deploy to Production

```bash
# Create Pull Request: staging → main
# After testing in staging, merge to main

git checkout main
git pull origin main
git merge staging
git push origin main

# Tag the release
git tag -a v1.0.0 -m "Release v1.0.0: Initial production release"
git push origin v1.0.0
```

---

## 📝 Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples

```bash
# Feature
git commit -m "feat(monitoring): add drift detection with KS test"

# Bug fix
git commit -m "fix(api): resolve feature engineer loading issue"

# Documentation
git commit -m "docs: add troubleshooting guide"

# Refactor
git commit -m "refactor(training): improve beautiful output formatting"

# Performance
git commit -m "perf(prediction): optimize feature transformation"
```

---

## 🔒 Branch Protection Rules

### `main` Branch
- ✅ Require pull request reviews (2 approvals)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Include administrators
- ✅ Restrict who can push
- ✅ Require signed commits

### `staging` Branch
- ✅ Require pull request reviews (1 approval)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date

### `dev` Branch
- ✅ Require pull request reviews (1 approval)
- ✅ Require status checks to pass

---

## 🏷️ Tagging Strategy

### Semantic Versioning

```
v<MAJOR>.<MINOR>.<PATCH>
```

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Examples

```bash
# Initial release
git tag -a v1.0.0 -m "Release v1.0.0: Initial production release"

# New feature
git tag -a v1.1.0 -m "Release v1.1.0: Add drift detection"

# Bug fix
git tag -a v1.1.1 -m "Release v1.1.1: Fix API server port issue"

# Breaking change
git tag -a v2.0.0 -m "Release v2.0.0: New model architecture"

# Push tags
git push origin --tags
```

---

## 🔄 Hotfix Workflow

For urgent production fixes:

```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/critical-bug

# Fix the issue
git add .
git commit -m "fix: resolve critical production bug"

# Merge to main
git checkout main
git merge hotfix/critical-bug
git push origin main

# Tag the hotfix
git tag -a v1.0.1 -m "Hotfix v1.0.1: Critical bug fix"
git push origin v1.0.1

# Merge back to staging and dev
git checkout staging
git merge main
git push origin staging

git checkout dev
git merge staging
git push origin dev

# Delete hotfix branch
git branch -d hotfix/critical-bug
```

---

## 📊 Release Process

### 1. Prepare Release

```bash
# Update version in files
# Update CHANGELOG.md
# Update README.md if needed

git add .
git commit -m "chore: prepare release v1.1.0"
```

### 2. Create Release Branch

```bash
git checkout -b release/v1.1.0
git push origin release/v1.1.0
```

### 3. Test Release

```bash
# Run all tests
python test_pipeline.py

# Test training
python train_beautiful.py

# Test production server
python 8_deployment/serve.py
```

### 4. Merge to Main

```bash
# Create PR: release/v1.1.0 → main
# After approval, merge

git checkout main
git merge release/v1.1.0
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin main --tags
```

### 5. Merge Back to Dev

```bash
git checkout dev
git merge main
git push origin dev

# Delete release branch
git branch -d release/v1.1.0
git push origin --delete release/v1.1.0
```

---

## 🔍 Code Review Checklist

### For Reviewers

- [ ] Code follows project style guidelines
- [ ] Tests pass (`python test_pipeline.py`)
- [ ] Documentation updated (README, TROUBLESHOOTING)
- [ ] No hardcoded credentials or secrets
- [ ] Error handling is appropriate
- [ ] Performance impact considered
- [ ] Backward compatibility maintained
- [ ] Commit messages follow convention

### For Authors

- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No debug code or comments
- [ ] Branch is up to date with target
- [ ] CI/CD checks pass
- [ ] Screenshots added (if UI changes)

---

## 🚦 CI/CD Pipeline

### On Push to `dev`

```yaml
- Run linting (flake8, black)
- Run tests (pytest)
- Build artifacts
- Deploy to dev environment
```

### On PR to `staging`

```yaml
- Run all tests
- Run integration tests
- Performance benchmarks
- Security scan
- Deploy to staging environment
```

### On PR to `main`

```yaml
- Run all tests
- Run integration tests
- Run end-to-end tests
- Security scan
- Manual approval required
- Deploy to production
- Create GitHub release
```

---

## 📦 Repository Setup

### Initial Setup

```bash
# Clone repository
git clone https://github.com/ayush488-bit/mlops-pipeline.git
cd mlops-pipeline

# Set up branches
git checkout -b dev
git push origin dev

git checkout -b staging
git push origin staging

git checkout -b main
git push origin main

# Set main as default branch on GitHub
```

### Local Setup

```bash
# Clone repository
git clone https://github.com/ayush488-bit/mlops-pipeline.git
cd mlops-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up git hooks (optional)
pre-commit install
```

---

## 🛠️ Useful Git Commands

### Branch Management

```bash
# List all branches
git branch -a

# Switch branches
git checkout dev

# Create and switch
git checkout -b feature/new-feature

# Delete local branch
git branch -d feature/old-feature

# Delete remote branch
git push origin --delete feature/old-feature
```

### Syncing

```bash
# Update local branches
git fetch --all

# Pull latest changes
git pull origin dev

# Rebase on dev
git rebase dev

# Push with force (after rebase)
git push --force-with-lease
```

### Stashing

```bash
# Stash changes
git stash

# List stashes
git stash list

# Apply stash
git stash apply

# Pop stash
git stash pop
```

### History

```bash
# View commit history
git log --oneline --graph --all

# View file history
git log --follow filename.py

# View changes
git diff dev..staging
```

---

## 📋 Quick Reference

### Feature Development

```bash
git checkout dev
git pull origin dev
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
# Create PR to dev
```

### Bug Fix

```bash
git checkout dev
git pull origin dev
git checkout -b fix/bug-description
# ... fix bug ...
git add .
git commit -m "fix: resolve bug description"
git push origin fix/bug-description
# Create PR to dev
```

### Release

```bash
git checkout staging
git pull origin staging
# Create PR to main
# After approval and merge:
git checkout main
git pull origin main
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin --tags
```

---

## 🎯 Best Practices

1. **Keep branches up to date**: Regularly merge from parent branch
2. **Small, focused commits**: One logical change per commit
3. **Descriptive commit messages**: Follow conventional commits
4. **Test before pushing**: Run tests locally
5. **Review your own PR**: Self-review before requesting review
6. **Delete merged branches**: Keep repository clean
7. **Use tags for releases**: Semantic versioning
8. **Document breaking changes**: In commit message and CHANGELOG
9. **Never force push to protected branches**: Use `--force-with-lease` on feature branches only
10. **Keep feature branches short-lived**: Merge within 1-2 days

---

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Git Best Practices](https://sethrobertson.github.io/GitBestPractices/)

---

**Follow this workflow to maintain a professional, organized codebase!** 🚀
