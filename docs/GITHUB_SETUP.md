# GitHub Repository Setup Guide

## Current Project State

✅ **Git initialized** - Repository is ready  
✅ **Initial commit created** - All essential files committed  
✅ **.gitignore configured** - Prevents committing unnecessary files  
✅ **README.md created** - Project documentation ready  
✅ **Requirements.txt added** - Dependencies documented  

## Next Steps: Push to GitHub

### Option 1: Create Repository on GitHub Website

1. **Go to GitHub**: https://github.com/new
2. **Create a new repository**:
   - Repository name: `HooperAI` (or your preferred name)
   - Description: "Basketball detection and analytics using YOLO"
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. **Copy the repository URL** (e.g., `https://github.com/YOUR_USERNAME/HooperAI.git`)

### Option 2: Use GitHub CLI (if installed)

```bash
gh repo create HooperAI --public --source=. --remote=origin --push
```

### Option 3: Manual Setup (Command Line)

After creating the repository on GitHub:

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/HooperAI.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## What's Included in the Repository

### ✅ Committed Files:
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `models/` - Trained model weights (4 .pt files)
- `Hooper-2/` - Dataset 1 (ball, hoop, player)
- `HooperAnalytic-1/` - Dataset 2 (ball, basket, player)
- `new_vids/` - Video files for processing
- `CLEANUP_SUMMARY.md` - Cleanup documentation
- `UNNECESSARY_FILES.md` - File analysis

### ❌ Excluded (via .gitignore):
- `.venv1/` - Virtual environment
- `runs/` - Training outputs
- `archives/` - Archived files
- `.idea/` - IDE files
- `*.cache` - Cache files
- `__pycache__/` - Python cache

## Repository Size Considerations

⚠️ **Note**: The repository includes:
- **Model weights** (~100-500 MB total)
- **Dataset images** (~50-200 MB)
- **Video files** (size varies)

If the repository becomes too large, consider:
1. Using Git LFS for large files (`.pt`, `.mp4`)
2. Hosting datasets separately
3. Using GitHub Releases for model weights

## Setting Up Git LFS (Optional - for large files)

If you want to use Git LFS for model weights:

```bash
# Install Git LFS (if not installed)
# Windows: Download from https://git-lfs.github.com/
# Or: winget install GitHub.GitLFS

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.mp4"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Add Git LFS tracking for large files"
git push
```

## Verification

After pushing, verify your repository:
1. Visit your GitHub repository URL
2. Check that all files are present
3. Verify README displays correctly
4. Ensure .gitignore is working (no unnecessary files)

## Future Workflow

```bash
# Make changes
git add .
git commit -m "Description of changes"
git push
```

## Troubleshooting

### Authentication Issues
- Use Personal Access Token instead of password
- Or set up SSH keys for GitHub

### Large File Issues
- Use Git LFS for files > 100MB
- Or exclude large files from repository

### Push Rejected
- Pull first: `git pull origin main --rebase`
- Resolve conflicts if any
- Push again: `git push origin main`
