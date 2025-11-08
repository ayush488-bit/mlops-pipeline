#!/bin/bash

# MLOps Pipeline - GitHub Setup Script
# This script sets up the repository with professional branching

echo "🚀 Setting up MLOps Pipeline on GitHub..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Add all files
echo -e "${BLUE}Step 1: Adding all files...${NC}"
git add .
echo -e "${GREEN}✅ Files added${NC}"
echo ""

# Step 2: Initial commit
echo -e "${BLUE}Step 2: Creating initial commit...${NC}"
git commit -m "feat: initial commit - complete MLOps pipeline with 12 phases

Features:
- All 12 MLOps phases implemented
- Beautiful terminal output with Rich library
- Production API with FastAPI
- Monitoring with drift detection
- Automated rollback system
- Continuous learning pipeline
- Comprehensive documentation (README, QUICKSTART, TROUBLESHOOTING)
- Professional Git workflow guide"

echo -e "${GREEN}✅ Initial commit created${NC}"
echo ""

# Step 3: Rename to main
echo -e "${BLUE}Step 3: Renaming branch to main...${NC}"
git branch -M main
echo -e "${GREEN}✅ Branch renamed to main${NC}"
echo ""

# Step 4: Add remote
echo -e "${BLUE}Step 4: Adding remote repository...${NC}"
git remote add origin https://github.com/ayush488-bit/mlops-pipeline.git
echo -e "${GREEN}✅ Remote added${NC}"
echo ""

# Step 5: Push to main
echo -e "${BLUE}Step 5: Pushing to main branch...${NC}"
git push -u origin main
echo -e "${GREEN}✅ Pushed to main${NC}"
echo ""

# Step 6: Create dev branch
echo -e "${BLUE}Step 6: Creating dev branch...${NC}"
git checkout -b dev
git push -u origin dev
echo -e "${GREEN}✅ Dev branch created and pushed${NC}"
echo ""

# Step 7: Create staging branch
echo -e "${BLUE}Step 7: Creating staging branch...${NC}"
git checkout -b staging
git push -u origin staging
echo -e "${GREEN}✅ Staging branch created and pushed${NC}"
echo ""

# Step 8: Create initial tag
echo -e "${BLUE}Step 8: Creating initial release tag...${NC}"
git checkout main
git tag -a v1.0.0 -m "Release v1.0.0: Complete MLOps Pipeline

Features:
- All 12 MLOps phases implemented
- Beautiful terminal output with Rich
- Production API with FastAPI
- Monitoring with drift detection
- Automated rollback system
- Continuous learning pipeline
- Comprehensive documentation"

git push origin v1.0.0
echo -e "${GREEN}✅ Tag v1.0.0 created and pushed${NC}"
echo ""

# Step 9: Switch back to dev
echo -e "${BLUE}Step 9: Switching to dev branch...${NC}"
git checkout dev
echo -e "${GREEN}✅ Now on dev branch${NC}"
echo ""

# Summary
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ GitHub Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📊 Repository Structure:${NC}"
echo "   main (production)    ← Protected, production-ready"
echo "   staging (pre-prod)   ← Protected, testing"
echo "   dev (development)    ← Default, active development"
echo ""
echo -e "${YELLOW}🔗 Repository URL:${NC}"
echo "   https://github.com/ayush488-bit/mlops-pipeline"
echo ""
echo -e "${YELLOW}🏷️  Release Tag:${NC}"
echo "   v1.0.0"
echo ""
echo -e "${YELLOW}📚 Next Steps:${NC}"
echo "   1. Set up branch protection on GitHub"
echo "   2. Create GitHub release from v1.0.0 tag"
echo "   3. Add repository description and topics"
echo "   4. Review GITHUB_SETUP.md for details"
echo ""
echo -e "${BLUE}🎯 Current branch: $(git branch --show-current)${NC}"
echo ""
