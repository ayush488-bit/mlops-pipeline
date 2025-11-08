#!/bin/bash

# Reset Git Workflow Script
# This script resets your local Git state to match GitHub exactly

echo "🔄 Resetting Git Workflow..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Switch to dev (safest branch)
echo -e "${BLUE}Step 1: Switching to dev branch...${NC}"
git checkout dev
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to switch to dev${NC}"
    exit 1
fi
echo -e "${GREEN}✅ On dev branch${NC}"
echo ""

# Step 2: Delete local main and staging
echo -e "${BLUE}Step 2: Deleting local main and staging branches...${NC}"
git branch -D main staging 2>/dev/null
echo -e "${GREEN}✅ Local branches deleted${NC}"
echo ""

# Step 3: Fetch latest from GitHub
echo -e "${BLUE}Step 3: Fetching latest from GitHub...${NC}"
git fetch origin
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to fetch from GitHub${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Fetched latest from GitHub${NC}"
echo ""

# Step 4: Recreate main from remote
echo -e "${BLUE}Step 4: Recreating main branch from GitHub...${NC}"
git checkout -b main origin/main
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to create main branch${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Main branch created${NC}"
echo ""

# Step 5: Recreate staging from remote
echo -e "${BLUE}Step 5: Recreating staging branch from GitHub...${NC}"
git checkout -b staging origin/staging
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to create staging branch${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Staging branch created${NC}"
echo ""

# Step 6: Switch back to dev
echo -e "${BLUE}Step 6: Switching back to dev...${NC}"
git checkout dev
echo -e "${GREEN}✅ Back on dev branch${NC}"
echo ""

# Step 7: Sync all branches (make them identical)
echo -e "${BLUE}Step 7: Syncing all branches...${NC}"

# Update staging to match dev
git checkout staging
git merge dev --no-ff -m "sync: merge dev into staging"
git push origin staging

# Update main to match staging
git checkout main
git merge staging --no-ff -m "sync: merge staging into main"
git push origin main

# Back to dev
git checkout dev

echo -e "${GREEN}✅ All branches synced${NC}"
echo ""

# Step 8: Verify
echo -e "${BLUE}Step 8: Verifying setup...${NC}"
echo ""
echo -e "${YELLOW}Branch status:${NC}"
git branch -vv
echo ""

# Summary
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Git Workflow Reset Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📊 Current State:${NC}"
echo "   • All branches synced with GitHub"
echo "   • All branches have identical code"
echo "   • Currently on: $(git branch --show-current)"
echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo "   1. Read COMPLETE_GIT_GUIDE.md for detailed workflow"
echo "   2. Create a feature branch: git checkout -b feature/my-feature"
echo "   3. Make changes and commit"
echo "   4. Push and create Pull Request"
echo ""
echo -e "${BLUE}🎯 You're ready to start developing!${NC}"
echo ""
