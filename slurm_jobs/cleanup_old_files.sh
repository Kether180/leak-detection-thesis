#!/bin/bash
echo "Cleaning up old files..."

# Remove old failed generation logs
rm -f generation_36095.err
rm -f generation_36095.log
rm -f generation_test_36044.err
rm -f generation_test_36044.log
rm -f generation_full_36540.err
rm -f generation_full_36540.log

# Remove old cleanup scripts
rm -f 01_quarantine_corrupted.sh
rm -f cleanup_project.sh
rm -f project_cleanup.sh
rm -f run_all_generation.sh
rm -f setup_environment.sh.save

# Remove old test directory (if exists and you don't need it)
# rm -rf no_leak_images_clean/

echo "✅ Cleanup complete!"
echo ""
echo "Kept important files:"
echo "  - Current generation job files (36542)"
echo "  - Filter scripts for tomorrow"
echo "  - All datasets and experiments"
