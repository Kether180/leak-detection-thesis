#!/bin/bash
clear
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Synthetic Generation Progress                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

no=$(ls synthetic_generation/experiment_b/no_leak/*.jpg 2>/dev/null | wc -l)
oil=$(ls synthetic_generation/experiment_b/oil_leak/*.jpg 2>/dev/null | wc -l)
water=$(ls synthetic_generation/experiment_b/water_leak/*.jpg 2>/dev/null | wc -l)
total=$((no + oil + water))

printf "no_leak:    %4d / 1400  [%3d%%] " $no $((no * 100 / 1400))
[ $no -eq 1400 ] && echo "✅" || echo "⏳"

printf "oil_leak:   %4d / 1400  [%3d%%] " $oil $((oil * 100 / 1400))
[ $oil -eq 1400 ] && echo "✅" || echo "⏳"

printf "water_leak: %4d / 1400  [%3d%%] " $water $((water * 100 / 1400))
[ $water -eq 1400 ] && echo "✅" || echo "⏳"

echo "────────────────────────────────────────────"
printf "Total:      %4d / 4200  [%3d%%]\n" $total $((total * 100 / 4200))
echo ""

echo "Running Jobs:"
squeue -u gega --format="%.8i %.10P %.12j %.2t %.10M %.6D %R" | head -6

echo ""
echo "Last updated: $(date)"
